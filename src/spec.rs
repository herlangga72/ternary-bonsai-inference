//! Speculative decoding loop: dspark draft + target verify.
//!
//! The target model stays authoritative; the drafter only proposes. With greedy
//! target sampling the emitted sequence must be token-identical to plain decode,
//! which is the correctness oracle (it does not depend on draft quality).
//!
//! Per round, following the reference convention (`common/speculative.cpp`):
//!
//! ```text
//!   forward  pending                    at position n        -> L0
//!   draft    [pending, MASK x (k-1)]    at positions n+1..    -> d0..d(k-1)
//!   verify   d_j while argmax(L_j) == d_j, forwarding only what is accepted
//!   emit     d0..d(a-1) plus the target token e = argmax(La)
//!   next     pending = e, n = n + a + 1
//! ```
//!
//! Verification is **streaming**: drafts are checked and forwarded one at a
//! time, stopping at the first rejection. Rejected drafts are never forwarded,
//! so the caches already end at the accepted prefix and no rollback (nor GDN
//! state snapshot) is needed. The cost is one forward per emitted token, so
//! this is correctness-first: the speedup needs a *batched* verify pass, which
//! only pays where decode is weight-bandwidth bound (see `notes/dspark-port-plan.md`).

#![allow(dead_code)]

use crate::dspark::{DraftCache, Dspark};
use crate::forward::Decoder;

/// Proposes draft blocks from the sidecar and mirrors the target's features.
pub struct Drafter {
    pub ds: Dspark,
    pub dcache: DraftCache,
    /// target layer indices whose inputs feed the encoder
    pub taps_want: Vec<u32>,
    pub p_min: f32,
}

impl Drafter {
    pub fn new(dspark_path: &str, n_ctx: usize, p_min: f32) -> Result<Drafter, String> {
        let ds = Dspark::open(dspark_path)?;
        let taps_want = ds.cfg.target_layers.clone();
        // the draft cache only needs the sidecar's trained context length
        let n_ctx = n_ctx.min(ds.cfg.context_length);
        let dcache = DraftCache::new(&ds.cfg, n_ctx);
        Ok(Drafter {
            ds,
            dcache,
            taps_want,
            p_min,
        })
    }

    /// Inject the target features for one committed position into the draft KV
    /// cache. `taps[i]` is layer `taps_want[i]`'s input hidden state.
    pub fn observe(&mut self, taps: &[Vec<f32>], pos: usize) -> Result<(), String> {
        let ne = self.ds.cfg.n_embd;
        let mut feat = Vec::with_capacity(self.taps_want.len() * ne);
        for t in taps {
            if t.len() != ne {
                return Err(format!("drafter observe: tap width {} != {ne}", t.len()));
            }
            feat.extend_from_slice(t);
        }
        let inp_g = self.ds.encode(&feat, 1)?;
        self.ds.inject(&mut self.dcache, &inp_g, &[pos])
    }

    /// Draft up to `n_draft` tokens for the block starting at `block_start`
    /// (the position right after `id_last`), anchored on `id_last`. Greedy per
    /// position, truncated where confidence < p_min. The draft K/V written at
    /// the block positions is discarded again (accepted positions are re-observed
    /// from the target during verification).
    pub fn propose(
        &mut self,
        id_last: u32,
        block_start: usize,
        n_draft: usize,
    ) -> Result<Vec<u32>, String> {
        if n_draft == 0 || n_draft > self.ds.cfg.block_size {
            return Err(format!(
                "drafter propose: n_draft {n_draft} out of 1..={}",
                self.ds.cfg.block_size
            ));
        }
        let block = self.ds.draft_block(&mut self.dcache, id_last, block_start)?;
        self.dcache.truncate(block_start);
        let n_vocab = self.ds.cfg.n_vocab;
        // anchor-first drafts read block positions 0..; the anchorless convention
        // treats position 0 as a bonus anchor and reads 1..
        let i_beg = if std::env::var("BONSAI_DSPARK_ANCHORLESS").is_ok() {
            1
        } else {
            0
        };
        let mut out = Vec::with_capacity(n_draft);
        for t in 0..n_draft {
            let idx = i_beg + t;
            if idx >= block.conf.len() {
                break;
            }
            if block.conf[idx] < self.p_min {
                break;
            }
            out.push(argmax(&block.logits[idx * n_vocab..(idx + 1) * n_vocab]) as u32);
        }
        Ok(out)
    }
}

/// Greedy argmax of a logit vector.
pub fn argmax(v: &[f32]) -> usize {
    let mut best = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, x) in v.iter().enumerate() {
        if *x > bv {
            bv = *x;
            best = i;
        }
    }
    best
}

/// Result of one speculative round.
pub struct RoundOut {
    /// tokens to emit, in order (accepted drafts followed by the target token)
    pub emitted: Vec<u32>,
    /// the raw draft proposals (before verification)
    pub drafts: Vec<u32>,
    /// number of draft tokens accepted
    pub accepted: usize,
    /// the target's own next token (the new pending), with its logits
    pub pending: u32,
    pub logits: Vec<f32>,
    /// target forwards performed this round
    pub target_forwards: usize,
}

/// One speculative round. `dec` must hold positions `0..n_past`; `pending` is
/// the last emitted token (not yet forwarded); `logits` are the target logits
/// at `n_past - 1` that predict `pending`. Leaves the caches holding the
/// accepted prefix, and returns the new pending token and its logits.
pub fn round(
    dec: &mut Decoder,
    drafter: &mut Drafter,
    pending: u32,
    n_past: usize,
    n_draft: usize,
) -> Result<RoundOut, String> {
    let taps_want = drafter.taps_want.clone();
    let n_taps = taps_want.len();
    let mut taps: Vec<Vec<f32>> = vec![Vec::new(); n_taps];

    // forward the pending token, mirroring its features into the drafter
    let h = dec.forward_hidden_taps(pending, n_past, &taps_want, &mut taps)?;
    drafter.observe(&taps, n_past)?;
    let mut lg = dec.head_logits(&h)?;
    let mut forwards = 1usize;

    // draft the block (positions n_past+1 ..)
    let drafts = drafter.propose(pending, n_past + 1, n_draft)?;

    // streaming verify: forward accepted drafts, stop at the first mismatch
    let mut a = 0usize;
    for (i, &d) in drafts.iter().enumerate() {
        if argmax(&lg) != d as usize {
            break;
        }
        a += 1;
        let ht = dec.forward_hidden_taps(d, n_past + 1 + i, &taps_want, &mut taps)?;
        drafter.observe(&taps, n_past + 1 + i)?;
        lg = dec.head_logits(&ht)?;
        forwards += 1;
    }

    let mut emitted: Vec<u32> = drafts[..a].to_vec();
    let pending_next = argmax(&lg) as u32;
    emitted.push(pending_next);

    Ok(RoundOut {
        emitted,
        drafts,
        accepted: a,
        pending: pending_next,
        logits: lg,
        target_forwards: forwards,
    })
}
