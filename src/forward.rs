//! qwen35 forward-pass pieces (M6).
//!
//! M6-3: full-attention layers (il % 4 == 3). Given the pre-normed input of a
//! single token, compute the gated multi-head attention of llama.cpp's
//! `build_layer_attn`:
//!
//!   qfull = wq @ x                    # rows: per head [q(256) | gate(256)]
//!   q     = rms_norm_rows(q, q_norm)  # per q head, head-major
//!   gate  = second half of each head  # head-major, 6144
//!   k     = rms_norm_rows(wk @ x, k_norm); rope each kv head
//!   v     = wv @ x                    # no norm, no rope
//!   attn  = softmax(q·k^T / sqrt(256)) @ v     # GQA: kv = q / 6 (grouped)
//!   out   = wo @ (attn ⊙ sigmoid(gate))
//!
//! The KV cache is f32, one contiguous `[kv_head][head_dim]` block per token
//! per full-attention layer. Only every `interval`-th layer participates.

#![allow(dead_code)]

use crate::kernels;
use crate::rope::rope_imrope;
use crate::weights::{Qwen35, Weights};

/// Per-layer KV caches for the full-attention layers.
///
/// `k[layer]` and `v[layer]` grow by `n_kv_elems` (n_head_kv * head_dim) floats
/// per appended token; token `t`'s block sits at `[t * n_kv_elems ..)`, laid
/// out as `[kv_head 0 head vector][kv_head 1 head vector]...`.
pub struct AttnCache {
    pub n_kv_elems: usize,
    pub k: Vec<Vec<f32>>,
    pub v: Vec<Vec<f32>>,
}

impl AttnCache {
    pub fn new(cfg: &Qwen35) -> AttnCache {
        let n_kv_elems = cfg.n_head_kv * cfg.n_embd_head;
        AttnCache {
            n_kv_elems,
            k: vec![Vec::new(); cfg.n_layer],
            v: vec![Vec::new(); cfg.n_layer],
        }
    }

    /// Number of cached tokens for one full-attention layer.
    pub fn n_pos(&self, il: usize) -> usize {
        self.k[il].len() / self.n_kv_elems
    }

    /// Append the current token's k/v rows (each n_kv_elems floats, kv-head
    /// major) for layer `il`.
    pub fn append(&mut self, il: usize, krow: &[f32], vrow: &[f32]) {
        debug_assert_eq!(krow.len(), self.n_kv_elems);
        debug_assert_eq!(vrow.len(), self.n_kv_elems);
        self.k[il].extend_from_slice(krow);
        self.v[il].extend_from_slice(vrow);
    }
}

/// Scratch buffers reused across attention calls (avoids per-head allocs).
pub struct AttnScratch {
    pub qfull: Vec<f32>,
    pub q: Vec<f32>,
    pub gate: Vec<f32>,
    pub krow: Vec<f32>,
    pub vrow: Vec<f32>,
    pub scores: Vec<f32>,
    pub probs: Vec<f32>,
    pub out: Vec<f32>,
    pub head_out: Vec<f32>,
}

impl AttnScratch {
    pub fn new(cfg: &Qwen35) -> AttnScratch {
        let hd = cfg.n_embd_head;
        let nh = cfg.n_head;
        AttnScratch {
            qfull: vec![0.0; 2 * nh * hd],
            q: vec![0.0; nh * hd],
            gate: vec![0.0; nh * hd],
            krow: vec![0.0; cfg.n_head_kv * hd],
            vrow: vec![0.0; cfg.n_head_kv * hd],
            scores: Vec::new(),
            probs: Vec::new(),
            out: vec![0.0; nh * hd],
            head_out: vec![0.0; hd],
        }
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// One full-attention layer for a single token.
///
/// `x` is the already `attn_norm`-ed token embedding (length n_embd), `pos` is
/// the absolute token position (its KV block index equals `pos`).
///
/// Returns the layer output before the attention residual is added.
pub fn full_attention_layer(
    w: &mut Weights,
    cfg: &Qwen35,
    il: usize,
    x: &[f32],
    pos: usize,
    cache: &mut AttnCache,
    s: &mut AttnScratch,
) -> Result<Vec<f32>, String> {
    debug_assert!(cfg.is_full_attention(il), "layer {il} is not full attention");
    let n_embd = cfg.n_embd;
    let hd = cfg.n_embd_head;
    let nh = cfg.n_head;
    let n_kv = cfg.n_head_kv;
    let n_rep = nh / n_kv; // grouped GQA: kv head = q head / 6
    let eps = cfg.eps;

    let name = |suffix: &str| cfg.blk_name(il, suffix);

    // ---- projections -------------------------------------------------------
    let qfull = w.matvec(&name("attn_q.weight"), x)?;
    let kraw = w.matvec(&name("attn_k.weight"), x)?;
    let vraw = w.matvec(&name("attn_v.weight"), x)?;
    if qfull.len() != 2 * nh * hd || kraw.len() != n_kv * hd || vraw.len() != n_kv * hd {
        return Err(format!(
            "attention_layer({il}): unexpected projection rows ({} / {} / {})",
            qfull.len(),
            kraw.len(),
            vraw.len()
        ));
    }
    debug_assert_eq!(s.qfull.len(), 2 * nh * hd);
    s.qfull.copy_from_slice(&qfull);
    s.krow.copy_from_slice(&kraw);
    s.vrow.copy_from_slice(&vraw);

    // ---- split [q | gate] per head, then per-head RMSNorm + IMROPE ---------
    // wq rows are interleaved per head: [q(hd) | gate(hd)] head-major
    for h in 0..nh {
        let base = h * 2 * hd;
        s.q[h * hd..(h + 1) * hd].copy_from_slice(&s.qfull[base..base + hd]);
        s.gate[h * hd..(h + 1) * hd].copy_from_slice(&s.qfull[base + hd..base + 2 * hd]);
    }
    let q_norm_w = w.vec_f32(&name("attn_q_norm.weight"))?;
    let mut q = kernels::rms_norm_rows(&s.q, &q_norm_w, hd, eps)?;
    let k_norm_w = w.vec_f32(&name("attn_k_norm.weight"))?;
    let mut k = kernels::rms_norm_rows(&s.krow, &k_norm_w, hd, eps)?;

    for h in 0..nh {
        rope_imrope(
            &mut q[h * hd..(h + 1) * hd],
            pos as i64,
            pos as i64,
            pos as i64,
            pos as i64,
            cfg.n_rot,
            cfg.sections,
            cfg.freq_base,
        );
    }
    for kv in 0..n_kv {
        rope_imrope(
            &mut k[kv * hd..(kv + 1) * hd],
            pos as i64,
            pos as i64,
            pos as i64,
            pos as i64,
            cfg.n_rot,
            cfg.sections,
            cfg.freq_base,
        );
    }

    // ---- store in KV cache (token index == pos) ----------------------------
    cache.append(il, &k, &s.vrow);
    debug_assert_eq!(cache.n_pos(il), pos + 1);

    // ---- GQA attention ------------------------------------------------------
    let scale = 1.0 / (hd as f32).sqrt();
    let n_pos = pos + 1;
    s.scores.resize(n_pos, 0.0);
    s.probs.resize(n_pos, 0.0);
    s.out.fill(0.0);
    let k_cache = &cache.k[il];
    let v_cache = &cache.v[il];
    let n_kv_elems = cache.n_kv_elems;

    for hq in 0..nh {
        let qh = &q[hq * hd..(hq + 1) * hd];
        let kv = hq / n_rep; // grouped GQA, matches ggml mul_mat broadcast
        let kv_off = kv * hd;
        for j in 0..n_pos {
            let kj = &k_cache[j * n_kv_elems + kv_off..j * n_kv_elems + kv_off + hd];
            s.scores[j] = dot(qh, kj) * scale;
        }
        let probs = kernels::softmax_rows(&s.scores, n_pos)?;
        s.probs.copy_from_slice(&probs);

        s.head_out.fill(0.0);
        for j in 0..n_pos {
            let pj = s.probs[j];
            if pj == 0.0 {
                continue;
            }
            let vj = &v_cache[j * n_kv_elems + kv_off..j * n_kv_elems + kv_off + hd];
            for d in 0..hd {
                s.head_out[d] += pj * vj[d];
            }
        }
        s.out[hq * hd..(hq + 1) * hd].copy_from_slice(&s.head_out);
    }

    // ---- gate + output projection ------------------------------------------
    // out = wo @ (attn ⊙ sigmoid(gate))
    kernels::map_inplace(&mut s.gate, kernels::sigmoid);
    for i in 0..s.out.len() {
        s.out[i] *= s.gate[i];
    }
    let wo = w.tensor(&name("attn_output.weight"))?.clone();
    let mut y = vec![0.0f32; n_embd];
    w.matvec_into(&wo, 0, n_embd, &s.out, &mut y)?;
    Ok(y)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mini_cfg() -> Qwen35 {
        Qwen35 {
            n_layer: 4,
            n_embd: 128,
            n_ff: 256,
            n_head: 8,
            n_head_kv: 2,
            n_embd_head: 16,
            n_ctx_train: 4096,
            eps: 1e-6,
            interval: 4,
            n_rot: 8,
            sections: [2, 1, 1, 0],
            freq_base: 1e4,
            ssm_conv_kernel: 4,
            ssm_state: 16,
            ssm_group_count: 2,
            ssm_dt_rank: 8,
            ssm_inner: 128,
        }
    }

    #[test]
    fn cache_grows_per_token_and_indexes() {
        let cfg = mini_cfg();
        let mut c = AttnCache::new(&cfg);
        assert_eq!(c.n_kv_elems, 32);
        assert_eq!(c.n_pos(3), 0);
        let krow: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let vrow: Vec<f32> = (0..32).map(|i| -i as f32).collect();
        c.append(3, &krow, &vrow);
        c.append(3, &krow, &vrow);
        assert_eq!(c.n_pos(3), 2);
        // contiguous per-token blocks, kv-head major
        assert_eq!(c.k[3][0], 0.0);
        assert_eq!(c.k[3][31], 31.0);
        assert_eq!(c.k[3][32], 0.0); // token 1 starts here
        assert_eq!(c.v[3][32 + 16], -16.0); // kv head 1 of token 1
        assert_eq!(c.n_pos(0), 0);
    }
}
