//! P1: N-wide prompt-prefill plumbing (activation tile + batched embed gather).
//!
//! This is the first phase of `notes/prefill-plan.md`. Today prefill feeds the
//! prompt through the single-stream decode loop one token at a time, so an
//! N-token prompt reads the whole ~7.1 GB weight stream N times. The plan is to
//! batch prefill so each weight block is read once and produces N outputs. That
//! is *later* phases (P2..P6, which touch the PQ2 kernels and change working
//! paths). **This module ships only correctness plumbing**:
//!
//! * an N-wide activation-tile layout, and
//! * a batched `token_embd.weight` gather that fills one tile from N token ids.
//!
//! plus a pure-Rust CPU reference (`ref_batched_hidden`) that reproduces the
//! per-token forward result so later phases can diff a batched GPU forward
//! against it.
//!
//! Everything here is **additive and inert** when `BONSAI_BATCH` is unset or 0:
//! no decode/forward path calls into it, so default behavior is byte-identical
//! and untouched.
//!
//! ## Tile layout
//!
//! The tile is row-major `[token][feature]` (one contiguous `n_feat` row per
//! token). That is the cheapest layout for today's host-side, per-row fetch of
//! `token_embd.weight` (each embedding row is contiguous, matching
//! `Weights::row_f32`) and it is what the P2 PQ2 N-column GEMM will address
//! with the weight row held in registers across the N activation columns. If a
//! later kernel proves cheaper with `[feature][token]`, the transpose is
//! confined to this one type.

#![allow(dead_code)]

use crate::forward::Decoder;
use crate::weights::Weights;

/// GGUF tensor holding the token embeddings (host-side row fetch, per
/// `forward.rs` convention: embeddings are never uploaded to the device).
pub const EMBED_TENSOR: &str = "token_embd.weight";

/// Batch width requested via `BONSAI_BATCH` (the number of prompt tokens batched
/// per prefill pass). Default 0 = today's single-token behavior is untouched.
/// Values 0 and 1 mean "no batching" (a 1-token batch has nothing to gain), so
/// they both read as 0 (disabled).
pub fn batch_width() -> usize {
    std::env::var("BONSAI_BATCH")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&n| n >= 2)
        .unwrap_or(0)
}

/// Whether the batched prefill path is enabled (`BONSAI_BATCH` >= 2).
pub fn batch_enabled() -> bool {
    batch_width() > 1
}

/// An N-wide activation tile over one model width (e.g. `n_embd` = 5120).
///
/// Layout: row-major `[token][feature]`, so `data[t * n_feat + f]` is token
/// `t`, feature `f`, and token `t`'s row is the contiguous slice
/// `data[t*n_feat .. (t+1)*n_feat]`.
#[derive(Debug, Clone, PartialEq)]
pub struct Tile {
    n_tokens: usize,
    n_feat: usize,
    data: Vec<f32>,
}

impl Tile {
    /// An empty tile (all zeros) of `n_tokens` x `n_feat`.
    pub fn new(n_tokens: usize, n_feat: usize) -> Tile {
        Tile {
            n_tokens,
            n_feat,
            data: vec![0.0; n_tokens.saturating_mul(n_feat)],
        }
    }

    /// Build a tile from per-token rows, validating their widths.
    pub fn from_rows(n_feat: usize, rows: Vec<Vec<f32>>) -> Result<Tile, String> {
        let n_tokens = rows.len();
        let mut t = Tile::new(n_tokens, n_feat);
        for (k, row) in rows.into_iter().enumerate() {
            t.set_row(k, &row)?;
        }
        Ok(t)
    }

    pub fn n_tokens(&self) -> usize {
        self.n_tokens
    }
    pub fn n_feat(&self) -> usize {
        self.n_feat
    }
    pub fn len(&self) -> usize {
        self.data.len()
    }
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
    /// Flat slice in `[token][feature]` order.
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }
    /// Flat view as little-endian f32 bytes (for a later single device upload).
    pub fn bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(self.data.as_ptr() as *const u8, self.data.len() * 4)
        }
    }

    /// The contiguous activation row for token `t`.
    pub fn row(&self, t: usize) -> Result<&[f32], String> {
        if t >= self.n_tokens {
            return Err(format!("prefill::Tile: token {t} out of range ({} rows)", self.n_tokens));
        }
        let a = t * self.n_feat;
        Ok(&self.data[a..a + self.n_feat])
    }

    /// Overwrite token `t`'s row, validating length.
    pub fn set_row(&mut self, t: usize, row: &[f32]) -> Result<(), String> {
        if t >= self.n_tokens {
            return Err(format!("prefill::Tile: token {t} out of range ({} rows)", self.n_tokens));
        }
        if row.len() != self.n_feat {
            return Err(format!(
                "prefill::Tile: row {t} width {} != tile width {}",
                row.len(),
                self.n_feat
            ));
        }
        let a = t * self.n_feat;
        self.data[a..a + self.n_feat].copy_from_slice(row);
        Ok(())
    }
}

/// Batched embedding gather: read the N `token_embd.weight` rows for `tokens`
/// into one tile. Still host-side and per-row (one `row_f32` fetch per token,
/// exactly as the single-token decode path does) but collected once into a
/// contiguous N-wide tile, ready for a single later upload / GEMM.
pub fn batch_embed(w: &mut Weights, tokens: &[u32]) -> Result<Tile, String> {
    if tokens.is_empty() {
        return Ok(Tile::new(0, 0));
    }
    let n_feat = w.config().n_embd;
    let mut t = Tile::new(tokens.len(), n_feat);
    for (k, &tok) in tokens.iter().enumerate() {
        let row = w.row_f32(EMBED_TENSOR, tok as u64)?;
        t.set_row(k, &row)?;
    }
    Ok(t)
}

/// Pure-Rust CPU reference for a batched prompt prefill.
///
/// Runs the existing single-stream `Decoder::forward_hidden` once per token at
/// positions `0..N` (the numeric anchor the plan preserves) and gathers the N
/// output-normalized hidden vectors into one `[token][feature]` tile. Later
/// phases diff the batched GPU prefill against this tile.
pub fn ref_batched_hidden(dec: &mut Decoder, tokens: &[u32]) -> Result<Tile, String> {
    if tokens.is_empty() {
        return Ok(Tile::new(0, 0));
    }
    let n_feat = dec.cfg.n_embd;
    let mut t = Tile::new(tokens.len(), n_feat);
    for (k, &tok) in tokens.iter().enumerate() {
        let h = dec.forward_hidden(tok, k)?;
        t.set_row(k, &h)?;
    }
    Ok(t)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tile_layout_roundtrips_rows() {
        let mut t = Tile::new(3, 4);
        assert_eq!(t.n_tokens(), 3);
        assert_eq!(t.n_feat(), 4);
        assert_eq!(t.len(), 12);
        assert_eq!(t.row(2).unwrap(), &[0.0; 4]);

        // set one row, read it back, and confirm the flat layout index.
        t.set_row(1, &[1.0, 2.0, 3.0, 4.0]).unwrap();
        assert_eq!(t.row(1).unwrap(), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(t.as_slice()[4..8], [1.0, 2.0, 3.0, 4.0]);
        // bytes are little-endian f32, 4 bytes per element
        assert_eq!(t.bytes().len(), 12 * 4);
    }

    #[test]
    fn tile_validates_row_width_and_range() {
        let mut t = Tile::new(2, 4);
        assert!(t.set_row(0, &[1.0, 2.0, 3.0]).is_err()); // wrong width
        assert!(t.set_row(2, &[0.0; 4]).is_err()); // out of range
        assert!(t.row(5).is_err());
        assert!(t.set_row(1, &[0.0; 4]).is_ok());
    }

    #[test]
    fn from_rows_gathers_equal_width_rows() {
        let rows = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let t = Tile::from_rows(2, rows).unwrap();
        assert_eq!(t.n_tokens(), 3);
        assert_eq!(t.row(0).unwrap(), &[1.0, 2.0]);
        assert_eq!(t.row(2).unwrap(), &[5.0, 6.0]);
        let err = Tile::from_rows(2, vec![vec![1.0, 2.0], vec![9.0]]);
        assert!(err.is_err());
    }

    #[test]
    fn batch_env_gates_off_by_default() {
        // unset => disabled; BONSAI_BATCH=1 => disabled; >=2 => enabled.
        let _old = std::env::var("BONSAI_BATCH").ok();
        unsafe { std::env::remove_var("BONSAI_BATCH") };
        assert!(!batch_enabled());
        assert_eq!(batch_width(), 0);
        unsafe { std::env::set_var("BONSAI_BATCH", "1") };
        assert!(!batch_enabled());
        unsafe { std::env::set_var("BONSAI_BATCH", "4") };
        assert!(batch_enabled());
        assert_eq!(batch_width(), 4);
        unsafe { std::env::set_var("BONSAI_BATCH", "junk") };
        assert!(!batch_enabled());
        match _old {
            Some(v) => unsafe { std::env::set_var("BONSAI_BATCH", v) },
            None => unsafe { std::env::remove_var("BONSAI_BATCH") },
        }
    }
}
