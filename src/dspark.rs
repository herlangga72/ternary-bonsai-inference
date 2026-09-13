//! Dspark speculator (PrismML `dflash`/`dspark` block-denoising drafter).
//!
//! This module ports the draft half of the speculative-decoding path: loading
//! the sidecar GGUF, the mixed-precision kernels it needs (F32, BF16, Q4_1,
//! PQ2_0), and the forward graph. The target half (tapped hidden states,
//! batched verify, recurrent-state rollback) lives in `spec.rs`.
//!
//! Graph (see `notes/dspark-port-plan.md`), reference `src/models/dflash.cpp`:
//!
//!   encoder:  taps[n_tok, 5*n_embd] --fc--> [n_tok, n_embd] --rmsnorm--> inp_g
//!   inject:   per draft layer, inp_g --wk/wv--> K/V (RMSNorm on K + rope),
//!             written into the draft KV cache at the committed positions
//!   decode:   [id_last, MASK x (block-1)] --token_embd--> (+ log-SNR embed)
//!             -> 6 layers -> output_norm -> LM head -> markov bias
//!
//! The sidecar's on-disk names are the `dspark.*` originals; nothing here
//! depends on the `bonsai-gguf dspark-convert` rewrite used for the fork.

#![allow(dead_code)]

use crate::gguf::{half_to_f32, GGUF, TensorInfo, TYPE_BF16, TYPE_F32, TYPE_PQ2_0, TYPE_Q4_1};
use crate::kernels;

/// Sidecar metadata resolved into the numbers the graph needs.
#[derive(Debug, Clone)]
pub struct DsparkCfg {
    pub n_layer: usize,
    pub n_embd: usize,
    pub n_ff: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub n_vocab: usize,
    pub eps: f32,
    pub rope_freq_base: f32,
    pub block_size: usize,
    pub mask_token_id: u32,
    pub target_layers: Vec<u32>,
    pub markov_rank: usize,
    pub confidence_head: bool,
    pub log_snr: Option<(f32, f32)>,
}

impl DsparkCfg {
    pub fn from_gguf(g: &GGUF) -> Result<DsparkCfg, String> {
        let u32v = |k: &str| -> Result<u32, String> {
            g.get(k)
                .and_then(|v| v.as_u64())
                .map(|v| v as u32)
                .ok_or_else(|| format!("dspark cfg: missing/invalid '{k}'"))
        };
        let f32v = |k: &str| -> Result<f32, String> {
            g.get(k)
                .and_then(|v| v.as_f32())
                .ok_or_else(|| format!("dspark cfg: missing/invalid '{k}'"))
        };
        let arch = g
            .get("general.architecture")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if arch != "dspark" {
            return Err(format!("dspark cfg: architecture is '{arch}', expected 'dspark'"));
        }
        let conf = g
            .get("dspark.dspark.confidence_head")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let log_snr = if g
            .get("dspark.dspark.log_snr_conditioning")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
        {
            Some((
                f32v("dspark.dspark.min_log_snr")?,
                f32v("dspark.dspark.max_log_snr")?,
            ))
        } else {
            None
        };
        let target_layers = g
            .get("dspark.dspark.target_layers")
            .and_then(|v| v.as_array())
            .ok_or("dspark cfg: missing 'dspark.dspark.target_layers'")?
            .iter()
            .map(|v| v.as_u64().map(|x| x as u32))
            .collect::<Option<Vec<u32>>>()
            .ok_or("dspark cfg: target_layers must be integers")?;
        Ok(DsparkCfg {
            n_layer: u32v("dspark.block_count")? as usize,
            n_embd: u32v("dspark.embedding_length")? as usize,
            n_ff: u32v("dspark.feed_forward_length")? as usize,
            n_head: u32v("dspark.attention.head_count")? as usize,
            n_head_kv: u32v("dspark.attention.head_count_kv")? as usize,
            head_dim: u32v("dspark.attention.key_length")? as usize,
            n_vocab: u32v("dspark.vocab_size")? as usize,
            eps: f32v("dspark.attention.layer_norm_rms_epsilon")?,
            rope_freq_base: f32v("dspark.rope.freq_base")?,
            block_size: u32v("dspark.dspark.block_size")? as usize,
            mask_token_id: u32v("dspark.dspark.mask_token_id")?,
            target_layers,
            markov_rank: u32v("dspark.dspark.markov_rank")? as usize,
            confidence_head: conf,
            log_snr,
        })
    }

    /// Width of the encoder input: `len(target_layers) * target n_embd`.
    pub fn n_embd_enc(&self) -> usize {
        self.target_layers.len() * self.n_embd
    }
}

// ---------------------------------------------------------------------------
// Mixed-precision dequant / matvec
// ---------------------------------------------------------------------------

const Q4_1_QK: usize = 32;
const Q4_1_BLOCK: usize = 20; // 2x fp16 (d, m) + 16 nibble bytes

/// Bytes in one row of a tensor with `ne0` columns.
pub fn row_bytes(ty: u32, ne0: usize) -> usize {
    match ty {
        TYPE_F32 => ne0 * 4,
        TYPE_BF16 => ne0 * 2,
        TYPE_Q4_1 => ne0.div_ceil(Q4_1_QK) * Q4_1_BLOCK,
        TYPE_PQ2_0 => kernels::pq2_row_bytes(ne0),
        _ => ne0 * 4, // caller validates the type first
    }
}

pub fn type_supported(ty: u32) -> bool {
    matches!(ty, TYPE_F32 | TYPE_BF16 | TYPE_Q4_1 | TYPE_PQ2_0)
}

fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// Dequantize one tensor row into `out` (`out.len() == ne0`).
pub fn dequant_row(ty: u32, raw: &[u8], ne0: usize, out: &mut [f32]) {
    match ty {
        TYPE_F32 => {
            for (i, o) in out.iter_mut().enumerate().take(ne0) {
                let b = &raw[i * 4..i * 4 + 4];
                *o = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
            }
        }
        TYPE_BF16 => {
            for (i, o) in out.iter_mut().enumerate().take(ne0) {
                let b = u16::from_le_bytes([raw[i * 2], raw[i * 2 + 1]]);
                *o = bf16_to_f32(b);
            }
        }
        TYPE_Q4_1 => {
            // ggml block_q4_1: elements 0..16 use the low nibble of qs[j],
            // elements 16..32 use the high nibble.
            for (blk, chunk) in raw.chunks_exact(Q4_1_BLOCK).enumerate() {
                let d = half_to_f32(u16::from_le_bytes([chunk[0], chunk[1]]));
                let m = half_to_f32(u16::from_le_bytes([chunk[2], chunk[3]]));
                let qs = &chunk[4..4 + 16];
                let base = blk * Q4_1_QK;
                for j in 0..Q4_1_QK / 2 {
                    if base + j < ne0 {
                        out[base + j] = d * (qs[j] & 0x0f) as f32 + m;
                    }
                    if base + j + Q4_1_QK / 2 < ne0 {
                        out[base + j + Q4_1_QK / 2] = d * (qs[j] >> 4) as f32 + m;
                    }
                }
                if base + Q4_1_QK >= ne0 {
                    break;
                }
            }
        }
        TYPE_PQ2_0 => {
            let dec = kernels::decode_pq2_0_row(raw, ne0);
            out[..ne0].copy_from_slice(&dec);
        }
        _ => {}
    }
}

/// Scalar `y = W @ x` for one mixed-precision tensor (rows contiguous).
/// Correctness-oriented; the speculative path runs this a few times per block.
pub fn matvec_scalar(
    ty: u32,
    payload: &[u8],
    ne0: usize,
    x: &[f32],
    y: &mut [f32],
) -> Result<(), String> {
    if !type_supported(ty) {
        return Err(format!("dspark matvec: unsupported tensor type {ty}"));
    }
    if x.len() != ne0 {
        return Err(format!("dspark matvec: x len {} != ne0 {ne0}", x.len()));
    }
    let rb = row_bytes(ty, ne0);
    let rows = y.len();
    if payload.len() < rows * rb {
        return Err(format!(
            "dspark matvec: payload {} < {} rows * {rb}",
            payload.len(),
            rows
        ));
    }
    let mut row = vec![0.0f32; ne0];
    for (r, y_r) in y.iter_mut().enumerate() {
        dequant_row(ty, &payload[r * rb..(r + 1) * rb], ne0, &mut row);
        let mut acc = 0.0f32;
        for (a, b) in x.iter().zip(row.iter()) {
            acc += a * b;
        }
        *y_r = acc;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Draft weight context
// ---------------------------------------------------------------------------

pub struct Dspark {
    pub cfg: DsparkCfg,
    pub gguf: GGUF,
}

impl Dspark {
    pub fn open(path: &str) -> Result<Dspark, String> {
        let gguf = GGUF::open(path)?;
        let cfg = DsparkCfg::from_gguf(&gguf)?;
        Ok(Dspark { cfg, gguf })
    }

    pub fn tensor(&self, name: &str) -> Result<&TensorInfo, String> {
        self.gguf
            .tensors
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| format!("dspark: tensor '{name}' missing"))
    }

    /// Fetch a full tensor as f32 (small tensors only).
    pub fn read_f32(&self, name: &str) -> Result<Vec<f32>, String> {
        let t = self.tensor(name)?.clone();
        if !type_supported(t.ty) {
            return Err(format!("dspark {name}: unsupported type {}", t.ty));
        }
        let ne0 = t.dims.first().copied().unwrap_or(0) as usize;
        let rows = (t.n_elem() as usize) / ne0.max(1);
        let payload = self.gguf.payload_slice(&t)?;
        let mut out = vec![0.0f32; rows * ne0];
        for r in 0..rows {
            let rb = row_bytes(t.ty, ne0);
            dequant_row(t.ty, &payload[r * rb..(r + 1) * rb], ne0, &mut out[r * ne0..(r + 1) * ne0]);
        }
        Ok(out)
    }

    /// `y = W @ x` for a named matrix tensor. `x.len()` must equal `ne0`.
    pub fn matvec(&self, name: &str, x: &[f32]) -> Result<Vec<f32>, String> {
        let t = self.tensor(name)?.clone();
        let ne0 = t.dims.first().copied().unwrap_or(0) as usize;
        let rows = (t.n_elem() as usize) / ne0.max(1);
        let payload = self.gguf.payload_slice(&t)?;
        let mut y = vec![0.0f32; rows];
        matvec_scalar(t.ty, payload, ne0, x, &mut y)?;
        Ok(y)
    }

    /// Encoder: fuse tapped target features into the draft hidden width.
    /// `features` is row-major `[n_tok, n_embd_enc]` (per token, the tapped
    /// layers' hidden vectors concatenated in `target_layers` order).
    pub fn encode(&self, features: &[f32], n_tok: usize) -> Result<Vec<f32>, String> {
        let n_enc = self.cfg.n_embd_enc();
        if features.len() != n_tok * n_enc {
            return Err(format!(
                "dspark encode: features len {} != {n_tok} * {n_enc}",
                features.len()
            ));
        }
        let fc = self.read_f32("dspark.fc.weight")?; // [n_embd, n_embd_enc] as ne0=n_enc rows=n_embd
        let hidden_norm = self.read_f32("dspark.hidden_norm.weight")?;
        let n_embd = self.cfg.n_embd;
        let mut out = vec![0.0f32; n_tok * n_embd];
        for t in 0..n_tok {
            let x = &features[t * n_enc..(t + 1) * n_enc];
            let mut y = vec![0.0f32; n_embd];
            for r in 0..n_embd {
                let row = &fc[r * n_enc..(r + 1) * n_enc];
                let mut acc = 0.0f32;
                for (a, b) in x.iter().zip(row.iter()) {
                    acc += a * b;
                }
                y[r] = acc;
            }
            out[t * n_embd..(t + 1) * n_embd]
                .copy_from_slice(&kernels::rms_norm(&y, &hidden_norm, self.cfg.eps));
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_decode_roundtrip() {
        // 1.0f32 in bf16 is 0x3F80
        assert_eq!(bf16_to_f32(0x3F80), 1.0);
        assert_eq!(bf16_to_f32(0xBF80), -1.0);
        // 2.0f32 = 0x4000
        assert_eq!(bf16_to_f32(0x4000), 2.0);
    }

    #[test]
    fn q4_1_block_decodes() {
        // one block: d=1.0, m=0.0, nibbles 0..15 low/high
        let mut raw = Vec::new();
        raw.extend_from_slice(&crate::gguf::f32_to_half(1.0).to_le_bytes());
        raw.extend_from_slice(&crate::gguf::f32_to_half(0.0).to_le_bytes());
        for b in 0..16u8 {
            raw.push(b | (b << 4));
        }
        let mut out = vec![0.0f32; 32];
        dequant_row(TYPE_Q4_1, &raw, 32, &mut out);
        for j in 0..16 {
            assert_eq!(out[j], j as f32, "low element {j}");
            assert_eq!(out[16 + j], j as f32, "high element {j}");
        }
    }

    #[test]
    fn row_bytes_match_block_rules() {
        assert_eq!(row_bytes(TYPE_F32, 100), 400);
        assert_eq!(row_bytes(TYPE_BF16, 100), 200);
        assert_eq!(row_bytes(TYPE_Q4_1, 64), 2 * Q4_1_BLOCK);
        assert_eq!(row_bytes(TYPE_PQ2_0, 128), 34);
    }
}
