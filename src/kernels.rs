//! Pure-Rust compute kernels (M5).
//!
//! Small operators needed by the qwen35 forward pass, validated against ggml
//! reference graphs before they get wired into the engine:
//!   - RMSNorm
//!   - PQ2_0 (ternary, group-128) row dequant + dot product
//!
//! Numerical conventions mirror llama.cpp / ggml.

#![allow(dead_code)]

use crate::gguf::{GGUF, TensorInfo, TYPE_PQ2_0};

/// RMSNorm: out_i = x_i * w_i * rsqrt(mean(x^2) + eps)
pub fn rms_norm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let mut out = vec![0.0f32; n];
    if n == 0 {
        return out;
    }
    let mut ss = 0.0f32;
    for &v in x {
        ss += v * v;
    }
    let mean = ss / n as f32;
    let scale = 1.0 / (mean + eps).sqrt();
    for i in 0..n {
        out[i] = x[i] * scale * w[i];
    }
    out
}

/// PQ2_0 block constants (see ggml-common.h: QK_PQ2_0 = 128, block = 34 bytes).
const PQ2_QK: usize = 128;
const PQ2_BLOCK: usize = 34;

/// Row stride in bytes for a PQ2_0 tensor with `ne0` columns.
pub fn pq2_row_bytes(ne0: usize) -> usize {
    (ne0.div_ceil(PQ2_QK)) * PQ2_BLOCK
}

/// Decode one PQ2_0 tensor row into f32 (mirrors `dequantize_row_pq2_0`).
pub fn decode_pq2_0_row(raw: &[u8], ne0: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(ne0);
    for block in 0..ne0.div_ceil(PQ2_QK) {
        let base = block * PQ2_BLOCK;
        let scale = crate::gguf::half_to_f32(u16::from_le_bytes([raw[base], raw[base + 1]]));
        let qs = &raw[base + 2..base + 2 + PQ2_QK / 4];
        for j in 0..PQ2_QK {
            if block * PQ2_QK + j >= ne0 {
                break;
            }
            let code = (qs[j / 4] >> ((j % 4) * 2)) & 0x03;
            out.push((code as i32 - 1) as f32 * scale);
        }
    }
    out
}

/// Dot product of a row of a PQ2_0 tensor with `x` (length ne0).
pub fn dot_pq2_0_row(
    gguf: &mut GGUF,
    info: &TensorInfo,
    row: u64,
    x: &[f32],
) -> Result<f32, String> {
    if info.ty != TYPE_PQ2_0 {
        return Err("dot_pq2_0_row: tensor is not PQ2_0".into());
    }
    let ne0 = info.dims[0] as usize;
    if x.len() != ne0 {
        return Err(format!(
            "dot_pq2_0_row: x length {} != ne0 {}",
            x.len(),
            ne0
        ));
    }
    let row_bytes = pq2_row_bytes(ne0);
    let mut raw = vec![0u8; row_bytes];
    gguf.read_bytes(info.offset + row * row_bytes as u64, &mut raw)?;

    let mut acc = 0.0f32;
    for (v, w) in x.iter().zip(decode_pq2_0_row(&raw, ne0).iter()) {
        acc += v * w;
    }
    Ok(acc)
}

/// Batched PQ2_0 matrix-vector product over a contiguous range of rows.
/// Computes y[r] = sum_j x[j] * W[base_row + r][j].
pub fn pq2_matvec_range(
    gguf: &mut GGUF,
    info: &TensorInfo,
    base_row: u64,
    n_rows: usize,
    x: &[f32],
    y: &mut [f32],
) -> Result<(), String> {
    if info.ty != TYPE_PQ2_0 {
        return Err("pq2_matvec_range: tensor is not PQ2_0".into());
    }
    let ne0 = info.dims[0] as usize;
    if x.len() != ne0 {
        return Err(format!("pq2_matvec_range: x length {} != ne0 {}", x.len(), ne0));
    }
    if y.len() < n_rows {
        return Err("pq2_matvec_range: y too small".into());
    }
    let row_bytes = pq2_row_bytes(ne0);
    let mut raw = vec![0u8; row_bytes];
    for r in 0..n_rows {
        let off = info.offset + (base_row + r as u64) * row_bytes as u64;
        gguf.read_bytes(off, &mut raw)?;
        let mut acc = 0.0f32;
        // decode + dot in one pass
        for block in 0..ne0.div_ceil(PQ2_QK) {
            let b = block * PQ2_BLOCK;
            let scale = crate::gguf::half_to_f32(u16::from_le_bytes([raw[b], raw[b + 1]]));
            let qs = &raw[b + 2..b + 2 + PQ2_QK / 4];
            let start = block * PQ2_QK;
            let end = (start + PQ2_QK).min(ne0);
            for j in start..end {
                let code = (qs[(j - start) / 4] >> (((j - start) % 4) * 2)) & 0x03;
                acc += x[j] * ((code as i32 - 1) as f32 * scale);
            }
        }
        y[r] = acc;
    }
    Ok(())
}

/// Number of rows (ne1) of a tensor.
pub fn n_rows(info: &TensorInfo) -> u64 {
    if info.dims.is_empty() {
        0
    } else {
        info.dims[1..].iter().product()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rms_norm_matches_reference() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let w = vec![0.5, 1.0, 1.5, 2.0];
        let y = rms_norm(&x, &w, 1e-6);
        // mean of squares = (1+4+9+16)/4 = 7.5; scale = 1/sqrt(7.5)
        let s = 1.0 / (7.5f32 + 1e-6).sqrt();
        let expect: Vec<f32> = x.iter().zip(&w).map(|(a, b)| a * b * s).collect();
        for (a, b) in y.iter().zip(&expect) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn pq2_decode_known_row() {
        // one block: scale 0x4000 = 2.0, then 32 bytes; codes 3 at j=0, 2 at j=5
        let mut raw = vec![0u8; PQ2_BLOCK];
        raw[0] = 0x00;
        raw[1] = 0x40;
        raw[2] = 0b11; // j=0 -> code 3 -> +2*2 = 4
        raw[3] = 0b10_01; // j=4 -> code 1 -> 0; j=5 -> code 2 -> +2
        let v = decode_pq2_0_row(&raw, PQ2_QK);
        assert_eq!(v[0], 4.0);
        assert_eq!(v[1], -2.0); // code 0 -> -2
        assert_eq!(v[4], 0.0);
        assert_eq!(v[5], 2.0);
    }
}
