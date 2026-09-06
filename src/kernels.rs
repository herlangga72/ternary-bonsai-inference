//! Pure-Rust compute kernels (M5).
//!
//! Small operators needed by the qwen35 forward pass, validated against ggml
//! reference graphs before they get wired into the engine:
//!   - RMSNorm (+ row-wise blocks for per-head norms)
//!   - L2 norm over rows
//!   - SiLU / softplus / sigmoid activations
//!   - softmax over rows
//!   - PQ2_0 (ternary, group-128) row dequant + dot product
//!
//! Numerical conventions mirror llama.cpp / ggml (M6-2 probes keep them honest).

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

/// RMSNorm applied to consecutive rows of `row_len`, each sharing the same
/// `row_len`-long weight (per-head norms: q/k heads of 256, v-heads of 128).
pub fn rms_norm_rows(x: &[f32], w: &[f32], row_len: usize, eps: f32) -> Result<Vec<f32>, String> {
    if w.len() != row_len {
        return Err(format!(
            "rms_norm_rows: weight len {} != row_len {row_len}",
            w.len()
        ));
    }
    if x.len() % row_len != 0 {
        return Err(format!(
            "rms_norm_rows: x len {} not divisible by row_len {row_len}",
            x.len()
        ));
    }
    let mut out = vec![0.0f32; x.len()];
    for (src, dst) in x.chunks_exact(row_len).zip(out.chunks_exact_mut(row_len)) {
        let mut ss = 0.0f32;
        for &v in src {
            ss += v * v;
        }
        let scale = 1.0 / (ss / row_len as f32 + eps).sqrt();
        for i in 0..row_len {
            dst[i] = src[i] * scale * w[i];
        }
    }
    Ok(out)
}

/// L2 normalize consecutive rows of `row_len` (used on q/k of the recurrent
/// branch). Mirrors ggml_l2_norm: sums in f64, scale = 1 / max(sqrt(sum), eps).
pub fn l2_norm_rows(x: &[f32], row_len: usize, eps: f32) -> Result<Vec<f32>, String> {
    if x.len() % row_len != 0 {
        return Err(format!(
            "l2_norm_rows: x len {} not divisible by row_len {row_len}",
            x.len()
        ));
    }
    let mut out = x.to_vec();
    for chunk in out.chunks_exact_mut(row_len) {
        let mut sum = 0.0f64;
        for &v in chunk.iter() {
            sum += (v as f64) * (v as f64);
        }
        let scale = 1.0 / (sum as f32).sqrt().max(eps);
        for v in chunk.iter_mut() {
            *v *= scale;
        }
    }
    Ok(out)
}

/// SiLU (Sigmoid Linear Unit): x / (1 + exp(-x)), as in ggml.
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Softplus: log(1 + exp(x)), clamped to identity above 20 (ggml threshold).
pub fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}

/// Sigmoid: 1 / (1 + exp(-x)).
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Apply a scalar fn elementwise in place.
pub fn map_inplace(x: &mut [f32], f: fn(f32) -> f32) {
    for v in x.iter_mut() {
        *v = f(*v);
    }
}

/// Numerically stable softmax over consecutive rows of `row_len` (masked
/// entries come in as -inf and naturally get exp(-inf) = 0).
pub fn softmax_rows(x: &[f32], row_len: usize) -> Result<Vec<f32>, String> {
    if x.len() % row_len != 0 {
        return Err(format!(
            "softmax_rows: x len {} not divisible by row_len {row_len}",
            x.len()
        ));
    }
    let mut out = vec![0.0f32; x.len()];
    for (src, dst) in x.chunks_exact(row_len).zip(out.chunks_exact_mut(row_len)) {
        let mut max = f32::NEG_INFINITY;
        for &v in src {
            if v > max {
                max = v;
            }
        }
        let mut sum = 0.0f64;
        for (s, d) in src.iter().zip(dst.iter_mut()) {
            let e = (*s - max).exp();
            *d = e;
            sum += e as f64;
        }
        let inv = 1.0 / sum as f32;
        for d in dst.iter_mut() {
            *d *= inv;
        }
    }
    Ok(out)
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

    #[test]
    fn rms_norm_rows_matches_single_row_math() {
        let x = vec![1.0f32, 2.0, 3.0, 4.0, 0.5, -0.5, 2.0, -3.0];
        let w = vec![0.5f32, 1.0, 1.5, 2.0];
        let eps = 1e-6f32;
        let y = rms_norm_rows(&x, &w, 4, eps).unwrap();
        for (row, src) in x.chunks_exact(4).enumerate() {
            let mut ss = 0.0f32;
            for v in src {
                ss += v * v;
            }
            let scale = 1.0 / (ss / 4.0 + eps).sqrt();
            for (i, v) in src.iter().enumerate() {
                let want = v * scale * w[i];
                assert!(
                    (y[row * 4 + i] - want).abs() < 1e-6,
                    "row {row} col {i}: {} vs {want}",
                    y[row * 4 + i]
                );
            }
        }
        assert!(rms_norm_rows(&x, &w, 3, eps).is_err()); // non-divisible
        assert!(rms_norm_rows(&x, &w[..2], 4, eps).is_err()); // bad weight len
    }

    #[test]
    fn l2_norm_rows_produce_unit_rows() {
        let x = vec![3.0f32, 4.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let y = l2_norm_rows(&x, 4, 1e-12).unwrap();
        assert!((y[0] - 0.6).abs() < 1e-6); // 3/5
        assert!((y[1] - 0.8).abs() < 1e-6); // 4/5
        for row in y[4..8].chunks_exact(4) {
            let ss: f32 = row.iter().map(|v| v * v).sum();
            assert!((ss - 1.0).abs() < 1e-6);
        }
        // eps floor: a zero row stays zero (1/max(0, eps) capped)
        let z = l2_norm_rows(&[0.0f32; 4], 4, 1e-6).unwrap();
        assert_eq!(z, vec![0.0; 4]);
        assert!(l2_norm_rows(&x, 3, 1e-12).is_err());
    }

    #[test]
    fn activations_hit_expected_points() {
        assert_eq!(silu(0.0), 0.0);
        assert!((silu(1000.0) - 1000.0).abs() < 1e-3); // saturates to identity
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-7);
        assert!(sigmoid(-1000.0) == 0.0);
        assert!((softplus(0.0) - 2f32.ln()).abs() < 1e-7);
        assert_eq!(softplus(100.0), 100.0); // > 20: identity branch
    }

    #[test]
    fn softmax_rows_are_normalized_and_mask_safe() {
        let x = vec![1.0f32, 2.0, 3.0, f32::NEG_INFINITY, 0.0, 0.0, 0.0, 0.0];
        let y = softmax_rows(&x, 4).unwrap();
        let sum0: f32 = y[0..4].iter().sum();
        assert!((sum0 - 1.0).abs() < 1e-6);
        assert_eq!(y[3], 0.0); // masked entry gets exp(-inf) = 0
        let sum1: f32 = y[4..8].iter().sum();
        assert!((sum1 - 1.0).abs() < 1e-6);
        for v in &y[4..8] {
            assert!((v - 0.25).abs() < 1e-6);
        }
        assert!(softmax_rows(&x, 3).is_err());
    }
}
