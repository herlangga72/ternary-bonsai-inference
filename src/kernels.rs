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

use crate::gguf::{half_to_f32, TensorInfo};

// ---------------------------------------------------------------------------
// Resource restraint (BONSAI_BW_PCT)
//
// Long validation runs saturate the machine's memory bandwidth. Setting the
// environment variable BONSAI_BW_PCT (1..100, default 100) throttles the
// engine to roughly that fraction of its normal bandwidth use:
//   - the CPU matvec fan-out is capped to ~pct% of the available cores, and
//   - decode loops that call `pause_for_budget` sleep so the average token
//     pace runs at pct% of the unpaced rate.
// ---------------------------------------------------------------------------

/// Fraction (0..=1) of normal resources to use, from BONSAI_BW_PCT.
pub fn bw_fraction() -> f32 {
    let v = std::env::var("BONSAI_BW_PCT")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(100.0);
    (v.clamp(1.0, 100.0)) / 100.0
}

/// Cap `avail` worker threads at ~bw_fraction() of the available cores.
/// Always keeps at least one thread.
pub fn scaled_threads(avail: usize) -> usize {
    ((avail as f32 * bw_fraction()).round() as usize).clamp(1, avail.max(1))
}

/// Sleep so the caller's per-token pace uses only `bw_fraction()` of the
/// unpaced budget. `elapsed_s` is the just-finished (paced or unpaced) token;
/// `baseline_s` is the measured unpaced token duration.
pub fn pause_for_budget(elapsed_s: f32, baseline_s: f32) {
    let f = bw_fraction();
    if f >= 1.0 || baseline_s <= 0.0 {
        return;
    }
    let target = baseline_s / f;
    if elapsed_s < target {
        let extra = (target - elapsed_s) as u64;
        std::thread::sleep(std::time::Duration::from_millis(extra * 1000));
    }
}

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
/// `payload` must be the tensor's whole contiguous data (see
/// `GGUF::payload_slice`); row `row` starts at `payload[row * row_bytes]`.
pub fn dot_pq2_0_row(
    payload: &[u8],
    ne0: usize,
    row: u64,
    x: &[f32],
) -> Result<f32, String> {
    if x.len() != ne0 {
        return Err(format!(
            "dot_pq2_0_row: x length {} != ne0 {}",
            x.len(),
            ne0
        ));
    }
    let row_bytes = pq2_row_bytes(ne0);
    let start = row as usize * row_bytes;
    let end = start + row_bytes;
    if end > payload.len() {
        return Err(format!(
            "dot_pq2_0_row: row {row} needs bytes {start}..{end}, payload is {}",
            payload.len()
        ));
    }
    let raw = &payload[start..end];

    let mut acc = 0.0f32;
    for (v, w) in x.iter().zip(decode_pq2_0_row(raw, ne0).iter()) {
        acc += v * w;
    }
    Ok(acc)
}

/// Batched PQ2_0 matrix-vector product over a contiguous range of rows.
/// Computes y[r] = sum_j x[j] * W[base_row + r][j].
/// `payload` is the tensor's whole contiguous data section; rows are laid out
/// back to back with `row_bytes` stride (see `GGUF::payload_slice`). No file
/// I/O and no per-row scratch: the caller passes an mmap-backed slice.
///
/// Large row ranges are split across the available cores (scoped threads,
/// read-only payload + x, disjoint y chunks). Small ranges run inline to keep
/// per-matvec launch overhead out of the dozens of small tensors per layer.
pub fn pq2_matvec_range(
    payload: &[u8],
    ne0: usize,
    base_row: u64,
    n_rows: usize,
    x: &[f32],
    y: &mut [f32],
) -> Result<(), String> {
    if x.len() != ne0 {
        return Err(format!("pq2_matvec_range: x length {} != ne0 {}", x.len(), ne0));
    }
    if y.len() < n_rows {
        return Err("pq2_matvec_range: y too small".into());
    }
    let row_bytes = pq2_row_bytes(ne0);
    let byte_len = (base_row as usize + n_rows)
        .checked_mul(row_bytes)
        .ok_or("pq2_matvec_range: row range byte length overflows")?;
    if byte_len > payload.len() {
        return Err(format!(
            "pq2_matvec_range: rows {base_row}..{} need {byte_len} bytes, payload is {}",
            base_row + n_rows as u64,
            payload.len()
        ));
    }
    let base = base_row as usize;
    // Minimum rows per matvec before spawning threads (the tiny ssm_alpha /
    // ssm_beta / norm-side projections stay inline).
    const MIN_PARALLEL_ROWS: usize = 1024;
    let avail = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let n_cores = scaled_threads(avail);
    if n_rows < MIN_PARALLEL_ROWS || n_cores <= 1 {
        dot_rows_into(payload, ne0, base, 0..n_rows, x, &mut y[..n_rows]);
        return Ok(());
    }

    // Split rows into ~one contiguous chunk per core; each chunk writes a
    // disjoint, contiguous y range (no false sharing) and reads its own rows.
    let n_workers = n_cores.min(n_rows);
    let chunk = n_rows.div_ceil(n_workers);
    let ranges: Vec<std::ops::Range<usize>> = (0..n_workers)
        .map(|w| {
            let s = w * chunk;
            let e = (s + chunk).min(n_rows);
            s..e
        })
        .collect();
    std::thread::scope(|scope| {
        let mut y_rest = &mut y[..n_rows];
        for range in ranges {
            let (head, tail) = y_rest.split_at_mut(range.len());
            y_rest = tail;
            let head: &mut [f32] = head;
            scope.spawn(move || dot_rows_into(payload, ne0, base, range, x, head));
        }
    });
    Ok(())
}

/// N-column PQ2_0 GEMM: `Y[t][r] = sum_j X[t][j] * W[base_row + r][j]`.
///
/// `x` is row-major `[n_tok, ne0]`, `y` row-major `[n_tok, n_rows]`. Each weight
/// row is fetched and decoded once and reused for all `n_tok` activations, so
/// the weight stream is read once per pass instead of once per token - the
/// primitive that makes batched target verification (and prefill) worth doing.
/// An `n_tok` of 1 is identical to `pq2_matvec_range`.
pub fn pq2_matmul_n(
    payload: &[u8],
    ne0: usize,
    base_row: u64,
    n_rows: usize,
    x: &[f32],
    n_tok: usize,
    y: &mut [f32],
) -> Result<(), String> {
    if x.len() != n_tok * ne0 {
        return Err(format!(
            "pq2_matmul_n: x length {} != n_tok {n_tok} * ne0 {ne0}",
            x.len()
        ));
    }
    if y.len() < n_tok * n_rows {
        return Err("pq2_matmul_n: y too small".into());
    }
    let row_bytes = pq2_row_bytes(ne0);
    let need = (base_row as usize + n_rows)
        .checked_mul(row_bytes)
        .ok_or("pq2_matmul_n: overflow")?;
    if need > payload.len() {
        return Err("pq2_matmul_n: row range exceeds payload".into());
    }
    if n_tok == 1 {
        return pq2_matvec_range(payload, ne0, base_row, n_rows, x, &mut y[..n_rows]);
    }

    let base = base_row as usize;
    #[cfg(target_arch = "x86_64")]
    let use_simd = std::arch::is_x86_feature_detected!("avx2")
        && std::arch::is_x86_feature_detected!("fma")
        && ne0 % PQ2_QK == 0;
    let row_dot = |r: usize, t: usize| -> f32 {
        let raw = &payload[(base + r) * row_bytes..(base + r + 1) * row_bytes];
        let xr = &x[t * ne0..(t + 1) * ne0];
        #[cfg(target_arch = "x86_64")]
        {
            if use_simd {
                return unsafe { row_dot_avx2(raw, ne0, xr) };
            }
        }
        row_dot_scalar(raw, ne0, xr)
    };

    const MIN_PARALLEL_ROWS: usize = 1024;
    let avail = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let n_cores = scaled_threads(avail);
    if n_rows < MIN_PARALLEL_ROWS || n_cores <= 1 {
        for k in 0..n_rows {
            for t in 0..n_tok {
                y[t * n_rows + k] = row_dot(k, t);
            }
        }
        return Ok(());
    }
    let n_workers = n_cores.min(n_rows);
    let chunk = n_rows.div_ceil(n_workers);
    let ranges: Vec<std::ops::Range<usize>> = (0..n_workers)
        .map(|w| {
            let s = w * chunk;
            s..(s + chunk).min(n_rows)
        })
        .collect();
    // rows are disjoint per worker, so the strided writes into the token-major
    // output never alias; the pointer is sent as usize because *mut is !Send.
    let yp = y.as_mut_ptr() as usize;
    std::thread::scope(|scope| {
        for range in ranges {
            let rd = &row_dot;
            scope.spawn(move || {
                let yp = yp as *mut f32;
                for r in range {
                    for t in 0..n_tok {
                        let v = rd(r, t);
                        unsafe { *yp.add(t * n_rows + r) = v };
                    }
                }
            });
        }
    });
    Ok(())
}

// ---------------------------------------------------------------------------
// Q4_1 (used only by the dspark drafter sidecar)
// ---------------------------------------------------------------------------

const Q4_1_QK: usize = 32;
const Q4_1_BLOCK: usize = 20; // fp16 d, fp16 m, 16 nibble bytes

/// Row stride in bytes for a Q4_1 tensor with `ne0` columns.
pub fn q4_1_row_bytes(ne0: usize) -> usize {
    ne0.div_ceil(Q4_1_QK) * Q4_1_BLOCK
}

/// Scalar Q4_1 row dot: `sum_j x[j] * (d*q_j + m)`, laid out as ggml
/// (`block_q4_1`: elements 0..16 low nibbles, 16..32 high nibbles).
fn q4_1_row_dot_scalar(raw: &[u8], ne0: usize, x: &[f32], xbs: &[f32]) -> f32 {
    let nblk = ne0.div_ceil(Q4_1_QK);
    let mut acc = 0.0f32;
    for blk in 0..nblk {
        let chunk = &raw[blk * Q4_1_BLOCK..(blk + 1) * Q4_1_BLOCK];
        let d = half_to_f32(u16::from_le_bytes([chunk[0], chunk[1]]));
        let m = half_to_f32(u16::from_le_bytes([chunk[2], chunk[3]]));
        let qs = &chunk[4..20];
        let base = blk * Q4_1_QK;
        let mut bs = 0.0f32;
        for j in 0..16 {
            if base + j < ne0 {
                bs += x[base + j] * (qs[j] & 0x0f) as f32;
            }
            if base + 16 + j < ne0 {
                bs += x[base + 16 + j] * (qs[j] >> 4) as f32;
            }
        }
        acc += d * bs + m * xbs[blk];
    }
    acc
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn q4_1_row_dot_avx2(raw: &[u8], ne0: usize, x: &[f32], xbs: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    debug_assert_eq!(ne0 % Q4_1_QK, 0);
    let nblk = ne0 / Q4_1_QK;
    let mask = _mm_set1_epi8(0x0f);
    let mut scalar_acc = 0.0f32;
    for blk in 0..nblk {
        let chunk = raw.as_ptr().add(blk * Q4_1_BLOCK);
        let d = half_to_f32(u16::from_le_bytes([*chunk, *chunk.add(1)]));
        let m = half_to_f32(u16::from_le_bytes([*chunk.add(2), *chunk.add(3)]));
        let qs = chunk.add(4);
        let mut acc = _mm256_setzero_ps();
        for k in 0..2 {
            let v = _mm_loadl_epi64(qs.add(k * 8) as *const __m128i);
            let lo = _mm_and_si128(v, mask);
            let hi = _mm_and_si128(_mm_srli_epi16(v, 4), mask);
            let lo_f = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(lo));
            let hi_f = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(hi));
            let xlo = _mm256_loadu_ps(x.as_ptr().add(blk * Q4_1_QK + k * 8));
            let xhi = _mm256_loadu_ps(x.as_ptr().add(blk * Q4_1_QK + 16 + k * 8));
            acc = _mm256_fmadd_ps(xlo, lo_f, acc);
            acc = _mm256_fmadd_ps(xhi, hi_f, acc);
        }
        let mut lanes = [0.0f32; 8];
        _mm256_storeu_ps(lanes.as_mut_ptr(), acc);
        let bs = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3])
            + (lanes[4] + lanes[5]) + (lanes[6] + lanes[7]);
        scalar_acc += d * bs + m * xbs[blk];
    }
    scalar_acc
}

/// Per-32-block sums of `x`, used to fold Q4_1's per-block `m` term.
fn block_sums(x: &[f32]) -> Vec<f32> {
    x.chunks(Q4_1_QK)
        .map(|c| c.iter().sum::<f32>())
        .collect()
}

fn q4_1_row_dot(raw: &[u8], ne0: usize, x: &[f32], xbs: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma")
            && ne0 % Q4_1_QK == 0
        {
            return unsafe { q4_1_row_dot_avx2(raw, ne0, x, xbs) };
        }
    }
    q4_1_row_dot_scalar(raw, ne0, x, xbs)
}

/// `y[..n_rows] = W[base_row..] @ x` for a Q4_1 matrix, threaded over rows.
pub fn q4_1_matvec_range(
    payload: &[u8],
    ne0: usize,
    base_row: u64,
    n_rows: usize,
    x: &[f32],
    y: &mut [f32],
) -> Result<(), String> {
    if x.len() != ne0 {
        return Err("q4_1_matvec_range: x length mismatch".into());
    }
    if y.len() < n_rows {
        return Err("q4_1_matvec_range: y too small".into());
    }
    let rb = q4_1_row_bytes(ne0);
    let need = (base_row as usize + n_rows)
        .checked_mul(rb)
        .ok_or("q4_1_matvec_range: overflow")?;
    if need > payload.len() {
        return Err("q4_1_matvec_range: row range exceeds payload".into());
    }
    let base = base_row as usize;
    let xbs = block_sums(x);
    let dot = |r: usize| -> f32 { q4_1_row_dot(&payload[r * rb..(r + 1) * rb], ne0, x, &xbs) };

    const MIN_PARALLEL_ROWS: usize = 1024;
    let avail = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let n_cores = scaled_threads(avail);
    if n_rows < MIN_PARALLEL_ROWS || n_cores <= 1 {
        for (k, o) in y[..n_rows].iter_mut().enumerate() {
            *o = dot(base + k);
        }
        return Ok(());
    }
    let n_workers = n_cores.min(n_rows);
    let chunk = n_rows.div_ceil(n_workers);
    let ranges: Vec<std::ops::Range<usize>> = (0..n_workers)
        .map(|w| {
            let s = w * chunk;
            s..(s + chunk).min(n_rows)
        })
        .collect();
    let mut rest = &mut y[..n_rows];
    std::thread::scope(|scope| {
        for range in ranges {
            let (head, tail) = rest.split_at_mut(range.len());
            rest = tail;
            scope.spawn(move || {
                for (k, o) in head.iter_mut().enumerate() {
                    *o = dot(base + range.start + k);
                }
            });
        }
    });
    Ok(())
}

/// Single-threaded variant of `pq2_matvec_range` using the production row
/// kernel (AVX2 when available). Used by microbenchmarks so kernel throughput
/// can be measured without thread-scheduling noise.
pub fn pq2_matvec_range_single(
    payload: &[u8],
    ne0: usize,
    base_row: u64,
    n_rows: usize,
    x: &[f32],
    y: &mut [f32],
) -> Result<(), String> {
    if x.len() != ne0 {
        return Err("pq2_matvec_range_single: x length mismatch".into());
    }
    if y.len() < n_rows {
        return Err("pq2_matvec_range_single: y too small".into());
    }
    let row_bytes = pq2_row_bytes(ne0);
    let need = (base_row as usize + n_rows)
        .checked_mul(row_bytes)
        .ok_or("pq2_matvec_range_single: overflow")?;
    if need > payload.len() {
        return Err("pq2_matvec_range_single: row range exceeds payload".into());
    }
    dot_rows_into(payload, ne0, base_row as usize, 0..n_rows, x, &mut y[..n_rows]);
    Ok(())
}

/// Dot rows `base + range` (local row index within `range`) into `y`, one
/// output float per row. Shared by the scalar and the per-thread paths so the
/// decode math is bit-identical either way. On x86-64 with AVX2+FMA the inner
/// loop uses a 4KB code->float lookup (values -1,0,1,2 per 2-bit code) with
/// packed FMAs; otherwise it falls back to the scalar decode.
fn dot_rows_into(
    payload: &[u8],
    ne0: usize,
    base: usize,
    rows: std::ops::Range<usize>,
    x: &[f32],
    y: &mut [f32],
) {
    debug_assert_eq!(y.len(), rows.len());
    let row_bytes = pq2_row_bytes(ne0);
    #[cfg(target_arch = "x86_64")]
    let use_simd = std::arch::is_x86_feature_detected!("avx2")
        && std::arch::is_x86_feature_detected!("fma");
    for (out, r) in y.iter_mut().zip(rows) {
        let raw = &payload[(base + r) * row_bytes..(base + r + 1) * row_bytes];
        #[cfg(target_arch = "x86_64")]
        if use_simd && ne0 % PQ2_QK == 0 {
            *out = unsafe { row_dot_avx2(raw, ne0, x) };
            continue;
        }
        *out = row_dot_scalar(raw, ne0, x);
    }
}

/// Scalar reference row dot (code-1 mapping, scale per 128-block). This is
/// also what the exactness tests and the SIMD fallback compare against.
fn row_dot_scalar(raw: &[u8], ne0: usize, x: &[f32]) -> f32 {
    let n_blocks = ne0.div_ceil(PQ2_QK);
    let mut acc = 0.0f32;
    for block in 0..n_blocks {
        let b = block * PQ2_BLOCK;
        let scale = half_to_f32(u16::from_le_bytes([raw[b], raw[b + 1]]));
        let qs = &raw[b + 2..b + 2 + PQ2_QK / 4];
        let start = block * PQ2_QK;
        let end = (start + PQ2_QK).min(ne0);
        let mut block_acc = 0.0f32;
        for j in start..end {
            let code = (qs[(j - start) / 4] >> (((j - start) % 4) * 2)) & 0x03;
            block_acc += x[j] * ((code as i32 - 1) as f32);
        }
        acc += scale * block_acc;
    }
    acc
}

/// AVX2+FMA row dot. For each 128-weight block a 4KB table maps every byte
/// (4 two-bit codes) to four f32 multipliers in {-1,0,1,2}; four lanes are
/// multiplied per FMA. The accumulator is kept in a 256-bit register across
/// every block and the block scale is folded in with one FMA per block, so the
/// only horizontal reduction is a single 8-lane sum per row. The result
/// differs from the scalar path only by fp rounding order (~1e-6 relative).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn row_dot_avx2(raw: &[u8], ne0: usize, x: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    static LUT: std::sync::OnceLock<Vec<f32>> = std::sync::OnceLock::new();
    let lut = LUT.get_or_init(|| {
        let mut v = vec![0.0f32; 256 * 4];
        for (byte, group) in v.chunks_exact_mut(4).enumerate() {
            for k in 0..4 {
                let code = (byte >> (2 * k)) & 0x03;
                group[k] = (code as i32 - 1) as f32;
            }
        }
        v
    });
    let lutp = lut.as_ptr();

    // Running vector sum of scale * (block dot). Reduced to a scalar once.
    let mut total = _mm256_setzero_ps();
    let n_blocks = ne0 / PQ2_QK;
    for block in 0..n_blocks {
        let b = block * PQ2_BLOCK;
        let scale = half_to_f32(u16::from_le_bytes([raw[b], raw[b + 1]]));
        let qs = raw.as_ptr().add(b + 2);
        let xoff = block * PQ2_QK;
        // Eight code bytes (=32 columns) per iteration into four independent
        // 256-bit accumulators, so the FMA latency chain does not serialize the
        // block. The code bytes are fetched with one unaligned 64-bit load.
        let mut a0 = _mm256_setzero_ps();
        let mut a1 = _mm256_setzero_ps();
        let mut a2 = _mm256_setzero_ps();
        let mut a3 = _mm256_setzero_ps();
        let mut k = 0usize;
        while k < 32 {
            let word = (qs.add(k) as *const u64).read_unaligned();
            let c = |sh: u32| ((word >> sh) & 0xff) as usize;
            let w0 = _mm256_insertf128_ps(
                _mm256_castps128_ps256(_mm_loadu_ps(lutp.add(c(0) * 4))),
                _mm_loadu_ps(lutp.add(c(8) * 4)),
                1,
            );
            let w1 = _mm256_insertf128_ps(
                _mm256_castps128_ps256(_mm_loadu_ps(lutp.add(c(16) * 4))),
                _mm_loadu_ps(lutp.add(c(24) * 4)),
                1,
            );
            let w2 = _mm256_insertf128_ps(
                _mm256_castps128_ps256(_mm_loadu_ps(lutp.add(c(32) * 4))),
                _mm_loadu_ps(lutp.add(c(40) * 4)),
                1,
            );
            let w3 = _mm256_insertf128_ps(
                _mm256_castps128_ps256(_mm_loadu_ps(lutp.add(c(48) * 4))),
                _mm_loadu_ps(lutp.add(c(56) * 4)),
                1,
            );
            let xp = x.as_ptr().add(xoff + 4 * k);
            a0 = _mm256_fmadd_ps(_mm256_loadu_ps(xp), w0, a0);
            a1 = _mm256_fmadd_ps(_mm256_loadu_ps(xp.add(8)), w1, a1);
            a2 = _mm256_fmadd_ps(_mm256_loadu_ps(xp.add(16)), w2, a2);
            a3 = _mm256_fmadd_ps(_mm256_loadu_ps(xp.add(24)), w3, a3);
            k += 8;
        }
        let s01 = _mm256_add_ps(a0, a1);
        let s23 = _mm256_add_ps(a2, a3);
        let vacc = _mm256_add_ps(s01, s23);
        total = _mm256_fmadd_ps(vacc, _mm256_set1_ps(scale), total);
    }
    let mut lanes = [0.0f32; 8];
    _mm256_storeu_ps(lanes.as_mut_ptr(), total);
    let mut acc = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3])
        + (lanes[4] + lanes[5]) + (lanes[6] + lanes[7]);
    // scalar tail for ne0 not a multiple of 128
    let tail_start = n_blocks * PQ2_QK;
    if tail_start < ne0 {
        let qs = &raw[n_blocks * PQ2_BLOCK + 2..];
        let mut tail_acc = 0.0f32;
        for (j, src) in (tail_start..ne0).enumerate() {
            let code = (qs[src / 4] >> ((src % 4) * 2)) & 0x03;
            tail_acc += x[tail_start + j] * ((code as i32 - 1) as f32);
        }
        let scale = half_to_f32(u16::from_le_bytes([
            raw[n_blocks * PQ2_BLOCK],
            raw[n_blocks * PQ2_BLOCK + 1],
        ]));
        acc += scale * tail_acc;
    }
    acc
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

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn avx2_row_dot_matches_scalar() {
        use super::*;
        if !(std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma"))
        {
            return;
        }
        let ne0 = 5120usize; // model row width, whole 128-blocks only
        let n_rows = 8usize;
        let row_bytes = pq2_row_bytes(ne0);
        let mut raw = vec![0u8; row_bytes * n_rows];
        // deterministic pseudo-random codes + positive fp16 scales
        let mut s: u64 = 0x1234_5678_9abc_def0;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        // f32 -> f16 (finite positive values only)
        let f16 = |v: f32| -> u16 {
            let b = v.to_bits();
            let sign = ((b >> 16) & 0x8000) as u16;
            let exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
            let mant = (b & 0x7f_ffff) >> 13;
            sign | (((exp as u16) << 10) | mant as u16)
        };
        for r in 0..n_rows {
            for b in 0..ne0 / PQ2_QK {
                let base = r * row_bytes + b * PQ2_BLOCK;
                let sc = 0.5 + ((next() >> 8) % 1000) as f32 / 1000.0;
                let h = f16(sc);
                raw[base] = (h & 0xff) as u8;
                raw[base + 1] = (h >> 8) as u8;
                for k in 0..32 {
                    raw[base + 2 + k] = (next() % 256) as u8;
                }
            }
        }
        let x: Vec<f32> = (0..ne0).map(|i| ((next() % 2001) as f32 - 1000.0) / 500.0).collect();
        for r in 0..n_rows {
            let row = &raw[r * row_bytes..(r + 1) * row_bytes];
            let scalar = row_dot_scalar(row, ne0, &x);
            let simd = unsafe { row_dot_avx2(row, ne0, &x) };
            let scale = scalar.abs().max(1e-6);
            assert!(
                (scalar - simd).abs() / scale < 1e-4,
                "row {r}: scalar {scalar} simd {simd}"
            );
        }
    }

    #[test]
    fn pq2_matmul_n_matches_per_token_matvec() {
        let ne0 = 256usize;
        let n_rows = 12usize;
        let n_tok = 3usize;
        let rb = pq2_row_bytes(ne0);
        let mut seed = 0x1234_5678u32;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed
        };
        let mut payload = vec![0u8; n_rows * rb];
        for r in 0..n_rows {
            for blk in 0..ne0 / PQ2_QK {
                let base = r * rb + blk * PQ2_BLOCK;
                let h = crate::gguf::f32_to_half(((next() % 500) as f32) / 250.0 + 0.01);
                payload[base] = (h & 0xff) as u8;
                payload[base + 1] = (h >> 8) as u8;
                for k in 0..32 {
                    payload[base + 2 + k] = (next() % 256) as u8;
                }
            }
        }
        let x: Vec<f32> = (0..n_tok * ne0)
            .map(|_| ((next() % 2001) as f32 - 1000.0) / 500.0)
            .collect();
        let mut y_n = vec![0.0f32; n_tok * n_rows];
        pq2_matmul_n(&payload, ne0, 0, n_rows, &x, n_tok, &mut y_n).unwrap();
        for t in 0..n_tok {
            let mut y1 = vec![0.0f32; n_rows];
            pq2_matvec_range(&payload, ne0, 0, n_rows, &x[t * ne0..(t + 1) * ne0], &mut y1).unwrap();
            for r in 0..n_rows {
                assert_eq!(y_n[t * n_rows + r], y1[r], "t {t} row {r}");
            }
        }
    }

    #[test]
    fn q4_1_matvec_matches_dequantized_dot() {
        let ne0 = 64usize;
        let n_rows = 5usize;
        let rb = q4_1_row_bytes(ne0);
        let mut seed = 0x9e37_79b9u32;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed
        };
        let mut payload = vec![0u8; n_rows * rb];
        for r in 0..n_rows {
            for blk in 0..ne0 / Q4_1_QK {
                let b = r * rb + blk * Q4_1_BLOCK;
                let d = crate::gguf::f32_to_half(((next() % 400) as f32) / 200.0 + 0.005);
                let m = crate::gguf::f32_to_half(((next() % 200) as f32 - 100.0) / 400.0);
                payload[b] = (d & 0xff) as u8;
                payload[b + 1] = (d >> 8) as u8;
                payload[b + 2] = (m & 0xff) as u8;
                payload[b + 3] = (m >> 8) as u8;
                for k in 0..16 {
                    payload[b + 4 + k] = (next() % 256) as u8;
                }
            }
        }
        let x: Vec<f32> = (0..ne0)
            .map(|_| ((next() % 2001) as f32 - 1000.0) / 500.0)
            .collect();
        let mut y = vec![0.0f32; n_rows];
        q4_1_matvec_range(&payload, ne0, 0, n_rows, &x, &mut y).unwrap();
        for (r, y_r) in y.iter().enumerate() {
            let row = &payload[r * rb..(r + 1) * rb];
            let mut want = 0.0f32;
            for blk in 0..ne0 / Q4_1_QK {
                let b = blk * Q4_1_BLOCK;
                let d = half_to_f32(u16::from_le_bytes([row[b], row[b + 1]]));
                let m = half_to_f32(u16::from_le_bytes([row[b + 2], row[b + 3]]));
                for j in 0..16 {
                    let qlo = (row[b + 4 + j] & 0x0f) as f32;
                    let qhi = (row[b + 4 + j] >> 4) as f32;
                    want += x[blk * 32 + j] * (d * qlo + m);
                    want += x[blk * 32 + 16 + j] * (d * qhi + m);
                }
            }
            let scale = want.abs().max(1.0);
            assert!((want - y_r).abs() / scale < 1e-4, "row {r}: {want} != {y_r}");
        }
    }
}
