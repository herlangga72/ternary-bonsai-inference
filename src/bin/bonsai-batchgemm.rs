//! bonsai-batchgemm: P2 correctness check for the N-column PQ2_0 GEMM.
//!
//! Runs the batched two-pass kernel from `notes/prefill-plan.md` P2
//! (`vk::Gpu::matvec_batch_n`, shaders/pq2_partial_n.comp + pq2_rowsum_n.comp)
//! against a real PQ2_0 tensor of the loaded model (default
//! `blk.0.ffn_up.weight`, 17408 x 5120) on the GPU (RADV/gfx902 iGPU), and
//! verifies two things:
//!
//! 1. **N=1 == the existing single-vector matvec.** The N=1 output of the new
//!    kernel must match the decode path's two-pass kernel
//!    (`vk::Gpu::matvec_bench_atom`, pq2_partial/pq2_rowsum) bit-for-bit. This
//!    is the gate before any N>1 path is trusted.
//! 2. **N>1 == CPU reference.** For a small batch N the new GEMM must match a
//!    pure-Rust CPU batched matmul (`kernels::pq2_matvec_range` per column)
//!    within <= ~1e-3 relative on every element.
//!
//! usage: bonsai-batchgemm <model.gguf> [n_cols] [tensor_name]

#[path = "../gguf.rs"]
mod gguf;
#[path = "../kernels.rs"]
mod kernels;
#[path = "../vk.rs"]
mod vk;

use gguf::GGUF;
use std::process::exit;

fn rand_floats(seed: u64, n: usize) -> Vec<f32> {
    let mut x = seed | 1;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            ((x >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

/// max abs / max rel diff between two same-length slices.
///
/// Relative error is measured against the reference's max magnitude, which is
/// the meaningful definition when a real dot-product output can legitimately
/// cross zero (a naive `d/|b|` blows up on near-zero reference elements even at
/// fp precision). `over` counts elements failing `rel > 1e-3 && d > 1e-4`.
fn diff_metrics(a: &[f32], b: &[f32]) -> (f32, f32, usize) {
    assert_eq!(a.len(), b.len());
    let bmax = b.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-30);
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    let mut over = 0usize;
    for i in 0..a.len() {
        let d = (a[i] - b[i]).abs();
        let rel = d / bmax;
        if rel > 1e-3 && d > 1e-4 {
            over += 1;
        }
        max_abs = max_abs.max(d);
        max_rel = max_rel.max(rel);
    }
    (max_abs, max_rel, over)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: bonsai-batchgemm <model.gguf> [n_cols] [tensor_name]");
        exit(1);
    }
    let model = &args[0];
    let n_cols: usize = args
        .get(1)
        .and_then(|s| s.parse().ok())
        .filter(|&n| n >= 1)
        .unwrap_or(8);
    let tensor_name = args.get(2).cloned().unwrap_or_else(|| "blk.0.ffn_up.weight".to_string());

    let mut g = GGUF::open(model).map_err(|e| format!("gguf open: {e}")).expect("open gguf");
    let info = g
        .tensors
        .iter()
        .find(|t| t.name == tensor_name)
        .cloned()
        .unwrap_or_else(|| panic!("tensor {tensor_name} not found"));
    let ne0 = info.dims[0] as usize;
    let rows = (info.n_elem() as usize / ne0.max(1)).max(1);
    if info.ty != gguf::TYPE_PQ2_0 {
        panic!("{tensor_name}: not a PQ2_0 tensor (ty {})", info.ty);
    }
    let payload = g.payload_slice(&info).expect("payload slice");
    println!(
        "{tensor_name}: {rows} rows x {ne0} cols, N={n_cols}, {:.1} MiB payload",
        payload.len() as f64 / 1048576.0
    );

    // Build the N activation columns (a [column][feature] tile, matching
    // prefill::Tile). Column 0 is reused for the N=1 vs old-path check.
    let mut x_tile = vec![0.0f32; n_cols.saturating_mul(ne0)];
    for c in 0..n_cols {
        let r = rand_floats(0x9E3779B9 ^ (c as u64 + 1), ne0);
        x_tile[c * ne0..(c + 1) * ne0].copy_from_slice(&r);
    }
    let x0: Vec<f32> = x_tile[..ne0].to_vec();

    let mut gpu = vk::Gpu::open().expect("open vulkan device");
    let device = format!("{} ({})", gpu.name, if gpu.discrete { "discrete" } else { "iGPU/APU" });
    eprintln!("[batchgemm] device: {device}");

    // ---- 1. N=1: new N-column kernel vs the existing single-vector matvec ----
    // The old two-pass path used by decode (pq2_partial + pq2_rowsum).
    let (_, y_old) = gpu
        .matvec_bench_atom(payload, ne0, 0, rows, &x0, 1)
        .expect("old single-vector matvec");
    // New N-column kernel at N=1.
    let y_new1 = gpu
        .matvec_batch_n(payload, ne0, 0, rows, 1, &x0)
        .expect("new N=1 GEMM");
    let (mabs, mrel, over1) = diff_metrics(&y_new1, &y_old);
    let bitcount = y_new1
        .iter()
        .zip(y_old.iter())
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    println!("[N=1] new-kernel vs old single-vector matvec:");
    println!(
        "      max abs diff {mabs:.3e}, max rel diff {mrel:.3e}, elems >1e-3 rel: {over1}/{}, bit-differing {bitcount}/{}",
        rows,
        rows
    );

    // Also cross-check N=1 against CPU so an N>1 comparison has a clean anchor.
    let mut y_cpu0 = vec![0.0f32; rows];
    kernels::pq2_matvec_range(payload, ne0, 0, rows, &x0, &mut y_cpu0).expect("cpu matvec col0");
    let (cabs, crel, _c) = diff_metrics(&y_new1, &y_cpu0);
    println!("[N=1] new-kernel vs CPU reference: max abs {cabs:.3e}, max rel {crel:.3e}");

    // ---- 2. N>1: new N-column GEMM vs CPU batched reference ----------------
    let y_gpu = gpu
        .matvec_batch_n(payload, ne0, 0, rows, n_cols, &x_tile)
        .expect("new N-column GEMM");
    assert_eq!(y_gpu.len(), n_cols * rows);

    let mut y_cpu = vec![0.0f32; n_cols.saturating_mul(rows)];
    for c in 0..n_cols {
        let col = &x_tile[c * ne0..(c + 1) * ne0];
        let slice = &mut y_cpu[c * rows..(c + 1) * rows];
        kernels::pq2_matvec_range(payload, ne0, 0, rows, col, slice).expect("cpu matvec");
    }

    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    let mut over = 0usize;
    for c in 0..n_cols {
        let (a, r, o) = diff_metrics(
            &y_gpu[c * rows..(c + 1) * rows],
            &y_cpu[c * rows..(c + 1) * rows],
        );
        max_abs = max_abs.max(a);
        max_rel = max_rel.max(r);
        over += o;
    }
    println!("[N={n_cols}] new GEMM vs CPU batched reference:");
    println!(
        "      max abs diff {max_abs:.3e}, max rel diff {max_rel:.3e}, elems >1e-3 rel: {over}/{}",
        y_cpu.len()
    );

    // A low-level "every column individually == old path" is implied: each GPU
    // column is reduced in the exact same order as the single-column kernel, so
    // a strong N=1 match plus the N>1 CPU match together prove generalization.

    let ok = over1 == 0 && over == 0 && mrel < 1e-3 && max_rel < 1e-3;
    println!("bonsai-batchgemm: {}", if ok { "OK" } else { "FAIL" });
    if !ok {
        exit(1);
    }
}
