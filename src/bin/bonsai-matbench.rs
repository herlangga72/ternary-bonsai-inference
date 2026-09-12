//! bonsai-matbench: PQ2_0 mat-vec throughput microbenchmark.
//!
//! Measures the **production** row kernel (`kernels::pq2_matvec_range`) plus a
//! scalar reference, single- and multi-threaded, on one real tensor. Each
//! variant runs several times and the best (min) time is reported to reduce
//! noise from other processes sharing the machine.

#[path = "../gguf.rs"]
mod gguf;
#[path = "../kernels.rs"]
mod kernels;

use gguf::GGUF;
use std::time::Instant;

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

/// Scalar reference (old kernel): per-element shift/mask + float multiply.
fn matvec_rows_scalar(data: &[u8], ne0: usize, rows: &[u64], x: &[f32]) -> Vec<f32> {
    let row_bytes = kernels::pq2_row_bytes(ne0);
    let mut y = Vec::with_capacity(rows.len());
    for &r in rows {
        let raw = &data[r as usize * row_bytes..(r as usize + 1) * row_bytes];
        let mut acc = 0.0f32;
        for block in 0..ne0.div_ceil(128) {
            let b = block * 34;
            let scale = gguf::half_to_f32(u16::from_le_bytes([raw[b], raw[b + 1]]));
            let qs = &raw[b + 2..b + 2 + 32];
            let start = block * 128;
            let end = (start + 128).min(ne0);
            for j in start..end {
                let code = (qs[(j - start) / 4] >> (((j - start) % 4) * 2)) & 0x03;
                acc += x[j] * ((code as i32 - 1) as f32 * scale);
            }
        }
        y.push(acc);
    }
    y
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let model = args.first().cloned().unwrap();
    let reps: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(5);
    let mut g = GGUF::open(&model).unwrap();
    let name = "blk.0.ffn_up.weight";
    let t = g
        .tensors
        .iter()
        .find(|t| t.name == name)
        .cloned()
        .expect("tensor not found");
    let ne0 = t.dims[0] as usize;
    let n_rows = t.n_elem() / ne0 as u64;
    let nbytes = g.tensor_nbytes(&t) as usize;
    let mut data = vec![0u8; nbytes];
    g.read_bytes(g.tensor_data_offset(&t), &mut data).unwrap();
    drop(g);

    let x = rand_floats(3, ne0);
    let total_mac = n_rows * ne0 as u64;
    let gmac = |s: f64| total_mac as f64 / s / 1e9;
    println!(
        "{name}: {n_rows} rows x {ne0} cols, {:.1} MiB, {:.3} G MACs, best of {reps}",
        nbytes as f64 / 1048576.0,
        total_mac as f64 / 1e9
    );

    let all_rows: Vec<u64> = (0..n_rows).collect();
    // warmup (first-touch the payload + LUT)
    let _ = matvec_rows_scalar(&data, ne0, &all_rows[..256], &x);
    let mut y = vec![0.0f32; n_rows as usize];

    // ---- scalar, single thread ----
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let t0 = Instant::now();
        let _ = matvec_rows_scalar(&data, ne0, &all_rows, &x);
        best = best.min(t0.elapsed().as_secs_f64());
    }
    let scalar1 = best;
    println!("scalar  1-thread: {scalar1:.3}s -> {:.2} GMAC/s", gmac(scalar1));

    // ---- production kernel, single thread ----
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let t0 = Instant::now();
        kernels::pq2_matvec_range_single(&data, ne0, 0, n_rows as usize, &x, &mut y).unwrap();
        best = best.min(t0.elapsed().as_secs_f64());
    }
    let prod1 = best;
    println!(
        "kernel  1-thread: {prod1:.3}s -> {:.2} GMAC/s  ({:.2}x scalar)",
        gmac(prod1),
        scalar1 / prod1
    );

    // ---- production kernel, multi thread ----
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let t0 = Instant::now();
        kernels::pq2_matvec_range(&data, ne0, 0, n_rows as usize, &x, &mut y).unwrap();
        best = best.min(t0.elapsed().as_secs_f64());
    }
    let n_threads = kernels::scaled_threads(
        std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4),
    );
    println!("kernel {n_threads}-thread: {best:.3}s -> {:.2} GMAC/s", gmac(best));
}
