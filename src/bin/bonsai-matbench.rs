//! bonsai-matbench: microbenchmark PQ2_0 mat-vec throughput to size the Rust
//! engine. Loads one tensor fully into RAM and computes y = W @ x across rows,
//! single-threaded and multi-threaded.

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

fn matvec_rows(data: &[u8], ne0: usize, rows: &[u64], x: &[f32]) -> Vec<f32> {
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
    g.read_bytes(t.offset, &mut data).unwrap();
    drop(g);

    let x = rand_floats(3, ne0);
    let total_mac = n_rows * ne0 as u64;
    println!(
        "{name}: {n_rows} rows x {ne0} cols, {:.1} MiB, {:.3} G MACs",
        nbytes as f64 / 1048576.0,
        total_mac as f64 / 1e9
    );

    // warmup: 8 rows
    let _ = matvec_rows(&data, ne0, &[0, 1, 2, 3, 4, 5, 6, 7], &x);

    let all_rows: Vec<u64> = (0..n_rows).collect();

    let t0 = Instant::now();
    let _ = matvec_rows(&data, ne0, &all_rows, &x);
    let dt = t0.elapsed();
    println!(
        "single-thread: {:.3}s -> {:.2} GMAC/s",
        dt.as_secs_f32(),
        total_mac as f64 / dt.as_secs_f64() / 1e9
    );

    let n_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let t0 = Instant::now();
    let mut handles = Vec::new();
    let chunk = all_rows.len() / n_threads + 1;
    for th in 0..n_threads {
        let rows: Vec<u64> = all_rows[th * chunk..((th + 1) * chunk).min(all_rows.len())].to_vec();
        let data = data.clone();
        let x = x.clone();
        handles.push(std::thread::spawn(move || matvec_rows(&data, ne0, &rows, &x)));
    }
    for h in handles {
        let _ = h.join().unwrap();
    }
    let dt = t0.elapsed();
    println!(
        "{n_threads}-thread:  {:.3}s -> {:.2} GMAC/s",
        dt.as_secs_f32(),
        total_mac as f64 / dt.as_secs_f64() / 1e9
    );
}
