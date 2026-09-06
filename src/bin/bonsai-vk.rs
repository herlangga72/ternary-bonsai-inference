//! bonsai-vk (G0): PQ2_0 matvec on Vulkan (RADV) vs the slice-based CPU
//! kernel. Same sweep and tolerances as bonsai-opencl, so both GPU backends
//! validate against the same CPU reference.
//!
//! Usage:
//!   bonsai-vk <model.gguf> [tensor_name]

#[path = "../gguf.rs"]
mod gguf;
#[path = "../kernels.rs"]
mod kernels;
#[path = "../vk.rs"]
mod vk;

use gguf::GGUF;
use std::time::Instant;

const DEFAULT_TENSORS: &[&str] = &[
    "blk.0.ffn_up.weight",   // 17408 x 5120 (recurrent FFN gate/up shape)
    "blk.0.attn_qkv.weight", // 10240 x 5120 (fused q|k|v)
    "blk.0.ffn_down.weight", // 5120 x 17408 (wide-ne0 shape)
    "blk.3.attn_q.weight",   // 12288 x 5120 (full-attention q, fused q|gate)
    "output.weight",         // 248320 x 5120 (LM head)
];

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

fn check_tensor(g: &GGUF, gpu: &vk::Gpu, name: &str) -> Result<(), String> {
    let info = g
        .tensors
        .iter()
        .find(|t| t.name == name)
        .ok_or_else(|| format!("tensor '{name}' not found"))?
        .clone();
    let ne0 = info.dims[0] as usize;
    let rows = kernels::n_rows(&info) as usize;
    if rows == 0 || info.ty != gguf::TYPE_PQ2_0 {
        return Err(format!("{name}: not a PQ2_0 matrix (ty {} rows {rows})", info.ty));
    }
    let payload = g.payload_slice(&info)?;
    let payload_len = payload.len();
    let x = rand_floats(0x9E3779B9 ^ name.len() as u64, ne0);

    // warmup once, then time 3 runs
    gpu.pq2_matvec(payload, ne0, 0, rows, &x)?;
    let n_iter = 3;
    let t0 = Instant::now();
    for _ in 0..n_iter {
        gpu.pq2_matvec(payload, ne0, 0, rows, &x)?;
    }
    let dt = t0.elapsed().as_secs_f64() / n_iter as f64;

    let ygpu = gpu.pq2_matvec(payload, ne0, 0, rows, &x)?;
    let mut ycpu = vec![0.0f32; rows];
    kernels::pq2_matvec_range(payload, ne0, 0, rows, &x, &mut ycpu)
        .map_err(|e| format!("cpu matvec: {e}"))?;

    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    let mut bad = 0usize;
    for i in 0..rows {
        let d = (ygpu[i] - ycpu[i]).abs();
        let rel = d / ycpu[i].abs().max(1e-30);
        if rel > 1e-4 && d > 1e-3 {
            bad += 1;
            if bad <= 5 {
                eprintln!("  row {i}: gpu {:.6} cpu {:.6} abs {d:.2e} rel {rel:.2e}", ygpu[i], ycpu[i]);
            }
        }
        max_abs = max_abs.max(d);
        max_rel = max_rel.max(rel);
    }
    let macs = rows as f64 * ne0 as f64;
    println!(
        "{name}: {rows} rows x {ne0} cols, {:.1} MiB payload",
        payload_len as f64 / 1048576.0
    );
    println!(
        "  max abs diff {max_abs:.3e}, max rel diff {max_rel:.3e}, rows over 1e-4 rel: {bad}/{}",
        rows
    );
    println!(
        "  {:.2} GMAC/s, {:.2} GB/s (payload, {n_iter} iters)",
        macs / dt / 1e9,
        payload_len as f64 / dt / 1e9
    );
    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: bonsai-vk <model.gguf> [tensor_name]");
        std::process::exit(1);
    }
    let model = &args[0];
    let g = match GGUF::open(model) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    };
    let gpu = match vk::Gpu::open() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("no usable Vulkan GPU: {e}");
            std::process::exit(1);
        }
    };
    println!(
        "device: {} ({})",
        gpu.name,
        if gpu.discrete { "discrete" } else { "integrated" }
    );

    let names: Vec<&str> = if args.len() > 1 {
        vec![args[1].as_str()]
    } else {
        DEFAULT_TENSORS.to_vec()
    };

    let mut failed = false;
    for name in names {
        if let Err(e) = check_tensor(&g, &gpu, name) {
            eprintln!("{name}: {e}");
            failed = true;
        }
    }
    if failed {
        eprintln!("G0 VULKAN CHECK FAILED");
        std::process::exit(1);
    }
    println!("G0 VULKAN CHECK PASSED");
}
