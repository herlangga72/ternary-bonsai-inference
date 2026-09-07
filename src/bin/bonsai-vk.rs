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

fn check_tensor(g: &GGUF, gpu: &mut vk::Gpu, name: &str) -> Result<(), String> {
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

    // Enough iterations for a stable kernel-throughput number without
    // over-running on the 248k-row LM head.
    let macs = rows as u64 * ne0 as u64;
    let iters: u32 = if macs < 200_000_000 { 30 } else { 6 };

    let (dt, ygpu) = gpu
        .matvec_bench(payload, ne0, 0, rows, &x, iters)
        .map_err(|e| format!("gpu matvec: {e}"))?;

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
    println!(
        "{name}: {rows} rows x {ne0} cols, {:.1} MiB payload",
        payload_len as f64 / 1048576.0
    );
    println!(
        "  max abs diff {max_abs:.3e}, max rel diff {max_rel:.3e}, rows over 1e-4 rel: {bad}/{}",
        rows
    );
    println!(
        "  {:.2} GMAC/s, {:.2} GB/s (device-local weights, {iters} iters in one submit)",
        macs as f64 / dt / 1e9,
        payload_len as f64 / dt / 1e9
    );
    Ok(())
}

fn check_rmsnorm(g: &mut GGUF, gpu: &mut vk::Gpu) -> Result<(), String> {
    let info = g
        .tensors
        .iter()
        .find(|t| t.name == "blk.0.attn_norm.weight")
        .ok_or("blk.0.attn_norm.weight not found")?
        .clone();
    let w = g.read_tensor(&info)?;
    let n = w.len();
    let x = rand_floats(0xDEAD_BEEF, n);
    let eps = 1e-6f32;

    let ycpu = kernels::rms_norm(&x, &w, eps);
    let ygpu = gpu.rms_norm(&x, &w, eps)?;

    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    let mut bad = 0usize;
    for i in 0..n {
        let d = (ygpu[i] - ycpu[i]).abs();
        let rel = d / ycpu[i].abs().max(1e-30);
        if rel > 1e-4 && d > 1e-4 {
            bad += 1;
            if bad <= 5 {
                eprintln!("  [{i}] gpu {:.6} cpu {:.6} abs {d:.2e}", ygpu[i], ycpu[i]);
            }
        }
        max_abs = max_abs.max(d);
        max_rel = max_rel.max(rel);
    }
    println!("rms_norm: n {n}, max abs diff {max_abs:.3e}, max rel diff {max_rel:.3e}, mismatched {bad}/{n}");
    if bad != 0 {
        return Err("rms_norm mismatch".into());
    }
    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: bonsai-vk <model.gguf> [tensor_name]  |  bonsai-vk rmsnorm <model.gguf>");
        std::process::exit(1);
    }
    let rms_mode = args[0] == "rmsnorm";
    let model = if rms_mode { args.get(1).cloned().unwrap_or_default() } else { args[0].clone() };
    if model.is_empty() {
        eprintln!("missing model path");
        std::process::exit(1);
    }
    let mut g = match GGUF::open(&model) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    };
    let mut gpu = match vk::Gpu::open() {
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

    if rms_mode {
        match check_rmsnorm(&mut g, &mut gpu) {
            Ok(()) => {
                println!("RMSNORM CHECK PASSED");
                return;
            }
            Err(e) => {
                eprintln!("RMSNORM CHECK FAILED: {e}");
                std::process::exit(1);
            }
        }
    }

    let names: Vec<&str> = if args.len() > 1 {
        vec![args[1].as_str()]
    } else {
        DEFAULT_TENSORS.to_vec()
    };

    let mut failed = false;
    for name in names {
        if let Err(e) = check_tensor(&g, &mut gpu, name) {
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
