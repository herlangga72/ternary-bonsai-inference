//! bonsai-opencl (G0): PQ2_0 matvec on the GPU vs the slice-based CPU kernel.
//!
//! Usage:
//!   bonsai-opencl <model.gguf> [tensor_name]
//!
//! With no tensor name it sweeps a representative set (recurrent qkv/ffn,
//! full-attention q, and the LM head). For each tensor it uploads the packed
//! payload + a deterministic activation vector, runs `pq2_matvec` on the GPU,
//! and compares every output row against `kernels::pq2_matvec_range`. Also
//! reports throughput (GMAC/s and GB/s) as a G0 data point.
//!
//! Everything goes through the runtime dlopen OpenCL binding in opencl.rs, so
//! this binary runs anywhere; it only fails when no OpenCL GPU is present.

#[path = "../gguf.rs"]
mod gguf;
#[path = "../kernels.rs"]
mod kernels;
#[path = "../opencl.rs"]
mod opencl;
#[path = "../cl_kernels.rs"]
mod cl_kernels;

use gguf::GGUF;
use opencl::{Buffer, Context, Program};
use std::time::Instant;

/// Representative tensors covering every ne0 / row shape the engine touches.
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

fn as_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn check_tensor(
    g: &GGUF,
    ctx: &Context,
    prog: &Program,
    name: &str,
    verbose: bool,
) -> Result<(), String> {
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

    // ---- upload -------------------------------------------------------------
    eprintln!("[opencl] {name}: upload {} MiB + x {} KiB...", payload_len as f64 / 1048576.0, ne0 * 4 / 1024);
    // Host-accessible everywhere: on gfx902 + ROCm 6.0, plain device-local
    // buffers hang on both large writes and readback. On the iGPU there is no
    // perf difference (weights live in system RAM anyway); G2 on the 7600 can
    // move the weight store to device-local VRAM once the supported driver is
    // in place.
    const HA: u64 = 0x1 | 0x10; // CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR
    let wbuf = Buffer::create_with_flags(ctx, HA, payload_len)?;
    wbuf.write(ctx, payload)?;
    let xbuf = Buffer::create_with_flags(ctx, HA, ne0 * 4)?;
    xbuf.write(ctx, unsafe {
        std::slice::from_raw_parts(x.as_ptr() as *const u8, ne0 * 4)
    })?;
    // Output must be host-accessible on this stack: plain device-local buffers
    // hang on clEnqueueReadBuffer (gfx902 + ROCm 6.0 defect). CL_MEM_READ_WRITE
    // | CL_MEM_ALLOC_HOST_PTR routes the readback through a working path and is
    // portable to the 7600.
    let ybuf = Buffer::create_with_flags(ctx, 0x1 | 0x10, rows * 4)?;

    let kernel = prog.kernel("pq2_matvec")?;
    let max_wg = kernel.max_work_group_size(ctx.dev).unwrap_or(256);
    kernel.arg_raw(0, std::mem::size_of::<opencl::cl_mem>(), &wbuf.as_mem() as *const _ as *const _)?;
    kernel.arg_raw(1, std::mem::size_of::<opencl::cl_mem>(), &xbuf.as_mem() as *const _ as *const _)?;
    kernel.arg_raw(2, std::mem::size_of::<opencl::cl_mem>(), &ybuf.as_mem() as *const _ as *const _)?;
    kernel.arg(3, &(ne0 as u32))?;
    kernel.arg(4, &0u32)?;
    eprintln!("[opencl] {name}: kernel args set, max_wg {max_wg}");
    let local = 256usize.min(max_wg).min(rows.max(1));

    // ---- warmup + timed runs ------------------------------------------------
    eprintln!("[opencl] {name}: launching {rows} work-items (local {local})...");
    opencl::run(ctx, &kernel, rows, local)?;
    let n_iter = 3;
    let t0 = Instant::now();
    for _ in 0..n_iter {
        opencl::run(ctx, &kernel, rows, local)?;
    }
    let dt = t0.elapsed().as_secs_f64() / n_iter as f64;

    // ---- read back + CPU reference ------------------------------------------
    eprintln!("[opencl] {name}: reading back {} rows...", rows);
    let mut ybytes = vec![0u8; rows * 4];
    ybuf.read(ctx, &mut ybytes)?;
    let ygpu = as_f32(&ybytes);

    let mut ycpu = vec![0.0f32; rows];
    kernels::pq2_matvec_range(payload, ne0, 0, rows, &x, &mut ycpu)
        .map_err(|e| format!("cpu matvec: {e}"))?;

    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    let mut bad = 0usize;
    for i in 0..rows {
        let d = (ygpu[i] - ycpu[i]).abs();
        // Relative error on near-zero outputs (catastrophic cancellation) is
        // meaningless; flag only rows with a real absolute mismatch too.
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
    if verbose {
        println!("  first 8 gpu: {:?}", &ygpu[..8.min(rows)]);
        println!("  first 8 cpu: {:?}", &ycpu[..8.min(rows)]);
    }
    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: bonsai-opencl <model.gguf> [tensor_name]");
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

    let ctx = match opencl::open_gpu() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no usable OpenCL GPU: {e}");
            std::process::exit(1);
        }
    };
    println!(
        "device: {} ({} CUs @ {} MHz, {} MiB local, {:.1} GiB global)",
        ctx.info.name,
        ctx.info.compute_units,
        ctx.info.max_clock_mhz,
        ctx.info.local_mem / 1048576,
        ctx.info.global_mem as f64 / (1 << 30) as f64
    );

    let prog = match Program::build(&ctx, cl_kernels::PQ2_MATVEC) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("kernel build failed:\n{e}");
            std::process::exit(1);
        }
    };

    let names: Vec<&str> = if args.len() > 1 {
        vec![args[1].as_str()]
    } else {
        DEFAULT_TENSORS.to_vec()
    };

    let mut failed = false;
    for name in names {
        if let Err(e) = check_tensor(&g, &ctx, &prog, name, false) {
            eprintln!("{name}: {e}");
            failed = true;
        }
    }
    if failed {
        eprintln!("G0 CHECK FAILED");
        std::process::exit(1);
    }
    println!("G0 CHECK PASSED");
}
