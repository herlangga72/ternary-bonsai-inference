//! bonsai-clprobe: minimal OpenCL execution probe (G0 bring-up).
//!
//! Walks through the pipeline with a trivial kernel so we can tell whether a
//! hang is in device execution, buffer transfers, or the real kernel. Each
//! step prints before it starts, so the last printed line names the offender.

#[path = "../opencl.rs"]
mod opencl;
#[path = "../cl_kernels.rs"]
mod cl_kernels;

use opencl::{Buffer, Program};

fn main() {
    eprintln!("[probe] open_gpu...");
    let ctx = match opencl::open_gpu() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no OpenCL GPU: {e}");
            std::process::exit(1);
        }
    };
    eprintln!(
        "[probe] device ok: {} ({} CUs)",
        ctx.info.name, ctx.info.compute_units
    );

    eprintln!("[probe] build add1...");
    let prog = Program::build(&ctx, cl_kernels::ADD1).expect("build add1");
    eprintln!("[probe] build ok");

    let n = 4096usize;
    eprintln!("[probe] create x/y buffers ({n} floats)...");
    let x = Buffer::create(&ctx, true, n * 4).expect("create x");
    // Host-accessible device allocation: some stacks only signal D2H reads
    // reliably on buffers created with CL_MEM_ALLOC_HOST_PTR.
    let y = Buffer::create_with_flags(&ctx, 0x1 | 0x10, n * 4).expect("create y");

    let data: Vec<f32> = (0..n).map(|i| i as f32 * 0.5).collect();
    eprintln!("[probe] write x...");
    x.write(&ctx, unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, n * 4)
    })
    .expect("write x");
    eprintln!("[probe] write x ok");

    eprintln!("[probe] create kernel + args...");
    let k = prog.kernel("add1").expect("kernel add1");
    k.arg_raw(0, std::mem::size_of::<opencl::cl_mem>(), &y.as_mem() as *const _ as *const _)
        .expect("arg0");
    k.arg_raw(1, std::mem::size_of::<opencl::cl_mem>(), &x.as_mem() as *const _ as *const _)
        .expect("arg1");

    eprintln!("[probe] run add1 (global {n}, local 256)...");
    opencl::run(&ctx, &k, n, 256).expect("run add1");
    eprintln!("[probe] run ok");

    let mut out = vec![0u8; n * 4];
    eprintln!("[probe] read y...");
    y.read(&ctx, &mut out).expect("read y");
    eprintln!("[probe] read ok");
    let vals: Vec<f32> = out
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .take(4)
        .collect();
    println!("[probe] y[0..4] = {vals:?} (expect [1.0, 1.5, 2.0, 2.5])");
    if (vals[0] - 1.0).abs() < 1e-6 {
        println!("PROBE OK: device executes kernels and copies");
    } else {
        println!("PROBE FAILED: wrong values");
        std::process::exit(1);
    }
}
