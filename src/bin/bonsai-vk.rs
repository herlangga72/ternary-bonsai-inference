//! bonsai-vk (G0): PQ2_0 matvec on Vulkan (RADV) vs the slice-based CPU
//! kernel. Same sweep and tolerances as bonsai-opencl, so both GPU backends
//! validate against the same CPU reference.
//!
//! Usage:
//!   bonsai-vk <model.gguf> [tensor_name]

#[path = "../gguf.rs"]
mod gguf;
#[path = "../gdn.rs"]
mod gdn;
#[path = "../kernels.rs"]
mod kernels;
#[path = "../rope.rs"]
mod rope;
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

fn check_elem(gpu: &mut vk::Gpu) -> Result<(), String> {
    let n = 17408usize; // n_ff shape, exercises multi-workgroup dispatch
    let a: Vec<f32> = rand_floats(0x1111, n).iter().map(|v| v * 4.0).collect(); // [-4,4]
    let b: Vec<f32> = rand_floats(0x2222, n).iter().map(|v| v * 2.0).collect();
    let mut worst_abs = 0.0f32;
    let mut worst_rel = 0.0f32;
    for mode in 0u32..=4u32 {
        let expected: Vec<f32> = a
            .iter()
            .zip(b.iter())
            .map(|(&x, &y)| match mode {
                0 => kernels::silu(x),
                1 => kernels::sigmoid(x),
                2 => kernels::softplus(x),
                3 => kernels::silu(x) * y,
                _ => kernels::sigmoid(x) * y,
            })
            .collect();
        let got = gpu.elem(&a, Some(&b), mode)?;
        let mut max_abs = 0.0f32;
        let mut max_rel = 0.0f32;
        for i in 0..n {
            let d = (got[i] - expected[i]).abs();
            max_abs = max_abs.max(d);
            max_rel = max_rel.max(d / expected[i].abs().max(1e-30));
        }
        worst_abs = worst_abs.max(max_abs);
        worst_rel = worst_rel.max(max_rel);
        println!(
            "elem mode {mode}: n {n}, max abs {max_abs:.3e}, max rel {max_rel:.3e}"
        );
    }
    if worst_abs > 1e-5 {
        return Err(format!("elem mismatch: worst abs {worst_abs:.3e}"));
    }
    Ok(())
}

fn check_normrows(gpu: &mut vk::Gpu) -> Result<(), String> {
    // rms rows: like q heads (24 rows x 256); l2 rows: like ssm q/k groups
    let eps = 1e-6f32;
    let xr: Vec<f32> = rand_floats(0x3333, 24 * 256);
    let wr: Vec<f32> = rand_floats(0x4444, 256);
    let yr_cpu = kernels::rms_norm_rows(&xr, &wr, 256, eps)?;
    let yr_gpu = gpu.norm_rows(&xr, Some(&wr), 0, 256, eps)?;
    let mut max_abs = 0.0f32;
    for i in 0..xr.len() {
        max_abs = max_abs.max((yr_gpu[i] - yr_cpu[i]).abs());
    }
    println!("norm_rows rms (24x256): max abs {max_abs:.3e}");
    if max_abs > 1e-5 {
        return Err("norm_rows rms mismatch".into());
    }

    let xl: Vec<f32> = rand_floats(0x5555, 16 * 128);
    let yl_cpu = kernels::l2_norm_rows(&xl, 128, 1e-12)?;
    let yl_gpu = gpu.norm_rows(&xl, None, 1, 128, 1e-12)?;
    let mut max_abs = 0.0f32;
    for i in 0..xl.len() {
        max_abs = max_abs.max((yl_gpu[i] - yl_cpu[i]).abs());
    }
    println!("norm_rows l2 (16x128): max abs {max_abs:.3e}");
    if max_abs > 1e-5 {
        return Err("norm_rows l2 mismatch".into());
    }
    Ok(())
}

fn check_softmax(gpu: &mut vk::Gpu) -> Result<(), String> {
    // attention-row shape with a masked tail (-inf like the causal mask)
    let n = 2048usize;
    let mut x: Vec<f32> = rand_floats(0x6666, n).iter().map(|v| v * 6.0).collect();
    for v in x[n - 512..].iter_mut() {
        *v = f32::NEG_INFINITY;
    }
    let y_cpu = kernels::softmax_rows(&x, n)?;
    let y_gpu = gpu.softmax_row(&x)?;
    let mut max_abs = 0.0f32;
    let mut bad = 0usize;
    for i in 0..n {
        let d = (y_gpu[i] - y_cpu[i]).abs();
        if d > 1e-5 {
            bad += 1;
        }
        max_abs = max_abs.max(d);
    }
    println!("softmax_row (n {n}, 512 masked): max abs {max_abs:.3e}, mismatched {bad}/{n}");
    if bad != 0 {
        return Err("softmax_row mismatch".into());
    }
    Ok(())
}

fn check_rope(gpu: &mut vk::Gpu) -> Result<(), String> {
    // full-attention q layout: 24 heads x 256, n_dims 64, sections [11,11,10,0]
    let hd = 256usize;
    let n_dims = 64usize;
    let nheads = 24usize;
    let pos: i64 = 37;
    let freq: f32 = 1e7;
    let sections_i32 = [11i32, 11, 10, 0];
    let heads: Vec<f32> = rand_floats(0x7777, nheads * hd);
    let mut y_cpu = heads.clone();
    for h in 0..nheads {
        rope::rope_imrope(
            &mut y_cpu[h * hd..(h + 1) * hd],
            pos,
            pos,
            pos,
            pos,
            n_dims,
            sections_i32,
            freq,
        );
    }
    let y_gpu = gpu.rope_imrope(&heads, hd, n_dims, pos as u32, freq, [11, 11, 10, 0])?;
    let mut max_abs = 0.0f32;
    for i in 0..heads.len() {
        max_abs = max_abs.max((y_gpu[i] - y_cpu[i]).abs());
    }
    println!("rope_imrope ({nheads} heads x {hd}, pos {pos}): max abs {max_abs:.3e}");
    if max_abs > 1e-4 {
        return Err("rope_imrope mismatch".into());
    }
    Ok(())
}

fn check_gdn(gpu: &mut vk::Gpu) -> Result<(), String> {
    use vk::{GDN_DI, GDN_DK, GDN_HV, GDN_INS_LEN, GDN_STATE_LEN};
    let s = 128usize;
    let small = |seed: u64, n: usize| -> Vec<f32> {
        rand_floats(seed, n).iter().map(|v| v * 0.05).collect()
    };
    let q = small(0xAAAA, GDN_DK);
    let k = small(0xBBBB, GDN_DK);
    let v = small(0xCCCC, GDN_DI);
    // decay gates in (-3, 0) so exp(gate) stays in (0, 1)
    let gate: Vec<f32> = rand_floats(0xDDDD, GDN_HV).iter().map(|v| v * 1.5 - 0.75).collect();
    let beta: Vec<f32> = rand_floats(0xEEEE, GDN_HV).iter().map(|v| v.abs() * 0.5 + 0.1).collect();
    let state = small(0xFFFF, GDN_STATE_LEN);

    let mut cpu_state = state.clone();
    let mut cpu_attn = vec![0.0f32; GDN_DI];
    gdn::gdn_step(&q, &k, &v, &gate, &beta, &mut cpu_state, &mut cpu_attn);

    let mut ins = Vec::with_capacity(GDN_INS_LEN);
    ins.extend_from_slice(&q);
    ins.extend_from_slice(&k);
    ins.extend_from_slice(&v);
    ins.extend_from_slice(&gate);
    ins.extend_from_slice(&beta);
    let (gpu_attn, gpu_state) = gpu.gdn_step(&ins, &state)?;

    let mut attn_max = 0.0f32;
    for i in 0..GDN_DI {
        attn_max = attn_max.max((gpu_attn[i] - cpu_attn[i]).abs());
    }
    let mut state_max = 0.0f32;
    for i in 0..GDN_STATE_LEN {
        state_max = state_max.max((gpu_state[i] - cpu_state[i]).abs());
    }
    println!(
        "gdn_step: attn max abs {attn_max:.3e}, state max abs {state_max:.3e} ({} heads x {s})",
        GDN_HV
    );
    if attn_max > 1e-4 || state_max > 1e-4 {
        return Err(format!("gdn_step mismatch (attn {attn_max:.3e}, state {state_max:.3e})"));
    }
    Ok(())
}

fn check_attn(gpu: &mut vk::Gpu) -> Result<(), String> {
    let nheads = 24usize;
    let kvheads = 4usize;
    let hd = 256usize;
    let n_pos = 37usize; // odd on purpose: exercises the +256 loop tail
    let kv_stride = kvheads * hd;
    let scale = 1.0 / (hd as f32).sqrt();

    let q: Vec<f32> = rand_floats(0x1010, nheads * hd);
    let kcache: Vec<f32> = rand_floats(0x2020, n_pos * kv_stride);
    let vcache: Vec<f32> = rand_floats(0x3030, n_pos * kv_stride);

    // CPU reference (mirrors forward.rs step 4)
    let mut expect = vec![0.0f32; nheads * hd];
    for h in 0..nheads {
        let kv = h / (nheads / kvheads);
        let qh = &q[h * hd..(h + 1) * hd];
        let mut scores = Vec::with_capacity(n_pos);
        for p in 0..n_pos {
            let kp = &kcache[p * kv_stride + kv * hd..p * kv_stride + (kv + 1) * hd];
            let mut s = 0.0f32;
            for i in 0..hd {
                s += qh[i] * kp[i];
            }
            scores.push(s * scale);
        }
        let mx = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = scores.iter().map(|s| (*s - mx).exp()).sum();
        for d in 0..hd {
            let mut acc = 0.0f32;
            for p in 0..n_pos {
                let pr = ((scores[p] - mx).exp()) / sum;
                acc += pr * vcache[p * kv_stride + kv * hd + d];
            }
            expect[h * hd + d] = acc;
        }
    }

    let got = gpu.attn_step(&q, &kcache, &vcache, n_pos, nheads, kvheads, hd)?;
    let mut max_abs = 0.0f32;
    let mut bad = 0usize;
    for i in 0..expect.len() {
        let d = (got[i] - expect[i]).abs();
        if d > 1e-4 {
            bad += 1;
        }
        max_abs = max_abs.max(d);
    }
    println!(
        "attn_step ({} heads x {hd}, {kvheads} kv, {n_pos} pos): max abs {max_abs:.3e}, mismatched {bad}/{}",
        nheads,
        expect.len()
    );
    if bad != 0 {
        return Err("attn_step mismatch".into());
    }
    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!(
            "usage: bonsai-vk <model.gguf> [tensor_name]\n   or: bonsai-vk rmsnorm <model.gguf>\n   or: bonsai-vk normrows | elem | softmax | rope"
        );
        std::process::exit(1);
    }
    let rms_mode = args[0] == "rmsnorm";
    let elem_mode = args[0] == "elem";
    let normrows_mode = args[0] == "normrows";
    let softmax_mode = args[0] == "softmax";
    let rope_mode = args[0] == "rope";
    let gdn_mode = args[0] == "gdn";
    let attn_mode = args[0] == "attn";
    let no_model = elem_mode || normrows_mode || softmax_mode || rope_mode || gdn_mode || attn_mode;
    let model = if rms_mode { args.get(1).cloned().unwrap_or_default() } else { args[0].clone() };
    if model.is_empty() {
        eprintln!("missing model path");
        std::process::exit(1);
    }
    let mut g: Option<GGUF> = None;
    if !no_model {
        g = Some(match GGUF::open(&model) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("error: {e}");
                std::process::exit(1);
            }
        });
    }
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

    if elem_mode {
        match check_elem(&mut gpu) {
            Ok(()) => {
                println!("ELEM CHECK PASSED");
                return;
            }
            Err(e) => {
                eprintln!("ELEM CHECK FAILED: {e}");
                std::process::exit(1);
            }
        }
    }

    if softmax_mode {
        match check_softmax(&mut gpu) {
            Ok(()) => {
                println!("SOFTMAX CHECK PASSED");
                return;
            }
            Err(e) => {
                eprintln!("SOFTMAX CHECK FAILED: {e}");
                std::process::exit(1);
            }
        }
    }

    if rope_mode {
        match check_rope(&mut gpu) {
            Ok(()) => {
                println!("ROPE CHECK PASSED");
                return;
            }
            Err(e) => {
                eprintln!("ROPE CHECK FAILED: {e}");
                std::process::exit(1);
            }
        }
    }

    if gdn_mode {
        match check_gdn(&mut gpu) {
            Ok(()) => {
                println!("GDN CHECK PASSED");
                return;
            }
            Err(e) => {
                eprintln!("GDN CHECK FAILED: {e}");
                std::process::exit(1);
            }
        }
    }

    if attn_mode {
        match check_attn(&mut gpu) {
            Ok(()) => {
                println!("ATTN CHECK PASSED");
                return;
            }
            Err(e) => {
                eprintln!("ATTN CHECK FAILED: {e}");
                std::process::exit(1);
            }
        }
    }

    if normrows_mode {
        match check_normrows(&mut gpu) {
            Ok(()) => {
                println!("NORMROWS CHECK PASSED");
                return;
            }
            Err(e) => {
                eprintln!("NORMROWS CHECK FAILED: {e}");
                std::process::exit(1);
            }
        }
    }

    if rms_mode {
        let g = g.as_mut().unwrap();
        match check_rmsnorm(g, &mut gpu) {
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

    let g = g.as_ref().unwrap();
    let mut failed = false;
    for name in names {
        if let Err(e) = check_tensor(g, &mut gpu, name) {
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
