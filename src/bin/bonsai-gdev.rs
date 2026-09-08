//! bonsai-gdev: full device (single-submit) decode vs the CPU Decoder.
//!
//! usage: bonsai-gdev <model.gguf> [first_token] ...
//! Runs the token sequence through GDev (all ops in one command buffer per
//! token) and Decoder::forward_hidden on the CPU, comparing each hidden and
//! the final LM-head logits.

#[path = "../gguf.rs"]
mod gguf;
#[path = "../gdn.rs"]
mod gdn;
#[path = "../kernels.rs"]
mod kernels;
#[path = "../rope.rs"]
mod rope;
#[path = "../weights.rs"]
mod weights;
#[path = "../vk.rs"]
mod vk;
#[path = "../forward.rs"]
mod forward;
#[path = "../gdev.rs"]
mod gdev;

use gguf::GGUF;
use std::time::Instant;

fn embed_of(g: &GGUF, token: u32) -> Vec<f32> {
    let info = g
        .tensors
        .iter()
        .find(|t| t.name == "token_embd.weight")
        .expect("token_embd.weight")
        .clone();
    let ne0 = info.dims[0] as usize;
    let row_bytes = kernels::pq2_row_bytes(ne0);
    let raw = g
        .slice_at(g.tensor_data_offset(&info) + token as u64 * row_bytes as u64, row_bytes)
        .expect("embed row");
    kernels::decode_pq2_0_row(raw, ne0)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: bonsai-gdev <model.gguf> [token ...]");
        std::process::exit(1);
    }
    let model = args[0].clone();
    let toks: Vec<u32> = if args.len() > 1 {
        args[1..].iter().filter_map(|s| s.parse().ok()).collect()
    } else {
        vec![248045, 846, 198, 3710, 369]
    };

    let g = GGUF::open(&model).expect("open gguf");
    let mut dec = forward::Decoder::open(&model).expect("cpu decoder");
    let mut dev = gdev::GDev::open(&model).expect("gdev open");

    let mut worst_rel = 0.0f32;
    let mut diffs = 0usize;
    for (pos, &tok) in toks.iter().enumerate() {
        let embed = embed_of(&g, tok);
        let t0 = Instant::now();
        let h_cpu = dec.forward_hidden(tok, pos).expect("cpu forward");
        let tc = t0.elapsed().as_secs_f32();
        let t0 = Instant::now();
        let h_gpu = dev.forward_token(pos, &embed).expect("gpu forward");
        let tg = t0.elapsed().as_secs_f32();

        let mut max_abs = 0.0f32;
        for i in 0..h_cpu.len() {
            max_abs = max_abs.max((h_cpu[i] - h_gpu[i]).abs());
        }
        let rel = max_abs / h_cpu.iter().fold(0.0f32, |a, v| a.max(v.abs())).max(1e-30);
        worst_rel = worst_rel.max(rel);

        // LM head on the CPU over both hidden vectors
        let l_cpu = dec.head_logits(&h_cpu).expect("cpu head");
        let l_gpu = dec.head_logits(&h_gpu).expect("gpu head");
        let argmax = |l: &[f32]| -> usize {
            l.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i).unwrap()
        };
        let a_cpu = argmax(&l_cpu);
        let a_gpu = argmax(&l_gpu);
        let lmax: f32 = l_cpu.iter().chain(l_gpu.iter()).fold(0.0f32, |a, v| a.max(v.abs())).max(1e-30);
        let ldiff: f32 = l_cpu
            .iter()
            .zip(l_gpu.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        println!(
            "pos {pos}: tok {tok}: cpu {tc:.2}s gpu {tg:.2}s hidden rel {rel:.2e} | argmax cpu {a_cpu} gpu {a_gpu} {} | logits rel {:.2e}",
            if a_cpu == a_gpu { "MATCH" } else { "DIFFER" },
            ldiff / lmax
        );
        if a_cpu != a_gpu {
            diffs += 1;
        }
    }
    println!("worst hidden rel: {worst_rel:.2e}, argmax diffs: {diffs}/{}", toks.len());
    if diffs != 0 {
        eprintln!("GDEV MISMATCH (argmax diverged on {diffs} prefix)");
        std::process::exit(1);
    }
    println!("GDEV OK");
}
