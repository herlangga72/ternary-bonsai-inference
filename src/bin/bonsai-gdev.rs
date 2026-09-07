//! bonsai-gdev: full device (single-submit) decode vs the CPU Decoder.
//!
//! usage: bonsai-gdev <model.gguf> [pos] [token]
//! Runs one token through GDev::forward_token (all ops recorded in one
//! command buffer) and through Decoder::forward_hidden on the CPU, then
//! compares the output-norm hidden vectors.

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

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: bonsai-gdev <model.gguf> [pos] [token]");
        std::process::exit(1);
    }
    let model = &args[0];
    let pos: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(0);
    let token: u32 = args
        .get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(248045);

    let g = GGUF::open(model).expect("open gguf");
    let emb_info = g
        .tensors
        .iter()
        .find(|t| t.name == "token_embd.weight")
        .expect("token_embd.weight")
        .clone();
    let ne0 = emb_info.dims[0] as usize;
    let row_bytes = kernels::pq2_row_bytes(ne0);
    let raw = g
        .slice_at(g.tensor_data_offset(&emb_info) + token as u64 * row_bytes as u64, row_bytes)
        .expect("embed row slice");
    let embed: Vec<f32> = kernels::decode_pq2_0_row(raw, ne0);

    let mut dec = forward::Decoder::open(model).expect("cpu decoder");
    let t0 = Instant::now();
    let h_cpu = dec.forward_hidden(token, pos).expect("cpu forward");
    println!("cpu hidden in {:.2}s", t0.elapsed().as_secs_f32());

    let mut dev = gdev::GDev::open(model).expect("gdev open");
    let t0 = Instant::now();
    let h_gpu = dev.forward_token(pos, &embed).expect("gpu forward");
    println!("gpu hidden in {:.2}s", t0.elapsed().as_secs_f32());

    let mut max_abs = 0.0f32;
    for i in 0..h_cpu.len() {
        max_abs = max_abs.max((h_cpu[i] - h_gpu[i]).abs());
    }
    let scale = h_cpu.iter().fold(0.0f32, |a, v| a.max(v.abs())).max(1e-30);
    println!(
        "hidden: max abs {max_abs:.4e} (rel {:.4e}) len {}",
        max_abs / scale,
        h_cpu.len()
    );
    if max_abs / scale > 1e-2 {
        eprintln!("GDEV MISMATCH");
        std::process::exit(1);
    }
    println!("GDEV OK");
}
