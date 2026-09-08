//! bonsai-grun: interactive decode with the single-submit device engine (GDev)
//! + CPU LM head + Rust sampler. Mirrors bonsai-run but replaces the matvec
//! heavy path with recorded GPU dispatches.

#[path = "../gguf.rs"]
mod gguf;
#[path = "../gdn.rs"]
mod gdn;
#[path = "../kernels.rs"]
mod kernels;
#[path = "../rope.rs"]
mod rope;
#[path = "../sampler.rs"]
mod sampler;
#[path = "../tokenizer.rs"]
mod tokenizer;
#[path = "../weights.rs"]
mod weights;
#[path = "../vk.rs"]
mod vk;
#[path = "../forward.rs"]
mod forward;
#[path = "../gdev.rs"]
mod gdev;

use std::io::{Read, Write};
use std::process::exit;
use std::time::Instant;

fn template(system: &str, user: &str) -> String {
    let mut out = String::new();
    if !system.is_empty() {
        out.push_str("<|im_start|>system\n");
        out.push_str(system);
        out.push_str("<|im_end|>\n");
    }
    out.push_str("<|im_start|>user\n");
    out.push_str(user);
    out.push_str("<|im_end|>\n");
    out.push_str("<|im_start|>assistant\n<think>\n");
    out
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 2 {
        eprintln!("usage: bonsai-grun <model.gguf> <prompt> [n_predict]");
        exit(1);
    }
    let model = args[0].clone();
    let prompt = args[1].clone();
    let n_predict: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(16);

    let gf = gguf::GGUF::open(&model).expect("open gguf");
    let vocab = tokenizer::Vocab::from_gguf(&gf).expect("tokenizer");
    let formatted = template("", &prompt);
    let toks: Vec<u32> = tokenizer::encode(
        &formatted,
        &vocab,
        &tokenizer::TokenizeOptions { add_special: false, parse_special: true },
    )
    .into_iter()
    .map(|t| t as u32)
    .collect();
    let mut stop_ids: Vec<u32> = ["<|im_end|>", "<|resp_end|>", "<|endoftext|>"]
        .iter()
        .filter_map(|w| {
            let t = tokenizer::encode(w, &vocab, &tokenizer::TokenizeOptions { add_special: false, parse_special: true });
            if t.len() == 1 { Some(t[0] as u32) } else { None }
        })
        .collect();
    if let Some(e) = gf.get("tokenizer.ggml.eos_token_id").and_then(|v| v.as_u32()) {
        stop_ids.push(e);
    }
    println!("prompt: {} chars -> {} tokens", formatted.len(), toks.len());

    let mut dec = forward::Decoder::open(&model).expect("cpu (head/embed)");
    let mut dev = gdev::GDev::open(&model).expect("gdev open");
    let mut sampler = sampler::Sampler::new(&sampler::SamplerConfig {
        top_k: 20,
        top_p: 0.9,
        min_p: 0.0,
        temp: 0.6,
        seed: 0xFFFF_FFFF,
    });

    let t0 = Instant::now();
    let mut hidden = Vec::new();
    for (pos, &tok) in toks.iter().enumerate() {
        let emb = dec.w.row_f32("token_embd.weight", tok as u64).expect("embed");
        hidden = dev.forward_token(pos, &emb).expect("prefill");
    }
    let prefill = t0.elapsed().as_secs_f32();
    println!("prefill done in {prefill:.1}s ({:.2} s/tok)", prefill / toks.len() as f32);

    let mut logits = dec.head_logits(&hidden).expect("head");
    let mut stdout = std::io::stdout();
    let t_gen = Instant::now();
    let mut n = 0usize;
    let mut pos = toks.len();
    loop {
        let id = sampler.sample(&logits, &sampler::SamplerConfig {
            top_k: 20, top_p: 0.9, min_p: 0.0, temp: 0.6, seed: 0xFFFF_FFFF,
        }) as u32;
        if stop_ids.contains(&id) || n >= n_predict {
            break;
        }
        let _ = stdout.write_all(&vocab.piece_bytes(id as i32));
        let _ = stdout.flush();
        n += 1;
        let emb = dec.w.row_f32("token_embd.weight", id as u64).expect("embed");
        hidden = dev.forward_token(pos, &emb).expect("decode");
        logits = dec.head_logits(&hidden).expect("head");
        pos += 1;
    }
    let el = t_gen.elapsed().as_secs_f32();
    println!("\n\n{n} tokens in {el:.1}s ({:.2} s/tok)", el / n.max(1) as f32);
}
