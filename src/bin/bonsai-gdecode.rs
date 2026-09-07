//! bonsai-gdecode: golden qa through the single-submit device decode (GDev)
//! with the LM head + sampling on the CPU. Final logits compared to the
//! llama.cpp golden capture (greedy 8160 expected).

#[path = "../gguf.rs"]
mod gguf;
#[path = "../gdn.rs"]
mod gdn;
#[path = "../kernels.rs"]
mod kernels;
#[path = "../rope.rs"]
mod rope;
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

use std::process::exit;
use std::time::Instant;

fn apply_qwen35_template(system: &str, user: &str) -> String {
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
    if args.len() != 3 {
        eprintln!("usage: bonsai-gdecode <model.gguf> <prompt.txt> <golden.logits.bin>");
        exit(1);
    }
    let (model, prompt_path, golden_path) = (&args[0], &args[1], &args[2]);

    let gf = gguf::GGUF::open(model).expect("open gguf");
    let vocab = tokenizer::Vocab::from_gguf(&gf).expect("tokenizer");
    let prompt_text = std::fs::read_to_string(prompt_path)
        .unwrap_or_else(|e| panic!("read {prompt_path}: {e}"))
        .trim()
        .to_string();
    let formatted = apply_qwen35_template("", &prompt_text);
    let toks: Vec<u32> = tokenizer::encode(
        &formatted,
        &vocab,
        &tokenizer::TokenizeOptions { add_special: false, parse_special: true },
    )
    .into_iter()
    .map(|t| t as u32)
    .collect();
    println!("prompt: {} chars -> {} tokens", formatted.len(), toks.len());

    let raw = std::fs::read(golden_path).expect("golden file");
    let rd = |off: usize| u32::from_le_bytes([raw[off], raw[off + 1], raw[off + 2], raw[off + 3]]);
    let n_vocab = rd(0) as usize;
    let golden: Vec<f32> = raw[8..8 + n_vocab * 4]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let greedy_golden = i32::from_le_bytes([
        raw[8 + n_vocab * 4],
        raw[8 + n_vocab * 4 + 1],
        raw[8 + n_vocab * 4 + 2],
        raw[8 + n_vocab * 4 + 3],
    ]);

    let mut dec = forward::Decoder::open(model).expect("cpu decoder (head/embed)");
    let mut dev = gdev::GDev::open(model).expect("gdev open");

    let t_all = Instant::now();
    let mut logits = Vec::new();
    for (pos, &tok) in toks.iter().enumerate() {
        let embed = dec.w.row_f32("token_embd.weight", tok as u64).expect("embed row");
        let t0 = Instant::now();
        let hidden = dev.forward_token(pos, &embed).expect("gpu forward");
        logits = dec.head_logits(&hidden).expect("cpu head");
        eprintln!(
            "tok {pos:>2} id {tok:<7} in {:.1}s",
            t0.elapsed().as_secs_f32()
        );
    }
    eprintln!("decode done in {:.0}s", t_all.elapsed().as_secs_f32());

    let argmax = logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    let scale = golden.iter().chain(logits.iter()).fold(0.0f32, |a, v| a.max(v.abs())).max(1e-30);
    let max_abs: f32 = golden
        .iter()
        .zip(logits.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("greedy: gpu {argmax} vs golden {greedy_golden} -> {}", if argmax as i32 == greedy_golden { "MATCH" } else { "DIFFER" });
    println!("logits: max abs diff {max_abs:.4e} (rel {:.4e})", max_abs / scale);
    if argmax as i32 != greedy_golden || max_abs / scale > 0.1 {
        eprintln!("GDEV GOLDEN FAILED");
        exit(1);
    }
    println!("GDEV GOLDEN PASSED");
}
