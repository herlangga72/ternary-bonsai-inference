//! bonsai-vkdecode (G2c): run the golden qa prompt through the GPU-accelerated
//! decoder (Decoder::open_gpu) and compare the final logits against the llama.cpp
//! golden capture. The PQ2_0 matvecs run on Vulkan; layer norms/attention/rope/
//! gdn stay on the CPU path. Success = greedy id 8160 + rel diff in the 1e-2
//! ballpark, proving the device matvec path reproduces decode end to end.

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

use forward::Decoder;
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
        eprintln!("usage: bonsai-vkdecode <model.gguf> <prompt.txt> <golden.logits.bin>");
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
    let n_prompt = rd(4) as usize;
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

    println!("opening GPU decoder (uploads weights to device-local memory)...");
    let t_load = Instant::now();
    let mut dec = Decoder::open_gpu(model).expect("open_gpu");
    println!("load + upload in {:.1}s", t_load.elapsed().as_secs_f32());
    if dec.vocab_size() != n_vocab || toks.len() != n_prompt {
        panic!("shape mismatch vs golden");
    }

    let t_all = Instant::now();
    let mut logits = Vec::new();
    for (pos, &tok) in toks.iter().enumerate() {
        let t0 = Instant::now();
        logits = dec.decode_token(tok, pos).expect("decode_token");
        eprintln!("tok {pos:>2} id {tok:<7} in {:.1}s", t0.elapsed().as_secs_f32());
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
    println!("greedy: rust {argmax} vs golden {greedy_golden} -> {}", if argmax as i32 == greedy_golden { "MATCH" } else { "DIFFER" });
    println!("logits: max abs diff {max_abs:.4e} (rel {:.4e})", max_abs / scale);
    if argmax as i32 != greedy_golden || max_abs / scale > 1e-2 {
        eprintln!("GPU DECODE VALIDATION FAILED");
        exit(1);
    }
    println!("GPU DECODE VALIDATION PASSED");
}
