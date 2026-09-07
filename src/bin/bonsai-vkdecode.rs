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
    if args.len() < 3 || args.len() > 4 {
        eprintln!(
            "usage: bonsai-vkdecode <model.gguf> <prompt.txt> <golden.logits.bin> [max_tokens]"
        );
        exit(1);
    }
    let (model, prompt_path, _golden_path) = (&args[0], &args[1], &args[2]);
    let max_tokens: Option<usize> = args.get(3).and_then(|s| s.parse().ok());

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
    let n_run = max_tokens.unwrap_or(toks.len()).min(toks.len());
    if n_run == 0 {
        eprintln!("nothing to decode");
        exit(1);
    }
    println!("decoding prefix of {n_run} tokens (CPU and GPU paths)");

    // ---- CPU reference -------------------------------------------------------
    let t0 = Instant::now();
    let mut dec_cpu = Decoder::open(model).expect("open cpu");
    let mut cpu_logits = Vec::new();
    for (pos, &tok) in toks.iter().take(n_run).enumerate() {
        cpu_logits = dec_cpu.decode_token(tok, pos).expect("cpu decode");
    }
    println!("cpu prefix done in {:.1}s", t0.elapsed().as_secs_f32());

    // ---- GPU-accelerated path -------------------------------------------------
    println!("opening GPU decoder (uploads weights to device-local memory)...");
    let t_load = Instant::now();
    let mut dec_gpu = Decoder::open_gpu(model).expect("open_gpu");
    println!("load + upload in {:.1}s", t_load.elapsed().as_secs_f32());
    let mut gpu_logits = Vec::new();
    let mut baseline: Option<f32> = None;
    for (pos, &tok) in toks.iter().take(n_run).enumerate() {
        let t = Instant::now();
        gpu_logits = dec_gpu.decode_token(tok, pos).expect("gpu decode");
        let el = t.elapsed().as_secs_f32();
        if let Some(b) = baseline {
            kernels::pause_for_budget(el, b);
        } else {
            baseline = Some(el);
        }
        eprintln!("gpu tok {pos:>2} id {tok:<7} in {el:.1}s");
    }

    // ---- compare cpu vs gpu ----------------------------------------------------
    let argmax_cpu = argmax_of(&cpu_logits);
    let argmax_gpu = argmax_of(&gpu_logits);
    let scale = cpu_logits.iter().chain(gpu_logits.iter()).fold(0.0f32, |a, v| a.max(v.abs())).max(1e-30);
    let max_abs: f32 = cpu_logits
        .iter()
        .zip(gpu_logits.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("greedy: cpu {argmax_cpu} vs gpu {argmax_gpu} -> {}", if argmax_cpu == argmax_gpu { "MATCH" } else { "DIFFER" });
    println!("cpu-vs-gpu logits: max abs diff {max_abs:.4e} (rel {:.4e})", max_abs / scale);
    if argmax_cpu != argmax_gpu || max_abs / scale > 1e-2 {
        eprintln!("GPU DECODE VALIDATION FAILED");
        exit(1);
    }
    println!("GPU DECODE VALIDATION PASSED (prefix {n_run}/{})", toks.len());
}

fn argmax_of(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap()
}
