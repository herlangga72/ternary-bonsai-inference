//! bonsai-golden: M6-6 validation. Runs the pure-Rust decoder over the prompt
//! tokens of a golden capture (tokenized exactly like bonsai-logits did) and
//! compares the final logits row to golden/<name>.logits.bin.
//!
//! Golden format (little-endian):
//!   u32 n_vocab; u32 n_prompt; f32 logits[n_vocab]; i32 greedy_id

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
#[path = "../kvquant.rs"]
mod kvquant;
#[path = "../forward.rs"]
mod forward;
#[path = "../vk.rs"]
mod vk;

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
        eprintln!("usage: bonsai-golden <model.gguf> <prompt.txt> <golden.logits.bin>");
        exit(1);
    }
    let (model, prompt_path, golden_path) = (&args[0], &args[1], &args[2]);

    // ---- tokenize the prompt exactly like the capture -----------------------
    let gf = match gguf::GGUF::open(model) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("error: {e}");
            exit(1);
        }
    };
    let vocab = match tokenizer::Vocab::from_gguf(&gf) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("tokenizer error: {e}");
            exit(1);
        }
    };
    let prompt_text = std::fs::read_to_string(prompt_path).unwrap_or_else(|e| {
        eprintln!("read {prompt_path}: {e}");
        exit(1);
    });
    let prompt_text = prompt_text.trim();
    let formatted = apply_qwen35_template("", prompt_text);
    let toks: Vec<u32> = tokenizer::encode(
        &formatted,
        &vocab,
        &tokenizer::TokenizeOptions { add_special: false, parse_special: true },
    )
    .into_iter()
    .map(|t| t as u32)
    .collect();
    println!("prompt: {} chars -> {} tokens", formatted.len(), toks.len());

    // ---- golden header -------------------------------------------------------
    let raw = std::fs::read(golden_path).unwrap_or_else(|e| {
        eprintln!("read {golden_path}: {e}");
        exit(1);
    });
    let rd = |off: usize| u32::from_le_bytes([raw[off], raw[off + 1], raw[off + 2], raw[off + 3]]);
    let n_vocab = rd(0) as usize;
    let n_prompt = rd(4) as usize;
    if raw.len() < 8 + n_vocab * 4 + 4 {
        eprintln!("golden file too short");
        exit(1);
    }
    let golden: Vec<f32> = raw[8..8 + n_vocab * 4]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let greedy_off = 8 + n_vocab * 4;
    let greedy_golden =
        i32::from_le_bytes([raw[greedy_off], raw[greedy_off + 1], raw[greedy_off + 2], raw[greedy_off + 3]]);
    println!("golden: n_vocab {n_vocab}, n_prompt {n_prompt}, greedy {greedy_golden}");

    let mut dec = match Decoder::open(model) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("error: {e}");
            exit(1);
        }
    };
    if dec.vocab_size() != n_vocab || toks.len() != n_prompt {
        eprintln!(
            "mismatch: rust vocab {} vs golden {n_vocab}; rust tokens {} vs golden {n_prompt}",
            dec.vocab_size(),
            toks.len()
        );
        exit(1);
    }

    // ---- decode the prompt token by token (causal), keep last logits --------
    let t_all = Instant::now();
    let mut logits = Vec::new();
    for (pos, &tok) in toks.iter().enumerate() {
        let t0 = Instant::now();
        logits = match dec.decode_token(tok, pos) {
            Ok(l) => l,
            Err(e) => {
                eprintln!("error at pos {pos}: {e}");
                exit(1);
            }
        };
        eprintln!(
            "tok {pos:>2} id {tok:<7} in {:.1}s",
            t0.elapsed().as_secs_f32()
        );
    }
    eprintln!("decode done in {:.0}s", t_all.elapsed().as_secs_f32());

    // ---- compare -------------------------------------------------------------
    let argmax = logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    let scale = golden
        .iter()
        .chain(logits.iter())
        .fold(0.0f32, |a, v| a.max(v.abs()))
        .max(1e-30);
    let max_abs: f32 = golden
        .iter()
        .zip(logits.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("greedy: rust {argmax} vs golden {greedy_golden} -> {}", if argmax as i32 == greedy_golden { "MATCH" } else { "DIFFER" });
    println!("logits: max abs diff {max_abs:.4e} (rel {:.4e})", max_abs / scale);
    if argmax as i32 != greedy_golden || max_abs / scale > 1e-2 {
        eprintln!("GOLDEN VALIDATION FAILED");
        exit(1);
    }
    println!("GOLDEN VALIDATION PASSED");
}
