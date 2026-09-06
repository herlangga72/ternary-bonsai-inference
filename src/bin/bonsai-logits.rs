//! bonsai-logits: capture golden logits from llama.cpp for a fixed prompt.
//!
//! Used by M6/M7 to validate the pure-Rust forward pass: run the same prompt
//! through llama_decode, then compare Rust-computed logits to this file.
//!
//! Output binary layout (little-endian):
//!   u32 n_vocab, u32 n_prompt, f32 logits[n_vocab], i32 greedy_id

#[path = "../llama.rs"]
mod llama;
#[path = "../gguf.rs"]
mod gguf;
#[path = "../tokenizer.rs"]
mod tokenizer;

use gguf::GGUF;
use llama::*;
use std::ffi::CString;
use std::fs;
use std::os::raw::c_char;
use std::process::exit;
use std::ptr;

fn cstr(s: &str) -> CString {
    CString::new(s).unwrap()
}

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
    if args.len() != 3 {
        eprintln!("usage: bonsai-logits <model.gguf> <prompt.txt> <out.bin>");
        exit(1);
    }
    let model_path = &args[0];
    let prompt_path = &args[1];
    let out_path = &args[2];
    let user = fs::read_to_string(prompt_path)
        .unwrap_or_else(|e| {
            eprintln!("error reading prompt: {e}");
            exit(1);
        })
        .trim()
        .to_string();
    let prompt = template("", &user);

    unsafe { llama_backend_init() }

    let mpath = cstr(model_path);
    let mut mp = unsafe { llama_model_default_params() };
    mp.n_gpu_layers = 0;
    let model = unsafe { llama_model_load_from_file(mpath.as_ptr(), mp) };
    if model.is_null() {
        eprintln!("error: model load failed");
        exit(1);
    }
    let vocab = unsafe { llama_model_get_vocab(model) };

    // Rust tokenizer for prompt tokens
    let gf = match GGUF::open(model_path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("error: {e}");
            exit(1);
        }
    };
    let rvocab = match tokenizer::Vocab::from_gguf(&gf) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("error: {e}");
            exit(1);
        }
    };
    let mut toks = tokenizer::encode(
        &prompt,
        &rvocab,
        &tokenizer::TokenizeOptions { add_special: false, parse_special: true },
    );
    if toks.is_empty() {
        eprintln!("error: tokenize failed");
        exit(1);
    }

    let mut cp = unsafe { llama_context_default_params() };
    cp.n_ctx = 4096;
    cp.n_batch = 512;
    cp.n_ubatch = 512;
    cp.n_threads = 4;
    cp.n_threads_batch = 4;
    let ctx = unsafe { llama_init_from_model(model, cp) };
    if ctx.is_null() {
        eprintln!("error: context creation failed");
        exit(1);
    }

    // prefill in one go (prompt is short)
    let n_prompt = toks.len() as i32;
    let mut pos: Vec<llama_pos> = (0..n_prompt).collect();
    let batch = llama_batch {
        n_tokens: n_prompt,
        token: toks.as_mut_ptr(),
        embd: ptr::null_mut(),
        pos: pos.as_mut_ptr(),
        n_seq_id: ptr::null_mut(),
        seq_id: ptr::null_mut(),
        logits: ptr::null_mut(),
    };
    if unsafe { llama_decode(ctx, batch) } != 0 {
        eprintln!("error: decode failed");
        exit(1);
    }

    let n_vocab = unsafe { llama_vocab_n_tokens(vocab) } as usize;
    let lp = unsafe { llama_get_logits_ith(ctx, n_prompt - 1) };
    if lp.is_null() {
        eprintln!("error: no logits");
        exit(1);
    }
    let logits = unsafe { std::slice::from_raw_parts(lp, n_vocab) };
    let greedy = logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i as i32)
        .unwrap_or(0);

    let mut out = Vec::with_capacity(8 + n_vocab * 4 + 4);
    out.extend_from_slice(&(n_vocab as u32).to_le_bytes());
    out.extend_from_slice(&(n_prompt as u32).to_le_bytes());
    for v in logits {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out.extend_from_slice(&greedy.to_le_bytes());
    fs::write(out_path, &out).unwrap();

    let greedy_text = rvocab.piece_bytes(greedy);
    eprintln!(
        "prompt {} tokens; greedy id {} text {:?}; logits saved to {} ({} bytes)",
        n_prompt,
        greedy,
        String::from_utf8_lossy(&greedy_text),
        out_path,
        out.len()
    );

    unsafe {
        llama_free(ctx);
        llama_model_free(model);
    }
}

#[allow(dead_code)]
fn _c(_: *const c_char) {}
