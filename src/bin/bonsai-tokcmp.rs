//! bonsai-tokcmp: compare Rust qwen35 BPE tokenizer (M3) against llama.cpp
//! `llama_tokenize` on a corpus.
//!
//! Usage: bonsai-tokcmp <model.gguf> [corpus lines on stdin]
//! Reads lines from stdin, tokenizes each with llama.cpp (C) and with the Rust
//! tokenizer, and reports the first mismatches plus a summary.

#[path = "../llama.rs"]
mod llama;
#[path = "../gguf.rs"]
mod gguf;
#[path = "../tokenizer.rs"]
mod tokenizer;

use gguf::GGUF;
use llama::*;
use std::ffi::CString;
use std::io::{BufRead, Write};
use std::process::exit;

fn cstr(s: &str) -> CString {
    CString::new(s).unwrap()
}

fn c_tokens(vocab: *const llama_vocab, text: &str, parse_special: bool) -> Vec<llama_token> {
    let c = cstr(text);
    let mut cap = (text.len() * 4 + 128) as i32;
    loop {
        let mut toks: Vec<llama_token> = vec![0; cap as usize];
        let n = unsafe {
            llama_tokenize(
                vocab,
                c.as_ptr(),
                text.len() as i32,
                toks.as_mut_ptr(),
                cap,
                0,
                parse_special as u8,
            )
        };
        if n < 0 {
            cap = -n;
            continue;
        }
        toks.truncate(n as usize);
        return toks;
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: bonsai-tokcmp <model.gguf> < corpus.txt");
        exit(1);
    }
    let path = &args[0];

    unsafe { llama_backend_init() }

    // C side: load vocab only (fast, no weights)
    let mpath = cstr(path);
    let mut mp = unsafe { llama_model_default_params() };
    mp.vocab_only = 1;
    let model = unsafe { llama_model_load_from_file(mpath.as_ptr(), mp) };
    if model.is_null() {
        eprintln!("error: could not load model (C side)");
        exit(1);
    }
    let cvocab = unsafe { llama_model_get_vocab(model) };

    // Rust side
    let g = match GGUF::open(path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("error: {e}");
            exit(1);
        }
    };
    let rvocab = match tokenizer::Vocab::from_gguf(&g) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("error building Rust vocab: {e}");
            exit(1);
        }
    };
    drop(g);

    let stdin = std::io::stdin();
    let mut total = 0usize;
    let mut mismatch_lines = 0usize;
    let mut shown = 0usize;
    for line in stdin.lock().lines().map_while(Result::ok) {
        if line.trim().is_empty() {
            continue;
        }
        total += 1;
        let c = c_tokens(cvocab, &line, true);
        let r = tokenizer::encode(
            &line,
            &rvocab,
            &tokenizer::TokenizeOptions { add_special: false, parse_special: true },
        );
        if c != r {
            mismatch_lines += 1;
            if shown < 10 {
                println!("MISMATCH line {total}: {line:?}");
                println!("  c  : {c:?}");
                println!("  rust: {r:?}");
                shown += 1;
            }
        }
    }

    let _ = std::io::stdout().flush();
    let ok = mismatch_lines == 0 && total > 0;
    println!(
        "\n{total} lines, {mismatch_lines} mismatching ({:.2}%) -> {}",
        if total > 0 { 100.0 * mismatch_lines as f64 / total as f64 } else { 0.0 },
        if ok { "MATCH" } else { "DIFFERS" }
    );
    unsafe { llama_model_free(model) }
    exit(if ok { 0 } else { 1 });
}
