//! bonsai-decode: smoke-test the M6-5 full Rust decoder (embeddings -> 64
//! layers -> LM head) on a real qwen35 GGUF over a short synthetic token
//! sequence. Prints per-token timing and the argmax token id.

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
#[path = "../kvquant.rs"]
mod kvquant;
#[path = "../forward.rs"]
mod forward;
#[path = "../vk.rs"]
mod vk;

use forward::Decoder;
use std::process::exit;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: bonsai-decode <model.gguf> [n_tokens]");
        exit(1);
    }
    let model = &args[0];
    let n_tokens: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(2);

    let mut dec = match Decoder::open(model) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("error: {e}");
            exit(1);
        }
    };
    let n_vocab = dec.vocab_size();
    println!(
        "model ready: {} layers, n_embd {}, vocab {}",
        dec.cfg.n_layer, dec.cfg.n_embd, n_vocab
    );

    // deterministic pseudo-token ids (skip specials)
    let mut x: u64 = 0x1234_5678;
    let mut next = move || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        (x % (n_vocab as u64 - 100) + 10) as u32
    };

    let t_all = Instant::now();
    let mut baseline: Option<f32> = None;
    for pos in 0..n_tokens {
        let tok = next();
        let t0 = Instant::now();
        let logits = match dec.decode_token(tok, pos) {
            Ok(l) => l,
            Err(e) => {
                eprintln!("error at pos {pos}: {e}");
                exit(1);
            }
        };
        let el = t0.elapsed().as_secs_f32();
        if let Some(b) = baseline {
            kernels::pause_for_budget(el, b);
        } else {
            baseline = Some(el);
        }
        assert_eq!(logits.len(), n_vocab);
        let mut argmax = 0usize;
        for (i, &v) in logits.iter().enumerate() {
            if v > logits[argmax] {
                argmax = i;
            }
        }
        println!(
            "pos {pos:>2}: tok {tok:<8} argmax {argmax:<8} (logit {:.3})  {el:.2}s",
            logits[argmax],
        );
    }
    println!(
        "done: {n_tokens} tokens in {:.1}s ({:.2} s/tok)",
        t_all.elapsed().as_secs_f32(),
        t_all.elapsed().as_secs_f32() / n_tokens as f32
    );
}
