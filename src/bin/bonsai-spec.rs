//! bonsai-spec: validate the dspark speculative loop against plain greedy
//! decode. With greedy target sampling the emitted sequence must be identical,
//! which is the correctness oracle and does not depend on draft quality.
//!
//! Usage: bonsai-spec <target.gguf> <sidecar.gguf> <tok,tok,...> [n_new] [n_draft]

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
#[path = "../vk.rs"]
mod vk;
#[path = "../forward.rs"]
mod forward;
#[path = "../dspark.rs"]
mod dspark;
#[path = "../spec.rs"]
mod spec;
#[path = "../tokenizer.rs"]
mod tokenizer;

use forward::Decoder;
use spec::{argmax, round, Drafter};
use std::process::exit;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 3 {
        eprintln!("usage: bonsai-spec <target.gguf> <sidecar.gguf> <tok,tok,...|text:...> [n_new] [n_draft]");
        exit(1);
    }
    let target = args[0].clone();
    let sidecar = args[1].clone();
    let prompt: Vec<u32> = if let Some(text) = args[2].strip_prefix("text:") {
        let gf = match gguf::GGUF::open(&target) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("open gguf: {e}");
                exit(1);
            }
        };
        let vocab = match tokenizer::Vocab::from_gguf(&gf) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("vocab: {e}");
                exit(1);
            }
        };
        tokenizer::encode(
            text,
            &vocab,
            &tokenizer::TokenizeOptions {
                add_special: false,
                parse_special: true,
            },
        )
        .into_iter()
        .map(|t| t as u32)
        .collect()
    } else {
        match args[2]
            .split(',')
            .map(|s| s.trim().parse::<u32>())
            .collect::<Result<Vec<u32>, _>>()
        {
            Ok(v) if !v.is_empty() => v,
            _ => {
                eprintln!("bad token list");
                exit(1);
            }
        }
    };
    let n_new: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(8);
    let n_draft: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(4);

    // ---- spec decoder ----------------------------------------------------
    let mut dec = match Decoder::open(&target) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("open target: {e}");
            exit(1);
        }
    };
    let ctx = 4096;
    let mut drafter = match Drafter::new(&sidecar, ctx, 0.0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("open sidecar: {e}");
            exit(1);
        }
    };

    // prefill: forward each prompt token, mirroring features into the drafter
    let taps_want = drafter.taps_want.clone();
    let mut taps: Vec<Vec<f32>> = vec![Vec::new(); taps_want.len()];
    let mut logits = Vec::new();
    let t0 = Instant::now();
    for (i, &tok) in prompt.iter().enumerate() {
        let h = match dec.forward_hidden_taps(tok, i, &taps_want, &mut taps) {
            Ok(h) => h,
            Err(e) => {
                eprintln!("prefill[{i}]: {e}");
                exit(1);
            }
        };
        if let Err(e) = drafter.observe(&taps, i) {
            eprintln!("observe[{i}]: {e}");
            exit(1);
        }
        logits = dec.head_logits(&h).unwrap();
    }
    println!("prefill {} tokens in {:.1}s", prompt.len(), t0.elapsed().as_secs_f32());

    let mut spec_tokens: Vec<u32> = Vec::new();
    let mut pending = argmax(&logits) as u32;
    spec_tokens.push(pending);
    let mut n_past = prompt.len();
    let mut n_forwards = 1usize;
    let mut n_accepted = 0usize;
    let mut n_rounds = 0usize;
    let t1 = Instant::now();
    while spec_tokens.len() < n_new {
        let out = match round(&mut dec, &mut drafter, pending, n_past, n_draft) {
            Ok(o) => o,
            Err(e) => {
                eprintln!("round: {e}");
                exit(1);
            }
        };
        n_rounds += 1;
        n_forwards += out.target_forwards;
        n_accepted += out.accepted;
        if std::env::var("BONSAI_SPEC_DEBUG").is_ok() {
            eprintln!(
                "round {n_rounds}: drafts {:?} accepted {} emitted {:?}",
                out.drafts, out.accepted, out.emitted
            );
        }
        // the round's emitted tokens are all new: accepted drafts + the target token
        spec_tokens.extend_from_slice(&out.emitted);
        pending = out.pending;
        n_past += out.accepted + 1;
    }
    let spec_elapsed = t1.elapsed().as_secs_f32();
    let spec_n = spec_tokens.len().min(n_new);
    spec_tokens.truncate(spec_n);

    // ---- plain greedy decode (fresh decoder) -----------------------------
    let mut plain = match Decoder::open(&target) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("open target(2): {e}");
            exit(1);
        }
    };
    let mut plogits = Vec::new();
    for (i, &tok) in prompt.iter().enumerate() {
        let h = plain.forward_hidden(tok, i).unwrap();
        plogits = plain.head_logits(&h).unwrap();
    }
    let t2 = Instant::now();
    let mut plain_tokens: Vec<u32> = Vec::new();
    for i in 0..n_new {
        let t = argmax(&plogits) as u32;
        plain_tokens.push(t);
        if i + 1 == n_new {
            break;
        }
        let h = plain.forward_hidden(t, prompt.len() + i).unwrap();
        plogits = plain.head_logits(&h).unwrap();
    }
    let plain_elapsed = t2.elapsed().as_secs_f32();

    println!("spec : {:?}", spec_tokens);
    println!("plain: {:?}", plain_tokens);
    let n = spec_tokens.len().min(plain_tokens.len());
    let identical = spec_tokens[..n] == plain_tokens[..n];
    println!(
        "{} ({n} tokens compared)  accepted {n_accepted}/{} drafts over {n_rounds} rounds",
        if identical { "IDENTICAL" } else { "MISMATCH" },
        n_rounds * n_draft
    );
    println!(
        "spec {:.1}s ({:.2} s/tok, {n_forwards} forwards)  plain {:.1}s ({:.2} s/tok)",
        spec_elapsed,
        spec_elapsed / spec_n.max(1) as f32,
        plain_elapsed,
        plain_elapsed / n_new.max(1) as f32
    );
    if !identical {
        exit(1);
    }
}
