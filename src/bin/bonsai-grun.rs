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
#[path = "../prefill.rs"]
mod prefill;

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

    // ---- prompt prefill: batched (P3-P5) by default, token loop fallback ----
    let (batch_enabled, window) = gdev::batch_cfg();
    let use_batch = batch_enabled && toks.len() >= 2;
    if use_batch {
        // gather all N prompt embeddings into one tile, then run the batched
        // 64-layer prefill (windowed if the prompt exceeds `window`) in a
        // single call. Returns the output-norm hidden of the final window; its
        // last row is position N-1.
        let tile = prefill::batch_embed(&mut dec.w, &toks).expect("batch embed");
        let n_feat = tile.n_feat();
        let n_tok = toks.len();
        let hidden_tile = dev
            .prefill_batch_windowed(tile.as_slice(), n_tok, window)
            .expect("batched prefill");
        let last = hidden_tile.len() / n_feat;
        hidden = hidden_tile[(last - 1) * n_feat..last * n_feat].to_vec();
        println!(
            "prefill done in {:.1}s via BATCHED prefill ({} tokens, window {window})",
            t0.elapsed().as_secs_f32(),
            n_tok
        );
    } else {
        for (pos, &tok) in toks.iter().enumerate() {
            let emb = dec.w.row_f32("token_embd.weight", tok as u64).expect("embed");
            hidden = dev.forward_token(pos, &emb).expect("prefill");
        }
        let prefill = t0.elapsed().as_secs_f32();
        println!(
            "prefill done in {prefill:.1}s via token loop ({:.2} s/tok)",
            prefill / toks.len() as f32
        );
    }

    let mut logits = dec.head_logits(&hidden).expect("head");
    let mut stdout = std::io::stdout();
    let t_gen = Instant::now();
    let mut baseline: Option<f32> = None;
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
        let t0 = Instant::now();
        hidden = dev.forward_token(pos, &emb).expect("decode");
        logits = dec.head_logits(&hidden).expect("head");
        let el = t0.elapsed().as_secs_f32();
        if let Some(b) = baseline {
            kernels::pause_for_budget(el, b);
        } else {
            baseline = Some(el);
        }
        pos += 1;
    }
    let el = t_gen.elapsed().as_secs_f32();
    println!("\n\n{n} tokens in {el:.1}s ({:.2} s/tok)", el / n.max(1) as f32);
}
