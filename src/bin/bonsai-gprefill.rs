//! bonsai-gprefill: P5 acceptance harness - end-to-end greedy continuation
//! equality between the sequential token-loop prefill and the batched
//! (P3-P5) prefill on a real Ternary-Bonsai-27B model via RADV.
//!
//! Runs the SAME full prompt two ways through the single-submit device engine
//! (GDev), each followed by K greedy decode tokens via the untouched
//! single-stream `forward_token` decode loop at `pos = N`:
//!
//! 1. **Token loop (reference)**  - `forward_token(pos, embed)` per prompt
//!    position, then greedy decode. This is the path P5 replaces.
//! 2. **Batched (P3-P5)**         - `prefill_batch_windowed(embeds, N, W)` runs
//!    the batched 64-layer forward over all N prompt positions (in windows of
//!    `W` columns when the prompt is long), then greedy decode.
//!
//! Reports whether the batched path reached the first generated token, and
//! whether the first K generated token ids are equal across the two prefill
//! paths (the acceptance check).
//!
//! usage: bonsai-gprefill <model.gguf> <prompt.txt> [K] [window]
//!   K      generated tokens to compare (default 8)
//!   window prefill window width (default = N = single window; force smaller to
//!          exercise the P5 windowed path)

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
#[path = "../kvquant.rs"]
mod kvquant;
#[path = "../forward.rs"]
mod forward;
#[path = "../gdev.rs"]
mod gdev;
#[path = "../prefill.rs"]
mod prefill;

use std::process::exit;
use std::time::Instant;

const N_EMBD: usize = 5120;

fn argmax(logits: &[f32]) -> u32 {
    let mut bi = 0usize;
    for i in 1..logits.len() {
        if logits[i] > logits[bi] {
            bi = i;
        }
    }
    bi as u32
}

/// Greedy-decode `k` continuation tokens starting from the position N-1 hidden
/// vector `h0` at `pos = n`. The KV / conv / GDN state left by the prefill is
/// appended to exactly as the decode path does after a real prompt.
fn decode_greedy(
    dec: &mut forward::Decoder,
    dev: &mut gdev::GDev,
    n: usize,
    h0: &[f32],
    k: usize,
) -> Result<Vec<u32>, String> {
    let mut h = h0.to_vec();
    let mut ids = Vec::with_capacity(k);
    for j in 0..k {
        let logits = dec.head_logits(&h)?;
        let id = argmax(&logits);
        ids.push(id);
        let pos = n + j;
        let emb = dec.w.row_f32("token_embd.weight", id as u64).map_err(|e| format!("embed {id}: {e}"))?;
        h = dev.forward_token(pos, &emb)?;
    }
    Ok(ids)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 2 {
        eprintln!("usage: bonsai-gprefill <model.gguf> <prompt.txt> [K] [window]");
        exit(1);
    }
    let (model, prompt_path) = (&args[0], &args[1]);
    let k: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(8);
    let window: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(0);

    let gf = gguf::GGUF::open(model).expect("open gguf");
    let vocab = tokenizer::Vocab::from_gguf(&gf).expect("tokenizer");
    let text = std::fs::read_to_string(prompt_path).expect("read prompt");
    let toks: Vec<u32> = tokenizer::encode(
        &text,
        &vocab,
        &tokenizer::TokenizeOptions { add_special: false, parse_special: true },
    )
    .into_iter()
    .map(|t| t as u32)
    .collect();
    let n = toks.len();
    eprintln!("[gprefill] model={model} prompt={prompt_path} ({n} tokens), K={k}, window={}", if window == 0 { "full".to_string() } else { window.to_string() });

    let mut dec = forward::Decoder::open(model).expect("cpu weights (head/embed)");
    let mut dev = gdev::GDev::open(model).expect("gdev open");

    let emb = prefill::batch_embed(&mut dec.w, &toks).expect("batch embed");

    // ---- 1. sequential token-loop prefill (reference) -----------------------
    let t0 = Instant::now();
    let mut ref_hidden = Vec::new();
    for p in 0..n {
        ref_hidden = dev
            .forward_token(p, &emb.as_slice()[p * N_EMBD..(p + 1) * N_EMBD])
            .expect("token forward");
    }
    eprintln!("[gprefill] token-loop prefill ({} tok): {:.1}s", n, t0.elapsed().as_secs_f32());
    let ref_ids = decode_greedy(&mut dec, &mut dev, n, &ref_hidden, k).expect("decode (token-loop)");

    // ---- 2. batched prefill (P3-P5), window W ------------------------------
    let t0 = Instant::now();
    let bat_tile = dev
        .prefill_batch_windowed(emb.as_slice(), n, if window == 0 { n } else { window })
        .expect("batched prefill");
    eprintln!(
        "[gprefill] batched prefill ({} tok): {:.1}s",
        n,
        t0.elapsed().as_secs_f32()
    );
    let last = bat_tile.len() / N_EMBD;
    let bat_hidden = bat_tile[(last - 1) * N_EMBD..last * N_EMBD].to_vec();
    let bat_ids = decode_greedy(&mut dec, &mut dev, n, &bat_hidden, k).expect("decode (batched)");

    // ---- report -------------------------------------------------------------
    let first_equal = ref_ids.first() == bat_ids.first();
    let first = bat_ids.first().copied();
    let neq = ref_ids.iter().zip(bat_ids.iter()).filter(|(a, b)| a == b).count();
    let all_equal = ref_ids == bat_ids;
    println!(
        "BATCHED-PREFILL PATH TAKEN: yes (prefill_batch_windowed n={n}) -> first generated token {first:?}"
    );
    println!(
        "Greedy continuation equality over K={k}: {} / {k} tokens equal, ids all-equal={all_equal}",
        neq
    );
    eprintln!("[gprefill] ref ids:  {:?}", ref_ids);
    eprintln!("[gprefill] batch ids: {:?}", bat_ids);
    if !(first_equal && all_equal) {
        eprintln!("[gprefill] FAIL: batched prefill continuation != token loop");
        exit(1);
    }
    println!("GPREFILL CONTINUATION PASSED (K={k}, first={first:?}, equal={all_equal})");
}
