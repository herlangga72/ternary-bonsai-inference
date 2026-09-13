//! bonsai-bprefill: P3+P4 correctness check - batched prefill vs the sequential
//! token loop on a real Ternary-Bonsai-27B model via RADV.
//!
//! Runs the SAME N prompt tokens two ways through the single-submit device
//! engine (GDev) and diffs the results:
//!
//! 1. **Sequential (reference)** - `forward_token(pos, embed)` per token, the
//!    current decode path that is the numeric anchor.
//! 2. **Batched (P3+P4)** - `prefill_batch(embeds, N)` (full-attention layers
//!    batched over the N columns through the P2 GEMM + causal attention; FFN
//!    batched; and, since P4, the recurrent GDN layers also batched: their
//!    attn_qkv / attn_gate / ssm_beta / ssm_alpha projections and ssm_out run
//!    through the N-column GEMM, with only the causal conv + gated-delta-net
//!    state recurrence kept sequential per position).
//!
//! Checks:
//! * output-normalized hidden at every position within ~1e-3 rel of the token
//!   loop, and its max abs/rel diff reported;
//! * the full-attention KV caches (positions 0..N-1, [pos][kv_head][head_dim])
//!   match the token loop, max abs/rel diff reported;
//! * the GDN state and conv caches of every recurrent layer match the token
//!   loop, max abs/rel diff reported;
//! * a decode-append at pos = N is greedy-equal after each path;
//! * the greedy first generated token after each path is equal.
//!
//! usage: bonsai-bprefill <model.gguf> [prompt.txt] [n_tokens]

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

use std::process::exit;

const N_EMBD: usize = 5120;
const GDN_HV: usize = 48;
const GDN_CH: usize = 10240;
const STATE_ELEMS: usize = GDN_HV * 128 * 128; // per recurrent layer
const CONV_ELEMS: usize = 3 * GDN_CH; // per recurrent layer

fn diff_metrics(a: &[f32], b: &[f32]) -> (f32, f32, usize) {
    assert_eq!(a.len(), b.len());
    let bmax = b.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-30);
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    let mut over = 0usize;
    for i in 0..a.len() {
        let d = (a[i] - b[i]).abs();
        let rel = d / bmax;
        if rel > 1e-3 && d > 1e-4 {
            over += 1;
        }
        max_abs = max_abs.max(d);
        max_rel = max_rel.max(rel);
    }
    (max_abs, max_rel, over)
}

fn greedy(logits: &[f32]) -> u32 {
    let mut bi = 0usize;
    for i in 1..logits.len() {
        if logits[i] > logits[bi] {
            bi = i;
        }
    }
    bi as u32
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: bonsai-bprefill <model.gguf> [prompt.txt] [n_tokens]");
        exit(1);
    }
    let model = args[0].clone();
    let prompt_path = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "golden/prompts/qa.txt".to_string());
    let n_req: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(20);

    let gf = gguf::GGUF::open(&model).expect("open gguf");
    let vocab = tokenizer::Vocab::from_gguf(&gf).expect("tokenizer");
    let text = std::fs::read_to_string(&prompt_path).expect("read prompt");
    let toks: Vec<u32> = tokenizer::encode(
        &text,
        &vocab,
        &tokenizer::TokenizeOptions { add_special: false, parse_special: true },
    )
    .into_iter()
    .map(|t| t as u32)
    .collect();
    let n = n_req.min(toks.len()).max(1);
    let toks = toks[..n].to_vec();
    eprintln!("[bprefill] model={model} prompt={prompt_path} ({n} tokens)");
    eprintln!("[bprefill] tokens: {:?}", toks);

    let mut dec = forward::Decoder::open(&model).expect("cpu weights (head/embed)");
    let mut dev = gdev::GDev::open(&model).expect("gdev open");

    // gather N embeddings (host row reads, forward.rs convention)
    let mut embeds = Vec::with_capacity(n * N_EMBD);
    for &t in &toks {
        let r = dec.w.row_f32("token_embd.weight", t as u64).expect("embed row");
        embeds.extend_from_slice(&r);
    }

    let n_layer = dev.cfg.n_layer;

    // ---- 1. sequential token loop (reference) ------------------------------
    let mut ref_hidden = vec![vec![0f32; N_EMBD]; n];
    let mut last = vec![0f32; N_EMBD];
    let t0 = std::time::Instant::now();
    for p in 0..n {
        let h = dev.forward_token(p, &embeds[p * N_EMBD..(p + 1) * N_EMBD]).expect("token forward");
        ref_hidden[p].copy_from_slice(&h);
        last = h;
    }
    eprintln!(
        "[bprefill] sequential loop ({} tok): {:.1}s",
        n,
        t0.elapsed().as_secs_f32()
    );
    let ref_greedy = greedy(&dec.head_logits(&last).expect("head"));
    let mut ref_kv: Vec<(Vec<f32>, Vec<f32>)> = Vec::new();
    for il in 0..n_layer {
        if dev.cfg.is_full_attention(il) {
            ref_kv.push(dev.dump_kv(il, n).expect("dump kv"));
        }
    }
    // GDN state + conv caches left by the sequential loop (positions 0..N-1).
    let mut ref_state: Vec<Option<Vec<f32>>> = Vec::with_capacity(n_layer);
    let mut ref_conv: Vec<Option<Vec<f32>>> = Vec::with_capacity(n_layer);
    for il in 0..n_layer {
        if dev.cfg.is_recurrent(il) {
            ref_state.push(dev.dump_state(il));
            ref_conv.push(dev.dump_conv(il));
        }
    }
    // decode continuation after the sequential loop: append one token at pos=n.
    let probe = embeds[0..N_EMBD].to_vec();
    let cont_a = dev.forward_token(n, &probe).expect("cont (seq)");
    let cont_a_greedy = greedy(&dec.head_logits(&cont_a).expect("head"));

    // ---- 2. batched prefill (P3+P4) ----------------------------------------
    let t0 = std::time::Instant::now();
    let bat = dev.prefill_batch(&embeds, n).expect("batched prefill");
    eprintln!("[bprefill] batched prefill ({} tok): {:.1}s", n, t0.elapsed().as_secs_f32());
    let bat_greedy = greedy(&dec.head_logits(&bat[(n - 1) * N_EMBD..n * N_EMBD]).expect("head"));
    let mut bat_kv: Vec<(Vec<f32>, Vec<f32>)> = Vec::new();
    for il in 0..n_layer {
        if dev.cfg.is_full_attention(il) {
            bat_kv.push(dev.dump_kv(il, n).expect("dump kv"));
        }
    }
    let mut bat_state: Vec<Option<Vec<f32>>> = Vec::with_capacity(n_layer);
    let mut bat_conv: Vec<Option<Vec<f32>>> = Vec::with_capacity(n_layer);
    for il in 0..n_layer {
        if dev.cfg.is_recurrent(il) {
            bat_state.push(dev.dump_state(il));
            bat_conv.push(dev.dump_conv(il));
        }
    }
    // decode continuation after the batched prefill: append one token at pos=n.
    let cont_b = dev.forward_token(n, &probe).expect("cont (batched)");
    let cont_b_greedy = greedy(&dec.head_logits(&cont_b).expect("head"));
    let (cabs, crel, cover) = diff_metrics(&cont_a, &cont_b);
    eprintln!(
        "[bprefill] decode append@pos=n hidden: max abs {:.3e} rel {:.3e} >1e-3: {} ; greedy seq={cont_a_greedy} batched={cont_b_greedy}",
        cabs, crel, cover
    );

    // ---- report hidden diffs per position ----------------------------------
    let mut worst = (0.0f32, 0.0f32, 0usize);
    for p in 0..n {
        let rb = &bat[p * N_EMBD..(p + 1) * N_EMBD];
        let m = diff_metrics(&ref_hidden[p], rb);
        eprintln!(
            "[bprefill] hidden pos {p:2}: max abs {:.3e}  max rel {:.3e}  >1e-3 rel: {}",
            m.0, m.1, m.2
        );
        if m.1 > worst.1 {
            worst = m;
        }
    }
    eprintln!(
        "[bprefill] hidden WORST: max abs {:.3e}  max rel {:.3e}  >1e-3 rel: {}",
        worst.0, worst.1, worst.2
    );

    // ---- report KV diffs (k and v for every full-attention layer) ----------
    let mut kworst = (0.0f32, 0.0f32, 0usize);
    let mut ai = 0usize;
    for il in 0..n_layer {
        if !dev.cfg.is_full_attention(il) {
            continue;
        }
        let (rk, rv) = &ref_kv[ai];
        let (bk, bv) = &bat_kv[ai];
        let mk = diff_metrics(rk, bk);
        let mv = diff_metrics(rv, bv);
        eprintln!(
            "[bprefill] KV layer {il:2}: k max abs {:.3e} rel {:.3e} over {} ; v max abs {:.3e} rel {:.3e} over {}",
            mk.0, mk.1, mk.2, mv.0, mv.1, mv.2
        );
        kworst.0 = kworst.0.max(mk.0.max(mv.0));
        kworst.1 = kworst.1.max(mk.1.max(mv.1));
        kworst.2 += mk.2 + mv.2;
        ai += 1;
    }
    eprintln!(
        "[bprefill] KV WORST: max abs {:.3e}  max rel {:.3e}  >1e-3 rel total {}",
        kworst.0, kworst.1, kworst.2
    );

    // ---- report GDN state + conv cache diffs (every recurrent layer) -------
    let mut sworst = (0.0f32, 0.0f32, 0usize); // state
    let mut cworst = (0.0f32, 0.0f32, 0usize); // conv
    let mut si = 0usize;
    for il in 0..n_layer {
        if !dev.cfg.is_recurrent(il) {
            continue;
        }
        let rs = ref_state[si].as_ref().unwrap();
        let bs = bat_state[si].as_ref().unwrap();
        let rc = ref_conv[si].as_ref().unwrap();
        let bc = bat_conv[si].as_ref().unwrap();
        assert_eq!(rs.len(), STATE_ELEMS);
        assert_eq!(rc.len(), CONV_ELEMS);
        let ms = diff_metrics(rs, bs);
        let mc = diff_metrics(rc, bc);
        eprintln!(
            "[bprefill] GDN layer {il:2}: state max abs {:.3e} rel {:.3e} over {} ; conv max abs {:.3e} rel {:.3e} over {}",
            ms.0, ms.1, ms.2, mc.0, mc.1, mc.2
        );
        if ms.1 > sworst.1 {
            sworst = ms;
        }
        if mc.1 > cworst.1 {
            cworst = mc;
        }
        sworst.2 += ms.2;
        cworst.2 += mc.2;
        si += 1;
    }
    eprintln!(
        "[bprefill] GDN-state WORST: max abs {:.3e}  max rel {:.3e}  >1e-3 rel total {}",
        sworst.0, sworst.1, sworst.2
    );
    eprintln!(
        "[bprefill] GDN-conv  WORST: max abs {:.3e}  max rel {:.3e}  >1e-3 rel total {}",
        cworst.0, cworst.1, cworst.2
    );

    // ---- greedy first generated token --------------------------------------
    eprintln!(
        "[bprefill] greedy first token: sequential={ref_greedy} batched={bat_greedy} equal={}",
        ref_greedy == bat_greedy
    );

    // ---- pass / fail ---------------------------------------------------------
    let hidden_ok = worst.1 < 1e-3 && worst.2 == 0;
    let kv_ok = kworst.1 < 1e-3 && kworst.2 == 0;
    let state_ok = sworst.1 < 1e-3 && sworst.2 == 0;
    let conv_ok = cworst.1 < 1e-3 && cworst.2 == 0;
    let greedy_ok = ref_greedy == bat_greedy;
    let cont_ok = crel < 1e-3 && cover == 0 && cont_a_greedy == cont_b_greedy;
    if hidden_ok && kv_ok && state_ok && conv_ok && greedy_ok && cont_ok {
        eprintln!("[bprefill] PASS");
    } else {
        eprintln!(
            "[bprefill] FAIL hidden_ok={hidden_ok} kv_ok={kv_ok} state_ok={state_ok} conv_ok={conv_ok} greedy_ok={greedy_ok} cont_ok={cont_ok}"
        );
        exit(1);
    }
}
