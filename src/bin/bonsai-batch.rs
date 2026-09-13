//! bonsai-batch: P1 correctness check for the batched prefill plumbing.
//!
//! Verifies the pure-Rust additions from `notes/prefill-plan.md` P1 against a
//! real qwen35 GGUF:
//!
//! 1. **Batched embed gather is bit-exact**: the N-wide tile from
//!    `prefill::batch_embed` equals N individual `Weights::row_f32` reads of
//!    `token_embd.weight` element-for-element (byte-identical f32).
//! 2. *(optional, `--ref`)* **CPU reference reproduces per-token forward**:
//!    `prefill::ref_batched_hidden` returns the same per-token hidden rows the
//!    sequential `Decoder::forward_hidden` decode loop produces, so a later
//!    batched GPU prefill has a stable CPU anchor to diff against.
//!
//! This is CPU-only (embeddings and the reference stay host-side per the
//! forward.rs convention). It does not run the GPU engine.
//!
//! usage: bonsai-batch <model.gguf> [n_tokens] [--ref]

#[path = "../gguf.rs"]
mod gguf;
#[path = "../gdn.rs"]
mod gdn;
#[path = "../kernels.rs"]
mod kernels;
#[path = "../prefill.rs"]
mod prefill;
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

use std::process::exit;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: bonsai-batch <model.gguf> [n_tokens] [--ref]");
        exit(1);
    }
    let model = args[0].clone();
    let mut n = 4usize;
    let mut do_ref = false;
    for a in &args[1..] {
        if a == "--ref" {
            do_ref = true;
        } else if let Ok(v) = a.parse::<usize>() {
            n = v;
        } else {
            eprintln!("bonsai-batch: unknown argument {a}");
            exit(2);
        }
    }
    if n < 2 {
        eprintln!("bonsai-batch: n_tokens must be >= 2 for a meaningful gather check");
        exit(2);
    }

    // ---- 1. batched embed gather == N individual row_f32 reads (bit-exact) ----
    let mut w = weights::Weights::open(&model).expect("open weights");
    let vocab = w
        .tensor(prefill::EMBED_TENSOR)
        .map(|t| t.n_elem())
        .expect("model has token_embd.weight");
    let n_feat = w.config().n_embd;
    let tokens: Vec<u32> = (0..n as u32).map(|t| (t * 7919 + 101) % vocab as u32).collect();
    let listed = tokens
        .iter()
        .map(|t| t.to_string())
        .collect::<Vec<_>>()
        .join(",");
    eprintln!("[batch] gathering N={n} token rows ({listed}) from {model}");

    let tile = prefill::batch_embed(&mut w, &tokens).expect("batched gather");
    assert_eq!(tile.n_tokens(), n);
    assert_eq!(tile.n_feat(), n_feat);
    assert_eq!(tile.len(), n * n_feat);

    let mut n_diff = 0usize;
    let mut first: Option<(usize, usize, f32, f32)> = None;
    for (k, &tok) in tokens.iter().enumerate() {
        let single = w
            .row_f32(prefill::EMBED_TENSOR, tok as u64)
            .expect("individual row read");
        let batched = tile.row(k).unwrap();
        assert_eq!(batched.len(), single.len(), "row {k} length mismatch");
        for (f, (&a, &b)) in batched.iter().zip(single.iter()).enumerate() {
            if a.to_bits() != b.to_bits() {
                n_diff += 1;
                if first.is_none() {
                    first = Some((k, f, a, b));
                }
                break;
            }
        }
    }
    if n_diff == 0 {
        println!(
            "GATHER BIT-EXACT: batch_embed({n} rows) == {n} x row_f32, all {} elements identical",
            tile.len()
        );
    } else {
        let (k, f, a, b) = first.unwrap();
        eprintln!(
            "[batch] FAIL {n_diff} rows differ; first at row {k} feat {f}: {a} (bits {:08x}) vs {b} (bits {:08x})",
            a.to_bits(),
            b.to_bits()
        );
        exit(1);
    }

    // ---- 2. (optional) CPU reference rows == sequential per-token forward ----
    if do_ref {
        let mut dec = forward::Decoder::open(&model).expect("cpu decoder");
        let tile_ref = prefill::ref_batched_hidden(&mut dec, &tokens).expect("ref hidden");
        let mut oracle = forward::Decoder::open(&model).expect("cpu decoder");
        let mut n_diff = 0usize;
        let mut first: Option<(usize, usize, f32, f32)> = None;
        for k in 0..tokens.len() {
            let h = oracle.forward_hidden(tokens[k], k).expect("sequential forward");
            let row = tile_ref.row(k).unwrap();
            assert_eq!(row.len(), h.len(), "token {k} hidden length mismatch");
            for (f, (&a, &b)) in row.iter().zip(h.iter()).enumerate() {
                if a.to_bits() != b.to_bits() {
                    n_diff += 1;
                    if first.is_none() {
                        first = Some((k, f, a, b));
                    }
                    break;
                }
            }
        }
        if n_diff == 0 {
            println!(
                "REF BIT-EXACT: ref_batched_hidden({n}) == sequential per-token forward rows"
            );
        } else {
            let (k, f, a, b) = first.unwrap();
            eprintln!(
                "[ref] FAIL {n_diff} rows differ; first at token {k} feat {f}: {a} (bits {:08x}) vs {b} (bits {:08x})",
                a.to_bits(),
                b.to_bits()
            );
            exit(1);
        }
    }

    println!("bonsai-batch: OK");
}
