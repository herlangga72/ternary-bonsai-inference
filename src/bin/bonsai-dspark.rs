//! bonsai-dspark: smoke-test the dspark drafter skeleton against the sidecar
//! GGUF. Checks the config, the expected tensor manifest (names, types, dims),
//! and runs the encoder on synthetic tapped features.

#[path = "../gguf.rs"]
mod gguf;
#[path = "../kernels.rs"]
mod kernels;
#[path = "../rope.rs"]
mod rope;
#[path = "../dspark.rs"]
mod dspark;

use dspark::Dspark;
use std::process::exit;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: bonsai-dspark <sidecar.gguf>");
        exit(1);
    }
    let d = match Dspark::open(&args[0]) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("open: {e}");
            exit(1);
        }
    };
    let c = &d.cfg;
    println!(
        "dspark: {} layers, n_embd {}, n_ff {}, {}q/{}kv x {} head_dim, vocab {}",
        c.n_layer, c.n_embd, c.n_ff, c.n_head, c.n_head_kv, c.head_dim, c.n_vocab
    );
    println!(
        "  block_size {}, mask {}, target_layers {:?}, markov_rank {}, conf {}, log_snr {:?}",
        c.block_size, c.mask_token_id, c.target_layers, c.markov_rank, c.confidence_head, c.log_snr
    );
    println!("  encoder input width {}", c.n_embd_enc());

    // --- expected tensor manifest ----------------------------------------
    let n_embd = c.n_embd as u64;
    let hd = c.head_dim as u64;
    let q_rows = (c.n_head * c.head_dim) as u64;
    let kv_rows = (c.n_head_kv * c.head_dim) as u64;
    let mut expect: Vec<(String, Vec<u64>)> = vec![
        ("dspark.fc.weight".into(), vec![c.n_embd_enc() as u64, n_embd]),
        ("dspark.hidden_norm.weight".into(), vec![n_embd]),
        ("token_embd.weight".into(), vec![n_embd, c.n_vocab as u64]),
        ("output.weight".into(), vec![n_embd, c.n_vocab as u64]),
        ("output_norm.weight".into(), vec![n_embd]),
        ("dspark.markov_head_a.weight".into(), vec![c.markov_rank as u64, c.n_vocab as u64]),
        ("dspark.markov_head_b.weight".into(), vec![c.markov_rank as u64, c.n_vocab as u64]),
        ("dspark.confidence_head.weight".into(), vec![n_embd + c.markov_rank as u64, 1]),
        ("dspark.confidence_head.bias".into(), vec![1]),
        ("dspark.log_snr_fc1.weight".into(), vec![128, n_embd]),
        ("dspark.log_snr_fc1.bias".into(), vec![n_embd]),
        ("dspark.log_snr_fc2.weight".into(), vec![n_embd, n_embd]),
        ("dspark.log_snr_fc2.bias".into(), vec![n_embd]),
    ];
    for il in 0..c.n_layer {
        let p = |s: &str| format!("blk.{il}.{s}");
        expect.push((p("attn_norm.weight"), vec![n_embd]));
        expect.push((p("attn_q.weight"), vec![n_embd, q_rows]));
        expect.push((p("attn_k.weight"), vec![n_embd, kv_rows]));
        expect.push((p("attn_v.weight"), vec![n_embd, kv_rows]));
        expect.push((p("attn_output.weight"), vec![q_rows, n_embd]));
        expect.push((p("attn_q_norm.weight"), vec![hd]));
        expect.push((p("attn_k_norm.weight"), vec![hd]));
        expect.push((p("ffn_norm.weight"), vec![n_embd]));
        expect.push((p("ffn_gate.weight"), vec![n_embd, c.n_ff as u64]));
        expect.push((p("ffn_up.weight"), vec![n_embd, c.n_ff as u64]));
        expect.push((p("ffn_down.weight"), vec![c.n_ff as u64, n_embd]));
    }

    let mut bad = 0;
    for (name, dims) in &expect {
        match d.tensor(name) {
            Err(e) => {
                eprintln!("MISSING {name}: {e}");
                bad += 1;
            }
            Ok(t) => {
                if t.dims != *dims {
                    eprintln!("DIMS {name}: {:?} != {:?}", t.dims, dims);
                    bad += 1;
                } else if !dspark::type_supported(t.ty) {
                    eprintln!("TYPE {name}: unsupported type {}", t.ty);
                    bad += 1;
                }
            }
        }
    }
    println!(
        "manifest: {}/{} expected tensors ok",
        expect.len() - bad,
        expect.len()
    );
    if bad > 0 {
        exit(1);
    }

    // --- encoder smoke (synthetic features, zeroed) -----------------------
    let n_tok = 3;
    let feats = vec![0.05f32; n_tok * c.n_embd_enc()];
    match d.encode(&feats, n_tok) {
        Ok(out) => {
            let finite = out.iter().all(|v| v.is_finite());
            let l2: f32 = out.iter().map(|v| v * v).sum::<f32>().sqrt();
            println!(
                "encoder: {} -> [{} x {}], all finite {finite}, out l2 {:.4}",
                n_tok * c.n_embd_enc(),
                n_tok,
                c.n_embd,
                l2
            );
            if !finite {
                exit(1);
            }
        }
        Err(e) => {
            eprintln!("encode: {e}");
            exit(1);
        }
    }
    // --- draft block smoke (real weights, empty context) ------------------
    let mut cache = dspark::DraftCache::new(c, 64);
    // inject a little fake context so attention has non-block positions to read
    let ctx_tok = 4;
    let feats2 = vec![0.01f32; ctx_tok * c.n_embd_enc()];
    let inp_g = d.encode(&feats2, ctx_tok).unwrap();
    let positions: Vec<usize> = (0..ctx_tok).collect();
    if let Err(e) = d.inject(&mut cache, &inp_g, &positions) {
        eprintln!("inject: {e}");
        exit(1);
    }
    let t0 = std::time::Instant::now();
    match d.draft_block(&mut cache, 1, ctx_tok) {
        Ok(b) => {
            let finite = b.logits.iter().all(|v| v.is_finite());
            let argmax: Vec<i64> = (0..c.block_size)
                .map(|t| {
                    let row = &b.logits[t * c.n_vocab..(t + 1) * c.n_vocab];
                    let mut bi = 0usize;
                    let mut bv = f32::NEG_INFINITY;
                    for (i, v) in row.iter().enumerate() {
                        if *v > bv {
                            bv = *v;
                            bi = i;
                        }
                    }
                    bi as i64
                })
                .collect();
            println!(
                "draft block in {:.2}s: logits finite {finite}, conf {:?}, argmax {:?}",
                t0.elapsed().as_secs_f32(),
                b.conf.iter().map(|v| (v * 1000.0).round() / 1000.0).collect::<Vec<_>>(),
                argmax
            );
            if !finite {
                exit(1);
            }
        }
        Err(e) => {
            eprintln!("draft_block: {e}");
            exit(1);
        }
    }
    println!("dspark skeleton: OK");
}
