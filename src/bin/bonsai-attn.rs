//! bonsai-attn: smoke-test the M6-3 full-attention layer on a real qwen35
//! GGUF over a short synthetic single-token sequence. Not a numeric oracle;
//! it exercises projections, per-head norms, IMROPE, KV cache indexing, GQA
//! scoring and the gated output projection end to end.

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
#[path = "../forward.rs"]
mod forward;
#[path = "../vk.rs"]
mod vk;

use forward::{full_attention_layer, AttnCache, AttnScratch};
use std::process::exit;
use std::time::Instant;
use weights::Weights;

struct XorShift(u64);
impl XorShift {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: bonsai-attn <model.gguf> [il] [n_tokens]");
        exit(1);
    }
    let model = &args[0];
    let il: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(3);
    let n_tokens: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(4);

    let mut w = match Weights::open(model) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("error: {e}");
            exit(1);
        }
    };
    let cfg = w.config().clone();
    if !cfg.is_full_attention(il) {
        eprintln!("layer {il} is not a full-attention layer (il % 4 == 3 needed)");
        exit(1);
    }
    let mut cache = AttnCache::new(&cfg);
    let mut scratch = AttnScratch::new(&cfg);
    let mut rng = XorShift(0x5EED_1234);

    let mut x: Vec<f32> = (0..cfg.n_embd)
        .map(|_| {
            let u = (rng.next() >> 40) as f32 / (1u64 << 24) as f32;
            u - 0.5
        })
        .collect();
    let t_all = Instant::now();
    for pos in 0..n_tokens {
        let t0 = Instant::now();
        let out = match full_attention_layer(&mut w, &cfg, il, &x, pos, &mut cache, &mut scratch) {
            Ok(y) => y,
            Err(e) => {
                eprintln!("error at pos {pos}: {e}");
                exit(1);
            }
        };
        assert_eq!(out.len(), cfg.n_embd);
        let norm: f32 = out.iter().map(|v| v * v).sum::<f32>().sqrt();
        let dt = t0.elapsed();
        println!(
            "pos {pos:>2}: kv_pos {}  out[0..4] = {:8.3} {:8.3} {:8.3} {:8.3}  |out| {:.1}  {:.2}s",
            cache.n_pos(il),
            out[0],
            out[1],
            out[2],
            out[3],
            norm,
            dt.as_secs_f32()
        );
        // next token's input is unrelated synthetic data; re-randomize
        for v in x.iter_mut() {
            let u = (rng.next() >> 40) as f32 / (1u64 << 24) as f32;
            *v = u - 0.5;
        }
    }
    println!(
        "done: {n_tokens} attention steps on layer {il} in {:.2}s",
        t_all.elapsed().as_secs_f32()
    );
}
