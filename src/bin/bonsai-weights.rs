//! bonsai-weights: M6-1 weight-context checks against a real qwen35 GGUF.
//!
//! Usage:
//!   bonsai-weights config <model.gguf>   print parsed qwen35 hyperparameters
//!   bonsai-weights check  <model.gguf>   verify the per-layer tensor manifest
//!                                        and run a smoke matvec/vec access

#[path = "../gguf.rs"]
mod gguf;
#[path = "../kernels.rs"]
mod kernels;
#[path = "../weights.rs"]
mod weights;
#[path = "../kvquant.rs"]
mod kvquant;
#[path = "../vk.rs"]
mod vk;

use std::process::exit;
use std::time::Instant;
use weights::{Qwen35, Weights};

fn print_config(cfg: &Qwen35) {
    println!("architecture: qwen35");
    println!("layers: {} (full-attention every {} layers)", cfg.n_layer, cfg.interval);
    println!(
        "model: n_embd {}  n_ff {}  heads {} q / {} kv / head {}  ctx {}",
        cfg.n_embd, cfg.n_ff, cfg.n_head, cfg.n_head_kv, cfg.n_embd_head, cfg.n_ctx_train
    );
    println!(
        "rope: n_rot {} sections {:?} freq_base {}",
        cfg.n_rot, cfg.sections, cfg.freq_base
    );
    println!(
        "ssm: conv_kernel {} state {} groups {} dt_rank {} inner {}",
        cfg.ssm_conv_kernel, cfg.ssm_state, cfg.ssm_group_count, cfg.ssm_dt_rank, cfg.ssm_inner
    );
    println!(
        "ssm branch derived: key_dim {} conv_channels {}",
        cfg.ssm_key_dim(),
        cfg.ssm_conv_channels()
    );
    let (n_full, n_rec) = (0..cfg.n_layer)
        .fold((0, 0), |(a, b), il| {
            if cfg.is_full_attention(il) {
                (a + 1, b)
            } else {
                (a, b + 1)
            }
        });
    println!("layer split: {n_full} full-attention + {n_rec} recurrent");
}

fn check(path: &str) -> bool {
    let mut w = match Weights::open(path) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("error: {e}");
            return false;
        }
    };
    println!("model: {path}");
    print_config(w.config());

    let t0 = Instant::now();
    match w.verify() {
        Ok(n) => println!(
            "verify: {n}/{} expected layer tensors match ({:.2}s)",
            w.n_indexed(),
            t0.elapsed().as_secs_f32()
        ),
        Err(e) => {
            eprintln!("verify FAILED:\n{e}");
            return false;
        }
    }

    // smoke: vec_f32 + full matvec through the name-based API (row access)
    let cfg = w.config().clone();
    let x = vec![0.25f32; cfg.n_embd];
    let t_vec = Instant::now();
    let norm = w
        .vec_f32("blk.0.attn_norm.weight")
        .expect("blk.0.attn_norm.weight must decode");
    assert_eq!(norm.len(), cfg.n_embd);
    let dt_vec = t_vec.elapsed().as_secs_f32();

    let t_mv = Instant::now();
    let y = w
        .matvec("blk.0.ffn_up.weight", &x)
        .expect("blk.0.ffn_up.weight matvec must run");
    let dt_mv = t_mv.elapsed().as_secs_f32();
    let macs = cfg.n_embd as u64 * cfg.n_ff as u64;
    println!(
        "smoke: blk.0.attn_norm.weight len {} ({dt_vec:.3}s); blk.0.ffn_up.weight {} rows -> {:.4} GMAC/s ({dt_mv:.2}s); y[0..4] = {:?}",
        norm.len(),
        y.len(),
        macs as f64 / dt_mv as f64 / 1e9,
        &y[..4.min(y.len())]
    );

    // row_f32 path (embedding/LM-head style lookup) on the largest tensor
    let t_row = Instant::now();
    let row = w.row_f32("output.weight", 3).expect("output.weight row 3");
    println!(
        "smoke: output.weight row 3 len {} ({:.3}s), first 4 = {:?}",
        row.len(),
        t_row.elapsed().as_secs_f32(),
        &row[..4.min(row.len())]
    );
    true
}

fn usage() {
    eprintln!(
        "usage:\n  bonsai-weights config <model.gguf>\n  bonsai-weights check  <model.gguf>"
    );
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 2 {
        usage();
        exit(1);
    }
    match args[0].as_str() {
        "config" => {
            let w = match Weights::open(&args[1]) {
                Ok(w) => w,
                Err(e) => {
                    eprintln!("error: {e}");
                    exit(1);
                }
            };
            print_config(w.config());
        }
        "check" => {
            if !check(&args[1]) {
                exit(1);
            }
        }
        _ => {
            usage();
            exit(1);
        }
    }
}
