//! bonsai-kerncmp: validate pure-Rust kernels (M5) against ggml reference
//! graphs executed by tools/ggml_probe on real model tensors.
//!
//! Usage: bonsai-kerncmp <model.gguf> [path to ggml_probe]

#[path = "../gguf.rs"]
mod gguf;
#[path = "../kernels.rs"]
mod kernels;

use gguf::{GGUF, Value};
use std::fs;
use std::path::PathBuf;
use std::process::{exit, Command};

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

fn rand_floats(seed: u64, n: usize) -> Vec<f32> {
    let mut r = XorShift(seed | 1);
    (0..n)
        .map(|_| ((r.next() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0)
        .collect()
}

fn write_f32(path: &PathBuf, v: &[f32]) {
    let mut b = Vec::with_capacity(v.len() * 4);
    for x in v {
        b.extend_from_slice(&x.to_le_bytes());
    }
    fs::write(path, b).unwrap();
}

fn read_f32(path: &PathBuf) -> Vec<f32> {
    let b = fs::read(path).unwrap();
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: bonsai-kerncmp <model.gguf> [ggml_probe path]");
        exit(1);
    }
    let model = &args[0];
    let probe = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "target/ggml_probe".to_string());

    let scratch = std::env::temp_dir().join("bonsai-kerncmp");
    fs::create_dir_all(&scratch).unwrap();
    let x_path = scratch.join("x.bin");
    let w_path = scratch.join("w.bin");
    let out_path = scratch.join("out.bin");
    let row_path = scratch.join("row.bin");

    let mut ok = true;
    let mut g = match GGUF::open(model) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("error: {e}");
            exit(1);
        }
    };

    let report = |name: &str, diff: f32, tol: f32| -> bool {
        let pass = diff <= tol;
        println!(
            "{:<44} max|diff| = {:.3e} (tol {:.1e}) -> {}",
            name,
            diff,
            tol,
            if pass { "PASS" } else { "FAIL" }
        );
        pass
    };

    // ---- RMSNorm -----------------------------------------------------------
    let norm_info = g
        .tensors
        .iter()
        .find(|t| t.name == "blk.0.attn_norm.weight")
        .cloned();
    if let Some(t) = norm_info {
        if let Ok(w) = g.read_tensor(&t) {
            let n = w.len();
            let x = rand_floats(1, n);
            write_f32(&x_path, &x);
            write_f32(&w_path, &w);
            let eps = g
                .get("qwen35.attention.layer_norm_rms_epsilon")
                .and_then(|v| v.as_f32())
                .unwrap_or(1e-6);
            let run = Command::new(&probe)
                .args(["rmsnorm"])
                .arg(&x_path)
                .arg(&w_path)
                .arg(eps.to_string())
                .arg(&out_path)
                .output();
            match run {
                Ok(o) if o.status.success() => {
                    let ref_y = read_f32(&out_path);
                    let rust_y = kernels::rms_norm(&x, &w, eps);
                    let scale = ref_y
                        .iter()
                        .chain(&rust_y)
                        .fold(0.0f32, |a, b| a.max(b.abs()))
                        .max(1.0);
                    ok &= report(
                        "rms_norm(blk.0.attn_norm.weight)",
                        max_diff(&ref_y, &rust_y) / scale,
                        1e-5,
                    );
                }
                Ok(o) => {
                    eprintln!("probe rmsnorm failed: {}", String::from_utf8_lossy(&o.stderr));
                    ok = false;
                }
                Err(e) => {
                    eprintln!("probe rmsnorm error: {e}");
                    ok = false;
                }
            }
        }
    }

    // ---- PQ2_0 exact dequant (vs ggml type-traits to_float) ------------------
    let deq_cases: &[(&str, u64)] = &[
        ("blk.0.ffn_up.weight", 0),
        ("blk.0.ffn_up.weight", 1),
        ("blk.0.ffn_up.weight", 100),
        ("blk.0.ffn_up.weight", 5000),
        ("output.weight", 0),
        ("output.weight", 3),
        ("output.weight", 100_000),
    ];
    for &(name, row) in deq_cases {
        let Some(t) = g.tensors.iter().find(|t| t.name == name).cloned() else {
            continue;
        };
        if t.ty != gguf::TYPE_PQ2_0 {
            continue;
        }
        let ne0 = t.dims[0] as usize;
        let row_bytes = kernels::pq2_row_bytes(ne0);
        let mut raw = vec![0u8; row_bytes];
        if g.read_bytes(t.offset + row * row_bytes as u64, &mut raw).is_err() {
            continue;
        }
        fs::write(&row_path, &raw).unwrap();
        let run = Command::new(&probe)
            .args(["pq2deq"])
            .arg(&row_path)
            .arg(ne0.to_string())
            .arg(&out_path)
            .output();
        match run {
            Ok(o) if o.status.success() => {
                let ref_y = read_f32(&out_path);
                let rust_y = kernels::decode_pq2_0_row(&raw, ne0);
                let scale = ref_y
                    .iter()
                    .chain(&rust_y)
                    .fold(0.0f32, |a, b| a.max(b.abs()))
                    .max(1e-30);
                ok &= report(
                    &format!("dequant_pq2_0({name}, row {row})"),
                    max_diff(&ref_y, &rust_y) / scale,
                    1e-6,
                );
            }
            Ok(o) => {
                eprintln!("probe pq2deq failed: {}", String::from_utf8_lossy(&o.stderr));
                ok = false;
            }
            Err(e) => {
                eprintln!("probe pq2deq error: {e}");
                ok = false;
            }
        }
    }

    println!("\n{}", if ok { "ALL KERNELS MATCH" } else { "MISMATCHES FOUND" });
    exit(if ok { 0 } else { 1 });
}

#[allow(dead_code)]
fn _use(_v: &Value) {}
