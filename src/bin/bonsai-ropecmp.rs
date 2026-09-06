//! bonsai-ropecmp: validate Rust IMROPE against ggml_rope_multi.

#[path = "../rope.rs"]
mod rope;

use std::fs;
use std::process::{exit, Command};

fn rand_floats(seed: u64, n: usize) -> Vec<f32> {
    let mut x = seed | 1;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            ((x >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let probe = args.first().cloned().unwrap_or_else(|| "target/ggml_probe".to_string());
    let scratch = std::env::temp_dir().join("bonsai-ropecmp");
    fs::create_dir_all(&scratch).unwrap();

    let head = rand_floats(9, 256);
    fs::write(scratch.join("in.bin"), {
        let mut b = Vec::new();
        for v in &head {
            b.extend_from_slice(&v.to_le_bytes());
        }
        b
    })
    .unwrap();

    let cases: [(i32, i32, i32, i32); 3] = [(5, 5, 5, 5), (17, 17, 17, 17), (3, 5, 7, 0)];
    for (pt, ph, pw, pe) in cases {
        let run = Command::new(&probe)
            .arg("rope")
            .arg(scratch.join("out.bin").to_str().unwrap())
            .arg(scratch.join("in.bin").to_str().unwrap())
            .arg(pt.to_string())
            .arg(ph.to_string())
            .arg(pw.to_string())
            .arg(pe.to_string())
            .output();
        let o = run.unwrap();
        if !o.status.success() {
            eprintln!("probe failed: {}", String::from_utf8_lossy(&o.stderr));
            exit(1);
        }
        let b = fs::read(scratch.join("out.bin")).unwrap();
        let ref_v: Vec<f32> = b
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        let mut rust_v = head.clone();
        rope::rope_imrope(
            &mut rust_v,
            pt as i64,
            ph as i64,
            pw as i64,
            pe as i64,
            64,
            [11, 11, 10, 0],
            1e7,
        );

        let scale = ref_v.iter().chain(&rust_v).fold(0.0f32, |a, b| a.max(b.abs())).max(1e-30);
        let diff = ref_v
            .iter()
            .zip(&rust_v)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max)
            / scale;
        println!("rope pos=({pt},{ph},{pw},{pe}) rel-diff {diff:.3e} -> {}", if diff <= 1e-5 { "PASS" } else { "FAIL" });
        if diff > 1e-5 {
            exit(1);
        }
    }
}
