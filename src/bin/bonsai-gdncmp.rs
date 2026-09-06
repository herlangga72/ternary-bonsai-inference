//! bonsai-gdncmp: validate the Rust gated-delta-net step against ggml
//! (`tools/ggml_probe gdn`) on synthetic single-token inputs.

#[path = "../gdn.rs"]
mod gdn;

use std::fs;
use std::path::PathBuf;
use std::process::{exit, Command};

const S: usize = 128;
const H_K: usize = 16;
const H_V: usize = 48;

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
fn read_f32(path: &PathBuf, n: usize) -> Vec<f32> {
    let b = fs::read(path).unwrap();
    assert!(b.len() >= n * 4);
    b.chunks_exact(4)
        .take(n)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let probe = args
        .first()
        .cloned()
        .unwrap_or_else(|| "target/ggml_probe".to_string());
    let scratch = std::env::temp_dir().join("bonsai-gdncmp");
    fs::create_dir_all(&scratch).unwrap();

    let q = rand_floats(1, H_K * S);
    let k = rand_floats(2, H_K * S);
    let v = rand_floats(3, H_V * S);
    let gate = rand_floats(4, H_V).iter().map(|x| x * 0.5).collect::<Vec<_>>();
    let beta = rand_floats(5, H_V);
    let mut state = rand_floats(6, H_V * S * S);

    for (name, v) in [
        ("q", &q),
        ("k", &k),
        ("v", &v),
        ("g", &gate),
        ("beta", &beta),
        ("state", &state),
    ] {
        write_f32(&scratch.join(format!("{name}.bin")), v);
    }

    // ggml reference: K=2 -> output = attn(S*H_v) + 2 state snapshots; slot 0 = final
    let run = Command::new(&probe)
        .arg("gdn")
        .arg(scratch.join("out.bin").to_str().unwrap())
        .arg(scratch.join("q.bin").to_str().unwrap())
        .arg(scratch.join("k.bin").to_str().unwrap())
        .arg(scratch.join("v.bin").to_str().unwrap())
        .arg(scratch.join("g.bin").to_str().unwrap())
        .arg(scratch.join("beta.bin").to_str().unwrap())
        .arg(scratch.join("state.bin").to_str().unwrap())
        .arg("2")
        .output();
    if let Err(e) = run {
        eprintln!("probe error: {e}");
        exit(1);
    }
    let run = run.unwrap();
    if !run.status.success() {
        eprintln!("probe failed: {}", String::from_utf8_lossy(&run.stderr));
        exit(1);
    }

    let attn_n = S * H_V;
    let ref_out = read_f32(&scratch.join("out.bin"), attn_n + 2 * H_V * S * S);
    let ref_attn = &ref_out[..attn_n];
    let ref_state = &ref_out[attn_n..attn_n + H_V * S * S]; // snapshot slot 0

    let mut rust_state = state.clone();
    let mut rust_attn = vec![0.0f32; attn_n];
    gdn::gdn_step(&q, &k, &v, &gate, &beta, &mut rust_state, &mut rust_attn);

    let maxdiff = |a: &[f32], b: &[f32], scale: f32| {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
            / scale
    };
    let scale_a = ref_attn.iter().chain(&rust_attn).fold(0.0f32, |a, b| a.max(b.abs())).max(1e-30);
    let scale_s = ref_state.iter().chain(&rust_state).fold(0.0f32, |a, b| a.max(b.abs())).max(1e-30);
    let da = maxdiff(ref_attn, &rust_attn, scale_a);
    let ds = maxdiff(ref_state, &rust_state, scale_s);
    println!("gdn attn  rel-diff {:.3e} -> {}", da, if da <= 1e-5 { "PASS" } else { "FAIL" });
    println!("gdn state rel-diff {:.3e} -> {}", ds, if ds <= 1e-5 { "PASS" } else { "FAIL" });
    exit(if da <= 1e-5 && ds <= 1e-5 { 0 } else { 1 });
}
