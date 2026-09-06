//! bonsai-actcmp: validate M6-2 activation helpers against ggml reference ops
//! executed by tools/ggml_probe on synthetic inputs (row RMSNorm, row L2 norm,
//! SiLU / softplus / sigmoid, row softmax).

#[path = "../gguf.rs"]
mod gguf;
#[path = "../kernels.rs"]
mod kernels;

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

fn rand_range(seed: u64, n: usize, lo: f32, hi: f32) -> Vec<f32> {
    let mut r = XorShift(seed | 1);
    (0..n)
        .map(|_| {
            let u = (r.next() >> 40) as f32 / (1u64 << 24) as f32;
            lo + u * (hi - lo)
        })
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

fn rel_max_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
        / a.iter()
            .chain(b)
            .fold(0.0f32, |acc, v| acc.max(v.abs()))
            .max(1e-30)
}

fn run_probe(probe: &str, args: &[&str]) -> Result<(), String> {
    let out = Command::new(probe)
        .args(args)
        .output()
        .map_err(|e| format!("probe error: {e}"))?;
    if !out.status.success() {
        return Err(format!("probe failed: {}", String::from_utf8_lossy(&out.stderr)));
    }
    Ok(())
}

fn report(name: &str, diff: f32, tol: f32) -> bool {
    let pass = diff <= tol;
    println!(
        "{name:<38} rel max|diff| = {diff:.3e} (tol {tol:.0e}) -> {}",
        if pass { "PASS" } else { "FAIL" }
    );
    pass
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let probe = args
        .first()
        .cloned()
        .unwrap_or_else(|| "target/ggml_probe".to_string());
    let scratch = std::env::temp_dir().join("bonsai-actcmp");
    fs::create_dir_all(&scratch).unwrap();

    let xp = |name: &str| scratch.join(format!("{name}.bin"));
    let op = |name: &str| scratch.join(format!("{name}_out.bin"));
    let s = |p: &PathBuf| p.to_str().unwrap().to_string();
    let eps = 1e-6f32;
    let mut ok = true;

    // ---- RMSNorm over rows: (256, 24) like q heads; (128, 48) like v heads ---
    for (tag, n_rows, row_len) in [("rms_q", 24usize, 256usize), ("rms_v", 48usize, 128usize)] {
        let x = rand_range(11, n_rows * row_len, -1.0, 1.0);
        let w = rand_range(12, row_len, 0.5, 2.0);
        write_f32(&xp(tag), &x);
        write_f32(&xp(&format!("{tag}_w")), &w);
        run_probe(
            &probe,
            &[
                "rmsrows",
                &s(&xp(tag)),
                &s(&xp(&format!("{tag}_w"))),
                &n_rows.to_string(),
                &row_len.to_string(),
                &eps.to_string(),
                &s(&op(tag)),
            ],
        )
        .unwrap();
        let ref_y = read_f32(&op(tag), n_rows * row_len);
        let rust_y = kernels::rms_norm_rows(&x, &w, row_len, eps).unwrap();
        ok &= report(&format!("rms_norm_rows({tag})"), rel_max_diff(&ref_y, &rust_y), 1e-5);
    }

    // ---- L2 norm over rows: (128, 16) like recurrent q/k ----------------------
    let n_rows = 16usize;
    let row_len = 128usize;
    let x = rand_range(21, n_rows * row_len, -1.0, 1.0);
    write_f32(&xp("l2"), &x);
    run_probe(
        &probe,
        &[
            "l2norm",
            &s(&xp("l2")),
            &n_rows.to_string(),
            &row_len.to_string(),
            &eps.to_string(),
            &s(&op("l2")),
        ],
    )
    .unwrap();
    let ref_y = read_f32(&op("l2"), n_rows * row_len);
    let rust_y = kernels::l2_norm_rows(&x, row_len, eps).unwrap();
    ok &= report("l2_norm_rows", rel_max_diff(&ref_y, &rust_y), 1e-6);

    // ---- elementwise activations ---------------------------------------------
    let n = 4096usize;
    let silu_x = rand_range(31, n, -8.0, 8.0);
    let sp_x = rand_range(32, n, -8.0, 24.0); // straddles the x>20 identity branch
    let sig_x = rand_range(33, n, -8.0, 8.0);
    for (tag, src, f) in [
        ("silu", &silu_x, kernels::silu as fn(f32) -> f32),
        ("softplus", &sp_x, kernels::softplus),
        ("sigmoid", &sig_x, kernels::sigmoid),
    ] {
        write_f32(&xp(tag), src);
        run_probe(
            &probe,
            &["unary", tag, &s(&xp(tag)), &s(&op(tag))],
        )
        .unwrap();
        let ref_y = read_f32(&op(tag), n);
        let rust_y: Vec<f32> = src.iter().map(|&v| f(v)).collect();
        ok &= report(&format!("{tag}"), rel_max_diff(&ref_y, &rust_y), 1e-5);
    }

    // ---- softmax over rows with -inf masked entries --------------------------
    let row_len = 2048usize;
    let n_rows = 8usize;
    let mut soft_x = rand_range(41, n_rows * row_len, -12.0, 2.0);
    // mask ~25% of each row like an attention mask
    let mut r = XorShift(42 | 1);
    for row in 0..n_rows {
        for i in 0..row_len {
            if (r.next() >> 62) == 0 {
                soft_x[row * row_len + i] = f32::NEG_INFINITY;
            }
        }
    }
    write_f32(&xp("softmax"), &soft_x);
    run_probe(
        &probe,
        &[
            "softmax",
            &s(&xp("softmax")),
            &n_rows.to_string(),
            &row_len.to_string(),
            &s(&op("softmax")),
        ],
    )
    .unwrap();
    let ref_y = read_f32(&op("softmax"), n_rows * row_len);
    let rust_y = kernels::softmax_rows(&soft_x, row_len).unwrap();
    ok &= report("softmax_rows(masked)", rel_max_diff(&ref_y, &rust_y), 1e-5);

    println!(
        "\n{}",
        if ok {
            "ALL ACTIVATION KERNELS MATCH"
        } else {
            "MISMATCHES FOUND"
        }
    );
    exit(if ok { 0 } else { 1 });
}
