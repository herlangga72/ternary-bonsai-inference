//! bonsai-gguf: inspect and retag GGUF files, pure Rust (M2).
//!
//! Usage:
//!   bonsai-gguf inspect <file>
//!   bonsai-gguf probe  <file> <tensor-name> [count] [start]
//!   bonsai-gguf retag  <legacy-in.gguf> <out.gguf>
//!   bonsai-gguf dspark-convert <sidecar-in.gguf> <out.gguf>

#[path = "../gguf.rs"]
mod gguf;

use gguf::{convert_dspark_sidecar, retag_legacy_ternary, GGUF, Value};
use std::process::exit;

fn type_name(t: u32) -> &'static str {
    match t {
        0 => "f32",
        1 => "f16",
        2 => "q4_0",
        3 => "q4_1",
        6 => "q5_0",
        7 => "q5_1",
        8 => "q8_0",
        10 => "q2_k",
        11 => "q3_k",
        12 => "q4_k",
        13 => "q5_k",
        14 => "q6_k",
        15 => "q8_k",
        16 => "iq2_xxs",
        17 => "iq2_xs",
        18 => "iq3_xxs",
        19 => "iq1_s",
        20 => "iq4_nl",
        21 => "iq3_s",
        22 => "iq2_s",
        23 => "iq4_xs",
        24 => "i8",
        25 => "iq1_m",
        26 => "bf16",
        30 => "q1_0",
        34 => "tq1_0",
        41 => "q1_0",
        42 => "q2_0(legacy-g128)",
        142 => "pq2_0",
        143 => "ptq1_0",
        _ => "unknown",
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 2 {
        usage();
        exit(1);
    }
    match args[0].as_str() {
        "inspect" => inspect(&args[1]),
        "probe" => probe(&args),
        "retag" => {
            if args.len() != 3 {
                usage();
                exit(1);
            }
            retag(&args[1], &args[2]);
        }
        "dspark-convert" => {
            if args.len() != 3 {
                usage();
                exit(1);
            }
            match convert_dspark_sidecar(&args[1], &args[2]) {
                Ok((n_kv, n_ty)) => println!(
                    "converted {}: {n_kv} metadata keys -> dflash naming, {n_ty} tensors renamed",
                    args[1]
                ),
                Err(e) => {
                    eprintln!("dspark-convert error: {e}");
                    exit(1);
                }
            }
        }
        _ => {
            usage();
            exit(1);
        }
    }
}

fn usage() {
    eprintln!(
        "usage:\n  bonsai-gguf inspect <file>\n  bonsai-gguf probe <file> <tensor-name> [count]\n  bonsai-gguf retag <legacy-in.gguf> <out.gguf>\n  bonsai-gguf dspark-convert <sidecar-in.gguf> <out.gguf>"
    );
}

fn inspect(path: &str) {
    let mut g = match GGUF::open(path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("error: {e}");
            exit(1);
        }
    };
    let arch = g
        .get("general.architecture")
        .and_then(|v| v.as_str())
        .unwrap_or("?")
        .to_string();
    println!("file: {path}");
    println!("gguf version: {}, tensors: {}", g.version, g.n_tensors);

    use std::collections::HashMap;
    let mut hist: HashMap<u32, usize> = HashMap::new();
    for t in &g.tensors {
        *hist.entry(t.ty).or_insert(0) += 1;
    }
    let mut counts: Vec<_> = hist.into_iter().collect();
    counts.sort();
    for (ty, n) in counts {
        println!("  type {ty:>3} ({:<16}): {n}", type_name(ty));
    }
    println!("architecture: {arch}");

    match g.check_layout() {
        Ok((data_bytes, file_len)) => println!(
            "layout ok: data section {:.1} MiB, file {:.1} MiB",
            data_bytes as f64 / (1024.0 * 1024.0),
            file_len as f64 / (1024.0 * 1024.0)
        ),
        Err(e) => {
            eprintln!("layout error: {e}");
            exit(1);
        }
    }

    // print a few interesting metadata values
    for key in ["general.name", "general.size_label", "general.file_type"] {
        if let Some(v) = g.get(key) {
            println!("meta {key}: {v:?}");
        }
    }
    let _ = &mut g;
}

fn probe(args: &[String]) {
    let (path, name) = (&args[1], &args[2]);
    let count: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(16);
    let start: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(0);
    let mut g = match GGUF::open(path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("error: {e}");
            exit(1);
        }
    };
    let info = match g.tensors.iter().find(|t| t.name == *name) {
        Some(t) => t.clone(),
        None => {
            eprintln!("tensor '{name}' not found");
            exit(1);
        }
    };
    println!(
        "{}: dims {:?}, type {} ({}), n_elem {}",
        info.name,
        info.dims,
        info.ty,
        type_name(info.ty),
        info.n_elem()
    );
    if start + count > info.n_elem() as usize {
        eprintln!("range out of bounds (start+count > n_elem)");
        exit(1);
    }
    let vals: Vec<f32> = match info.ty {
        142 => match g.read_pq2_0_range(&info, start, count) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("decode error: {e}");
                exit(1);
            }
        },
        0 | 1 => match g.read_tensor(&info) {
            Ok(v) => v[start..start + count].to_vec(),
            Err(e) => {
                eprintln!("decode error: {e}");
                exit(1);
            }
        },
        other => {
            eprintln!("probe: unsupported tensor type {other}");
            exit(1);
        }
    };
    for v in &vals {
        println!("{v:.7e}");
    }
}

fn retag(src: &str, dst: &str) {
    match retag_legacy_ternary(src, dst) {
        Ok((n_ty, n_ft)) => {
            println!("retagged {n_ty} tensors 42->142, {n_ft} file_type fields");
            if n_ty == 0 {
                eprintln!("warning: no tensors needed retagging (already current format?)");
            }
        }
        Err(e) => {
            eprintln!("retag error: {e}");
            exit(1);
        }
    }
}

// silence unused-import warnings when Value is not referenced directly
#[allow(dead_code)]
fn _value_ty(_v: &Value) {}
