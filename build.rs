use std::env;
use std::path::PathBuf;

// Links against the PrismML llama.cpp fork build, but ONLY for the legacy
// `llama-backend` comparison tools (golden-logit capture, oracle tokenizer).
// The pure-Rust engine never links llama.cpp, so without that feature this
// script is a no-op and no external build tree is required.
fn main() {
    if env::var("CARGO_FEATURE_LLAMA_BACKEND").is_err() {
        return;
    }

    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());

    // Candidates for the llama.cpp cmake build tree, in priority order.
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Ok(dir) = env::var("BONSAI_LLAMA_DIR") {
        candidates.push(PathBuf::from(dir));
    }
    candidates.push(manifest.join("../llama.cpp/build"));
    candidates.push(manifest.join("llama.cpp/build"));

    let lib_dir = candidates
        .iter()
        .find(|p| p.join("src").join("libllama.a").is_file())
        .map(|p| p.join("src"))
        .or_else(|| {
            candidates
                .iter()
                .find(|p| {
                    p.join("libllama.a").is_file()
                        || p.join("lib").join("libllama.a").is_file()
                })
                .cloned()
        })
        .unwrap_or_else(|| {
            panic!(
                "llama.cpp static lib not found. Set BONSAI_LLAMA_DIR to the cmake build tree \
                 of the PrismML llama.cpp fork (prism branch). Looked in: {:?}",
                candidates
            )
        });

    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=static=llama");

    // ggml archives live next to libllama.a or in a sibling ggml/src dir.
    let mut ggml_dirs = vec![lib_dir.clone(), lib_dir.join("ggml/src")];
    if let Some(parent) = lib_dir.parent() {
        ggml_dirs.push(parent.join("ggml/src"));
    }
    for d in &ggml_dirs {
        for lib in ["ggml-cpu", "ggml-base", "ggml"] {
            if d.join(format!("lib{lib}.a")).is_file() {
                println!("cargo:rustc-link-search=native={}", d.display());
                println!("cargo:rustc-link-lib=static={lib}");
            }
        }
    }

    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rustc-link-lib=dylib=gomp");
    println!("cargo:rustc-link-lib=dylib=pthread");
    println!("cargo:rustc-link-lib=dylib=dl");
    println!("cargo:rustc-link-lib=dylib=m");

    println!("cargo:rerun-if-env-changed=BONSAI_LLAMA_DIR");
}
