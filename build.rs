use std::env;
use std::path::PathBuf;
use std::process::Command;

/// Compile every GLSL compute shader under `shaders/` to SPIR-V in OUT_DIR.
/// `glslangValidator` ships with the Vulkan SDK; the pure-CPU engine never
/// needs it, but the Vulkan backend (`bonsai-vk`) fails to build without the
/// generated modules.
fn compile_shaders() {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let shader_dir = manifest.join("shaders");
    let tool = find_in_path("glslangValidator");
    let Some(tool) = tool else {
        println!(
            "cargo:warning=glslangValidator not found in PATH: Vulkan SPIR-V modules will not \
             be compiled (the bonsai-vk binary needs it)"
        );
        return;
    };
    let Ok(entries) = std::fs::read_dir(&shader_dir) else {
        return;
    };
    for e in entries.flatten() {
        let path = e.path();
        if path.extension().and_then(|s| s.to_str()) != Some("comp") {
            continue;
        }
        let name = path.file_stem().unwrap().to_string_lossy().into_owned();
        let dst = out.join(format!("{name}.spv"));
        println!("cargo:rerun-if-changed={}", path.display());
        let status = Command::new(&tool)
            .arg("-V")
            .arg("--target-env")
            .arg("vulkan1.2")
            .arg("-o")
            .arg(&dst)
            .arg(&path)
            .status();
        match status {
            Ok(s) if s.success() => {
                println!("cargo:rerun-if-changed={}", dst.display());
            }
            other => {
                println!(
                    "cargo:warning=glslangValidator failed for {}: {:?}",
                    path.display(),
                    other
                );
            }
        }
    }
}

fn find_in_path(name: &str) -> Option<PathBuf> {
    let path = env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        let cand = dir.join(name);
        if cand.is_file() {
            return Some(cand);
        }
    }
    None
}

// Links against the PrismML llama.cpp fork build, but ONLY for the legacy
// `llama-backend` comparison tools (golden-logit capture, oracle tokenizer).
// The pure-Rust engine never links llama.cpp, so without that feature this
// script is a no-op and no external build tree is required.
fn main() {
    compile_shaders();

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
