//! bonsai-server: OpenAI-compatible HTTP endpoint for the pure-Rust engine.
//!
//!   bonsai-server --model <target.gguf> [--host 127.0.0.1] [--port 8080]
//!                 [--think] [--keep-think] [--id <model-name>]
//!
//! Env: BONSAI_DSPARK=<sidecar.gguf> enables speculative greedy decoding,
//! BONSAI_DSPARK_N=<n> the draft length. Nothing else is required.

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
#[path = "../kvquant.rs"]
mod kvquant;
#[path = "../vk.rs"]
mod vk;
#[path = "../forward.rs"]
mod forward;
#[path = "../dspark.rs"]
mod dspark;
#[path = "../spec.rs"]
mod spec;
#[path = "../tokenizer.rs"]
mod tokenizer;
#[path = "../sampler.rs"]
mod sampler;
#[path = "../json.rs"]
mod json;
#[path = "../server.rs"]
mod server;

use forward::Decoder;
use server::Engine;
use std::process::exit;

struct Cli {
    model: String,
    host: String,
    port: u16,
    think: bool,
    strip: bool,
    id: Option<String>,
}

fn parse(args: &[String]) -> Result<Cli, String> {
    let mut c = Cli {
        model: String::new(),
        host: "127.0.0.1".into(),
        port: 8080,
        think: false,
        strip: true,
        id: None,
    };
    let mut i = 0;
    while i < args.len() {
        let need = |i: &mut usize, f: &str| -> Result<String, String> {
            *i += 1;
            args.get(*i).cloned().ok_or_else(|| format!("missing value for {f}"))
        };
        match args[i].as_str() {
            "-m" | "--model" => c.model = need(&mut i, "--model")?,
            "--host" => c.host = need(&mut i, "--host")?,
            "--port" => c.port = need(&mut i, "--port")?.parse().map_err(|_| "--port must be an int")?,
            "--think" => c.think = true,
            "--keep-think" => c.strip = false,
            "--id" => c.id = Some(need(&mut i, "--id")?),
            "-h" | "--help" => {
                println!(
                    "usage: bonsai-server --model <target.gguf> [options]\n\
                     \n  --host <ip>     bind address (default 127.0.0.1)\n\
                     \x20 --port <n>      port (default 8080)\n\
                     \x20 --id <name>     model id reported to clients\n\
                     \x20 --no-think      omit <think> from the chat prompt\n                     \x20 --keep-think    keep the <think>...</think> block in the reply\n\
                     \n  env BONSAI_DSPARK=<sidecar.gguf>  speculative greedy decode\n\
                     \x20     BONSAI_DSPARK_N=<n>            draft length (default 4)"
                );
                exit(0);
            }
            other => return Err(format!("unknown argument: {other}")),
        }
        i += 1;
    }
    if c.model.is_empty() {
        return Err("missing --model".into());
    }
    Ok(c)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cli = match parse(&args) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: {e}");
            exit(1);
        }
    };

    let dec = match Decoder::open(&cli.model) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("error: {e}");
            exit(1);
        }
    };
    let gf = match gguf::GGUF::open(&cli.model) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("error: {e}");
            exit(1);
        }
    };
    let vocab = match tokenizer::Vocab::from_gguf(&gf) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("error: {e}");
            exit(1);
        }
    };
    let eos = gf
        .get("tokenizer.ggml.eos_token_id")
        .and_then(|v| v.as_u64())
        .map(|x| x as u32);
    drop(gf);

    let mut stop: Vec<u32> = ["<|im_end|>", "<|resp_end|>", "<|endoftext|>"]
        .iter()
        .filter_map(|w| {
            let t = tokenizer::encode(
                w,
                &vocab,
                &tokenizer::TokenizeOptions { add_special: false, parse_special: true },
            );
            if t.len() == 1 {
                Some(t[0] as u32)
            } else {
                None
            }
        })
        .collect();
    if let Some(e) = eos {
        stop.push(e);
    }

    // optional dspark drafter
    let drafter = match std::env::var("BONSAI_DSPARK") {
        Ok(p) if !p.is_empty() => {
            let n_ctx = std::env::var("BONSAI_CTX")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(4096);
            match spec::Drafter::new(&p, Some(&cli.model), n_ctx, 0.0) {
                Ok(d) => {
                    eprintln!("[dspark] speculative greedy decode enabled ({p})");
                    Some(d)
                }
                Err(e) => {
                    eprintln!("[dspark] disabled: {e}");
                    None
                }
            }
        }
        _ => None,
    };

    let model_id = cli.id.clone().unwrap_or_else(|| {
        std::path::Path::new(&cli.model)
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "ternary-bonsai-27b".into())
    });

    let engine = Engine {
        dec,
        drafter,
        vocab,
        stop,
        model_id: model_id.clone(),
        think: cli.think,
        strip_think: cli.strip,
    };

    eprintln!("model: {model_id}  ({} layers)", engine.dec.cfg.n_layer);
    eprintln!(
        "try: curl -s http://{}:{}/v1/chat/completions -H 'Content-Type: application/json' \\\n\
         \x20     -d '{{\"messages\":[{{\"role\":\"user\",\"content\":\"hi\"}}],\"stream\":true}}'",
        cli.host, cli.port
    );
    let addr = format!("{}:{}", cli.host, cli.port);
    if let Err(e) = server::serve(&addr, engine) {
        eprintln!("error: {e}");
        exit(1);
    }
}
