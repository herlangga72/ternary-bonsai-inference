//! bonsai-run: run Ternary-Bonsai-27B (ternary qwen35 GGUF) on CPU entirely in
//! Rust (M7). No llama.cpp is linked or called: GGUF reading, weight context,
//! tokenizer/detokenizer, the 64-layer forward pass and the sampler all live
//! in this crate.

mod forward;
mod kvquant;
mod gguf;
mod vk;
mod gdn;
mod kernels;
mod prefill;
mod rope;
mod sampler;
mod tokenizer;
mod weights;

use forward::Decoder;
use sampler::{Sampler, SamplerConfig};
use std::io::{Read, Write};
use std::process::exit;
use std::time::Instant;

// ---------------------------------------------------------------------------
// tiny CLI parsing
// ---------------------------------------------------------------------------
struct Cli {
    model: String,
    prompt: String,
    system: String,
    n_predict: usize,
    temp: f32,
    top_k: i32,
    top_p: f32,
    min_p: f32,
    seed: u32,
    verbose: bool,
}

impl Cli {
    fn parse(args: &[String]) -> Result<Cli, String> {
        let mut c = Cli {
            model: String::new(),
            prompt: String::new(),
            system: String::new(),
            n_predict: 8,
            temp: 0.6,
            top_k: 20,
            top_p: 0.9,
            min_p: 0.0,
            seed: 0xFFFF_FFFF, // random
            verbose: false,
        };
        let mut i = 0;
        let need = |i: &mut usize, flag: &str| -> Result<String, String> {
            *i += 1;
            args.get(*i)
                .cloned()
                .ok_or_else(|| format!("missing value for {flag}"))
        };
        while i < args.len() {
            match args[i].as_str() {
                "-m" | "--model" => c.model = need(&mut i, "--model")?,
                "-p" | "--prompt" => c.prompt = need(&mut i, "--prompt")?,
                "-s" | "--system" => c.system = need(&mut i, "--system")?,
                "-n" | "--n-predict" => {
                    c.n_predict = need(&mut i, "--n-predict")?.parse().map_err(|_| "--n-predict must be an int")?
                }
                "-c" | "--ctx" | "-t" | "--threads" | "--n-batch" => {
                    let flag = args[i].clone();
                    let _ = need(&mut i, &flag);
                }
                "--temp" => c.temp = need(&mut i, "--temp")?.parse().map_err(|_| "--temp must be a float")?,
                "--top-k" => c.top_k = need(&mut i, "--top-k")?.parse().map_err(|_| "--top-k must be an int")?,
                "--top-p" => c.top_p = need(&mut i, "--top-p")?.parse().map_err(|_| "--top-p must be a float")?,
                "--min-p" => c.min_p = need(&mut i, "--min-p")?.parse().map_err(|_| "--min-p must be a float")?,
                "--seed" => c.seed = need(&mut i, "--seed")?.parse().map_err(|_| "--seed must be a u32")?,
                "-v" | "--verbose" => c.verbose = true,
                "-h" | "--help" => {
                    print_usage();
                    exit(0);
                }
                other => return Err(format!("unknown argument: {other}")),
            }
            i += 1;
        }
        if c.model.is_empty() {
            return Err("missing --model".into());
        }
        if c.prompt.is_empty() {
            let mut buf = String::new();
            std::io::stdin()
                .read_to_string(&mut buf)
                .map_err(|e| format!("stdin: {e}"))?;
            c.prompt = buf.trim().to_string();
        }
        if c.prompt.is_empty() {
            return Err("missing prompt (use --prompt or pipe stdin)".into());
        }
        Ok(c)
    }
}

fn print_usage() {
    eprintln!(
        "usage: bonsai-run --model <path.gguf> [options]\n\
         \n\
         options:\n\
         \x20 -m, --model <file>     ternary PQ2_0 GGUF (arch qwen35)\n\
         \x20 -p, --prompt <text>     user prompt (default: read stdin)\n\
         \x20 -s, --system <text>     system prompt (optional)\n\
         \x20 -n, --n-predict <n>     max tokens to generate (default 8)\n\
         \x20     --temp <f>          temperature (default 0.6)\n\
         \x20     --top-k <n>         top-k (default 20)\n\
         \x20     --top-p <f>         top-p (default 0.9)\n\
         \x20     --min-p <f>         min-p (default 0)\n\
         \x20     --seed <u32>        sampler seed (default random)\n\
         \x20 -v, --verbose           print timing to stderr\n\
         \x20 -h, --help              this help"
    );
}

// Plain-text path of the qwen35 chat template (from the GGUF jinja): wraps
// messages in <|im_start|> markers and starts the assistant turn in thinking
// mode with a <think> tag. Text-only chats, no tools/vision.
fn apply_qwen35_template(system: &str, user: &str) -> String {
    let mut out = String::new();
    if !system.is_empty() {
        out.push_str("<|im_start|>system\n");
        out.push_str(system);
        out.push_str("<|im_end|>\n");
    }
    out.push_str("<|im_start|>user\n");
    out.push_str(user);
    out.push_str("<|im_end|>\n");
    out.push_str("<|im_start|>assistant\n<think>\n");
    out
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cli = match Cli::parse(&args) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: {e}\n");
            print_usage();
            exit(1);
        }
    };

    // ---- load model + tokenizer (pure Rust) ---------------------------------
    let mut dec = match Decoder::open(&cli.model) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("error: {e}");
            exit(1);
        }
    };
    let n_vocab = dec.vocab_size();
    eprintln!(
        "model: {} layers, n_embd {}, vocab {}",
        dec.cfg.n_layer, dec.cfg.n_embd, n_vocab
    );

    let gf = match gguf::GGUF::open(&cli.model) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("error: {e}");
            exit(1);
        }
    };
    let rvocab = match tokenizer::Vocab::from_gguf(&gf) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("error: {e}");
            exit(1);
        }
    };
    let eos_id = gf
        .get("tokenizer.ggml.eos_token_id")
        .and_then(|v| v.as_u32())
        .map(|x| x as u32);

    // ---- build the formatted prompt (thinking mode) --------------------------
    let prompt = apply_qwen35_template(&cli.system, &cli.prompt);
    if cli.verbose {
        eprintln!("{prompt}\n");
    }
    let toks: Vec<u32> = tokenizer::encode(
        &prompt,
        &rvocab,
        &tokenizer::TokenizeOptions { add_special: false, parse_special: true },
    )
    .into_iter()
    .map(|t| t as u32)
    .collect();
    if toks.is_empty() {
        eprintln!("error: tokenize failed");
        exit(1);
    }
    eprintln!("prompt: {} chars -> {} tokens", prompt.len(), toks.len());

    // stop tokens: chat end markers + eos
    let mut stop_ids: Vec<u32> = ["<|im_end|>", "<|resp_end|>", "<|endoftext|>"]
        .iter()
        .filter_map(|w| {
            let t = tokenizer::encode(
                w,
                &rvocab,
                &tokenizer::TokenizeOptions { add_special: false, parse_special: true },
            );
            if t.len() == 1 {
                Some(t[0] as u32)
            } else {
                None
            }
        })
        .collect();
    if let Some(e) = eos_id {
        stop_ids.push(e);
    }
    if cli.verbose {
        eprintln!("stops: {stop_ids:?}");
    }

    let sampler_cfg = SamplerConfig {
        top_k: cli.top_k,
        top_p: cli.top_p,
        min_p: cli.min_p,
        temp: cli.temp,
        seed: cli.seed as u64,
    };
    let mut sampler = Sampler::new(&sampler_cfg);
    eprintln!(
        "sampler (rust): temp={} top_k={} top_p={} min_p={}",
        cli.temp, cli.top_k, cli.top_p, cli.min_p
    );

    // ---- prefill (forward each prompt token, cache the final hidden) ---------
    eprintln!("prefill ...");
    let t_gen0 = Instant::now();
    let mut last_h = Vec::new();
    for (pos, &tok) in toks.iter().enumerate() {
        last_h = match dec.forward_hidden(tok, pos) {
            Ok(h) => h,
            Err(e) => {
                eprintln!("error at prefill pos {pos}: {e}");
                exit(1);
            }
        };
    }
    let t_prefill = t_gen0.elapsed();
    eprintln!(
        "prefill done in {:.0}s ({:.2} s/tok)",
        t_prefill.as_secs_f32(),
        t_prefill.as_secs_f32() / toks.len() as f32
    );

    // ---- decode loop ----------------------------------------------------------
    let mut stdout = std::io::stdout();
    let t0 = Instant::now();
    let mut n_gen: u64 = 0;
    let mut baseline: Option<f32> = None;
    let mut logits = match dec.head_logits(&last_h) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("error: {e}");
            exit(1);
        }
    };

    loop {
        let id = sampler.sample(&logits, &sampler_cfg) as u32;
        if stop_ids.contains(&id) {
            break;
        }
        let piece = rvocab.piece_bytes(id as i32);
        let _ = stdout.write_all(&piece);
        let _ = stdout.flush();
        n_gen += 1;
        if n_gen >= cli.n_predict as u64 {
            break;
        }

        let pos = toks.len() + n_gen as usize - 1;
        let tk = Instant::now();
        let h = match dec.forward_hidden(id, pos) {
            Ok(h) => h,
            Err(e) => {
                eprintln!("\nerror: decode failed at pos {pos}: {e}");
                exit(1);
            }
        };
        logits = match dec.head_logits(&h) {
            Ok(l) => l,
            Err(e) => {
                eprintln!("\nerror: head failed: {e}");
                exit(1);
            }
        };
        let el = tk.elapsed().as_secs_f32();
        if let Some(b) = baseline {
            kernels::pause_for_budget(el, b);
        } else {
            baseline = Some(el);
        }
    }

    let el = t0.elapsed();
    let _ = stdout.flush();
    eprintln!(
        "\n\n{n_gen} tokens in {:.0}s ({:.2} s/tok)",
        el.as_secs_f32(),
        el.as_secs_f32() / n_gen.max(1) as f32,
    );
}
