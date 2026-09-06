//! bonsai-run: run Ternary-Bonsai-27B (ternary Q2_0 GGUF, arch qwen35) on CPU
//! from Rust, using the PrismML llama.cpp fork as the compute engine.
//!
//! The GGUF file is read and the ternary tensors are executed inside llama.cpp;
//! everything around it (model load, chat template, tokenization, sampling
//! policy, streaming, timing) lives here in Rust.

mod llama;

use llama::*;
use std::ffi::CString;
use std::io::{Read, Write};
use std::os::raw::{c_char, c_void};
use std::process::exit;
use std::ptr;
use std::time::Instant;

// ---------------------------------------------------------------------------
// tiny CLI parsing
// ---------------------------------------------------------------------------
struct Cli {
    model: String,
    prompt: String,
    system: String,
    n_predict: usize,
    n_ctx: u32,
    n_batch: u32,
    temp: f32,
    top_k: i32,
    top_p: f32,
    min_p: f32,
    seed: u32,
    n_threads: i32,
    verbose: bool,
}

impl Cli {
    fn parse(args: &[String]) -> Result<Cli, String> {
        let mut c = Cli {
            model: String::new(),
            prompt: String::new(),
            system: String::new(),
            n_predict: 64,
            n_ctx: 4096,
            n_batch: 512,
            temp: 0.6,
            top_k: 20,
            top_p: 0.9,
            min_p: 0.0,
            seed: 0xFFFF_FFFF, // random
            n_threads: std::thread::available_parallelism()
                .map(|n| n.get() as i32)
                .unwrap_or(4),
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
                "-c" | "--ctx" => {
                    c.n_ctx = need(&mut i, "--ctx")?.parse().map_err(|_| "--ctx must be an int")?
                }
                "-t" | "--threads" => {
                    c.n_threads = need(&mut i, "--threads")?.parse().map_err(|_| "--threads must be an int")?
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
         \x20 -m, --model <file>     ternary Q2_0 GGUF (e.g. Ternary-Bonsai-27B-Q2_0.gguf)\n\
         \x20 -p, --prompt <text>     user prompt (default: read stdin)\n\
         \x20 -s, --system <text>     system prompt (optional)\n\
         \x20 -n, --n-predict <n>     max tokens to generate (default 64)\n\
         \x20 -c, --ctx <n>           context window (default 4096)\n\
         \x20 -t, --threads <n>       cpu threads (default: all cores)\n\
         \x20     --temp <f>          temperature (default 0.6)\n\
         \x20     --top-k <n>         top-k (default 20)\n\
         \x20     --top-p <f>         top-p (default 0.9)\n\
         \x20     --min-p <f>         min-p (default 0)\n\
         \x20     --seed <u32>        sampler seed (default random)\n\
         \x20 -v, --verbose           print timing/debug to stderr\n\
         \x20 -h, --help              this help"
    );
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------
fn cstr(s: &str) -> CString {
    CString::new(s).unwrap_or_else(|_| CString::new("<invalid utf8>").unwrap())
}

fn unsafe_cstr<'a>(p: *const c_char) -> &'a str {
    if p.is_null() {
        return "";
    }
    let s = unsafe { std::ffi::CStr::from_ptr(p) };
    s.to_str().unwrap_or("<non-utf8>")
}

// forward llama.cpp log output to stderr
unsafe extern "C" fn log_cb(level: i32, text: *const c_char, _user: *mut c_void) {
    if level != GGML_LOG_LEVEL_CONT {
        eprint!("{}", unsafe_cstr(text));
    }
}

// Fallback chatml prompt when the model template cannot be applied.
fn fallback_chatml(system: &str, user: &str) -> String {
    let mut out = String::new();
    if !system.is_empty() {
        out.push_str("<|im_start|>system\n");
        out.push_str(system);
        out.push_str("<|im_end|>\n");
    }
    out.push_str("<|im_start|>user\n");
    out.push_str(user);
    out.push_str("<|im_end|>\n");
    out.push_str("<|im_start|>assistant\n");
    out
}

// Scan the vocabulary for special token ids used as generation stops.
fn find_stop_tokens(vocab: *const llama_vocab, wants: &[&str]) -> Vec<llama_token> {
    let n = unsafe { llama_vocab_n_tokens(vocab) };
    let mut found: Vec<Option<llama_token>> = vec![None; wants.len()];
    for tok in 0..n {
        let text = unsafe_cstr(unsafe { llama_vocab_get_text(vocab, tok) });
        for (i, w) in wants.iter().enumerate() {
            if found[i].is_none() && text == *w {
                found[i] = Some(tok);
            }
        }
        if found.iter().all(|f| f.is_some()) {
            break;
        }
    }
    found.into_iter().flatten().collect()
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
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

    if cli.verbose {
        eprintln!("llama.cpp {}", unsafe_cstr(unsafe { llama_version() }));
    }
    unsafe {
        llama_backend_init();
        llama_log_set(log_cb as *const c_void, ptr::null_mut());
    }

    // ---- load model -------------------------------------------------------
    let mpath = cstr(&cli.model);
    let mut mparams = unsafe { llama_model_default_params() };
    mparams.n_gpu_layers = 0; // CPU-only
    eprintln!("loading model {} ...", cli.model);
    let t_load = Instant::now();
    let model = unsafe { llama_model_load_from_file(mpath.as_ptr(), mparams) };
    if model.is_null() {
        eprintln!("error: failed to load model {}", cli.model);
        exit(1);
    }
    eprintln!("model loaded in {:.1}s", t_load.elapsed().as_secs_f32());

    let mut desc_buf = vec![0u8; 512];
    unsafe {
        llama_model_desc(model, desc_buf.as_mut_ptr() as *mut c_char, desc_buf.len());
    }
    let desc = unsafe_cstr(desc_buf.as_ptr() as *const c_char);
    let size_mb = unsafe { llama_model_size(model) } as f64 / (1024.0 * 1024.0);
    let n_params = unsafe { llama_model_n_params(model) };
    let n_ctx_train = unsafe { llama_model_n_ctx_train(model) };
    eprintln!(
        "{desc}: {:.0} MiB, {:.2}B params, {n_ctx_train} ctx",
        size_mb,
        n_params as f64 / 1e9
    );
    let vocab = unsafe { llama_model_get_vocab(model) };

    // ---- build the formatted prompt with the model's chat template ---------
    let role_sys = cstr("system");
    let role_user = cstr("user");
    let content_sys = cstr(&cli.system);
    let content_user = cstr(&cli.prompt);

    let prompt: String;
    let tmpl = unsafe { llama_model_chat_template(model, ptr::null()) };
    if !tmpl.is_null() {
        let mut messages: Vec<llama_chat_message> = Vec::new();
        if !cli.system.is_empty() {
            messages.push(llama_chat_message {
                role: role_sys.as_ptr(),
                content: content_sys.as_ptr(),
            });
        }
        messages.push(llama_chat_message {
            role: role_user.as_ptr(),
            content: content_user.as_ptr(),
        });
        let needed = unsafe {
            llama_chat_apply_template(tmpl, messages.as_ptr(), messages.len(), 1, ptr::null_mut(), 0)
        };
        if needed > 0 {
            let mut buf = vec![0u8; (needed + 1) as usize];
            let n = unsafe {
                llama_chat_apply_template(
                    tmpl,
                    messages.as_ptr(),
                    messages.len(),
                    1,
                    buf.as_mut_ptr() as *mut c_char,
                    needed + 1,
                )
            };
            if n > 0 {
                buf.truncate(n as usize);
                prompt = String::from_utf8_lossy(&buf).to_string();
            } else {
                prompt = fallback_chatml(&cli.system, &cli.prompt);
            }
        } else {
            prompt = fallback_chatml(&cli.system, &cli.prompt);
        }
    } else {
        prompt = fallback_chatml(&cli.system, &cli.prompt);
    }

    // ---- tokenize ----------------------------------------------------------
    let ptext = cstr(&prompt);
    let n_tok = unsafe {
        llama_tokenize(vocab, ptext.as_ptr(), -1, ptr::null_mut(), 0, 0, 1)
    };
    let n_tok = if n_tok < 0 { -n_tok } else { n_tok };
    if n_tok <= 0 {
        eprintln!("error: tokenize failed");
        exit(1);
    }
    let mut toks: Vec<llama_token> = vec![0; n_tok as usize];
    unsafe {
        llama_tokenize(vocab, ptext.as_ptr(), -1, toks.as_mut_ptr(), n_tok, 0, 1);
    }
    eprintln!("prompt: {} chars -> {} tokens", prompt.len(), toks.len());
    if cli.verbose {
        eprintln!("{prompt}\n");
    }

    // ---- context ------------------------------------------------------------
    let mut cparams = unsafe { llama_context_default_params() };
    let ctx_size = if cli.n_ctx == 0 {
        n_ctx_train as u32
    } else {
        cli.n_ctx.min(n_ctx_train as u32)
    };
    cparams.n_ctx = ctx_size;
    cparams.n_batch = cli.n_batch;
    cparams.n_ubatch = cli.n_batch;
    cparams.n_threads = cli.n_threads;
    cparams.n_threads_batch = cli.n_threads;
    let ctx = unsafe { llama_init_from_model(model, cparams) };
    if ctx.is_null() {
        eprintln!("error: failed to create context with n_ctx={ctx_size}");
        exit(1);
    }
    eprintln!("context created: n_ctx={ctx_size}, n_batch={}", cli.n_batch);

    // ---- sampler chain ------------------------------------------------------
    // Bonsai-flavored sampling: top-k -> top-p -> min-p -> temperature -> dist
    let sp = unsafe { llama_sampler_chain_default_params() };
    let smpl = unsafe { llama_sampler_chain_init(sp) };
    unsafe {
        llama_sampler_chain_add(smpl, llama_sampler_init_top_k(cli.top_k));
        llama_sampler_chain_add(smpl, llama_sampler_init_top_p(cli.top_p, 1));
        if cli.min_p > 0.0 {
            llama_sampler_chain_add(smpl, llama_sampler_init_min_p(cli.min_p, 1));
        }
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(cli.temp));
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(cli.seed));
    }

    // stop tokens: chatml end markers + eog handled separately
    let stop_ids = find_stop_tokens(vocab, &["<|im_end|>", "<|resp_end|>"]);
    eprintln!(
        "sampler: temp={} top_k={} top_p={} min_p={}; stops: {:?}",
        cli.temp, cli.top_k, cli.top_p, cli.min_p, stop_ids
    );

    // ---- prefill ------------------------------------------------------------
    let t_gen0 = Instant::now();
    let n_prompt = toks.len();
    let mut n_past: i32 = 0;
    let mut pos_buf: Vec<llama_pos> = Vec::new();
    let mut start = 0;
    while start < n_prompt {
        let chunk = (n_prompt - start).min(cli.n_batch as usize);
        pos_buf.clear();
        pos_buf.extend((n_past..n_past + chunk as i32).map(|p| p as llama_pos));
        let batch = llama_batch {
            n_tokens: chunk as i32,
            token: toks[start..start + chunk].as_mut_ptr(),
            embd: ptr::null_mut(),
            pos: pos_buf.as_mut_ptr(),
            n_seq_id: ptr::null_mut(),
            seq_id: ptr::null_mut(),
            // NULL logits => only the last token of the batch emits logits
            logits: ptr::null_mut(),
        };
        let rc = unsafe { llama_decode(ctx, batch) };
        if rc != 0 {
            eprintln!("error: llama_decode (prefill chunk) failed rc={rc}");
            exit(1);
        }
        n_past += chunk as i32;
        start += chunk;
    }
    let t_prefill = t_gen0.elapsed();
    eprintln!(
        "prefill done in {:.2}s ({:.1} tok/s)",
        t_prefill.as_secs_f32(),
        n_prompt as f32 / t_prefill.as_secs_f32().max(1e-6)
    );

    // ---- decode loop ----------------------------------------------------------
    let mut stdout = std::io::stdout();
    let t0 = Instant::now();
    let mut n_gen: i64 = 0;
    let mut piece_buf: Vec<u8> = vec![0; 1024];

    loop {
        // sample from the most recent decoded token (idx -1)
        let id = unsafe { llama_sampler_sample(smpl, ctx, -1) };

        if unsafe { llama_vocab_is_eog(vocab, id) != 0 } || stop_ids.contains(&id) {
            break;
        }

        // token -> text
        loop {
            let n = unsafe {
                llama_token_to_piece(
                    vocab,
                    id,
                    piece_buf.as_mut_ptr() as *mut c_char,
                    piece_buf.len() as i32,
                    0,
                    0,
                )
            };
            if n < 0 {
                piece_buf.resize((-n) as usize, 0);
                continue;
            }
            let _ = stdout.write_all(&piece_buf[..n as usize]);
            let _ = stdout.flush();
            break;
        }
        n_gen += 1;
        if n_gen >= cli.n_predict as i64 {
            break;
        }

        // decode the sampled token as a 1-token batch
        let mut tok_slot = id;
        let mut pos_slot = n_past;
        let batch = llama_batch {
            n_tokens: 1,
            token: &mut tok_slot,
            embd: ptr::null_mut(),
            pos: &mut pos_slot,
            n_seq_id: ptr::null_mut(),
            seq_id: ptr::null_mut(),
            logits: ptr::null_mut(),
        };
        let rc = unsafe { llama_decode(ctx, batch) };
        if rc != 0 {
            eprintln!("\nerror: llama_decode (generation) failed rc={rc}");
            break;
        }
        n_past += 1;
    }

    let el = t0.elapsed();
    let _ = stdout.flush();
    eprintln!(
        "\n\n{n_gen} tokens in {:.2}s ({:.1} tok/s, {:.1} ms/tok)",
        el.as_secs_f32(),
        n_gen as f32 / el.as_secs_f32().max(1e-6),
        el.as_secs_f32() * 1000.0 / n_gen.max(1) as f32,
    );

    unsafe {
        llama_sampler_free(smpl);
        llama_free(ctx);
        llama_model_free(model);
    }
}
