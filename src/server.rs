//! OpenAI-compatible HTTP server for the pure-Rust engine.
//!
//! Endpoints:
//!   GET  /health
//!   GET  /v1/models
//!   POST /v1/chat/completions   (streaming SSE and non-streaming)
//!   POST /v1/completions        (legacy text completion)
//!
//! Dependency-free: `std::net` for HTTP/1.1 (fixed Content-Length for normal
//! replies, chunked for SSE) and `crate::json` for the bodies. One connection
//! per thread, one generation at a time behind a mutex (the engine is
//! single-stream).

#![allow(dead_code)]

use crate::json::{escape, J};
use crate::sampler::{Sampler, SamplerConfig};
use crate::spec::{self, Drafter};
use crate::forward::Decoder;
use crate::tokenizer::{self, Vocab};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};

pub struct Engine {
    pub dec: Decoder,
    pub drafter: Option<Drafter>,
    pub vocab: Vocab,
    pub stop: Vec<u32>,
    pub model_id: String,
    /// emit `<think>` (qwen35 thinking mode) before the answer
    pub think: bool,
    /// strip a leading `<think>...</think>` block from the reply so agents get
    /// the answer (the model emits the block itself even without the scaffold)
    pub strip_think: bool,
}

/// Removes a leading reasoning block from streamed text. Until `</think>` is
/// seen the text is buffered; then the remainder flows (and is streamed live if
/// enough of it is already buffered).
struct ThinkFilter {
    strip: bool,
    seen_end: bool,
    buf: String,
}

impl ThinkFilter {
    fn new(strip: bool) -> ThinkFilter {
        ThinkFilter { strip, seen_end: !strip, buf: String::new() }
    }
    fn feed(&mut self, text: &str) -> Option<String> {
        if self.seen_end {
            return Some(text.to_string());
        }
        self.buf.push_str(text);
        if let Some(i) = self.buf.find("</think>") {
            self.seen_end = true;
            let rest = self.buf[i + "</think>".len()..].to_string();
            self.buf.clear();
            return Some(rest);
        }
        // Not (yet) a reasoning block: `think` off plus a model that does not
        // prefix one. Once enough text has arrived without a leading <think we
        // stream it rather than buffering the whole reply.
        if self.buf.len() >= 8 && !self.buf.starts_with("<think") {
            self.seen_end = true;
            // keep a short prefix in case "</think>" straddles the boundary
            return Some(std::mem::take(&mut self.buf));
        }
        if self.buf.len() > 8192 {
            self.seen_end = true;
            return Some(std::mem::take(&mut self.buf));
        }
        None
    }
    /// Whatever is left if the model never closed its reasoning block.
    fn flush(&mut self) -> Option<String> {
        if self.seen_end || self.buf.is_empty() {
            return None;
        }
        self.seen_end = true;
        Some(std::mem::take(&mut self.buf))
    }
}

impl Engine {
    /// Clear all state so a request starts from an empty context.
    pub fn reset(&mut self) {
        self.dec.reset();
        if let Some(d) = self.drafter.as_mut() {
            d.dcache.reset();
        }
    }

    /// qwen35 chat template over the message list.
    pub fn build_prompt(&self, msgs: &[(String, String)]) -> String {
        let mut out = String::new();
        for (role, content) in msgs {
            out.push_str("<|im_start|>");
            out.push_str(role);
            out.push('\n');
            out.push_str(content);
            out.push_str("<|im_end|>\n");
        }
        out.push_str("<|im_start|>assistant\n");
        if self.think {
            out.push_str("<think>\n");
        }
        out
    }

    /// Greedy/sampled generation. `on_piece` receives each token's raw bytes
    /// and returns `false` to stop early (client disconnect).
    /// Returns (prompt_tokens, completion_tokens, finish_reason).
    pub fn generate(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        sampler_cfg: &SamplerConfig,
        on_piece: &mut dyn FnMut(&[u8]) -> bool,
    ) -> Result<(usize, usize, String), String> {
        let toks: Vec<u32> = tokenizer::encode(
            prompt,
            &self.vocab,
            &tokenizer::TokenizeOptions {
                add_special: false,
                parse_special: true,
            },
        )
        .into_iter()
        .map(|t| t as u32)
        .collect();
        if toks.is_empty() {
            return Err("tokenize produced no tokens".into());
        }
        self.reset();

        // prefill
        let taps_want = self.drafter.as_ref().map(|d| d.taps_want.clone());
        let mut taps: Vec<Vec<f32>> = match &taps_want {
            Some(w) => vec![Vec::new(); w.len()],
            None => Vec::new(),
        };
        let mut last_h = Vec::new();
        for (pos, &tok) in toks.iter().enumerate() {
            match &taps_want {
                Some(w) => {
                    let h = self.dec.forward_hidden_taps(tok, pos, w, &mut taps)?;
                    if let Some(d) = self.drafter.as_mut() {
                        d.observe(&taps, pos)?;
                    }
                    last_h = h;
                }
                None => last_h = self.dec.forward_hidden(tok, pos)?,
            }
        }

        let mut sampler = Sampler::new(sampler_cfg);
        let greedy = sampler_cfg.temp <= 0.0;
        let mut n_out = 0usize;
        let mut finish = "length".to_string();

        if let Some(drafter) = self.drafter.as_mut().filter(|_| greedy) {
            // speculative greedy path
            let mut logits = self.dec.head_logits(&last_h)?;
            let mut pending = sampler.sample(&logits, sampler_cfg) as u32;
            let mut n_past = toks.len();
            let mut n_draft = 4usize;
            if let Ok(v) = std::env::var("BONSAI_DSPARK_N") {
                if let Ok(n) = v.parse() {
                    n_draft = n;
                }
            }
            loop {
                if self.stop.contains(&pending) {
                    finish = "stop".into();
                    break;
                }
                let piece = self.vocab.piece_bytes(pending as i32);
                if !on_piece(&piece) {
                    finish = "stop".into();
                    break;
                }
                n_out += 1;
                if n_out >= max_tokens {
                    break;
                }
                let out = spec::round(&mut self.dec, drafter, pending, n_past, n_draft)?;
                n_past += out.accepted + 1;
                let last = out.emitted.len() - 1;
                let mut stop = false;
                for &t in &out.emitted[..last] {
                    if self.stop.contains(&t) {
                        stop = true;
                        break;
                    }
                    let piece = self.vocab.piece_bytes(t as i32);
                    if !on_piece(&piece) {
                        stop = true;
                        break;
                    }
                    n_out += 1;
                    if n_out >= max_tokens {
                        stop = true;
                        break;
                    }
                }
                pending = out.pending;
                if stop {
                    finish = "stop".into();
                    break;
                }
            }
            return Ok((toks.len(), n_out, finish));
        }

        // plain path
        let mut logits = self.dec.head_logits(&last_h)?;
        let mut n_past = toks.len();
        loop {
            let id = sampler.sample(&logits, sampler_cfg) as u32;
            if self.stop.contains(&id) {
                finish = "stop".into();
                break;
            }
            let piece = self.vocab.piece_bytes(id as i32);
            if !on_piece(&piece) {
                finish = "stop".into();
                break;
            }
            n_out += 1;
            if n_out >= max_tokens {
                break;
            }
            let h = self.dec.forward_hidden(id, n_past)?;
            n_past += 1;
            logits = self.dec.head_logits(&h)?;
        }
        Ok((toks.len(), n_out, finish))
    }
}

// ---------------------------------------------------------------------------
// HTTP
// ---------------------------------------------------------------------------

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn header<'a>(hdrs: &'a str, name: &str) -> Option<&'a str> {
    hdrs.lines().find_map(|l| {
        let (k, v) = l.split_once(':')?;
        if k.trim().eq_ignore_ascii_case(name) {
            Some(v.trim())
        } else {
            None
        }
    })
}

/// Read one request: (method, path, body).
fn read_request(stream: &mut TcpStream) -> Result<(String, String, Vec<u8>), String> {
    let mut reader = BufReader::new(stream.try_clone().map_err(|e| e.to_string())?);
    let mut line = String::new();
    reader.read_line(&mut line).map_err(|e| e.to_string())?;
    let mut parts = line.split_whitespace();
    let method = parts.next().unwrap_or("").to_string();
    let path = parts.next().unwrap_or("/").to_string();
    if method.is_empty() {
        return Err("empty request line".into());
    }
    let mut hdrs = String::new();
    loop {
        let mut l = String::new();
        if reader.read_line(&mut l).map_err(|e| e.to_string())? == 0 {
            break;
        }
        if l == "\r\n" || l == "\n" {
            break;
        }
        hdrs.push_str(&l);
    }
    let len: usize = header(&hdrs, "content-length")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    if header(&hdrs, "expect")
        .map(|v| v.eq_ignore_ascii_case("100-continue"))
        .unwrap_or(false)
    {
        stream.write_all(b"HTTP/1.1 100 Continue\r\n\r\n").ok();
    }
    let mut body = vec![0u8; len];
    if len > 0 {
        reader.read_exact(&mut body).map_err(|e| e.to_string())?;
    }
    Ok((method, path, body))
}

fn write_head(stream: &mut TcpStream, status: &str, ctype: &str, len: Option<usize>, chunked: bool) {
    let mut h = format!(
        "HTTP/1.1 {status}\r\nContent-Type: {ctype}\r\nAccess-Control-Allow-Origin: *\r\n\
         Access-Control-Allow-Headers: *\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
    );
    match (len, chunked) {
        (_, true) => h.push_str("Transfer-Encoding: chunked\r\n"),
        (Some(n), _) => h.push_str(&format!("Content-Length: {n}\r\n")),
        (None, _) => h.push_str("Connection: close\r\n"),
    }
    h.push_str("\r\n");
    stream.write_all(h.as_bytes()).ok();
}

fn respond_json(stream: &mut TcpStream, status: &str, body: &str) {
    write_head(stream, status, "application/json", Some(body.len()), false);
    stream.write_all(body.as_bytes()).ok();
    stream.flush().ok();
}

fn sse_chunk(stream: &mut TcpStream, data: &str) -> bool {
    let payload = format!("data: {data}\n\n");
    let ok = stream
        .write_all(format!("{:x}\r\n", payload.len()).as_bytes())
        .and_then(|_| stream.write_all(payload.as_bytes()))
        .and_then(|_| stream.write_all(b"\r\n"))
        .is_ok();
    stream.flush().ok();
    ok
}

fn sse_end(stream: &mut TcpStream) {
    stream.write_all(b"0\r\n\r\n").ok();
    stream.flush().ok();
}

fn models_json(model_id: &str) -> String {
    format!(
        "{{\"object\":\"list\",\"data\":[{{\"id\":\"{id}\",\"object\":\"model\",\"created\":0,\"owned_by\":\"local\"}}]}}",
        id = escape(model_id)
    )
}

fn err_json(msg: &str) -> String {
    format!("{{\"error\":{{\"message\":\"{}\",\"type\":\"invalid_request_error\"}}}}", escape(msg))
}

fn parse_chat(body: &[u8], default_model: &str) -> Result<(Vec<(String, String)>, ChatOpts), String> {
    let txt = std::str::from_utf8(body).map_err(|_| "body is not UTF-8".to_string())?;
    let j = J::parse(txt).map_err(|e| format!("bad JSON: {e}"))?;
    let msgs = j
        .get("messages")
        .and_then(|m| m.as_arr())
        .ok_or("missing 'messages' array")?;
    let mut out = Vec::with_capacity(msgs.len());
    for m in msgs {
        let role = m.get("role").and_then(|r| r.as_str()).unwrap_or("user");
        // content may be a string or an array of parts
        let content = match m.get("content") {
            Some(J::Str(s)) => s.clone(),
            Some(J::Arr(parts)) => parts
                .iter()
                .filter_map(|p| p.get("text").and_then(|t| t.as_str()).map(|s| s.to_string()))
                .collect::<Vec<_>>()
                .join(""),
            _ => String::new(),
        };
        out.push((role.to_string(), content));
    }
    let _ = default_model;
    Ok((out, ChatOpts::from(&j)))
}

struct ChatOpts {
    max_tokens: usize,
    temp: f32,
    top_k: i32,
    top_p: f32,
    min_p: f32,
    stream: bool,
}

impl From<&J> for ChatOpts {
    fn from(j: &J) -> ChatOpts {
        let max_tokens = j
            .get("max_tokens")
            .or_else(|| j.get("max_completion_tokens"))
            .and_then(|v| v.as_i64())
            .filter(|n| *n > 0)
            .unwrap_or(256) as usize;
        ChatOpts {
            max_tokens: max_tokens.min(8192),
            temp: j.get("temperature").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
            top_k: j
                .get("top_k")
                .and_then(|v| v.as_i64())
                .map(|v| v as i32)
                .unwrap_or(20),
            top_p: j.get("top_p").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32,
            min_p: j.get("min_p").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
            stream: j.get("stream").and_then(|v| v.as_bool()).unwrap_or(false),
        }
    }
}

fn sampler_cfg(o: &ChatOpts) -> SamplerConfig {
    SamplerConfig {
        top_k: o.top_k,
        top_p: o.top_p,
        min_p: o.min_p,
        temp: o.temp,
        seed: 0xFFFF_FFFF,
    }
}

fn handle(stream: &mut TcpStream, engine: &Arc<Mutex<Engine>>) {
    let (method, path, body) = match read_request(stream) {
        Ok(r) => r,
        Err(_) => return,
    };
    let path_only = path.split('?').next().unwrap_or("/").to_string();

    if method == "OPTIONS" {
        write_head(stream, "204 No Content", "text/plain", Some(0), false);
        stream.flush().ok();
        return;
    }

    match (method.as_str(), path_only.as_str()) {
        ("GET", "/health") | ("GET", "/v1/health") => {
            respond_json(stream, "200 OK", "{\"status\":\"ok\"}");
        }
        ("GET", "/v1/models") => {
            let id = engine.lock().map(|e| e.model_id.clone()).unwrap_or_default();
            respond_json(stream, "200 OK", &models_json(&id));
        }
        ("POST", "/v1/chat/completions") => {
            let (msgs, opts) = {
                let e = match engine.lock() {
                    Ok(e) => e,
                    Err(_) => return,
                };
                match parse_chat(&body, &e.model_id) {
                    Ok(v) => v,
                    Err(err) => {
                        drop(e);
                        respond_json(stream, "400 Bad Request", &err_json(&err));
                        return;
                    }
                }
            };
            let prompt = match engine.lock() {
                Ok(e) => e.build_prompt(&msgs),
                Err(_) => return,
            };
            run_completion(stream, engine, &prompt, &opts, true);
        }
        ("POST", "/v1/completions") => {
            let txt = std::str::from_utf8(&body).unwrap_or("");
            let j = J::parse(txt).unwrap_or(J::Null);
            let prompt = j
                .get("prompt")
                .and_then(|p| match p {
                    J::Str(s) => Some(s.clone()),
                    J::Arr(a) => a.first().and_then(|x| x.as_str()).map(|s| s.to_string()),
                    _ => None,
                })
                .unwrap_or_default();
            let opts = ChatOpts::from(&j);
            run_completion(stream, engine, &prompt, &opts, false);
        }
        _ => respond_json(stream, "404 Not Found", &err_json("unknown endpoint")),
    }
}

fn run_completion(
    stream: &mut TcpStream,
    engine: &Arc<Mutex<Engine>>,
    prompt: &str,
    opts: &ChatOpts,
    chat: bool,
) {
    let cfg = sampler_cfg(opts);
    let model_id = engine.lock().map(|e| e.model_id.clone()).unwrap_or_default();
    let id = format!("chatcmpl-{}", now_secs());
    let created = now_secs();

    // The engine is single-stream: hold the lock for the whole generation.
    let mut e = match engine.lock() {
        Ok(e) => e,
        Err(_) => return,
    };

    if opts.stream {
        write_head(stream, "200 OK", "text/event-stream", None, true);
        let id2 = id.clone();
        let model2 = model_id.clone();
        let mut buf: Vec<u8> = Vec::new();
        let mut pending_bytes: Vec<u8> = Vec::new();
        let strip = e.strip_think;
        let mut filt = ThinkFilter::new(strip);
        // Buffer pieces so multi-byte UTF-8 is not split across SSE frames.
        let res = e.generate(prompt, opts.max_tokens, &cfg, &mut |piece| {
            buf.extend_from_slice(piece);
            // emit complete UTF-8 prefix, keep any trailing partial sequence
            let valid = match std::str::from_utf8(&buf) {
                Ok(_) => buf.len(),
                Err(er) => er.valid_up_to(),
            };
            if valid == 0 {
                return true;
            }
            let text = String::from_utf8_lossy(&buf[..valid]).to_string();
            pending_bytes.extend_from_slice(&buf[..valid]);
            buf.drain(..valid);
            let text = match filt.feed(&text) {
                Some(t) => t,
                None => return true,
            };
            if text.is_empty() {
                return true;
            }
            let delta = if chat {
                format!(
                    "{{\"id\":\"{id2}\",\"object\":\"chat.completion.chunk\",\"created\":{created},\
                     \"model\":\"{}\",\"choices\":[{{\"index\":0,\"delta\":{{\"content\":\"{}\"}},\
                     \"finish_reason\":null}}]}}",
                    escape(&model2),
                    escape(&text)
                )
            } else {
                format!(
                    "{{\"id\":\"{id2}\",\"object\":\"text_completion\",\"created\":{created},\
                     \"model\":\"{}\",\"choices\":[{{\"index\":0,\"text\":\"{}\",\"finish_reason\":null}}]}}",
                    escape(&model2),
                    escape(&text)
                )
            };
            sse_chunk(stream, &delta)
        });
        if let Some(tail) = filt.flush() {
            if !tail.is_empty() {
                let d = if chat {
                    format!(
                        "{{\"id\":\"{id2}\",\"object\":\"chat.completion.chunk\",\"created\":{created},\
                         \"model\":\"{}\",\"choices\":[{{\"index\":0,\"delta\":{{\"content\":\"{}\"}},\
                         \"finish_reason\":null}}]}}",
                        escape(&model2),
                        escape(&tail)
                    )
                } else {
                    format!(
                        "{{\"id\":\"{id2}\",\"object\":\"text_completion\",\"created\":{created},\
                         \"model\":\"{}\",\"choices\":[{{\"index\":0,\"text\":\"{}\",\"finish_reason\":null}}]}}",
                        escape(&model2),
                        escape(&tail)
                    )
                };
                sse_chunk(stream, &d);
            }
        }
        let (pt, ct, finish) = res.unwrap_or((0, 0, "error".into()));
        let last = if chat {
            format!(
                "{{\"id\":\"{id2}\",\"object\":\"chat.completion.chunk\",\"created\":{created},\
                 \"model\":\"{}\",\"choices\":[{{\"index\":0,\"delta\":{{}},\"finish_reason\":\"{finish}\"}}],\
                 \"usage\":{{\"prompt_tokens\":{pt},\"completion_tokens\":{ct},\"total_tokens\":{}}}}}",
                escape(&model2),
                pt + ct
            )
        } else {
            format!(
                "{{\"id\":\"{id2}\",\"object\":\"text_completion\",\"created\":{created},\
                 \"model\":\"{}\",\"choices\":[{{\"index\":0,\"text\":\"\",\"finish_reason\":\"{finish}\"}}]}}",
                escape(&model2)
            )
        };
        sse_chunk(stream, &last);
        sse_chunk(stream, "[DONE]");
        sse_end(stream);
        return;
    }

    // non-streaming
    let mut text = Vec::new();
    let res = e.generate(prompt, opts.max_tokens, &cfg, &mut |piece| {
        text.extend_from_slice(piece);
        true
    });
    let body = match res {
        Ok((pt, ct, finish)) => {
            let raw = String::from_utf8_lossy(&text).to_string();
            let content = if e.strip_think {
                match raw.find("</think>") {
                    Some(i) => raw[i + "</think>".len()..].trim_start().to_string(),
                    None => raw,
                }
            } else {
                raw
            };
            if chat {
                format!(
                    "{{\"id\":\"{id}\",\"object\":\"chat.completion\",\"created\":{created},\
                     \"model\":\"{}\",\"choices\":[{{\"index\":0,\"message\":{{\"role\":\"assistant\",\"content\":\"{}\"}},\
                     \"finish_reason\":\"{finish}\"}}],\
                     \"usage\":{{\"prompt_tokens\":{pt},\"completion_tokens\":{ct},\"total_tokens\":{}}}}}",
                    escape(&model_id),
                    escape(&content),
                    pt + ct
                )
            } else {
                format!(
                    "{{\"id\":\"{id}\",\"object\":\"text_completion\",\"created\":{created},\
                     \"model\":\"{}\",\"choices\":[{{\"index\":0,\"text\":\"{}\",\"finish_reason\":\"{finish}\"}}],\
                     \"usage\":{{\"prompt_tokens\":{pt},\"completion_tokens\":{ct},\"total_tokens\":{}}}}}",
                    escape(&model_id),
                    escape(&content),
                    pt + ct
                )
            }
        }
        Err(err) => err_json(&err),
    };
    respond_json(stream, "200 OK", &body);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn think_filter_strips_leading_reasoning() {
        let mut f = ThinkFilter::new(true);
        assert_eq!(f.feed("<think>\nreasoning"), None);
        assert_eq!(f.feed("\n</think>\nParis"), Some("\nParis".to_string()));
        assert_eq!(f.feed(" is nice"), Some(" is nice".to_string()));
    }

    #[test]
    fn think_filter_passes_plain_text() {
        // no leading <think: stream it once a few bytes have arrived
        let mut f = ThinkFilter::new(true);
        assert_eq!(f.feed("hello world"), Some("hello world".to_string()));
    }

    #[test]
    fn think_filter_flushes_unclosed_block() {
        let mut f = ThinkFilter::new(true);
        assert_eq!(f.feed("<think>still thinking"), None);
        assert_eq!(f.flush(), Some("<think>still thinking".to_string()));
        assert_eq!(f.flush(), None);
    }
}

/// Serve until the process is killed.
pub fn serve(addr: &str, engine: Engine) -> Result<(), String> {
    let listener = TcpListener::bind(addr).map_err(|e| format!("bind {addr}: {e}"))?;
    let engine = Arc::new(Mutex::new(engine));
    eprintln!("[server] listening on http://{addr} (OpenAI-compatible)");
    for conn in listener.incoming() {
        match conn {
            Ok(mut stream) => {
                let engine = Arc::clone(&engine);
                std::thread::spawn(move || {
                    handle(&mut stream, &engine);
                });
            }
            Err(e) => eprintln!("[server] accept: {e}"),
        }
    }
    Ok(())
}
