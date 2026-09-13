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
    /// split the `<think>...</think>` block into `reasoning_content` instead of
    /// inlining it in `content`
    pub split_reasoning: bool,
}

/// The exact tool preamble from the model's `tokenizer.chat_template`.
pub const TOOL_INSTRUCTIONS: &str = r#"If you choose to call a function ONLY reply in the following format with NO suffix:

<tool_call>
<function=example_function_name>
<parameter=example_parameter_1>
value_1
</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>
</tool_call>

<IMPORTANT>
Reminder:
- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags
- Required parameters MUST be specified
- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after
- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls
</IMPORTANT>"#;

/// One chat message as the template consumes it.
#[derive(Debug, Clone, Default)]
pub struct Msg {
    pub role: String,
    pub content: String,
    /// assistant tool calls: (function name, arguments as a JSON object string)
    pub tool_calls: Vec<(String, String)>,
}

/// Split an assistant message into (reasoning, answer) the way the template does.
fn split_think(content: &str) -> (String, String) {
    match content.split_once("</think>") {
        Some((before, after)) => (
            before.split("<think>").last().unwrap_or("").trim().to_string(),
            after.trim_start_matches('\n').to_string(),
        ),
        None => (String::new(), content.to_string()),
    }
}

/// Splits a reply into (reasoning, content) at `</think>`, streaming both.
/// The model emits a leading `<think>...</think>` block; agents get the answer
/// in `content` and the trace in `reasoning_content` (the DeepSeek/Qwen
/// convention). `keep` disables the split and passes everything as content.
struct ThinkSplitter {
    buf: String,
    /// 0 = undecided, 1 = inside the reasoning block, 2 = content
    phase: u8,
    keep: bool,
}

impl ThinkSplitter {
    fn new(keep: bool) -> ThinkSplitter {
        ThinkSplitter { buf: String::new(), phase: if keep { 2 } else { 0 }, keep }
    }

    fn feed(&mut self, text: &str) -> (String, String) {
        if self.keep {
            return (String::new(), text.to_string());
        }
        self.buf.push_str(text);
        let mut reason = String::new();
        let mut content = String::new();
        loop {
            match self.phase {
                2 => {
                    content.push_str(&self.buf);
                    self.buf.clear();
                    break;
                }
                0 => {
                    let t = self.buf.trim_start();
                    if t.len() < "<think".len() && "<think".starts_with(t) {
                        break; // need more bytes to decide
                    }
                    if t.starts_with("<think") {
                        self.buf = t["<think>".len()..].trim_start_matches('\n').to_string();
                        self.phase = 1;
                    } else {
                        self.phase = 2;
                    }
                }
                _ => {
                    if let Some(i) = self.buf.find("</think>") {
                        reason.push_str(&self.buf[..i]);
                        self.buf = self.buf[i + "</think>".len()..]
                            .trim_start_matches('\n')
                            .to_string();
                        self.phase = 2;
                        continue;
                    }
                    // hold back a possible split closing tag
                    let keep = "</think>".len() - 1;
                    if self.buf.len() > keep {
                        let cut = self.buf.len() - keep;
                        reason.push_str(&self.buf[..cut]);
                        self.buf = self.buf[cut..].to_string();
                    }
                    break;
                }
            }
        }
        (reason, content)
    }

    /// Whatever is left (an unterminated reasoning block stays reasoning).
    fn flush(&mut self) -> (String, String) {
        let s = std::mem::take(&mut self.buf);
        if self.keep {
            return (String::new(), s);
        }
        if self.phase == 1 {
            (s, String::new())
        } else {
            (String::new(), s)
        }
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

    /// Render the conversation with the model's own chat template, including
    /// the tool definitions and assistant tool-call blocks.
    pub fn build_prompt(&self, msgs: &[Msg], tools: &[String]) -> String {
        let mut out = String::new();
        let system = msgs
            .first()
            .filter(|m| m.role == "system")
            .map(|m| m.content.trim().to_string())
            .filter(|c| !c.is_empty());

        if !tools.is_empty() {
            out.push_str("<|im_start|>system\n");
            out.push_str("# Tools\n\nYou have access to the following functions:\n\n<tools>");
            for t in tools {
                out.push('\n');
                out.push_str(t);
            }
            out.push_str("\n</tools>\n\n");
            out.push_str(TOOL_INSTRUCTIONS);
            if let Some(c) = &system {
                out.push_str("\n\n");
                out.push_str(c);
            }
            out.push_str("<|im_end|>\n");
        } else if let Some(c) = &system {
            out.push_str("<|im_start|>system\n");
            out.push_str(c);
            out.push_str("<|im_end|>\n");
        }

        for m in msgs {
            match m.role.as_str() {
                // emitted in the preamble above
                "system" => {}
                "user" => {
                    out.push_str("<|im_start|>user\n");
                    out.push_str(m.content.trim());
                    out.push_str("<|im_end|>\n");
                }
                "assistant" => {
                    let (reasoning, content) = split_think(&m.content);
                    out.push_str("<|im_start|>assistant\n");
                    if self.think && !reasoning.is_empty() {
                        out.push_str("<think>\n");
                        out.push_str(&reasoning);
                        out.push_str("\n</think>\n\n");
                    }
                    out.push_str(&content);
                    for (i, (name, args)) in m.tool_calls.iter().enumerate() {
                        if i == 0 {
                            if content.trim().is_empty() {
                                out.push_str("<tool_call>\n");
                            } else {
                                out.push_str("\n\n<tool_call>\n");
                            }
                        } else {
                            out.push_str("\n<tool_call>\n");
                        }
                        out.push_str("<function=");
                        out.push_str(name);
                        out.push_str(">\n");
                        if let Ok(J::Obj(pairs)) = J::parse(args) {
                            for (k, v) in pairs {
                                out.push_str("<parameter=");
                                out.push_str(&k);
                                out.push_str(">\n");
                                match &v {
                                    J::Str(x) => out.push_str(x),
                                    other => out.push_str(&crate::json::to_json(other)),
                                }
                                out.push_str("\n</parameter>\n");
                            }
                        }
                        out.push_str("</function>\n</tool_call>");
                    }
                    out.push_str("<|im_end|>\n");
                }
                "tool" => {
                    out.push_str("<|im_start|>user\n<tool_response>\n");
                    out.push_str(m.content.trim());
                    out.push_str("\n</tool_response><|im_end|>\n");
                }
                _ => {
                    out.push_str("<|im_start|>user\n");
                    out.push_str(&m.content);
                    out.push_str("<|im_end|>\n");
                }
            }
        }
        out.push_str("<|im_start|>assistant\n");
        // the template closes an empty reasoning block when thinking is off
        out.push_str(if self.think {
            "<think>\n"
        } else {
            "<think>\n\n</think>\n\n"
        });
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

fn parse_chat(body: &[u8]) -> Result<(Vec<Msg>, Vec<String>, ChatOpts), String> {
    let txt = std::str::from_utf8(body).map_err(|_| "body is not UTF-8".to_string())?;
    let j = J::parse(txt).map_err(|e| format!("bad JSON: {e}"))?;
    let msgs = j
        .get("messages")
        .and_then(|m| m.as_arr())
        .ok_or("missing 'messages' array")?;
    let mut out = Vec::with_capacity(msgs.len());
    for m in msgs {
        let role = m.get("role").and_then(|r| r.as_str()).unwrap_or("user").to_string();
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
        // assistant tool calls carried back by the client
        let mut tool_calls = Vec::new();
        if let Some(tc) = m.get("tool_calls").and_then(|t| t.as_arr()) {
            for c in tc {
                let f = c.get("function").unwrap_or(c);
                let name = f.get("name").and_then(|n| n.as_str()).unwrap_or("");
                if name.is_empty() {
                    continue;
                }
                let args = match f.get("arguments") {
                    Some(J::Str(s)) => s.clone(),
                    Some(other) => crate::json::to_json(other),
                    None => "{}".to_string(),
                };
                tool_calls.push((name.to_string(), args));
            }
        }
        out.push(Msg { role, content, tool_calls });
    }
    let mut tools: Vec<String> = j
        .get("tools")
        .and_then(|t| t.as_arr())
        .map(|a| a.iter().map(crate::json::to_json).collect())
        .unwrap_or_default();
    if matches!(j.get("tool_choice").and_then(|v| v.as_str()), Some("none")) {
        tools.clear();
    }
    let _ = out.first();
    Ok((out, tools, ChatOpts::from(&j)))
}

/// Pull `<tool_call><function=NAME><parameter=K>V</parameter></function></tool_call>`
/// blocks out of a reply: returns (remaining content, calls).
pub fn parse_tool_calls(text: &str) -> (String, Vec<(String, String)>) {
    const OPEN: &str = "<tool_call>";
    const CLOSE: &str = "</tool_call>";
    let mut calls = Vec::new();
    let mut content = String::new();
    let mut rest = text;
    while let Some(i) = rest.find(OPEN) {
        content.push_str(&rest[..i]);
        let after = &rest[i + OPEN.len()..];
        match after.find(CLOSE) {
            Some(j) => {
                if let Some(c) = parse_tool_block(&after[..j]) {
                    calls.push(c);
                }
                rest = &after[j + CLOSE.len()..];
            }
            None => {
                rest = "";
                break;
            }
        }
    }
    content.push_str(rest);
    (content.trim().to_string(), calls)
}

/// Parse the inside of a `<tool_call>` block into (name, arguments JSON object).
fn parse_tool_block(block: &str) -> Option<(String, String)> {
    let fs = block.find("<function=")? + "<function=".len();
    let fe = block[fs..].find('>')? + fs;
    let name = block[fs..fe].trim().to_string();
    let body_end = block.find("</function>").unwrap_or(block.len());
    let body = block.get(fe + 1..body_end).unwrap_or("");
    let mut pairs: Vec<(String, J)> = Vec::new();
    let mut r = body;
    while let Some(i) = r.find("<parameter=") {
        let p0 = i + "<parameter=".len();
        let Some(gt) = r[p0..].find('>') else { break };
        let pname = r[p0..p0 + gt].trim().to_string();
        let vstart = p0 + gt + 1;
        let Some(vrel) = r[vstart..].find("</parameter>") else { break };
        let val = r[vstart..vstart + vrel].trim_matches('\n').to_string();
        // the template writes string values raw and everything else as JSON
        let v = J::parse(&val).unwrap_or(J::Str(val));
        pairs.push((pname, v));
        r = &r[vstart + vrel + "</parameter>".len()..];
    }
    Some((name, crate::json::to_json(&J::Obj(pairs))))
}

/// Streaming counterpart: holds back `<tool_call>` blocks (and a possible
/// partial opening tag) so they never leak into `content`.
struct ToolFilter {
    buf: String,
    calls: Vec<(String, String)>,
}

impl ToolFilter {
    fn new() -> ToolFilter {
        ToolFilter { buf: String::new(), calls: Vec::new() }
    }

    fn feed(&mut self, text: &str) -> String {
        const OPEN: &str = "<tool_call>";
        const CLOSE: &str = "</tool_call>";
        self.buf.push_str(text);
        let mut out = String::new();
        loop {
            if let Some(i) = self.buf.find(OPEN) {
                if let Some(jrel) = self.buf[i..].find(CLOSE) {
                    let end = i + jrel + CLOSE.len();
                    let block = self.buf[i + OPEN.len()..i + jrel].to_string();
                    if let Some(c) = parse_tool_block(&block) {
                        self.calls.push(c);
                    }
                    out.push_str(&self.buf[..i]);
                    self.buf = self.buf[end..].to_string();
                    continue;
                }
                out.push_str(&self.buf[..i]);
                self.buf = self.buf[i..].to_string();
                return out;
            }
            // hold back a possible split opening tag
            let keep = OPEN.len() - 1;
            if self.buf.len() > keep {
                let cut = self.buf.len() - keep;
                out.push_str(&self.buf[..cut]);
                self.buf = self.buf[cut..].to_string();
            }
            return out;
        }
    }

    fn flush(&mut self) -> String {
        std::mem::take(&mut self.buf)
    }
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
            let (msgs, tools, opts) = match parse_chat(&body) {
                Ok(v) => v,
                Err(err) => {
                    respond_json(stream, "400 Bad Request", &err_json(&err));
                    return;
                }
            };
            let prompt = match engine.lock() {
                Ok(e) => e.build_prompt(&msgs, &tools),
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
        let split = e.split_reasoning;
        let mut filt = ThinkSplitter::new(!split);
        let mut tfilt = ToolFilter::new();
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
            let (reason, content) = filt.feed(&text);
            let content = tfilt.feed(&content);
            let mut ok = true;
            if !reason.is_empty() && chat {
                let d = format!(
                    "{{\"id\":\"{id2}\",\"object\":\"chat.completion.chunk\",\"created\":{created},\
                     \"model\":\"{}\",\"choices\":[{{\"index\":0,\"delta\":{{\"reasoning_content\":\"{}\"}},\
                     \"finish_reason\":null}}]}}",
                    escape(&model2),
                    escape(&reason)
                );
                ok = sse_chunk(stream, &d);
            }
            if content.is_empty() {
                return ok;
            }
            let delta = if chat {
                format!(
                    "{{\"id\":\"{id2}\",\"object\":\"chat.completion.chunk\",\"created\":{created},\
                     \"model\":\"{}\",\"choices\":[{{\"index\":0,\"delta\":{{\"content\":\"{}\"}},\
                     \"finish_reason\":null}}]}}",
                    escape(&model2),
                    escape(&content)
                )
            } else {
                format!(
                    "{{\"id\":\"{id2}\",\"object\":\"text_completion\",\"created\":{created},\
                     \"model\":\"{}\",\"choices\":[{{\"index\":0,\"text\":\"{}\",\"finish_reason\":null}}]}}",
                    escape(&model2),
                    escape(&content)
                )
            };
            sse_chunk(stream, &delta)
        });
        let mut tail = filt.flush().1;
        tail.push_str(&tfilt.flush());
        {
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
        let (pt, ct, finish0) = res.unwrap_or((0, 0, "error".into()));
        if !tfilt.calls.is_empty() && chat {
            let tc: Vec<String> = tfilt
                .calls
                .iter()
                .enumerate()
                .map(|(i, (n, a))| {
                    format!(
                        "{{\"index\":{i},\"id\":\"call_{i}\",\"type\":\"function\",\
                         \"function\":{{\"name\":\"{}\",\"arguments\":\"{}\"}}}}",
                        escape(n),
                        escape(a)
                    )
                })
                .collect();
            let d = format!(
                "{{\"id\":\"{id2}\",\"object\":\"chat.completion.chunk\",\"created\":{created},\
                 \"model\":\"{}\",\"choices\":[{{\"index\":0,\"delta\":{{\"tool_calls\":[{tc}]}},\
                 \"finish_reason\":null}}]}}",
                escape(&model2),
                tc = tc.join(",")
            );
            sse_chunk(stream, &d);
        }
        let finish = if !tfilt.calls.is_empty() { "tool_calls".to_string() } else { finish0 };
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
        Ok((pt, ct, finish0)) => {
            let raw = String::from_utf8_lossy(&text).to_string();
            let (reasoning, body_text) = {
                let mut sp = ThinkSplitter::new(!e.split_reasoning);
                let (mut r, mut c) = sp.feed(&raw);
                let (r2, c2) = sp.flush();
                r.push_str(&r2);
                c.push_str(&c2);
                (r.trim().to_string(), c)
            };
            let (content, calls) = parse_tool_calls(&body_text);
            let content = content.trim_start().to_string();
            let reasoning_field = if reasoning.is_empty() {
                String::new()
            } else {
                format!("\"reasoning_content\":\"{}\",", escape(&reasoning))
            };
            let finish = if calls.is_empty() { finish0 } else { "tool_calls".to_string() };
            if !calls.is_empty() && chat {
                let tc: Vec<String> = calls
                    .iter()
                    .enumerate()
                    .map(|(i, (n, a))| {
                        format!(
                            "{{\"id\":\"call_{i}\",\"type\":\"function\",\
                             \"function\":{{\"name\":\"{}\",\"arguments\":\"{}\"}}}}",
                            escape(n),
                            escape(a)
                        )
                    })
                    .collect();
                format!(
                    "{{\"id\":\"{id}\",\"object\":\"chat.completion\",\"created\":{created},\
                     \"model\":\"{}\",\"choices\":[{{\"index\":0,\"message\":{{\"role\":\"assistant\",\
                     {}\"content\":\"{}\",\"tool_calls\":[{}]}},\"finish_reason\":\"tool_calls\"}}],\
                     \"usage\":{{\"prompt_tokens\":{pt},\"completion_tokens\":{ct},\"total_tokens\":{}}}}}",
                    escape(&model_id),
                    reasoning_field,
                    escape(&content),
                    tc.join(","),
                    pt + ct
                )
            } else if chat {
                format!(
                    "{{\"id\":\"{id}\",\"object\":\"chat.completion\",\"created\":{created},\
                     \"model\":\"{}\",\"choices\":[{{\"index\":0,\"message\":{{\"role\":\"assistant\",\
                     {}\"content\":\"{}\"}},\"finish_reason\":\"{finish}\"}}],\
                     \"usage\":{{\"prompt_tokens\":{pt},\"completion_tokens\":{ct},\"total_tokens\":{}}}}}",
                    escape(&model_id),
                    reasoning_field,
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
    fn parses_tool_call_blocks() {
        let text = "Let me check.\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>";
        let (content, calls) = parse_tool_calls(text);
        assert_eq!(content, "Let me check.");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].0, "get_weather");
        assert_eq!(calls[0].1, r#"{"city":"Paris","days":3}"#);
    }

    #[test]
    fn parses_two_tool_calls() {
        let text = "<tool_call>\n<function=a>\n</function>\n</tool_call><tool_call>\n<function=b>\n<parameter=x>\nhi\n</parameter>\n</function>\n</tool_call>";
        let (content, calls) = parse_tool_calls(text);
        assert_eq!(content, "");
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].0, "a");
        assert_eq!(calls[0].1, "{}");
        assert_eq!(calls[1], ("b".to_string(), r#"{"x":"hi"}"#.to_string()));
    }

    #[test]
    fn tool_filter_holds_back_a_split_tag() {
        let mut f = ToolFilter::new();
        // the filter always holds back the last 10 bytes (a split "<tool_call>")
        let out = f.feed("Hello there, how are");
        assert_eq!(out, "Hello ther");
        let out2 = f.feed(" you<tool_call>\n<function=f>\n<parameter=a>\n1\n</parameter>\n</function>\n</tool_call>");
        assert_eq!(out2, "e, how are you");
        assert_eq!(f.calls.len(), 1);
        assert_eq!(f.calls[0].0, "f");
        assert_eq!(f.calls[0].1, r#"{"a":1}"#);
    }

    #[test]
    fn split_think_separates_reasoning() {
        let (r, c) = split_think("<think>\nreason\n</think>\n\nParis");
        assert_eq!(r, "reason");
        assert_eq!(c, "Paris");
        let (r2, c2) = split_think("plain");
        assert_eq!(r2, "");
        assert_eq!(c2, "plain");
    }

    #[test]
    fn splitter_separates_reasoning_from_content() {
        let mut f = ThinkSplitter::new(false);
        let (mut reason, mut content) = f.feed("<think>\nreasoning...");
        assert!(content.is_empty());
        let (r2, c2) = f.feed("\n</think>\n\nParis");
        reason.push_str(&r2);
        content.push_str(&c2);
        assert_eq!(reason, "reasoning...\n");
        assert_eq!(content, "Paris");
    }

    #[test]
    fn splitter_passes_plain_text_as_content() {
        let mut f = ThinkSplitter::new(false);
        let (r, c) = f.feed("hello world");
        assert!(r.is_empty());
        assert_eq!(c, "hello world");
    }

    #[test]
    fn splitter_keeps_an_unterminated_block_as_reasoning() {
        let mut f = ThinkSplitter::new(false);
        let (mut reason, content) = f.feed("<think>still thinking");
        assert!(content.is_empty());
        let (r2, c2) = f.flush();
        reason.push_str(&r2);
        assert_eq!(reason, "still thinking");
        assert!(c2.is_empty());
    }

    #[test]
    fn splitter_disabled_inlines_everything() {
        let mut f = ThinkSplitter::new(true);
        let (r, c) = f.feed("<think>x</think>y");
        assert!(r.is_empty());
        assert_eq!(c, "<think>x</think>y");
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
