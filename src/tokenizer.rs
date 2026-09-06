//! Pure-Rust qwen35 BPE tokenizer (M3), a faithful port of llama.cpp's
//! `llm_tokenizer_bpe_session` path for the `qwen35` pre-tokenizer.
//!
//! Pipeline (mirrors llama.cpp):
//!   1. split special-token fragments (tokenizer_st_partition)
//!   2. per text fragment: split into words with the qwen35 codepoint scanner
//!   3. GPT-2 byte-encode each word (spaces -> U+0120 etc.)
//!   4. apply BPE merges with a (rank, left) min-priority queue
//!   5. map final symbols to ids, with single-byte fallback
//!
//! Text/category handling uses Rust std + unicode-general-category, which
//! should match llama.cpp's generated unicode tables for the categories the
//! qwen35 scanner uses (L, M, N, whitespace, punctuation/symbol presence).

#![allow(dead_code)]

use std::collections::HashMap;
use unicode_general_category::{get_general_category, GeneralCategory};

use crate::gguf::{GGUF, Value};

// ---------------------------------------------------------------------------
// vocabulary loaded from GGUF metadata
// ---------------------------------------------------------------------------
pub struct Vocab {
    pub tokens: Vec<String>,
    text_to_id: HashMap<String, i32>,
    /// tokenizer.ggml.token_type per id (0 unknown, 3 control, 4 user-defined)
    token_types: Vec<i32>,
    merges: HashMap<(String, String), usize>,
    byte_tokens: HashMap<u8, i32>, // single-raw-byte fallback tokens
    special_ids: Vec<i32>,         // control + user-defined, in id order
}

impl Vocab {
    pub fn from_gguf(g: &GGUF) -> Result<Vocab, String> {
        let tokens = match g.get("tokenizer.ggml.tokens") {
            Some(Value::Array { items, .. }) => {
                let mut v = Vec::with_capacity(items.len());
                for it in items {
                    match it {
                        Value::Str(s) => v.push(s.clone()),
                        _ => return Err("tokens array not strings".into()),
                    }
                }
                v
            }
            _ => return Err("missing tokenizer.ggml.tokens".into()),
        };

        let token_types: Vec<i32> = match g.get("tokenizer.ggml.token_type") {
            Some(Value::Array { items, .. }) => {
                let mut v = Vec::with_capacity(items.len());
                for it in items {
                    match it {
                        Value::I32(x) => v.push(*x),
                        Value::U32(x) => v.push(*x as i32),
                        _ => return Err("token_type array not i32".into()),
                    }
                }
                v
            }
            _ => Vec::new(),
        };

        let merges: HashMap<(String, String), usize> = match g.get("tokenizer.ggml.merges") {
            Some(Value::Array { items, .. }) => {
                let mut m = HashMap::new();
                for (i, it) in items.iter().enumerate() {
                    let s = match it {
                        Value::Str(s) => s,
                        _ => return Err("merges array not strings".into()),
                    };
                    // llama.cpp: word.find(' ', 1) over raw bytes
                    let bytes = s.as_bytes();
                    let rel = bytes[1..].iter().position(|&b| b == b' ');
                    let Some(rel) = rel else {
                        return Err("merge without separator".into());
                    };
                    let pos = 1 + rel;
                    let first = s[..pos].to_string();
                    let second = s[pos + 1..].to_string();
                    m.insert((first, second), i);
                }
                m
            }
            _ => return Err("missing tokenizer.ggml.merges".into()),
        };

        let mut text_to_id = HashMap::with_capacity(tokens.len());
        let mut byte_tokens = HashMap::new();
        let mut special_ids = Vec::new();
        for (i, t) in tokens.iter().enumerate() {
            text_to_id.insert(t.clone(), i as i32);
            let tt = token_types.get(i).copied().unwrap_or(0);
            if t.len() == 1 {
                byte_tokens.insert(t.as_bytes()[0], i as i32);
            }
            if tt == 3 || tt == 4 {
                special_ids.push(i as i32);
            }
        }

        Ok(Vocab { tokens, text_to_id, token_types, merges, byte_tokens, special_ids })
    }

    fn token(&self, s: &str) -> Option<i32> {
        self.text_to_id.get(s).copied()
    }

    fn find_bpe_rank(&self, left: &str, right: &str) -> Option<usize> {
        self.merges.get(&(left.to_string(), right.to_string())).copied()
    }

    /// Rendered bytes for one token id. Control tokens render empty when
    /// special tokens are not requested (mirrors `llama_token_to_piece(..., false)`).
    pub fn piece_bytes(&self, id: i32) -> Vec<u8> {
        if id < 0 || id as usize >= self.tokens.len() {
            return Vec::new();
        }
        let tt = self.token_types.get(id as usize).copied().unwrap_or(0);
        if tt == 3 {
            return Vec::new(); // control
        }
        decode_bytes(&self.tokens[id as usize])
    }
}

// ---------------------------------------------------------------------------
// GPT-2 byte encoding (matches unicode_byte_to_utf8_map)
// ---------------------------------------------------------------------------
fn byte_exempt(b: u32) -> bool {
    (0x21..=0x7E).contains(&b) || (0xA1..=0xAC).contains(&b) || (0xAE..=0xFF).contains(&b)
}

fn byte_to_char(b: u8) -> char {
    if byte_exempt(b as u32) {
        char::from_u32(b as u32).unwrap()
    } else {
        let n = (0..b).filter(|&x| !byte_exempt(x as u32)).count() as u32;
        char::from_u32(256 + n).unwrap()
    }
}

fn byte_encode_word(word: &str) -> String {
    word.as_bytes().iter().map(|&b| byte_to_char(b)).collect()
}

// inverse GPT-2 byte map: byte-encoded char -> original byte
fn byte_of_char(c: char) -> Option<u8> {
    static MAP: std::sync::OnceLock<HashMap<char, u8>> = std::sync::OnceLock::new();
    let map = MAP.get_or_init(|| {
        (0..=255u8).map(|b| (byte_to_char(b), b)).collect()
    });
    map.get(&c).copied()
}

/// Decode one byte-encoded piece into raw output bytes (mirrors llama.cpp
/// detokenization: each codepoint in the GPT-2 byte map becomes its byte;
/// anything else is kept as UTF-8).
pub fn decode_bytes(text: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(text.len());
    for c in text.chars() {
        match byte_of_char(c) {
            Some(b) => out.push(b),
            None => {
                let mut buf = [0u8; 4];
                out.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// codepoint flags
// ---------------------------------------------------------------------------
#[derive(Clone, Copy, Default)]
struct Flags {
    whitespace: bool,
    letter: bool,
    mark: bool,
    number: bool,
    punct: bool,
    symbol: bool,
}

impl Flags {
    fn any(&self) -> bool {
        self.whitespace || self.letter || self.mark || self.number || self.punct || self.symbol
    }
    fn any_of_main(&self) -> bool {
        self.whitespace || self.letter || self.mark || self.number
    }
}

fn flags_of(c: u32) -> Flags {
    let Some(ch) = char::from_u32(c) else {
        return Flags::default();
    };
    let gc = get_general_category(ch);
    use GeneralCategory::*;
    Flags {
        whitespace: ch.is_whitespace(),
        letter: ch.is_alphabetic(),
        mark: matches!(gc, NonspacingMark | SpacingMark | EnclosingMark),
        number: ch.is_numeric(),
        punct: matches!(
            gc,
            ConnectorPunctuation
                | DashPunctuation
                | ClosePunctuation
                | FinalPunctuation
                | InitialPunctuation
                | OtherPunctuation
                | OpenPunctuation
        ),
        symbol: matches!(
            gc,
            MathSymbol | CurrencySymbol | ModifierSymbol | OtherSymbol
        ),
    }
}

fn to_lower(c: u32) -> u32 {
    char::from_u32(c)
        .map(|ch| ch.to_lowercase().next().map(|l| l as u32).unwrap_or(c))
        .unwrap_or(c)
}

// ---------------------------------------------------------------------------
// qwen35 word scanner (port of unicode_regex_split_custom_qwen35)
// Returns a list of word lengths (in codepoints) partitioning the input.
// ---------------------------------------------------------------------------
fn scan_qwen35(cpts: &[u32]) -> Vec<usize> {
    const OOR: u32 = 0xFFFF_FFFF;
    let len = cpts.len();

    let get_cpt = |pos: usize| -> u32 {
        if pos < len {
            cpts[pos]
        } else {
            OOR
        }
    };
    let get_flags = |pos: usize| -> Flags {
        if pos < len {
            flags_of(cpts[pos])
        } else {
            Flags::default()
        }
    };

    let mut words = Vec::new();
    let mut prev_end = 0usize;
    let mut pos = 0usize;

    macro_rules! add_token {
        ($end:expr) => {{
            let e = $end;
            if e > prev_end {
                words.push(e - prev_end);
            }
            prev_end = e;
            e
        }};
    }

    while pos < len {
        let cpt = cpts[pos];
        let flags = get_flags(pos);

        // (?i:'s|'t|'re|'ve|'m|'ll|'d)
        if cpt == '\'' as u32 && pos + 1 < len {
            let next = to_lower(get_cpt(pos + 1));
            if next == 's' as u32 || next == 't' as u32 || next == 'm' as u32 || next == 'd' as u32 {
                pos = add_token!(pos + 2);
                continue;
            }
            if pos + 2 < len {
                let next2 = to_lower(get_cpt(pos + 2));
                if (next == 'r' as u32 && next2 == 'e' as u32)
                    || (next == 'v' as u32 && next2 == 'e' as u32)
                    || (next == 'l' as u32 && next2 == 'l' as u32)
                {
                    pos = add_token!(pos + 3);
                    continue;
                }
            }
        }

        // [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
        if !(cpt == '\r' as u32 || cpt == '\n' as u32 || flags.number) {
            let nf = get_flags(pos + 1);
            if flags.letter || flags.mark || nf.mark || nf.letter {
                let mut p = pos + 1;
                loop {
                    let f = get_flags(p);
                    if !(f.letter || f.mark) {
                        break;
                    }
                    p += 1;
                }
                pos = add_token!(p);
                continue;
            }
        }

        // \p{N}
        if flags.number {
            pos = add_token!(pos + 1);
            continue;
        }

        // <space>?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
        let f2 = if cpt == ' ' as u32 { get_flags(pos + 1) } else { flags };
        if !f2.any_of_main() && flags.any() {
            let mut p = pos + if cpt == ' ' as u32 { 1 } else { 0 };
            loop {
                let f = get_flags(p);
                if !(!f.any_of_main() && f.any()) {
                    break;
                }
                p += 1;
            }
            let mut c2 = get_cpt(p);
            while c2 == '\r' as u32 || c2 == '\n' as u32 {
                c2 = get_cpt(p + 1);
                p += 1;
            }
            pos = add_token!(p);
            continue;
        }

        // whitespace run accounting
        let mut n_ws = 0usize;
        let mut last_nl = 0usize;
        loop {
            let f = get_flags(pos + n_ws);
            if !f.whitespace {
                break;
            }
            let c2 = get_cpt(pos + n_ws);
            if c2 == '\r' as u32 || c2 == '\n' as u32 {
                last_nl = pos + n_ws + 1;
            }
            n_ws += 1;
        }

        // \s*[\r\n]+
        if last_nl > 0 {
            pos = add_token!(last_nl);
            continue;
        }

        // \s+(?!\S)
        if n_ws > 1 && get_cpt(pos + n_ws) != OOR {
            pos = add_token!(pos + n_ws - 1);
            continue;
        }

        // \s+
        if n_ws > 0 {
            pos = add_token!(pos + n_ws);
            continue;
        }

        // no match: single codepoint
        pos = add_token!(pos + 1);
    }

    words
}

// ---------------------------------------------------------------------------
// special-token partition (tokenizer_st_partition, text-only path)
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub enum Frag {
    Raw { start: usize, len: usize },
    Tok(i32),
}

fn split_fragment(
    raw: &[u8],
    needle: &[u8],
    start: usize,
    len: usize,
    out: &mut Vec<Frag>,
) {
    if len == 0 {
        return;
    }
    let window_end = start + len;
    // naive byte search
    let mut first = None;
    if !needle.is_empty() {
        let mut i = start;
        while i + needle.len() <= window_end {
            if &raw[i..i + needle.len()] == needle {
                first = Some(i);
                break;
            }
            i += 1;
        }
    }
    let Some(m) = first else {
        out.push(Frag::Raw { start, len });
        return;
    };

    // left part may still contain the token earlier (overlap), recurse
    split_fragment(raw, needle, start, m - start, out);
    out.push(Frag::Tok(0)); // id filled in below
    let right_start = m + needle.len();
    split_fragment(raw, needle, right_start, window_end - right_start, out);
}

fn partition(raw: &[u8], specials: &[(i32, Vec<u8>)], parse_special: bool, types: &[i32]) -> Vec<Frag> {
    // iterative: apply each special token over the fragment list
    let mut frags = vec![Frag::Raw { start: 0, len: raw.len() }];
    for &(id, ref text) in specials {
        let tt = types.get(id as usize).copied().unwrap_or(0);
        if !parse_special && (tt == 3 || tt == 0) {
            continue;
        }
        let mut next = Vec::new();
        for f in frags {
            match f {
                Frag::Tok(_) => next.push(f),
                Frag::Raw { start, len } => {
                    // decide whether any occurrence exists first (recursion handles it)
                    let mut tmp = Vec::new();
                    split_fragment(raw, text, start, len, &mut tmp);
                    let has_occ = tmp.iter().any(|f| matches!(f, Frag::Tok(_)));
                    if has_occ {
                        // replace the placeholder token ids
                        for frag in tmp {
                            match frag {
                                Frag::Tok(_) => next.push(Frag::Tok(id)),
                                Frag::Raw { start, len } => next.push(Frag::Raw { start, len }),
                            }
                        }
                    } else {
                        next.push(Frag::Raw { start, len });
                    }
                }
            }
        }
        frags = next;
    }
    frags
}

// ---------------------------------------------------------------------------
// BPE merge pass on a single byte-encoded word
// ---------------------------------------------------------------------------
struct Sym {
    prev: i32,
    next: i32,
    start: usize,
    n: usize,
}

fn sym_str<'a>(word: &'a str, s: &Sym) -> &'a str {
    &word[s.start..s.start + s.n]
}

fn add_new_bigram(
    word: &str,
    syms: &[Sym],
    vocab: &Vocab,
    left: i32,
    right: i32,
    queue: &mut std::collections::BinaryHeap<std::cmp::Reverse<(usize, usize, String)>>,
) {
    if left < 0 || right < 0 {
        return;
    }
    let l = sym_str(word, &syms[left as usize]);
    let r = sym_str(word, &syms[right as usize]);
    if let Some(rank) = vocab.find_bpe_rank(l, r) {
        queue.push(std::cmp::Reverse((rank, left as usize, format!("{l}{r}"))));
    }
}

pub fn encode_word(word: &str, vocab: &Vocab, out: &mut Vec<i32>) {
    let mut boundaries = vec![0usize];
    for c in word.chars() {
        boundaries.push(boundaries.last().unwrap() + c.len_utf8());
    }
    let n_chars = boundaries.len() - 1;
    let mut syms: Vec<Sym> = Vec::with_capacity(n_chars);
    for i in 0..n_chars {
        syms.push(Sym {
            prev: i as i32 - 1,
            next: if i + 1 == n_chars { -1 } else { i as i32 + 1 },
            start: boundaries[i],
            n: boundaries[i + 1] - boundaries[i],
        });
    }

    use std::cmp::Reverse;
    let mut queue: std::collections::BinaryHeap<Reverse<(usize, usize, String)>> =
        std::collections::BinaryHeap::new();

    for i in 1..n_chars {
        add_new_bigram(word, &syms, vocab, i as i32 - 1, i as i32, &mut queue);
    }

    while let Some(Reverse((_rank, left, text))) = queue.pop() {
        let right = syms[left].next;
        if right < 0 {
            continue;
        }
        let right = right as usize;
        if syms[left].n == 0 || syms[right].n == 0 {
            continue;
        }
        // skip stale bigram entries (neighbors changed since enqueue)
        let combined = format!(
            "{}{}",
            sym_str(word, &syms[left]),
            sym_str(word, &syms[right])
        );
        if combined != text {
            continue;
        }
        // merge right into left
        syms[left].n += syms[right].n;
        syms[right].n = 0;
        syms[left].next = syms[right].next;
        if syms[right].next >= 0 {
            let nn = syms[right].next as usize;
            syms[nn].prev = left as i32;
        }
        let pl = syms[left].prev;
        let nl = syms[left].next;
        add_new_bigram(word, &syms, vocab, pl, left as i32, &mut queue);
        add_new_bigram(word, &syms, vocab, left as i32, nl, &mut queue);
    }

    for s in &syms {
        if s.n == 0 {
            continue;
        }
        let str = sym_str(word, s);
        match vocab.token(str) {
            Some(id) => out.push(id),
            None => {
                for &b in str.as_bytes() {
                    if let Some(id) = vocab.byte_tokens.get(&b) {
                        out.push(*id);
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// top-level encode: llama_vocab::impl::tokenize for BPE
// ---------------------------------------------------------------------------
pub struct TokenizeOptions {
    pub add_special: bool,
    pub parse_special: bool,
}

pub fn encode(text: &str, vocab: &Vocab, opts: &TokenizeOptions) -> Vec<i32> {
    let mut out = Vec::new();

    let specials: Vec<(i32, Vec<u8>)> = vocab
        .special_ids
        .iter()
        .map(|&id| (id, vocab.tokens[id as usize].clone().into_bytes()))
        .collect();
    let frags = partition(text.as_bytes(), &specials, opts.parse_special, &vocab.token_types);

    for f in &frags {
        match f {
            Frag::Tok(id) => out.push(*id),
            Frag::Raw { start, len } => {
                let raw = &text.as_bytes()[*start..*start + *len];
                let s = String::from_utf8_lossy(raw);
                encode_fragment(&s, vocab, &mut out);
            }
        }
    }

    if opts.add_special {
        // qwen35 GGUF has tokenizer.ggml.add_bos_token = false; nothing to add.
    }
    out
}

fn encode_fragment(text: &str, vocab: &Vocab, out: &mut Vec<i32>) {
    let cpts: Vec<u32> = text.chars().map(|c| c as u32).collect();
    let lengths = scan_qwen35(&cpts);
    let mut start = 0usize;
    for &len in &lengths {
        let end = start + len;
        let word: String = cpts[start..end]
            .iter()
            .filter_map(|&c| char::from_u32(c))
            .collect();
        let encoded = byte_encode_word(&word);
        encode_word(&encoded, vocab, out);
        start = end;
    }
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_encoder_table() {
        assert_eq!(byte_to_char(b' '), '\u{0120}'); // Ġ
        assert_eq!(byte_to_char(b'H'), 'H');
        assert_eq!(byte_to_char(0xA1), '\u{A1}');
        assert_eq!(byte_to_char(0x00), '\u{0100}');
        assert_eq!(byte_to_char(0x1F), '\u{011F}');
        assert_eq!(byte_to_char(0x7F), '\u{0121}');
        assert_eq!(byte_encode_word(" Hello"), "ĠHello");
    }

    #[test]
    fn qwen35_scan_simple() {
        let text: Vec<u32> = "The capital of France".chars().map(|c| c as u32).collect();
        let lens = scan_qwen35(&text);
        let total: usize = lens.iter().sum();
        assert_eq!(total, text.len());
        let mut s = 0;
        let mut parts = Vec::new();
        for &l in &lens {
            let w: String = text[s..s + l].iter().filter_map(|&c| char::from_u32(c)).collect();
            parts.push(w);
            s += l;
        }
        // a single space prefixes the following word (regex:  ?\p{L}+)
        assert_eq!(parts, vec!["The", " capital", " of", " France"], "parts={parts:?}");
    }
}
