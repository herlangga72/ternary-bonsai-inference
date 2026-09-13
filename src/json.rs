//! Minimal JSON for the HTTP server: just enough to parse OpenAI request
//! bodies and emit responses without pulling in a JSON crate.

#![allow(dead_code)]

#[derive(Debug, Clone, PartialEq)]
pub enum J {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<J>),
    Obj(Vec<(String, J)>),
}

impl J {
    pub fn get(&self, key: &str) -> Option<&J> {
        match self {
            J::Obj(pairs) => pairs.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            J::Str(s) => Some(s),
            _ => None,
        }
    }
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            J::Num(n) => Some(*n),
            _ => None,
        }
    }
    pub fn as_i64(&self) -> Option<i64> {
        self.as_f64().map(|n| n as i64)
    }
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            J::Bool(b) => Some(*b),
            _ => None,
        }
    }
    pub fn as_arr(&self) -> Option<&[J]> {
        match self {
            J::Arr(v) => Some(v),
            _ => None,
        }
    }

    pub fn parse(s: &str) -> Result<J, String> {
        let b = s.as_bytes();
        let mut i = 0usize;
        let v = parse_value(b, &mut i)?;
        skip_ws(b, &mut i);
        if i != b.len() {
            return Err(format!("trailing data at byte {i}"));
        }
        Ok(v)
    }
}

fn skip_ws(b: &[u8], i: &mut usize) {
    while *i < b.len() && matches!(b[*i], b' ' | b'\t' | b'\n' | b'\r') {
        *i += 1;
    }
}

fn parse_value(b: &[u8], i: &mut usize) -> Result<J, String> {
    skip_ws(b, i);
    if *i >= b.len() {
        return Err("unexpected end of input".into());
    }
    match b[*i] {
        b'{' => parse_obj(b, i),
        b'[' => parse_arr(b, i),
        b'"' => Ok(J::Str(parse_str(b, i)?)),
        b't' => lit(b, i, "true", J::Bool(true)),
        b'f' => lit(b, i, "false", J::Bool(false)),
        b'n' => lit(b, i, "null", J::Null),
        _ => parse_num(b, i),
    }
}

fn lit(b: &[u8], i: &mut usize, word: &str, v: J) -> Result<J, String> {
    if b.len() >= *i + word.len() && &b[*i..*i + word.len()] == word.as_bytes() {
        *i += word.len();
        Ok(v)
    } else {
        Err(format!("bad literal at byte {i}"))
    }
}

fn parse_num(b: &[u8], i: &mut usize) -> Result<J, String> {
    let start = *i;
    while *i < b.len()
        && matches!(b[*i], b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9')
    {
        *i += 1;
    }
    if start == *i {
        return Err(format!("bad number at byte {start}"));
    }
    std::str::from_utf8(&b[start..*i])
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .map(J::Num)
        .ok_or_else(|| format!("bad number at byte {start}"))
}

fn parse_str(b: &[u8], i: &mut usize) -> Result<String, String> {
    if *i >= b.len() || b[*i] != b'"' {
        return Err("expected string".into());
    }
    *i += 1;
    let mut out = String::new();
    while *i < b.len() {
        let c = b[*i];
        *i += 1;
        match c {
            b'"' => return Ok(out),
            b'\\' => {
                if *i >= b.len() {
                    return Err("truncated escape".into());
                }
                let e = b[*i];
                *i += 1;
                match e {
                    b'"' => out.push('"'),
                    b'\\' => out.push('\\'),
                    b'/' => out.push('/'),
                    b'b' => out.push('\u{8}'),
                    b'f' => out.push('\u{c}'),
                    b'n' => out.push('\n'),
                    b'r' => out.push('\r'),
                    b't' => out.push('\t'),
                    b'u' => {
                        let cp = hex4(b, i)?;
                        // surrogate pair
                        if (0xD800..0xDC00).contains(&cp) && *i + 1 < b.len() && b[*i] == b'\\' && b[*i + 1] == b'u' {
                            *i += 2;
                            let lo = hex4(b, i)?;
                            let c = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                            out.push(char::from_u32(c).unwrap_or('\u{fffd}'));
                        } else {
                            out.push(char::from_u32(cp).unwrap_or('\u{fffd}'));
                        }
                    }
                    _ => return Err("bad escape".into()),
                }
            }
            _ => {
                // copy the raw UTF-8 byte(s)
                let start = *i - 1;
                while *i < b.len() && (b[*i] & 0xC0) == 0x80 {
                    *i += 1;
                }
                match std::str::from_utf8(&b[start..*i]) {
                    Ok(s) => out.push_str(s),
                    Err(_) => out.push('\u{fffd}'),
                }
            }
        }
    }
    Err("unterminated string".into())
}

fn hex4(b: &[u8], i: &mut usize) -> Result<u32, String> {
    if *i + 4 > b.len() {
        return Err("truncated \\u".into());
    }
    let s = std::str::from_utf8(&b[*i..*i + 4]).map_err(|_| "bad \\u")?;
    *i += 4;
    u32::from_str_radix(s, 16).map_err(|_| "bad \\u".to_string())
}

fn parse_arr(b: &[u8], i: &mut usize) -> Result<J, String> {
    *i += 1; // '['
    let mut items = Vec::new();
    skip_ws(b, i);
    if *i < b.len() && b[*i] == b']' {
        *i += 1;
        return Ok(J::Arr(items));
    }
    loop {
        items.push(parse_value(b, i)?);
        skip_ws(b, i);
        match b.get(*i) {
            Some(b',') => {
                *i += 1;
            }
            Some(b']') => {
                *i += 1;
                return Ok(J::Arr(items));
            }
            _ => return Err("expected , or ] in array".into()),
        }
    }
}

fn parse_obj(b: &[u8], i: &mut usize) -> Result<J, String> {
    *i += 1; // '{'
    let mut pairs = Vec::new();
    skip_ws(b, i);
    if *i < b.len() && b[*i] == b'}' {
        *i += 1;
        return Ok(J::Obj(pairs));
    }
    loop {
        skip_ws(b, i);
        let k = parse_str(b, i)?;
        skip_ws(b, i);
        if b.get(*i) != Some(&b':') {
            return Err("expected : in object".into());
        }
        *i += 1;
        let v = parse_value(b, i)?;
        pairs.push((k, v));
        skip_ws(b, i);
        match b.get(*i) {
            Some(b',') => {
                *i += 1;
            }
            Some(b'}') => {
                *i += 1;
                return Ok(J::Obj(pairs));
            }
            _ => return Err("expected , or } in object".into()),
        }
    }
}

/// Serialize a value back to compact JSON (order preserved).
pub fn to_json(v: &J) -> String {
    let mut out = String::new();
    write_json(v, &mut out);
    out
}

fn write_json(v: &J, out: &mut String) {
    match v {
        J::Null => out.push_str("null"),
        J::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        J::Num(n) => {
            if n.is_finite() {
                if n.fract() == 0.0 && n.abs() < 1e15 {
                    out.push_str(&format!("{}", *n as i64));
                } else {
                    out.push_str(&format!("{n}"));
                }
            } else {
                out.push_str("null");
            }
        }
        J::Str(s) => {
            out.push('"');
            out.push_str(&escape(s));
            out.push('"');
        }
        J::Arr(items) => {
            out.push('[');
            for (i, it) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_json(it, out);
            }
            out.push(']');
        }
        J::Obj(pairs) => {
            out.push('{');
            for (i, (k, val)) in pairs.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push('"');
                out.push_str(&escape(k));
                out.push_str("\":");
                write_json(val, out);
            }
            out.push('}');
        }
    }
}

/// JSON string escape (RFC 8259).
pub fn escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 8);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_objects_and_arrays() {
        let v = J::parse(r#"{"a": 1, "b": [true, null, "x\ny"], "c": {"d": -2.5}}"#).unwrap();
        assert_eq!(v.get("a").unwrap().as_i64(), Some(1));
        assert_eq!(v.get("b").unwrap().as_arr().unwrap().len(), 3);
        assert_eq!(v.get("b").unwrap().as_arr().unwrap()[2].as_str(), Some("x\ny"));
        assert_eq!(v.get("c").unwrap().get("d").unwrap().as_f64(), Some(-2.5));
    }

    #[test]
    fn parses_unicode_escapes() {
        let v = J::parse(r#""a\u00e9b""#).unwrap();
        assert_eq!(v.as_str(), Some("aéb"));
    }

    #[test]
    fn serializes_back_to_compact_json() {
        let src = r#"{"a":1,"b":[true,null,"x"],"c":{"d":-2.5},"e":"q\"z"}"#;
        let v = J::parse(src).unwrap();
        assert_eq!(to_json(&v), src);
    }

    #[test]
    fn escapes_roundtrip() {
        assert_eq!(escape("a\"b\\c\nd"), "a\\\"b\\\\c\\nd");
    }
}
