use std::io::{Write, Read};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

// ── Constants ─────────────────────────────────────────────────────────────

const DS4_TOOL_CALLS_START: &str = "<｜DSML｜tool_calls>";
const DS4_TOOL_CALLS_END: &str = "</｜DSML｜tool_calls>";
const DS4_INVOKE_START: &str = "<｜DSML｜invoke";
const DS4_INVOKE_END: &str = "</｜DSML｜invoke>";
const DS4_PARAM_START: &str = "<｜DSML｜parameter";
const DS4_PARAM_END: &str = "</｜DSML｜parameter>";

static STOP_REQUESTED: AtomicBool = AtomicBool::new(false);

// ── Think Mode ────────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq)]
enum ThinkMode { None, High, Max }

impl ThinkMode {
    fn enabled(self) -> bool { !matches!(self, ThinkMode::None) }
    fn from_enabled(enabled: bool, effort: ThinkMode) -> Self {
        if !enabled { ThinkMode::None }
        else if matches!(effort, ThinkMode::Max) { ThinkMode::Max }
        else { ThinkMode::High }
    }
}

// ── Stop List ─────────────────────────────────────────────────────────────

struct StopList {
    v: Vec<String>,
    max_len: usize,
}

impl StopList {
    fn new() -> Self { StopList { v: Vec::new(), max_len: 0 } }
    fn push(&mut self, s: &str) {
        if s.is_empty() { return; }
        self.max_len = self.max_len.max(s.len());
        self.v.push(s.to_string());
    }
    fn clear(&mut self) { self.v.clear(); self.max_len = 0; }
    fn find_from(&self, text: &str, from: usize) -> Option<(usize, usize)> {
        if self.v.is_empty() || text.is_empty() { return None; }
        let mut best: Option<(usize, usize)> = None;
        for stop in &self.v {
            if let Some(p) = text[from..].find(stop.as_str()) {
                let ppos = from + p;
                let plen = stop.len();
                match best {
                    Some((bp, _)) if ppos < bp => best = Some((ppos, plen)),
                    None => best = Some((ppos, plen)),
                    _ => {}
                }
            }
        }
        best
    }
    fn stream_safe_len(&self, text_len: usize) -> usize {
        if self.v.is_empty() || self.max_len <= 1 { return text_len; }
        let hold = self.max_len - 1;
        if text_len > hold { text_len - hold } else { 0 }
    }
}

// ── Data Structures ───────────────────────────────────────────────────────

#[derive(Default)]
struct ToolCall {
    id: String,
    name: String,
    arguments: String,
}

#[derive(Default)]
struct ToolCalls {
    v: Vec<ToolCall>,
    raw_dsml: String,
}

#[derive(Default)]
struct ChatMessage {
    role: String,
    content: String,
    reasoning: String,
    tool_call_id: String,
    calls: ToolCalls,
}

#[derive(Default)]
struct ToolSchemaOrder {
    name: String,
    prop: Vec<String>,
}

#[derive(Default)]
struct ChatMessages {
    v: Vec<ChatMessage>,
}

#[derive(Clone, Copy, PartialEq)]
enum ReqKind { Chat, Completion }

#[derive(Clone, Copy, PartialEq)]
enum ApiStyle { OpenAI, Anthropic }

struct Request {
    kind: ReqKind,
    api: ApiStyle,
    prompt: Vec<i32>,
    model: String,
    stops: StopList,
    raw_body: String,
    prompt_text: String,
    tool_orders: Vec<ToolSchemaOrder>,
    max_tokens: usize,
    top_k: usize,
    temperature: f32,
    top_p: f32,
    min_p: f32,
    seed: u64,
    stream: bool,
    stream_include_usage: bool,
    think_mode: ThinkMode,
    has_tools: bool,
    prompt_preserves_reasoning: bool,
}

impl Request {
    fn new(kind: ReqKind, max_tokens: usize) -> Self {
        Request {
            kind,
            api: ApiStyle::OpenAI,
            model: "deepseek-v4-flash".to_string(),
            max_tokens,
            top_k: 0,
            temperature: 1.0,
            top_p: 1.0,
            min_p: 0.0,
            seed: 0,
            stream: false,
            stream_include_usage: false,
            think_mode: ThinkMode::High,
            has_tools: false,
            prompt_preserves_reasoning: false,
            prompt: Vec::new(),
            stops: StopList::new(),
            raw_body: String::new(),
            prompt_text: String::new(),
            tool_orders: Vec::new(),
        }
    }
}

// ── JSON Parser ───────────────────────────────────────────────────────────

struct JsonParser<'a> {
    p: &'a [u8],
    pos: usize,
}

impl<'a> JsonParser<'a> {
    fn new(s: &'a str) -> Self { JsonParser { p: s.as_bytes(), pos: 0 } }

    fn remaining(&self) -> &[u8] { &self.p[self.pos..] }

    fn ws(&mut self) {
        while self.pos < self.p.len() && self.p[self.pos].is_ascii_whitespace() {
            self.pos += 1;
        }
    }

    fn peek(&self) -> u8 { self.p.get(self.pos).copied().unwrap_or(0) }

    fn lit(&mut self, lit: &str) -> bool {
        let b = lit.as_bytes();
        if self.p[self.pos..].starts_with(b) {
            self.pos += b.len();
            true
        } else { false }
    }

    fn string(&mut self) -> Option<String> {
        self.ws();
        if self.peek() != b'"' { return None; }
        self.pos += 1;
        let mut out = Vec::new();
        loop {
            let c = *self.p.get(self.pos)?;
            self.pos += 1;
            if c == b'"' { break; }
            if c != b'\\' { out.push(c); continue; }
            let c2 = *self.p.get(self.pos)?;
            self.pos += 1;
            match c2 {
                b'"' => out.push(b'"'),
                b'\\' => out.push(b'\\'),
                b'/' => out.push(b'/'),
                b'b' => out.push(8),
                b'f' => out.push(12),
                b'n' => out.push(b'\n'),
                b'r' => out.push(b'\r'),
                b't' => out.push(b'\t'),
                b'u' => {
                    self.pos -= 2;
                    let cp = self.json_u16()?;
                    let lo = if cp >= 0xd800 && cp <= 0xdbff {
                        let lo = self.json_u16()?;
                        if lo >= 0xdc00 && lo <= 0xdfff {
                            0x10000 + ((cp - 0xd800) << 10) + (lo - 0xdc00)
                        } else { 0 }
                    } else { 0 };
                    let cp = if lo != 0 { lo } else { cp };
                    let mut buf = [0u8; 4];
                    let s = std::char::from_u32(cp).unwrap_or('?').encode_utf8(&mut buf);
                    out.extend_from_slice(s.as_bytes());
                }
                _ => return None,
            }
        }
        Some(String::from_utf8(out).unwrap_or_default())
    }

    fn json_u16(&mut self) -> Option<u32> {
        if self.remaining().starts_with(b"\\u") {
            let s = std::str::from_utf8(&self.p[self.pos+2..self.pos+6]).ok()?;
            self.pos += 6;
            u32::from_str_radix(s, 16).ok()
        } else { None }
    }

    fn number(&mut self) -> Option<f64> {
        self.ws();
        let start = self.pos;
        if self.peek() == b'-' { self.pos += 1; }
        while self.pos < self.p.len() && self.p[self.pos].is_ascii_digit() { self.pos += 1; }
        if self.peek() == b'.' {
            self.pos += 1;
            while self.pos < self.p.len() && self.p[self.pos].is_ascii_digit() { self.pos += 1; }
        }
        if self.peek() == b'e' || self.peek() == b'E' {
            self.pos += 1;
            if self.peek() == b'+' || self.peek() == b'-' { self.pos += 1; }
            while self.pos < self.p.len() && self.p[self.pos].is_ascii_digit() { self.pos += 1; }
        }
        if self.pos == start { return None; }
        std::str::from_utf8(&self.p[start..self.pos]).ok()?.parse().ok()
    }

    fn int(&mut self) -> Option<usize> {
        let v = self.number()?;
        if v < 0.0 { Some(0) } else { Some(v as usize) }
    }

    fn bool_val(&mut self) -> Option<bool> {
        self.ws();
        if self.lit("true") { Some(true) }
        else if self.lit("false") { Some(false) }
        else { None }
    }

    fn skip_value(&mut self) -> bool { self.skip_value_depth(0) }

    fn skip_value_depth(&mut self, depth: usize) -> bool {
        if depth > 256 { return false; }
        self.ws();
        match self.peek() {
            b'"' => { self.string().is_some() }
            b'{' => self.skip_object(depth),
            b'[' => self.skip_array(depth),
            _ => {
                if self.lit("true") || self.lit("false") || self.lit("null") { return true; }
                self.number().is_some()
            }
        }
    }

    fn skip_object(&mut self, depth: usize) -> bool {
        if depth >= 256 { return false; }
        self.ws();
        if self.peek() != b'{' { return false; }
        self.pos += 1;
        self.ws();
        if self.peek() == b'}' { self.pos += 1; return true; }
        loop {
            self.ws();
            if self.peek() != b'"' { return false; }
            if self.string().is_none() { return false; }
            self.ws();
            if self.peek() != b':' { return false; }
            self.pos += 1;
            if !self.skip_value_depth(depth + 1) { return false; }
            self.ws();
            if self.peek() == b'}' { self.pos += 1; return true; }
            if self.peek() != b',' { return false; }
            self.pos += 1;
        }
    }

    fn skip_array(&mut self, depth: usize) -> bool {
        if depth >= 256 { return false; }
        self.ws();
        if self.peek() != b'[' { return false; }
        self.pos += 1;
        self.ws();
        if self.peek() == b']' { self.pos += 1; return true; }
        loop {
            if !self.skip_value_depth(depth + 1) { return false; }
            self.ws();
            if self.peek() == b']' { self.pos += 1; return true; }
            if self.peek() != b',' { return false; }
            self.pos += 1;
        }
    }

    fn raw_value(&mut self) -> Option<String> {
        self.ws();
        let start = self.pos;
        if !self.skip_value() { return None; }
        Some(String::from_utf8(self.p[start..self.pos].to_vec()).unwrap_or_default())
    }

    fn content(&mut self) -> Option<String> {
        self.ws();
        if self.peek() == b'"' { return self.string(); }
        if self.lit("null") { return Some(String::new()); }
        if self.peek() != b'[' { self.skip_value(); return Some(String::new()); }
        self.pos += 1;
        let mut out = String::new();
        loop {
            self.ws();
            if self.peek() == b']' { self.pos += 1; break; }
            if self.peek() == b'"' {
                if let Some(s) = self.string() { out.push_str(&s); }
            } else if self.peek() == b'{' {
                self.pos += 1;
                loop {
                    self.ws();
                    if self.peek() == b'}' { self.pos += 1; break; }
                    let key = self.string()?;
                    self.ws();
                    if self.peek() != b':' { return None; }
                    self.pos += 1;
                    if key == "text" {
                        if let Some(s) = self.string() { out.push_str(&s); }
                    } else { self.skip_value(); }
                    self.ws();
                    if self.peek() == b',' { self.pos += 1; }
                }
            } else { self.skip_value(); }
            self.ws();
            if self.peek() == b',' { self.pos += 1; }
        }
        Some(out)
    }

    fn key(&mut self) -> Option<String> {
        self.ws();
        self.string()
    }

    fn colon(&mut self) -> bool {
        self.ws();
        if self.peek() == b':' { self.pos += 1; true } else { false }
    }

    fn comma(&mut self) {
        self.ws();
        if self.peek() == b',' { self.pos += 1; }
    }
}

// ── Chat Template Rendering ───────────────────────────────────────────────

fn role_is_system(role: &str) -> bool {
    role == "system" || role == "developer"
}

fn role_is_user_like(role: &str) -> bool {
    role == "user" || role == "tool" || role == "function"
}

fn append_tools_prompt_text(b: &mut String, schemas: &str) {
    if schemas.is_empty() { return; }
    b.push_str(
        "## Tools\n\n\
         You have access to a set of tools to help answer the user question. \
         You can invoke tools by writing a \"<｜DSML｜tool_calls>\" block like the following:\n\n\
         <｜DSML｜tool_calls>\n\
         <｜DSML｜invoke name=\"$TOOL_NAME\">\n\
         <｜DSML｜parameter name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</｜DSML｜parameter>\n\
         ...\n\
         </｜DSML｜invoke>\n\
         </｜DSML｜tool_calls>\n\n\
         String parameters should be specified as raw text and set `string=\"true\"`. \
         Preserve characters such as `>`, `&`, and `&&` exactly; never replace normal string \
         characters with XML or HTML entity escapes. \
         Only if a string value itself contains the exact closing parameter tag `</｜DSML｜parameter>`, \
         write that tag as `&lt;/｜DSML｜parameter>` inside the value. \
         For all other types (numbers, booleans, arrays, objects), pass the value in JSON format \
         and set `string=\"false\"`.\n\n\
         If thinking_mode is enabled (triggered by <think>), you MUST output your complete \
         reasoning inside <think>...</think> BEFORE any tool calls or final response.\n\n\
         Otherwise, output directly after </think> with tool calls or final response.\n\n\
         ### Available Tool Schemas\n\n");
    b.push_str(schemas);
    b.push_str("\n\nYou MUST strictly follow the above defined tool name and parameter schemas \
                 to invoke tool calls. Use the exact parameter names from the schemas.");
}

fn text_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            _ => out.push(c),
        }
    }
    out
}

fn attr_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            _ => out.push(c),
        }
    }
    out
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' | '\\' => { out.push('\\'); out.push(c); }
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            _ => out.push(c),
        }
    }
    out.push('"');
    out
}

fn chat_history_uses_tool_context(msgs: &ChatMessages, schemas: &str) -> bool {
    if !schemas.is_empty() { return true; }
    for m in &msgs.v {
        if m.role == "assistant" && !m.calls.v.is_empty() { return true; }
        if m.role == "tool" || m.role == "function" { return true; }
    }
    false
}

fn append_dsml_tool_calls_text(b: &mut String, calls: &ToolCalls) {
    if calls.v.is_empty() { return; }
    if !calls.raw_dsml.is_empty() {
        b.push_str(&calls.raw_dsml);
        return;
    }
    b.push_str("\n\n<｜DSML｜tool_calls>\n");
    for tc in &calls.v {
        b.push_str("<｜DSML｜invoke name=\"");
        b.push_str(&attr_escape(&tc.name));
        b.push_str("\">\n");
        b.push_str("<｜DSML｜parameter name=\"arguments\" string=\"true\">");
        b.push_str(&text_escape(&tc.arguments));
        b.push_str("</｜DSML｜parameter>\n");
        b.push_str("</｜DSML｜invoke>\n");
    }
    b.push_str("</｜DSML｜tool_calls>");
}

fn render_chat_prompt_text(
    msgs: &ChatMessages,
    schemas: &str,
    think_mode: ThinkMode,
) -> String {
    let think = think_mode.enabled();
    let tool_context = chat_history_uses_tool_context(msgs, schemas);
    let mut last_user_idx = -1i32;
    for (i, m) in msgs.v.iter().enumerate() {
        if role_is_user_like(&m.role) { last_user_idx = i as i32; }
    }

    let mut system = String::new();
    for m in &msgs.v {
        if !role_is_system(&m.role) { continue; }
        if !system.is_empty() { system.push_str("\n\n"); }
        system.push_str(&m.content);
    }
    if !schemas.is_empty() {
        if !system.is_empty() { system.push_str("\n\n"); }
        append_tools_prompt_text(&mut system, schemas);
    }

    let mut out = String::new();
    out.push_str("<｜begin▁of▁sentence｜>");
    if think_mode == ThinkMode::Max {
        // ds4_think_max_prefix would be available from C bindings
    }
    out.push_str(&system);

    let mut pending_assistant = false;
    let mut pending_tool_result = false;
    let last_user = last_user_idx;

    for m in &msgs.v {
        if role_is_system(&m.role) {
            continue;
        } else if m.role == "user" {
            out.push_str("<｜User｜>");
            out.push_str(&m.content);
            pending_assistant = true;
            pending_tool_result = false;
        } else if m.role == "tool" || m.role == "function" {
            if !pending_tool_result { out.push_str("<｜User｜>"); }
            out.push_str("<tool_result>");
            out.push_str(&text_escape(&m.content));
            out.push_str("</tool_result>");
            pending_assistant = true;
            pending_tool_result = true;
        } else if m.role == "assistant" {
            if pending_assistant {
                out.push_str("<｜Assistant｜>");
                if think {
                    if tool_context || msgs.v.iter().position(|x| x as *const _ == m).unwrap_or(0) > last_user as usize {
                        out.push_str("<think>");
                        out.push_str(&m.reasoning);
                        out.push_str("</think>");
                    } else {
                        out.push_str("</think>");
                    }
                } else {
                    out.push_str("</think>");
                }
            }
            out.push_str(&m.content);
            append_dsml_tool_calls_text(&mut out, &m.calls);
            out.push_str("<｜end▁of▁sentence｜>");
            pending_assistant = false;
            pending_tool_result = false;
        }
    }

    if pending_assistant {
        out.push_str("<｜Assistant｜>");
        out.push_str(if think { "<think>" } else { "</think>" });
    }

    out
}

// ── Response Writing ──────────────────────────────────────────────────────

fn send_json(stream: &mut TcpStream, code: u16, body: &str) {
    let reason = match code {
        200 => "OK", 400 => "Bad Request", 404 => "Not Found",
        500 => "Internal Server Error", _ => "Error",
    };
    let resp = format!(
        "HTTP/1.1 {} {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        code, reason, body.len(), body);
    let _ = stream.write_all(resp.as_bytes());
}

fn send_json_keepalive(stream: &mut TcpStream, code: u16, body: &str) {
    let reason = match code {
        200 => "OK", 400 => "Bad Request", 404 => "Not Found",
        500 => "Internal Server Error", _ => "Error",
    };
    let resp = format!(
        "HTTP/1.1 {} {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        code, reason, body.len(), body);
    let _ = stream.write_all(resp.as_bytes());
}

fn send_error(stream: &mut TcpStream, code: u16, msg: &str) {
    let body = format!(r#"{{"error":{{"message":{},"type":"invalid_request_error"}}}}"#,
                       json_escape(msg));
    send_json(stream, code, &body);
}

fn start_sse(stream: &mut TcpStream) {
    let hdr = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n";
    let _ = stream.write_all(hdr.as_bytes());
    let _ = stream.flush();
}

fn sse_send(stream: &mut TcpStream, data: &str) {
    let msg = format!("data: {}\n\n", data);
    let _ = stream.write_all(msg.as_bytes());
    let _ = stream.flush();
}

fn sse_done(stream: &mut TcpStream) {
    let _ = stream.write_all(b"data: [DONE]\n\n");
    let _ = stream.flush();
}

fn now_secs() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs() as i64
}

// ── Tool Call Parsing from Generated DSML ─────────────────────────────────

fn find_any_tool_start(text: &str) -> Option<usize> {
    for pat in &[DS4_TOOL_CALLS_START, "<tool_calls>"] {
        if let Some(p) = text.find(pat) { return Some(p); }
    }
    None
}

fn dsml_unescape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] != b'&' { out.push(bytes[i] as char); i += 1; continue; }
        if s[i..].starts_with("&amp;") { out.push('&'); i += 5; }
        else if s[i..].starts_with("&lt;") { out.push('<'); i += 4; }
        else if s[i..].starts_with("&gt;") { out.push('>'); i += 4; }
        else if s[i..].starts_with("&quot;") { out.push('"'); i += 6; }
        else if s[i..].starts_with("&apos;") { out.push('\''); i += 6; }
        else { out.push('&'); i += 1; }
    }
    out
}

fn dsml_attr(tag: &str, name: &str) -> Option<String> {
    let pat = format!("{}=\"", name);
    let start = tag.find(&pat)?;
    let after = &tag[start + pat.len()..];
    let end = after.find('"')?;
    let raw = &after[..end];
    Some(dsml_unescape(raw))
}

fn parse_generated_message(text: &str) -> (String, String, ToolCalls) {
    let mut calls = ToolCalls::default();
    let mut reasoning = String::new();
    let mut content;
    let text = text.trim();

    // Split <think>...</think>
    let body = if let Some(ts) = text.find("<think>") {
        let after_think = &text[ts + 7..];
        if let Some(te) = after_think.find("</think>") {
            reasoning = after_think[..te].to_string();
            format!("{}{}", &text[..ts], &after_think[te + 8..])
        } else {
            text.to_string()
        }
    } else {
        text.to_string()
    };

    // Find tool_calls block
    let tool_start = find_any_tool_start(&body);
    if let Some(ts) = tool_start {
        content = body[..ts].trim_end().to_string();
        let block = &body[ts..];
        if let Some(tc_end) = block.find(DS4_TOOL_CALLS_END)
            .or_else(|| block.find("</tool_calls>"))
        {
            let end_len = if block[tc_end..].starts_with(DS4_TOOL_CALLS_END) { DS4_TOOL_CALLS_END.len() }
                           else { "</tool_calls>".len() };
            calls.raw_dsml = block[..tc_end + end_len].to_string();

            // Parse invokes
            let mut pos = ts;
            loop {
                let inv = match body[pos..].find(DS4_INVOKE_START)
                    .or_else(|| body[pos..].find("<invoke"))
                {
                    Some(p) => p,
                    None => break,
                };
                let inv_abs = pos + inv;
                let after_open = &body[inv_abs..];
                let tag_end = match after_open.find('>') {
                    Some(p) => p,
                    None => break,
                };
                let attrs = &after_open[..tag_end];
                let tool_name = dsml_attr(attrs, "name").unwrap_or_default();

                let value_start = tag_end + 1;
                let value_end = match after_open[value_start..].find(DS4_INVOKE_END)
                    .or_else(|| after_open[value_start..].find("</invoke>"))
                {
                    Some(p) => p,
                    None => break,
                };
                let body_content = &after_open[value_start..value_start + value_end];

                let mut args = String::from("{");
                let mut first = true;
                let mut bp = 0usize;
                loop {
                    let ps = match body_content[bp..].find(DS4_PARAM_START)
                        .or_else(|| body_content[bp..].find("<parameter"))
                    {
                        Some(p) => p,
                        None => break,
                    };
                    let pbody = &body_content[bp + ps..];
                    let close = match pbody.find('>') {
                        Some(p) => p,
                        None => break,
                    };
                    let pattrs = &pbody[..close];
                    let (pname, is_string) = parse_dsml_param(pattrs);
                    let pval_start = close + 1;
                    let pval_end = match pbody[pval_start..].find(DS4_PARAM_END)
                        .or_else(|| pbody[pval_start..].find("</parameter>"))
                    {
                        Some(p) => p,
                        None => break,
                    };
                    let pval = &pbody[pval_start..pval_start + pval_end];

                    if !first { args.push_str(", "); }
                    first = false;
                    if is_string {
                        args.push_str(&format!("{}: {}", json_escape(&pname), json_escape(pval)));
                    } else {
                        args.push_str(&format!("{}: {}", json_escape(&pname), pval));
                    }

                    bp += ps + pval_start + pval_end +
                        if pbody[pval_start + pval_end..].starts_with(DS4_PARAM_END) { DS4_PARAM_END.len() }
                        else { "</parameter>".len() };
                }
                args.push('}');
                calls.v.push(ToolCall {
                    name: tool_name,
                    arguments: args,
                    ..Default::default()
                });
                let inv_end_len = { let rest = &after_open[value_start + value_end..];
                    if rest.starts_with(DS4_INVOKE_END) { DS4_INVOKE_END.len() }
                    else { "</invoke>".len() } };
                pos = inv_abs + value_start + value_end + inv_end_len;
            }
        }
    } else {
        content = body;
    }

    content = content.trim_end().to_string();
    (content, reasoning, calls)
}

fn parse_dsml_param(attrs: &str) -> (String, bool) {
    let name = dsml_attr(attrs, "name").unwrap_or_default();
    let is_string = dsml_attr(attrs, "string")
        .map(|s| s == "true")
        .unwrap_or(true);
    (name, is_string)
}

// ── Request Parsing: OpenAI Chat ──────────────────────────────────────────

fn parse_chat_request(body: &str) -> Option<Request> {
    let mut r = Request::new(ReqKind::Chat, 256);
    let mut p = JsonParser::new(body);
    let mut got_messages = false;
    let mut tool_choice_none = false;
    let mut got_thinking = false;
    let mut thinking_enabled = true;
    let mut reasoning_effort = ThinkMode::High;
    let mut msgs = ChatMessages::default();
    let mut tool_schemas = String::new();

    p.ws();
    if p.peek() != b'{' { return None; }
    p.pos += 1;

    loop {
        p.ws();
        if p.peek() == b'}' { p.pos += 1; break; }
        let key = p.key()?;
        p.colon();

        match key.as_str() {
            "messages" => {
                msgs.v.clear();
                parse_messages(&mut p, &mut msgs)?;
                got_messages = true;
            }
            "tools" => {
                tool_schemas.clear();
                parse_tools_value(&mut p, &mut tool_schemas, &mut r.tool_orders);
            }
            "tool_choice" => {
                p.ws();
                if p.peek() == b'"' {
                    if let Some(choice) = p.string() {
                        tool_choice_none = choice == "none";
                    }
                } else { p.skip_value(); }
            }
            "model" => { r.model = p.string().unwrap_or_default(); }
            "max_tokens" | "max_completion_tokens" => {
                r.max_tokens = p.int().unwrap_or(256);
            }
            "temperature" => { r.temperature = p.number().unwrap_or(1.0) as f32; }
            "top_p" => { r.top_p = p.number().unwrap_or(1.0) as f32; }
            "min_p" => { r.min_p = p.number().unwrap_or(0.0) as f32; }
            "top_k" => { r.top_k = p.int().unwrap_or(0); }
            "seed" => { r.seed = p.number().unwrap_or(0.0) as u64; }
            "stream" => { r.stream = p.bool_val().unwrap_or(false); }
            "stream_options" => {
                parse_stream_options(&mut p, &mut r.stream_include_usage);
            }
            "thinking" => {
                parse_thinking_control(&mut p, &mut thinking_enabled);
                got_thinking = true;
            }
            "reasoning_effort" => {
                parse_reasoning_effort(&mut p, &mut reasoning_effort);
            }
            "think" => {
                thinking_enabled = p.bool_val().unwrap_or(true);
                got_thinking = true;
            }
            "stop" => { parse_stop(&mut p, &mut r.stops); }
            _ => { p.skip_value(); }
        }
        p.comma();
    }

    if !got_messages { return None; }

    r.has_tools = !tool_schemas.is_empty() && !tool_choice_none;

    // Model alias handling
    if !got_thinking {
        if r.model == "deepseek-chat" { thinking_enabled = false; }
        if r.model == "deepseek-reasoner" { thinking_enabled = true; }
    }
    r.think_mode = ThinkMode::from_enabled(thinking_enabled, reasoning_effort);

    let active_schemas = if r.has_tools { Some(&tool_schemas as &str) } else { None };
    r.prompt_preserves_reasoning = chat_history_uses_tool_context(&msgs, active_schemas.unwrap_or(""));
    r.prompt_text = render_chat_prompt_text(&msgs, active_schemas.unwrap_or(""), r.think_mode);
    Some(r)
}

fn parse_messages(p: &mut JsonParser, msgs: &mut ChatMessages) -> Option<()> {
    p.ws();
    if p.peek() != b'[' { return None; }
    p.pos += 1;
    loop {
        p.ws();
        if p.peek() == b']' { p.pos += 1; break; }
        if p.peek() != b'{' { return None; }
        p.pos += 1;
        let mut msg = ChatMessage::default();
        loop {
            p.ws();
            if p.peek() == b'}' { p.pos += 1; break; }
            let key = p.key()?;
            p.colon();
            match key.as_str() {
                "role" => { msg.role = p.string().unwrap_or_default(); }
                "content" => { msg.content = p.content().unwrap_or_default(); }
                "reasoning_content" => { msg.reasoning = p.content().unwrap_or_default(); }
                "tool_call_id" => { msg.tool_call_id = p.string().unwrap_or_default(); }
                "tool_calls" => { parse_tool_calls_value(p, &mut msg.calls); }
                _ => { p.skip_value(); }
            }
            p.comma();
        }
        if msg.role.is_empty() { msg.role = "user".to_string(); }
        msgs.v.push(msg);
        p.comma();
    }
    Some(())
}

fn parse_tool_calls_value(p: &mut JsonParser, calls: &mut ToolCalls) {
    p.ws();
    if p.lit("null") { return; }
    if p.peek() != b'[' { return; }
    p.pos += 1;
    loop {
        p.ws();
        if p.peek() == b']' { p.pos += 1; break; }
        if p.peek() != b'{' { return; }
        p.pos += 1;
        let mut tc = ToolCall::default();
        loop {
            p.ws();
            if p.peek() == b'}' { p.pos += 1; break; }
            let key = p.key().unwrap_or_default();
            p.colon();
            match key.as_str() {
                "id" => { tc.id = p.string().unwrap_or_default(); }
                "function" => { parse_function_call(p, &mut tc); }
                "type" => { p.skip_value(); }
                _ => { p.skip_value(); }
            }
            p.comma();
        }
        if !tc.name.is_empty() && !tc.arguments.is_empty() {
            calls.v.push(tc);
        }
        p.comma();
    }
}

fn parse_function_call(p: &mut JsonParser, tc: &mut ToolCall) {
    p.ws();
    if p.peek() != b'{' { return; }
    p.pos += 1;
    loop {
        p.ws();
        if p.peek() == b'}' { p.pos += 1; break; }
        let key = p.key().unwrap_or_default();
        p.colon();
        match key.as_str() {
            "name" => { tc.name = p.string().unwrap_or_default(); }
            "arguments" => {
                p.ws();
                if p.peek() == b'"' { tc.arguments = p.string().unwrap_or_default(); }
                else { tc.arguments = p.raw_value().unwrap_or_default(); }
            }
            _ => { p.skip_value(); }
        }
        p.comma();
    }
}

fn parse_tools_value(p: &mut JsonParser, schemas: &mut String, orders: &mut Vec<ToolSchemaOrder>) {
    p.ws();
    if p.lit("null") { return; }
    if p.peek() != b'[' { return; }
    p.pos += 1;
    loop {
        p.ws();
        if p.peek() == b']' { p.pos += 1; break; }
        let raw = p.raw_value().unwrap_or_default();
        let function_schema = openai_function_schema_from_tool(&raw);
        let schema = function_schema.as_deref().unwrap_or(&raw);
        if !schemas.is_empty() { schemas.push('\n'); }
        schemas.push_str(schema);
        parse_schema_properties(schema, orders);
        p.comma();
    }
}

fn openai_function_schema_from_tool(raw: &str) -> Option<String> {
    let mut p = JsonParser::new(raw);
    p.ws();
    if p.peek() != b'{' { return None; }
    p.pos += 1;
    loop {
        p.ws();
        if p.peek() == b'}' { break; }
        let key = p.key()?;
        p.colon();
        if key == "function" {
            return p.raw_value();
        } else { p.skip_value(); }
        p.comma();
    }
    None
}

fn parse_schema_properties(json: &str, orders: &mut Vec<ToolSchemaOrder>) {
    let mut p = JsonParser::new(json);
    p.ws();
    if p.peek() != b'{' { return; }
    p.pos += 1;
    let mut name = String::new();
    loop {
        p.ws();
        if p.peek() == b'}' { p.pos += 1; break; }
        let key = p.key().unwrap_or_default();
        p.colon();
        match key.as_str() {
            "name" => { name = p.string().unwrap_or_default(); }
            "input_schema" | "parameters" => {
                if let Some(raw) = p.raw_value() {
                    let mut pp = JsonParser::new(&raw);
                    pp.ws();
                    if pp.peek() == b'{' {
                        pp.pos += 1;
                        loop {
                            pp.ws();
                            if pp.peek() == b'}' { pp.pos += 1; break; }
                            let pk = pp.key().unwrap_or_default();
                            pp.colon();
                            if pk == "properties" {
                                if pp.peek() == b'{' {
                                    pp.pos += 1;
                                    loop {
                                        pp.ws();
                                        if pp.peek() == b'}' { pp.pos += 1; break; }
                                        let prop = pp.string().unwrap_or_default();
                                        pp.colon();
                                        pp.skip_value();
                                        let idx = orders.iter().position(|o| o.name == name);
                                        if let Some(idx) = idx {
                                            orders[idx].prop.push(prop);
                                        }
                                        pp.comma();
                                    }
                                }
                            } else { pp.skip_value(); }
                            pp.comma();
                        }
                    }
                }
            }
            _ => { p.skip_value(); }
        }
        p.comma();
    }
    if !name.is_empty() {
        if !orders.iter().any(|o| o.name == name) {
            orders.push(ToolSchemaOrder { name, prop: Vec::new() });
        }
    }
}

fn parse_stream_options(p: &mut JsonParser, include_usage: &mut bool) {
    p.ws();
    if p.peek() != b'{' { p.skip_value(); return; }
    p.pos += 1;
    loop {
        p.ws();
        if p.peek() == b'}' { p.pos += 1; break; }
        let key = p.key().unwrap_or_default();
        p.colon();
        if key == "include_usage" {
            *include_usage = p.bool_val().unwrap_or(false);
        } else { p.skip_value(); }
        p.comma();
    }
}

fn parse_thinking_control(p: &mut JsonParser, enabled: &mut bool) {
    p.ws();
    if p.lit("null") { return; }
    if p.peek() == b't' || p.peek() == b'f' {
        *enabled = p.bool_val().unwrap_or(true);
        return;
    }
    if p.peek() != b'{' { p.skip_value(); return; }
    p.pos += 1;
    loop {
        p.ws();
        if p.peek() == b'}' { p.pos += 1; break; }
        let key = p.key().unwrap_or_default();
        p.colon();
        if key == "type" {
            if let Some(t) = p.string() {
                *enabled = t == "enabled";
            }
        } else { p.skip_value(); }
        p.comma();
    }
}

fn parse_reasoning_effort(p: &mut JsonParser, effort: &mut ThinkMode) {
    p.ws();
    if p.lit("null") { return; }
    if let Some(s) = p.string() {
        match s.as_str() {
            "max" => *effort = ThinkMode::Max,
            _ => *effort = ThinkMode::High,
        }
    }
}

fn parse_stop(p: &mut JsonParser, stops: &mut StopList) {
    p.ws();
    stops.clear();
    if p.peek() == b'"' {
        if let Some(s) = p.string() { stops.push(&s); }
        return;
    }
    if p.peek() != b'[' { p.skip_value(); return; }
    p.pos += 1;
    loop {
        p.ws();
        if p.peek() == b']' { p.pos += 1; break; }
        if let Some(s) = p.string() { stops.push(&s); }
        else { p.skip_value(); }
        p.comma();
    }
}

// ── Request Parsing: Anthropic ────────────────────────────────────────────

fn parse_anthropic_request(body: &str) -> Option<Request> {
    let mut r = Request::new(ReqKind::Chat, 256);
    r.api = ApiStyle::Anthropic;
    let mut p = JsonParser::new(body);
    let mut got_messages = false;
    let mut tool_choice_none = false;
    let mut got_thinking = false;
    let mut thinking_enabled = true;
    let mut reasoning_effort = ThinkMode::High;
    let mut msgs = ChatMessages::default();
    let mut system = String::new();
    let mut tool_schemas = String::new();

    p.ws();
    if p.peek() != b'{' { return None; }
    p.pos += 1;

    loop {
        p.ws();
        if p.peek() == b'}' { p.pos += 1; break; }
        let key = p.key()?;
        p.colon();

        match key.as_str() {
            "messages" => {
                msgs.v.clear();
                parse_anthropic_messages(&mut p, &mut msgs)?;
                got_messages = true;
            }
            "system" => {
                system.clear();
                parse_anthropic_system(&mut p, &mut system);
            }
            "tools" => {
                tool_schemas.clear();
                parse_tools_value(&mut p, &mut tool_schemas, &mut r.tool_orders);
            }
            "tool_choice" => {
                p.ws();
                if p.peek() == b'{' {
                    p.pos += 1;
                    loop {
                        p.ws();
                        if p.peek() == b'}' { p.pos += 1; break; }
                        let ck = p.key().unwrap_or_default();
                        p.colon();
                        if ck == "type" {
                            if let Some(choice) = p.string() {
                                tool_choice_none = choice == "none";
                            }
                        } else { p.skip_value(); }
                        p.comma();
                    }
                } else { p.skip_value(); }
            }
            "model" => { r.model = p.string().unwrap_or_default(); }
            "max_tokens" => { r.max_tokens = p.int().unwrap_or(256); }
            "temperature" => { r.temperature = p.number().unwrap_or(1.0) as f32; }
            "top_p" => { r.top_p = p.number().unwrap_or(1.0) as f32; }
            "top_k" => { r.top_k = p.int().unwrap_or(0); }
            "stream" => { r.stream = p.bool_val().unwrap_or(false); }
            "stop_sequences" => { parse_stop(&mut p, &mut r.stops); }
            "thinking" => {
                parse_thinking_control(&mut p, &mut thinking_enabled);
                got_thinking = true;
            }
            "output_config" => {
                parse_output_config_effort(&mut p, &mut reasoning_effort);
            }
            "reasoning_effort" => {
                parse_reasoning_effort(&mut p, &mut reasoning_effort);
            }
            _ => { p.skip_value(); }
        }
        p.comma();
    }

    if !got_messages { return None; }

    // Prepend system message
    if !system.is_empty() {
        msgs.v.insert(0, ChatMessage {
            role: "system".to_string(),
            content: system,
            ..Default::default()
        });
    }

    r.has_tools = !tool_schemas.is_empty() && !tool_choice_none;
    if !got_thinking {
        if r.model == "deepseek-chat" { thinking_enabled = false; }
        if r.model == "deepseek-reasoner" { thinking_enabled = true; }
    }
    r.think_mode = ThinkMode::from_enabled(thinking_enabled, reasoning_effort);
    let active_schemas = if r.has_tools { Some(&tool_schemas as &str) } else { None };
    r.prompt_text = render_chat_prompt_text(&msgs, active_schemas.unwrap_or(""), r.think_mode);
    Some(r)
}

fn parse_anthropic_messages(p: &mut JsonParser, msgs: &mut ChatMessages) -> Option<()> {
    p.ws();
    if p.peek() != b'[' { return None; }
    p.pos += 1;
    loop {
        p.ws();
        if p.peek() == b']' { p.pos += 1; break; }
        if p.peek() != b'{' { return None; }
        p.pos += 1;
        let mut msg = ChatMessage::default();
        loop {
            p.ws();
            if p.peek() == b'}' { p.pos += 1; break; }
            let key = p.key()?;
            p.colon();
            match key.as_str() {
                "role" => { msg.role = p.string().unwrap_or_default(); }
                "content" => {
                    msg.content.clear();
                    parse_anthropic_content(p, &mut msg);
                }
                _ => { p.skip_value(); }
            }
            p.comma();
        }
        if msg.role.is_empty() { msg.role = "user".to_string(); }
        msgs.v.push(msg);
        p.comma();
    }
    Some(())
}

fn parse_anthropic_content(p: &mut JsonParser, msg: &mut ChatMessage) {
    p.ws();
    if p.peek() == b'"' {
        msg.content = p.string().unwrap_or_default();
        return;
    }
    if p.lit("null") { msg.content = String::new(); return; }
    if p.peek() != b'[' { p.skip_value(); return; }
    p.pos += 1;
    loop {
        p.ws();
        if p.peek() == b']' { p.pos += 1; break; }
        if p.peek() == b'"' {
            if let Some(s) = p.string() { msg.content.push_str(&s); }
        } else if p.peek() == b'{' {
            let role = msg.role.clone();
            parse_anthropic_content_block(p, &role, msg);
        } else { p.skip_value(); }
        p.comma();
    }
}

fn parse_anthropic_content_block(p: &mut JsonParser, role: &str, msg: &mut ChatMessage) {
    p.ws();
    if p.peek() != b'{' { return; }
    p.pos += 1;
    let mut block_type = String::new();
    let mut text = String::new();
    let mut thinking = String::new();
    let mut id = String::new();
    let mut name = String::new();
    let mut input = String::new();
    let mut tool_result = String::new();

    loop {
        p.ws();
        if p.peek() == b'}' { p.pos += 1; break; }
        let key = p.key().unwrap_or_default();
        p.colon();
        match key.as_str() {
            "type" => { block_type = p.string().unwrap_or_default(); }
            "text" => { text = p.content().unwrap_or_default(); }
            "thinking" => { thinking = p.content().unwrap_or_default(); }
            "id" | "tool_use_id" => { id = p.string().unwrap_or_default(); }
            "name" => { name = p.string().unwrap_or_default(); }
            "input" => { input = p.raw_value().unwrap_or_default(); }
            "content" => { tool_result = p.content().unwrap_or_default(); }
            _ => { p.skip_value(); }
        }
        p.comma();
    }

    if block_type == "tool_use" && role == "assistant" {
        msg.calls.v.push(ToolCall {
            id: if id.is_empty() { format!("toolu_{}", msg.calls.v.len()) } else { id },
            name,
            arguments: if input.is_empty() { "{}".to_string() } else { input },
        });
    } else if block_type == "tool_result" {
        let mut b = msg.content.clone();
        b.push_str("<tool_result>");
        b.push_str(&text_escape(&tool_result));
        b.push_str("</tool_result>");
        msg.content = b;
    } else {
        if !text.is_empty() { msg.content.push_str(&text); }
        if !thinking.is_empty() { msg.reasoning.push_str(&thinking); }
    }
}

fn parse_anthropic_system(p: &mut JsonParser, out: &mut String) {
    p.ws();
    if p.peek() == b'"' {
        if let Some(s) = p.string() { out.push_str(&s); }
        return;
    }
    if p.lit("null") { return; }
    if p.peek() != b'[' { p.skip_value(); return; }
    p.pos += 1;
    loop {
        p.ws();
        if p.peek() == b']' { p.pos += 1; break; }
        if p.peek() == b'"' {
            if let Some(s) = p.string() {
                if !out.is_empty() && !out.ends_with('\n') { out.push('\n'); }
                out.push_str(&s);
            }
        } else if p.peek() == b'{' {
            p.pos += 1;
            loop {
                p.ws();
                if p.peek() == b'}' { p.pos += 1; break; }
                let key = p.key().unwrap_or_default();
                p.colon();
                if key == "text" {
                    if let Some(s) = p.string() {
                        if !out.is_empty() && !out.ends_with('\n') { out.push('\n'); }
                        out.push_str(&s);
                    }
                } else { p.skip_value(); }
                p.comma();
            }
        } else { p.skip_value(); }
        p.comma();
    }
}

fn parse_output_config_effort(p: &mut JsonParser, effort: &mut ThinkMode) {
    p.ws();
    if p.lit("null") { return; }
    if p.peek() != b'{' { p.skip_value(); return; }
    p.pos += 1;
    loop {
        p.ws();
        if p.peek() == b'}' { p.pos += 1; break; }
        let key = p.key().unwrap_or_default();
        p.colon();
        if key == "effort" {
            parse_reasoning_effort(p, effort);
        } else { p.skip_value(); }
        p.comma();
    }
}

// ── Request Parsing: Completions ──────────────────────────────────────────

fn parse_completion_request(body: &str) -> Option<Request> {
    let mut r = Request::new(ReqKind::Completion, 256);
    let mut p = JsonParser::new(body);
    let mut prompt = String::new();
    let mut got_thinking = false;
    let mut thinking_enabled = true;
    let mut reasoning_effort = ThinkMode::High;

    p.ws();
    if p.peek() != b'{' { return None; }
    p.pos += 1;
    loop {
        p.ws();
        if p.peek() == b'}' { p.pos += 1; break; }
        let key = p.key()?;
        p.colon();
        match key.as_str() {
            "prompt" => {
                p.ws();
                if p.peek() == b'"' { prompt = p.string().unwrap_or_default(); }
                else { p.skip_value(); }
            }
            "model" => { r.model = p.string().unwrap_or_default(); }
            "max_tokens" => { r.max_tokens = p.int().unwrap_or(256); }
            "temperature" => { r.temperature = p.number().unwrap_or(1.0) as f32; }
            "top_p" => { r.top_p = p.number().unwrap_or(1.0) as f32; }
            "min_p" => { r.min_p = p.number().unwrap_or(0.0) as f32; }
            "top_k" => { r.top_k = p.int().unwrap_or(0); }
            "seed" => { r.seed = p.number().unwrap_or(0.0) as u64; }
            "stream" => { r.stream = p.bool_val().unwrap_or(false); }
            "stream_options" => { parse_stream_options(&mut p, &mut r.stream_include_usage); }
            "thinking" => {
                parse_thinking_control(&mut p, &mut thinking_enabled);
                got_thinking = true;
            }
            "reasoning_effort" => {
                parse_reasoning_effort(&mut p, &mut reasoning_effort);
            }
            "think" => {
                thinking_enabled = p.bool_val().unwrap_or(true);
                got_thinking = true;
            }
            "stop" => { parse_stop(&mut p, &mut r.stops); }
            _ => { p.skip_value(); }
        }
        p.comma();
    }

    if prompt.is_empty() { return None; }
    if !got_thinking {
        if r.model == "deepseek-chat" { thinking_enabled = false; }
        if r.model == "deepseek-reasoner" { thinking_enabled = true; }
    }
    r.think_mode = ThinkMode::from_enabled(thinking_enabled, reasoning_effort);

    let mut rendered = String::from("<｜begin▁of▁sentence｜>");
    if r.think_mode == ThinkMode::Max {
        // ds4_think_max_prefix()
    }
    rendered.push_str("You are a helpful assistant<｜User｜>");
    rendered.push_str(&prompt);
    rendered.push_str("<｜Assistant｜>");
    rendered.push_str(if r.think_mode.enabled() { "<think>" } else { "</think>" });
    r.prompt_text = rendered;
    Some(r)
}

// ── Response Formatting: OpenAI ───────────────────────────────────────────

fn format_openai_response(
    r: &Request,
    text: &str,
    reasoning: &str,
    calls: &ToolCalls,
    finish: &str,
    prompt_tokens: usize,
    completion_tokens: usize,
) -> String {
    let now = now_secs();
    let mut body = format!(
        r#"{{"id":"chatcmpl-{}","object":"chat.completion","created":{},"model":{},"choices":[{{"index":0,"message":{{"role":"assistant","content":{}"#,
        r.model, now, json_escape(&r.model), json_escape(text)
    );
    if !reasoning.is_empty() {
        body.push_str(&format!(",\"reasoning_content\":{}", json_escape(reasoning)));
    }
    if !calls.v.is_empty() {
        body.push_str(",\"tool_calls\":[");
        for (i, tc) in calls.v.iter().enumerate() {
            if i > 0 { body.push(','); }
            let id = if tc.id.is_empty() { format!("call_{}_{}", r.model, i) } else { tc.id.clone() };
            body.push_str(&format!(
                r#"{{"id":{},"type":"function","function":{{"name":{},"arguments":{}}}}}"#,
                json_escape(&id), json_escape(&tc.name), json_escape(&tc.arguments)
            ));
        }
        body.push(']');
    }
    body.push('}');
    body.push_str(&format!(",\"finish_reason\":{}", json_escape(finish)));
    body.push_str("}],\"usage\":{");
    body.push_str(&format!("\"prompt_tokens\":{},\"completion_tokens\":{},\"total_tokens\":{}",
                           prompt_tokens, completion_tokens, prompt_tokens + completion_tokens));
    body.push_str("}}");
    body
}

fn format_openai_completion_response(
    r: &Request,
    text: &str,
    finish: &str,
    prompt_tokens: usize,
    completion_tokens: usize,
) -> String {
    let now = now_secs();
    let mut body = String::new();
    body.push_str(&format!(
        "{{\"id\":\"cmpl-{0}\",\"object\":\"text_completion\",\"created\":{1},\"model\":{2},\"choices\":[{{\"text\":{3},\"index\":0,\"finish_reason\":{4}}}",
        r.model, now, json_escape(&r.model), json_escape(text), json_escape(finish)
    ));
    body.push_str("],\"usage\":{");
    body.push_str(&format!("\"prompt_tokens\":{},\"completion_tokens\":{},\"total_tokens\":{}",
                           prompt_tokens, completion_tokens, prompt_tokens + completion_tokens));
    body.push_str("}}");
    body
}

// ── Response Formatting: Anthropic ────────────────────────────────────────

fn format_anthropic_response(
    r: &Request,
    text: &str,
    reasoning: &str,
    calls: &ToolCalls,
    finish: &str,
    prompt_tokens: usize,
    completion_tokens: usize,
) -> String {
    let stop_reason = match finish {
        "tool_calls" => "tool_use",
        "length" => "max_tokens",
        _ => "end_turn",
    };
    let mut content = String::from("[");
    let mut wrote = false;

    if !reasoning.is_empty() {
        content.push_str(&format!(
            r#"{{"type":"thinking","thinking":{},"signature":"sig_{}"}}"#,
            json_escape(reasoning), reasoning.len()
        ));
        wrote = true;
    }
    if !text.is_empty() {
        if wrote { content.push(','); }
        content.push_str(&format!(r#"{{"type":"text","text":{}}}"#, json_escape(text)));
        wrote = true;
    }
    for (i, tc) in calls.v.iter().enumerate() {
        if wrote { content.push(','); }
        let id = if tc.id.is_empty() { format!("toolu_{}_{}", r.model, i) } else { tc.id.clone() };
        content.push_str(&format!(
            r#"{{"type":"tool_use","id":{},"name":{},"input":{}}}"#,
            json_escape(&id), json_escape(&tc.name), if tc.arguments.is_empty() { "{}".to_string() } else { tc.arguments.clone() }
        ));
        wrote = true;
    }
    if !wrote {
        content.push_str(r#"{"type":"text","text":""}"#);
    }
    content.push(']');

    let mut body = String::new();
    body.push_str(&format!(
        "{{\"id\":\"msg_{}\",\"type\":\"message\",\"role\":\"assistant\",\"model\":{},\"content\":{},\"stop_reason\":\"{}\",\"stop_sequence\":null,\"usage\":{{\"input_tokens\":{},\"output_tokens\":{}",
        r.model, json_escape(&r.model), content, stop_reason, prompt_tokens, completion_tokens
    ));
    body.push_str("}}");
    body
}

// ── Streaming: OpenAI Chat ────────────────────────────────────────────────

fn sse_chat_delta(stream: &mut TcpStream, r: &Request, field: &str, text: &str) {
    let now = now_secs();
    let data = format!(
        r#"{{"id":"chatcmpl-{}","object":"chat.completion.chunk","created":{},"model":{},"choices":[{{"index":0,"delta":{{"{}":{}}},"finish_reason":null}}]}}"#,
        r.model, now, json_escape(&r.model), field, json_escape(text)
    );
    sse_send(stream, &data);
}

fn sse_chat_delta_n(stream: &mut TcpStream, r: &Request, field: &str, text: &str, len: usize) {
    if len == 0 { return; }
    let now = now_secs();
    let t = &text[..len.min(text.len())];
    let data = format!(
        r#"{{"id":"chatcmpl-{}","object":"chat.completion.chunk","created":{},"model":{},"choices":[{{"index":0,"delta":{{"{}":{}}},"finish_reason":null}}]}}"#,
        r.model, now, json_escape(&r.model), field, json_escape(t)
    );
    sse_send(stream, &data);
}

fn sse_chat_tool_call_start(
    stream: &mut TcpStream,
    r: &Request,
    index: usize,
    tool_id: &str,
    name: &str,
) {
    let now = now_secs();
    let data = format!(
        r#"{{"id":"chatcmpl-{}","object":"chat.completion.chunk","created":{},"model":{},"choices":[{{"index":0,"delta":{{"tool_calls":[{{"index":{},"id":{},"type":"function","function":{{"name":{},"arguments":""}}}}]}},"finish_reason":null}}]}}"#,
        r.model, now, json_escape(&r.model), index, json_escape(tool_id), json_escape(name)
    );
    sse_send(stream, &data);
}

fn sse_chat_tool_call_args(
    stream: &mut TcpStream,
    r: &Request,
    index: usize,
    text: &str,
) {
    if text.is_empty() { return; }
    let now = now_secs();
    let data = format!(
        r#"{{"id":"chatcmpl-{}","object":"chat.completion.chunk","created":{},"model":{},"choices":[{{"index":0,"delta":{{"tool_calls":[{{"index":{},"function":{{"arguments":{}}}}}]}},"finish_reason":null}}]}}"#,
        r.model, now, json_escape(&r.model), index, json_escape(text)
    );
    sse_send(stream, &data);
}

fn sse_chat_finish(
    stream: &mut TcpStream,
    r: &Request,
    finish: &str,
    prompt_tokens: usize,
    completion_tokens: usize,
) {
    let now = now_secs();
    let data = format!(
        r#"{{"id":"chatcmpl-{}","object":"chat.completion.chunk","created":{},"model":{},"choices":[{{"index":0,"delta":{{}},"finish_reason":{}}}]}}"#,
        r.model, now, json_escape(&r.model), json_escape(finish)
    );
    sse_send(stream, &data);

    if r.stream_include_usage {
        let usage = format!(
            r#"{{"id":"chatcmpl-{}","object":"chat.completion.chunk","created":{},"model":{},"choices":[],"usage":{{"prompt_tokens":{},"completion_tokens":{},"total_tokens":{}}}}}"#,
            r.model, now, json_escape(&r.model), prompt_tokens, completion_tokens, prompt_tokens + completion_tokens
        );
        sse_send(stream, &usage);
    }

    sse_done(stream);
}

// ── Streaming: Anthropic ──────────────────────────────────────────────────

fn sse_event(stream: &mut TcpStream, event: &str, data: &str) {
    let msg = format!("event: {}\ndata: {}\n\n", event, data);
    let _ = stream.write_all(msg.as_bytes());
    let _ = stream.flush();
}

// ── Text Generation ───────────────────────────────────────────────────────

fn generate_text(
    sess: &mut crate::session::SessionState,
    prompt_tokens: &[i32],
    max_tokens: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    min_p: f32,
    stops: &StopList,
) -> String {
    let vocab_ptr: *const crate::model::Vocab = match sess.vocab.as_ref() {
        Some(v) => v,
        None => return String::new(),
    };

    if let Err(e) = sess.prefill(prompt_tokens) {
        eprintln!("prefill error: {}", e);
        return String::new();
    }

    let mut out = Vec::new();
    let mut text_buf = String::new();

    for _ in 0..max_tokens {
        let token = if temperature < 0.01 {
            let (t, _) = sess.argmax();
            t
        } else {
            sess.sample(temperature, top_k)
        };

        if sess.is_stop_token(token) { break; }
        out.push(token);

        // Check stop sequences
        if !stops.v.is_empty() {
            let decoded = unsafe { crate::tokenizer::token_decode(&*vocab_ptr, &[token]) };
            text_buf.push_str(&decoded);
            if let Some((pos, _)) = stops.find_from(&text_buf, 0) {
                text_buf.truncate(pos);
                // Remove tokens that triggered the stop
                while !out.is_empty() {
                    let tk = out.pop().unwrap();
                    let tk_str = unsafe { crate::tokenizer::token_decode(&*vocab_ptr, &[tk]) };
                    if text_buf.ends_with(&tk_str) {
                        text_buf.truncate(text_buf.len() - tk_str.len());
                    }
                    if text_buf.len() <= pos { break; }
                }
                break;
            }
        }

        if let Err(e) = sess.decode(token) {
            eprintln!("decode error: {}", e);
            break;
        }
    }

    unsafe { crate::tokenizer::token_decode(&*vocab_ptr, &out) }
}

fn count_tokens(sess: &crate::session::SessionState, text: &str) -> usize {
    if let Some(ref vocab) = sess.vocab {
        crate::tokenizer::bpe_tokenize(vocab, text).len()
    } else { 0 }
}

// ── HTTP Handlers ─────────────────────────────────────────────────────────

fn handle_models(stream: &mut TcpStream) {
    let body = r#"{"object":"list","data":[{"id":"deepseek-v4-flash","object":"model","created":1767225600,"owned_by":"ds4.c","name":"DeepSeek V4 Flash","context_length":65536,"top_provider":{"context_length":65536,"max_completion_tokens":32768,"is_moderated":false},"supported_parameters":["tools","tool_choice","max_tokens","temperature","top_p","top_k","min_p","stop","seed","stream","reasoning_effort"]}]}"#;
    send_json(stream, 200, body);
}

fn handle_model(stream: &mut TcpStream) {
    let body = r#"{"id":"deepseek-v4-flash","object":"model","created":1767225600,"owned_by":"ds4.c","name":"DeepSeek V4 Flash","context_length":65536,"top_provider":{"context_length":65536,"max_completion_tokens":32768,"is_moderated":false},"supported_parameters":["tools","tool_choice","max_tokens","temperature","top_p","top_k","min_p","stop","seed","stream","reasoning_effort"]}"#;
    send_json(stream, 200, body);
}

fn handle_chat(stream: &mut TcpStream, body: &str, is_anthropic: bool) {
    let r = if is_anthropic {
        match parse_anthropic_request(body) {
            Some(r) => r,
            None => { send_error(stream, 400, "invalid request"); return; }
        }
    } else {
        match parse_chat_request(body) {
            Some(r) => r,
            None => { send_error(stream, 400, "invalid request"); return; }
        }
    };

    if r.stream {
        handle_chat_stream(stream, r, is_anthropic);
    } else {
        handle_chat_nonstream(stream, r, is_anthropic);
    }
}

fn handle_chat_nonstream(stream: &mut TcpStream, r: Request, is_anthropic: bool) {
    if let Some(session) = crate::session::SESSION.get() {
        if let Ok(mut sess) = session.lock() {
            let prompt_tokens = count_tokens(&sess, &r.prompt_text);
            let tokens = match sess.vocab.as_ref() {
                Some(v) => crate::tokenizer::bpe_tokenize(v, &r.prompt_text),
                None => { send_error(stream, 500, "no vocab"); return; }
            };

            let text = generate_text(
                &mut sess, &tokens, r.max_tokens,
                r.temperature, r.top_k, r.top_p, r.min_p, &r.stops,
            );

            let (content, reasoning, calls) = parse_generated_message(&text);

            let finish = if !calls.v.is_empty() { "tool_calls" } else { "stop" };
            let completion_tokens = count_tokens(&sess, &text);

            let body = if is_anthropic {
                format_anthropic_response(&r, &content, &reasoning, &calls, finish,
                                          prompt_tokens, completion_tokens)
            } else {
                format_openai_response(&r, &content, &reasoning, &calls, finish,
                                       prompt_tokens, completion_tokens)
            };
            send_json(stream, 200, &body);
            return;
        }
    }
    send_error(stream, 500, "no session");
}

fn handle_chat_stream(stream: &mut TcpStream, r: Request, is_anthropic: bool) {
    if let Some(session) = crate::session::SESSION.get() {
        if let Ok(mut sess) = session.lock() {
            let prompt_tokens = count_tokens(&sess, &r.prompt_text);
            let tokens = match sess.vocab.as_ref() {
                Some(v) => crate::tokenizer::bpe_tokenize(v, &r.prompt_text),
                None => { send_error(stream, 500, "no vocab"); return; }
            };

            let vocab_ptr: *const crate::model::Vocab = match sess.vocab.as_ref() {
                Some(v) => v,
                None => { send_error(stream, 500, "no vocab"); return; }
            };

            if let Err(e) = sess.prefill(&tokens) {
                eprintln!("prefill error: {}", e);
                send_error(stream, 500, "prefill failed");
                return;
            }

            start_sse(stream);

            if is_anthropic {
                // Anthropic message_start
                let msg_start = format!(
                    r#"{{"type":"message_start","message":{{"id":"msg_{}","type":"message","role":"assistant","model":{},"content":[],"stop_reason":null,"stop_sequence":null,"usage":{{"input_tokens":{},"output_tokens":0}}}}"#,
                    r.model, json_escape(&r.model), prompt_tokens
                );
                sse_event(stream, "message_start", &msg_start);
            }

            // Streaming generation
            let mut text_buf = String::new();
            let mut reasoning_str = String::new();
            let mut sent_reasoning = false;
            let mut saw_think_open = false;
            let mut saw_think_close = false;
            let mut n = 0usize;
            let mut is_tool_mode = false;
            let mut tool_index = 0usize;
            let mut tool_name = String::new();
            let mut tool_args = String::new();

            for _ in 0..r.max_tokens {
                if STOP_REQUESTED.load(Ordering::Relaxed) { break; }
                let token = sess.argmax().0;
                if sess.is_stop_token(token) { break; }

                let txt = unsafe { crate::tokenizer::token_decode(&*vocab_ptr, &[token]) };
                text_buf.push_str(&txt);
                n += 1;

                // Check for tool call start
                if r.has_tools && !is_tool_mode {
                    if let Some(ts) = find_any_tool_start(&text_buf) {
                        // Emit any remaining text before the tool marker
                        if ts > 0 {
                            let before = &text_buf[..ts];
                            if is_anthropic {
                                sse_event(stream, "content_block_start", &format!(
                                    r#"{{"type":"content_block_start","index":{},"content_block":{{"type":"text","text":""}}}}"#,
                                    if sent_reasoning { 1 } else { 0 }
                                ));
                                sse_event(stream, "content_block_delta", &format!(
                                    r#"{{"type":"content_block_delta","index":{},"delta":{{"type":"text_delta","text":{}}}}}"#,
                                    if sent_reasoning { 1 } else { 0 },
                                    json_escape(before)
                                ));
                                sse_event(stream, "content_block_stop", &format!(
                                    r#"{{"type":"content_block_stop","index":{}}}"#,
                                    if sent_reasoning { 1 } else { 0 }
                                ));
                            } else {
                                sse_chat_delta_n(stream, &r, "content", &text_buf, ts);
                            }
                        }
                        is_tool_mode = true;
                        continue;
                    }
                }

                if is_tool_mode {
                    // Parse DSML tool calls for streaming
                    if let Some(inv) = text_buf.find(DS4_INVOKE_START)
                        .or_else(|| text_buf.find("<invoke"))
                    {
                        let after_inv = &text_buf[inv..];
                        if let Some(close) = after_inv.find('>') {
                            let tag = &after_inv[..=close];
                            if let Some(name) = dsml_attr(tag, "name") {
                                if name != tool_name {
                                    // New tool invoke
                                    if !is_anthropic {
                                        let tid = format!("call_{}_{}", r.model, tool_index);
                                        sse_chat_tool_call_start(stream, &r, tool_index, &tid, &name);
                                    }
                                    tool_name = name;
                                }
                            }
                        }
                        // Parse DSML parameters and emit as JSON arguments
                        let mut pos = 0usize;
                        while let Some(ps) = text_buf[pos..].find(DS4_PARAM_START)
                            .or_else(|| text_buf[pos..].find("<parameter"))
                        {
                            let pbody = &text_buf[pos + ps..];
                            let close = pbody.find('>').unwrap_or(0);
                            let pattrs = &pbody[..close];
                            let (pname, is_string) = parse_dsml_param(pattrs);
                            let pval_start = close + 1;
                            let pval_end = pbody[pval_start..].find(DS4_PARAM_END)
                                .or_else(|| pbody[pval_start..].find("</parameter>"))
                                .unwrap_or(0);
                            let pval = &pbody[pval_start..pval_start + pval_end];

                            if pval_start + pval_end < pbody.len() {
                                // Parameter complete - emit as JSON fragment
                                let mut frag = if tool_args.is_empty() { String::from("{") } else { String::from(", ") };
                                if is_string {
                                    frag.push_str(&format!("{}: {}", json_escape(&pname), json_escape(pval)));
                                } else {
                                    frag.push_str(&format!("{}: {}", json_escape(&pname), pval));
                                }
                                if !is_anthropic {
                                    sse_chat_tool_call_args(stream, &r, tool_index, &frag);
                                }
                                tool_args.push_str(&frag);
                                pos += ps + pval_start + pval_end +
                                    if pbody[pval_start + pval_end..].starts_with(DS4_PARAM_END) { DS4_PARAM_END.len() }
                                    else { "</parameter>".len() };
                            } else { break; }
                        }
                    }

                    // Check if tool block ends
                    if text_buf.contains(DS4_TOOL_CALLS_END) || text_buf.contains("</tool_calls>") {
                        if !is_anthropic && !tool_args.is_empty() {
                            sse_chat_tool_call_args(stream, &r, tool_index, "}");
                        }
                        is_tool_mode = false;
                    }
                } else {
                    // Handle think/reasoning for non-tool text
                    if r.think_mode.enabled() && !sent_reasoning {
                        if !saw_think_open && text_buf.contains("<think>") {
                            saw_think_open = true;
                            let pos = text_buf.find("<think>").unwrap();
                            // Emit any text before <think> as content
                            if pos > 0 {
                                if is_anthropic {
                                    sse_event(stream, "content_block_start", &format!(
                                        r#"{{"type":"content_block_start","index":0,"content_block":{{"type":"text","text":""}}}}"#
                                    ));
                                    sse_event(stream, "content_block_delta", &format!(
                                        r#"{{"type":"content_block_delta","index":0,"delta":{{"type":"text_delta","text":{}}}}}"#,
                                        json_escape(&text_buf[..pos])
                                    ));
                                    sse_event(stream, "content_block_stop", &format!(
                                        r#"{{"type":"content_block_stop","index":0}}"#
                                    ));
                                } else {
                                    sse_chat_delta_n(stream, &r, "content", &text_buf, pos);
                                }
                            }
                            reasoning_str.clear();
                            continue;
                        }
                        if saw_think_open {
                            let close_pos = text_buf.find("</think>");
                            if let Some(cp) = close_pos {
                                let think_content = &text_buf["<think>".len()..cp];
                                if is_anthropic {
                                    sse_event(stream, "content_block_start", &format!(
                                        r#"{{"type":"content_block_start","index":0,"content_block":{{"type":"thinking","thinking":"","signature":""}}}}"#
                                    ));
                                    if !think_content.is_empty() {
                                        sse_event(stream, "content_block_delta", &format!(
                                            r#"{{"type":"content_block_delta","index":0,"delta":{{"type":"thinking_delta","thinking":{}}}}}"#,
                                            json_escape(think_content)
                                        ));
                                    }
                                    sse_event(stream, "content_block_delta", &format!(
                                        r#"{{"type":"content_block_delta","index":0,"delta":{{"type":"signature_delta","signature":"sig_{}"}}}}"#,
                                        think_content.len()
                                    ));
                                    sse_event(stream, "content_block_stop", &format!(
                                        r#"{{"type":"content_block_stop","index":0}}"#
                                    ));
                                } else {
                                    if !think_content.is_empty() {
                                        sse_chat_delta(stream, &r, "reasoning_content", think_content);
                                    }
                                }
                                reasoning_str = think_content.to_string();
                                saw_think_close = true;
                                sent_reasoning = true;
                                // Emit remaining text after </think>
                                let after_think = &text_buf[cp + 7..];
                                if !after_think.is_empty() {
                                    if is_anthropic {
                                        sse_event(stream, "content_block_start", &format!(
                                            r#"{{"type":"content_block_start","index":1,"content_block":{{"type":"text","text":""}}}}"#
                                        ));
                                        sse_event(stream, "content_block_delta", &format!(
                                            r#"{{"type":"content_block_delta","index":1,"delta":{{"type":"text_delta","text":{}}}}}"#,
                                            json_escape(after_think)
                                        ));
                                    } else {
                                        sse_chat_delta_n(stream, &r, "content", after_think, after_think.len());
                                    }
                                }
                                continue;
                            } else {
                                // Partial think content
                                let start = "<think>".len();
                                if text_buf.len() > start {
                                    let partial = &text_buf[start..];
                                    // Hold last 7 bytes for </think> split
                                    let safe = if partial.len() > 7 { partial.len() - 7 } else { 0 };
                                    if safe > 0 {
                                        let to_emit = &partial[..safe];
                                        if is_anthropic {
                                            // Anthropic thinking blocks stream deltas
                                        } else {
                                            sse_chat_delta_n(stream, &r, "reasoning_content", to_emit, safe);
                                        }
                                    }
                                }
                            }
                            continue;
                        }
                    }

                    // Normal content streaming
                    if !is_tool_mode {
                        // Don't emit <think> markers to client
                        let emit = text_buf.replace("<think>", "").replace("</think>", "");
                        if !emit.is_empty() && r.has_tools {
                            // Hold last bytes that could be a tool marker start
                            let safe = emit.len().saturating_sub(80);
                            let to_emit = if safe > 0 { &emit[..safe] } else { "" };
                            if !to_emit.is_empty() {
                                if is_anthropic {
                                    // Already handled above via content block events
                                } else {
                                    sse_chat_delta_n(stream, &r, "content", to_emit, to_emit.len());
                                }
                            }
                        } else if !emit.is_empty() && !r.has_tools {
                            if is_anthropic {
                                // Content already emitted via block events
                            } else {
                                sse_chat_delta_n(stream, &r, "content", &emit, emit.len());
                            }
                        }
                    }
                }

                if let Err(e) = sess.decode(token) {
                    eprintln!("decode error: {}", e);
                    break;
                }
            }

            // Parse final message for tool calls
            let (content, _, calls) = parse_generated_message(&text_buf);
            let finish = if !calls.v.is_empty() { "tool_calls" } else { "stop" };
            let completion_tokens = count_tokens(&sess, &text_buf);

            if is_anthropic {
                // Emit tool use blocks
                for (i, tc) in calls.v.iter().enumerate() {
                    let idx = if sent_reasoning { 1 + i } else { i };
                    let tid = format!("toolu_{}_{}", r.model, i);
                    sse_event(stream, "content_block_start", &format!(
                        r#"{{"type":"content_block_start","index":{},"content_block":{{"type":"tool_use","id":{},"name":{},"input":{{}}}}}}"#,
                        idx, json_escape(&tid), json_escape(&tc.name)
                    ));
                    if !tc.arguments.is_empty() {
                        sse_event(stream, "content_block_delta", &format!(
                            r#"{{"type":"content_block_delta","index":{},"delta":{{"type":"input_json_delta","partial_json":{}}}}}"#,
                            idx, &tc.arguments
                        ));
                    }
                    sse_event(stream, "content_block_stop", &format!(
                        r#"{{"type":"content_block_stop","index":{}}}"#, idx
                    ));
                }
                let stop_reason = if !calls.v.is_empty() { "tool_use" } else { "end_turn" };
                sse_event(stream, "message_delta", &format!(
                    r#"{{"type":"message_delta","delta":{{"stop_reason":"{}","stop_sequence":null}},"usage":{{"output_tokens":{}}}}}"#,
                    stop_reason, completion_tokens
                ));
                sse_event(stream, "message_stop", r#"{"type":"message_stop"}"#);
            } else {
                // Emit final tool calls in the finish chunk for OpenAI
                let now = now_secs();
                let mut final_data = format!(
                    r#"{{"id":"chatcmpl-{}","object":"chat.completion.chunk","created":{},"model":{},"choices":[{{"index":0,"delta":{{}}"#,
                    r.model, now, json_escape(&r.model)
                );

                // If we have tool calls but didn't stream them (post-gen parsing), emit them now
                if !calls.v.is_empty() {
                    final_data.push_str(",\"tool_calls\":[");
                    for (i, tc) in calls.v.iter().enumerate() {
                        if i > 0 { final_data.push(','); }
                        let tid = format!("call_{}_{}", r.model, i);
                        final_data.push_str(&format!(
                            r#"{{"index":{},"id":{},"type":"function","function":{{"name":{},"arguments":{}}}}}"#,
                            i, json_escape(&tid), json_escape(&tc.name), json_escape(&tc.arguments)
                        ));
                    }
                    final_data.push(']');
                }

                final_data.push_str(&format!(r#","finish_reason":{}}}]}}"#, json_escape(finish)));
                sse_send(stream, &final_data);
                sse_chat_finish(stream, &r, finish, prompt_tokens, completion_tokens);
            }
            return;
        }
    }
    send_error(stream, 500, "no session");
}

fn handle_completions(stream: &mut TcpStream, body: &str) {
    let r = match parse_completion_request(body) {
        Some(r) => r,
        None => { send_error(stream, 400, "invalid request"); return; }
    };

    if r.stream {
        handle_completions_stream(stream, r);
    } else {
        handle_completions_nonstream(stream, r);
    }
}

fn handle_completions_nonstream(stream: &mut TcpStream, r: Request) {
    if let Some(session) = crate::session::SESSION.get() {
        if let Ok(mut sess) = session.lock() {
            let prompt_tokens = count_tokens(&sess, &r.prompt_text);
            let tokens = match sess.vocab.as_ref() {
                Some(v) => crate::tokenizer::bpe_tokenize(v, &r.prompt_text),
                None => { send_error(stream, 500, "no vocab"); return; }
            };

            let text = generate_text(
                &mut sess, &tokens, r.max_tokens,
                r.temperature, r.top_k, r.top_p, r.min_p, &r.stops,
            );

            let completion_tokens = count_tokens(&sess, &text);
            let body = format_openai_completion_response(&r, &text, "stop",
                                                          prompt_tokens, completion_tokens);
            send_json(stream, 200, &body);
            return;
        }
    }
    send_error(stream, 500, "no session");
}

fn handle_completions_stream(stream: &mut TcpStream, r: Request) {
    if let Some(session) = crate::session::SESSION.get() {
        if let Ok(mut sess) = session.lock() {
            let tokens = match sess.vocab.as_ref() {
                Some(v) => crate::tokenizer::bpe_tokenize(v, &r.prompt_text),
                None => { send_error(stream, 500, "no vocab"); return; }
            };

            let vocab_ptr: *const crate::model::Vocab = match sess.vocab.as_ref() {
                Some(v) => v,
                None => { send_error(stream, 500, "no vocab"); return; }
            };

            if let Err(e) = sess.prefill(&tokens) {
                eprintln!("prefill error: {}", e);
                send_error(stream, 500, "prefill failed");
                return;
            }

            start_sse(stream);
            let mut n = 0usize;

            for _ in 0..r.max_tokens {
                if STOP_REQUESTED.load(Ordering::Relaxed) { break; }
                let token = sess.argmax().0;
                if sess.is_stop_token(token) { break; }

                let txt = unsafe { crate::tokenizer::token_decode(&*vocab_ptr, &[token]) };
                let now = now_secs();
                let json = format!(
                    r#"{{"id":"cmpl-{}","object":"text_completion","created":{},"model":{},"choices":[{{"text":{},"index":0,"finish_reason":null}}]}}"#,
                    r.model, now, json_escape(&r.model), json_escape(&txt)
                );
                sse_send(stream, &json);
                n += 1;

                if let Err(e) = sess.decode(token) {
                    eprintln!("decode error: {}", e);
                    break;
                }
            }

            let now = now_secs();
            sse_send(stream, &format!(
                r#"{{"id":"cmpl-{}","object":"text_completion","created":{},"model":{},"choices":[{{"text":"","index":0,"finish_reason":"stop"}}]}}"#,
                r.model, now, json_escape(&r.model)
            ));
            sse_done(stream);
            return;
        }
    }
    send_error(stream, 500, "no session");
}

// ── Client Handler ────────────────────────────────────────────────────────

fn handle_client(mut stream: TcpStream) {
    let mut buf = vec![0u8; 131072];
    let mut offset = 0;
    loop {
        match stream.read(&mut buf[offset..]) {
            Ok(0) => break,
            Ok(n) => offset += n,
            Err(e) => {
                eprintln!("read error: {}", e);
                return;
            }
        }

        let request_str = match std::str::from_utf8(&buf[..offset]) {
            Ok(s) => s,
            Err(_) => { send_error(&mut stream, 400, "invalid utf8"); return; }
        };

        if let Some(pos) = request_str.find("\r\n\r\n") {
            let header = &request_str[..pos + 4];
            let body = &request_str[pos + 4..];
            let lines: Vec<&str> = header.lines().collect();
            if lines.is_empty() {
                send_error(&mut stream, 400, "Bad Request");
                return;
            }
            let parts: Vec<&str> = lines[0].split_whitespace().collect();
            if parts.len() < 2 {
                send_error(&mut stream, 400, "Bad Request");
                return;
            }
            let (method, path) = (parts[0], parts[1]);

            match (method, path) {
                ("GET", "/v1/models") => handle_models(&mut stream),
                ("GET", "/v1/models/deepseek-v4-flash") => handle_model(&mut stream),
                ("POST", "/v1/chat/completions") => handle_chat(&mut stream, body, false),
                ("POST", "/v1/messages") => handle_chat(&mut stream, body, true),
                ("POST", "/v1/completions") => handle_completions(&mut stream, body),
                _ => send_error(&mut stream, 404, "Not Found"),
            }
            return;
        }

        if offset >= buf.len() {
            send_error(&mut stream, 413, "Too Large");
            return;
        }
    }
}

// ── Public Entry Point ────────────────────────────────────────────────────

pub fn serve(port: u16) -> Result<(), Box<dyn std::error::Error>> {
    let addr = format!("0.0.0.0:{}", port);
    let listener = TcpListener::bind(&addr)?;
    eprintln!("ds4-server: listening on http://{}", addr);
    eprintln!("ds4-server: OpenAI-compatible at /v1/chat/completions, /v1/completions");
    eprintln!("ds4-server: Anthropic-compatible at /v1/messages");

    // Signal handler thread - sets the shutdown flag on stdin EOF (Ctrl+D)
    // or when the process receives SIGINT/SIGTERM.
    let running = Arc::new(AtomicBool::new(true));
    {
        let r = running.clone();
        std::thread::spawn(move || {
            let mut line = String::new();
            let _ = std::io::stdin().read_line(&mut line);
            r.store(false, Ordering::Relaxed);
            STOP_REQUESTED.store(true, Ordering::Relaxed);
        });
    }

    listener.set_nonblocking(true)?;
    use std::time::Duration;

    while running.load(Ordering::Relaxed) && !STOP_REQUESTED.load(Ordering::Relaxed) {
        match listener.accept() {
            Ok((stream, _)) => {
                handle_client(stream);
            }
            Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                std::thread::sleep(Duration::from_millis(100));
                continue;
            }
            Err(e) => {
                if running.load(Ordering::Relaxed) {
                    eprintln!("accept error: {}", e);
                }
                break;
            }
        }
    }

    eprintln!("ds4-server: shutting down");
    Ok(())
}
