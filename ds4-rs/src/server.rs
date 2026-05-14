use std::net::{TcpListener, TcpStream};
use std::io::{Write, Read};

fn handle_client(mut stream: TcpStream) {
    let mut buf = [0u8; 65536];
    let mut offset = 0;
    loop {
        let n = stream.read(&mut buf[offset..]).unwrap();
        if n == 0 { break; }
        offset += n;
        let request_str = std::str::from_utf8(&buf[..offset]).unwrap_or("");
        if let Some(pos) = request_str.find("\r\n\r\n") {
            let header = &request_str[..pos + 4];
            let body = &request_str[pos + 4..];
            let lines: Vec<&str> = header.lines().collect();
            if lines.is_empty() || lines[0].split_whitespace().count() < 2 {
                send_error(&mut stream, 400, "Bad Request"); return;
            }
            let parts: Vec<&str> = lines[0].split_whitespace().collect();
            let (method, path) = (parts[0], parts[1]);
            let is_stream = body.contains("\"stream\":true") || body.contains("\"stream\": true");
            match (method, path) {
                ("GET", "/health") => send_health(&mut stream),
                ("GET", "/v1/models") => send_models(&mut stream),
                ("POST", "/v1/chat/completions") => {
                    if is_stream { handle_chat_stream(&mut stream, body); }
                    else { handle_chat(&mut stream, body); }
                }
                ("POST", "/v1/completions") => {
                    if is_stream { handle_completions_stream(&mut stream, body); }
                    else { handle_completions(&mut stream, body); }
                }
                ("POST", "/v1/tokenize") => handle_tokenize(&mut stream, body),
                ("GET", "/v1/cache/save") => handle_cache_save(&mut stream),
                ("POST", "/v1/cache/load") => handle_cache_load(&mut stream, body),
                _ => send_error(&mut stream, 404, "Not Found"),
            }
            return;
        }
        if offset >= buf.len() { send_error(&mut stream, 413, "Too Large"); return; }
    }
}

fn send_models(s: &mut TcpStream) { send_json(s, 200, r#"{"data":[{"id":"ds4-flash","object":"model"}]}"#); }
fn send_health(s: &mut TcpStream) { send_json(s, 200, r#"{"status":"ok"}"#); }
fn send_error(s: &mut TcpStream, code: u16, msg: &str) {
    send_json(s, code, &format!(r#"{{"error":"{}"}}"#, msg));
}

fn send_json(s: &mut TcpStream, code: u16, body: &str) {
    let resp = format!(
        "HTTP/1.1 {} {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        code, if code == 200 { "OK" } else { "Error" }, body.len(), body);
    let _ = s.write_all(resp.as_bytes());
}

fn start_sse(s: &mut TcpStream) {
    let hdr = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: keep-alive\r\n\r\n";
    let _ = s.write_all(hdr.as_bytes());
    let _ = s.flush();
}

fn sse_send(s: &mut TcpStream, data: &str) {
    let msg = format!("data: {}\n\n", data);
    let _ = s.write_all(msg.as_bytes());
    let _ = s.flush();
}

fn sse_done(s: &mut TcpStream) {
    let _ = s.write_all(b"data: [DONE]\n\n");
    let _ = s.flush();
}

fn parse_json_str(body: &str, key: &str) -> Option<String> {
    let search = format!("\"{}\"", key);
    if let Some(start) = body.find(&search) {
        let after = &body[start + search.len()..];
        if let Some(colon) = after.find(':') {
            let val = after[colon + 1..].trim_start();
            if val.starts_with('"') {
                let inner = &val[1..];
                let end = inner.find('"')?;
                return Some(inner[..end].to_string());
            }
            let end = val.find(|c| c == ',' || c == '}' || c == '\n').unwrap_or(val.len());
            return Some(val[..end].trim().to_string());
        }
    }
    None
}

fn parse_json_int(body: &str, key: &str) -> Option<usize> {
    parse_json_str(body, key).and_then(|s| s.parse().ok())
}

fn parse_json_f32(body: &str, key: &str) -> Option<f32> {
    parse_json_str(body, key).and_then(|s| s.parse().ok())
}

// DeepSeek chat template: <｜User｜>prompt<｜Assistant｜>
const DSML_TOOL_CALLS_START: &str = "<｜DSML｜tool_calls>";
#[allow(dead_code)]
const DSML_TOOL_CALLS_END: &str = "</｜DSML｜tool_calls>";
const DSML_INVOKE_START: &str = "<｜DSML｜invoke";
const DSML_INVOKE_END: &str = "</｜DSML｜invoke>";
const DSML_PARAM_START: &str = "<｜DSML｜parameter";
const DSML_PARAM_END: &str = "</｜DSML｜parameter>";

struct ToolCall {
    name: String,
    arguments: String,
}

fn parse_dsml_param_name(s: &str) -> Option<(&str, bool)> {
    // Parse: name="..." string="true|false"
    let name_start = s.find("name=\"")?;
    let after_name = &s[name_start + 6..];
    let name_end = after_name.find('"')?;
    let name = &after_name[..name_end];

    let string_attr = if let Some(ss) = s.find("string=\"") {
        let after_ss = &s[ss + 8..];
        let se = after_ss.find('"')?;
        &after_ss[..se]
    } else {
        "true"
    };
    Some((name, string_attr == "true"))
}

#[allow(dead_code)]
fn dsml_extract_value(body: &str, tag_start: &str, tag_end: &str) -> Option<String> {
    let s = body.find(tag_start)?;
    let after = &body[s + tag_start.len()..];
    // Skip attributes for start tag
    let content_start = if tag_start.ends_with('>') { 0 }
        else { after.find('>')? + 1 };
    let content = &after[content_start..];
    let e = content.find(tag_end)?;
    Some(content[..e].to_string())
}

fn parse_dsml_tool_calls(text: &str) -> Vec<ToolCall> {
    let mut calls = Vec::new();
    let mut pos = 0;
    while let Some(start) = text[pos..].find(DSML_INVOKE_START) {
        let inv = &text[pos + start..];
        // Extract name attribute
        let after_open = &inv[DSML_INVOKE_START.len()..];
        let close = after_open.find('>').unwrap_or(0);
        let attrs = &after_open[..close];
        let name = attrs.split("name=\"")
            .nth(1)
            .and_then(|s| s.split('"').next())
            .unwrap_or("unknown");

        let content_start = close + 1;
        let rest = &after_open[content_start..];
        let end = rest.find(DSML_INVOKE_END).unwrap_or(rest.len());
        let body = &rest[..end];

        // Parse parameters
        let mut args = String::from("{");
        let mut first = true;
        let mut bp = 0;
        while let Some(ps) = body[bp..].find(DSML_PARAM_START) {
            let pbody = &body[bp + ps..];
            let close2 = pbody.find('>').unwrap_or(0);
            let pattrs = &pbody[DSML_PARAM_START.len()..close2];
            let (pname, is_string) = parse_dsml_param_name(pattrs).unwrap_or(("", true));
            let pval_start = close2 + 1;
            let pval_end = pbody[pval_start..].find(DSML_PARAM_END).unwrap_or(0);
            let pval = &pbody[pval_start..pval_start + pval_end];

            if !first { args.push_str(", "); }
            first = false;
            args.push_str(&format!("\"{}\": {}", pname,
                if is_string { format!("\"{}\"", pval) } else { pval.to_string() }));

            bp += ps + pval_start + pval_end + DSML_PARAM_END.len();
        }
        args.push('}');

        calls.push(ToolCall { name: name.to_string(), arguments: args });
        pos += start + DSML_INVOKE_START.len() + content_start + end + DSML_INVOKE_END.len();
    }
    calls
}

fn has_dsml_tool_call(text: &str) -> bool {
    text.contains(DSML_TOOL_CALLS_START)
}

fn tool_system_prompt(schemas: &str) -> String {
    format!(
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
        For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string=\"false\"`.\n\n\
        ### Available Tool Schemas\n\n{}\n\n\
        You MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls. \
        Use the exact parameter names from the schemas.",
        schemas)
}

#[allow(dead_code)]
fn apply_chat_template(prompt: &str) -> String {
    // If it already contains special tokens, use as-is
    if prompt.contains("｜") || prompt.contains("<｜") || prompt.contains("User") {
        return prompt.to_string();
    }
    format!("<｜User｜>{}\n<｜Assistant｜>", prompt)
}

fn extract_tools_schemas(body: &str) -> Option<String> {
    // Find the "tools" array in the JSON body and extract schema descriptions
    let tools_start = body.find("\"tools\"")?;
    let colon = body[tools_start..].find(':')?;
    let arr_start = body[tools_start + colon..].find('[')?;
    let after_arr = &body[tools_start + colon + arr_start + 1..];

    // Find the matching closing bracket (simple nesting count)
    let mut depth = 1;
    let mut end = 0;
    for (i, c) in after_arr.char_indices() {
        if c == '[' { depth += 1; }
        else if c == ']' { depth -= 1; if depth == 0 { end = i; break; } }
    }
    if end == 0 { return None; }
    let tools_json = &after_arr[..end+1];

    // Try to extract just the function schemas
    let schemas = if tools_json.contains("\"function\"") {
        Some(format!("{}\n", tools_json))
    } else {
        Some(format!("{}\n", tools_json))
    };
    schemas
}

fn apply_chat_template_with_tools(prompt: &str, tools_schemas: Option<&str>) -> String {
    let mut result = String::new();
    if let Some(schemas) = tools_schemas {
        result.push_str(&tool_system_prompt(schemas));
        result.push_str("\n\n");
    }
    result.push_str(&format!("<｜User｜>{}\n<｜Assistant｜>", prompt));
    result
}

fn handle_chat(stream: &mut TcpStream, body: &str) {
    let prompt = parse_json_str(body, "prompt")
        .or_else(|| parse_json_str(body, "content"))
        .unwrap_or_default();
    let max_tokens = parse_json_int(body, "max_tokens").unwrap_or(256);
    let _temp = parse_json_f32(body, "temperature").unwrap_or(0.7);
    let _top_k = parse_json_int(body, "top_k").unwrap_or(40);
    let logprobs = parse_json_int(body, "logprobs").unwrap_or(0);
    let tools_schemas = extract_tools_schemas(body);
    let stop_str = parse_json_str(body, "stop").unwrap_or_default();
    let stop_seqs: Vec<String> = if stop_str.is_empty() { vec![] }
        else { stop_str.split(',').map(|s| s.trim().trim_matches('"').to_string()).collect() };

    if let Some(session) = crate::SESSION.get() {
        if let Ok(mut sess) = session.lock() {
            let full_prompt = apply_chat_template_with_tools(&prompt, tools_schemas.as_deref());
            let tokens = match sess.vocab.as_ref() {
                Some(v) => crate::tokenizer::bpe_tokenize(v, &full_prompt),
                None => { send_json(stream, 200, r#"{"choices":[{"message":{"content":"no vocab"}}]}"#); return; }
            };
            let result = sess.generate_with_stops(&tokens, max_tokens, &stop_seqs, _temp, _top_k);
            let text = sess.vocab.as_ref()
                .map(|v| crate::tokenizer::token_decode(v, &result))
                .unwrap_or_default();

            // Check for tool calls in the output
            if has_dsml_tool_call(&text) {
                let tool_calls = parse_dsml_tool_calls(&text);
                let tc_json: String = tool_calls.iter().enumerate().map(|(i, tc)| {
                    format!(r#"{{"index":{},"id":"call_{}","type":"function","function":{{"name":"{}","arguments":"{}"}}}}"#,
                        i, i, tc.name, tc.arguments.replace('"', "\\\""))
                }).collect::<Vec<_>>().join(",");

                let body = format!(r#"{{"choices":[{{"message":{{"role":"assistant","content":"","tool_calls":[{}]}}}}]}}"#, tc_json);
                send_json(stream, 200, &body);
            } else if logprobs > 0 {
                let lps = sess.top_logprobs(logprobs);
                let lp_json: String = lps.iter().map(|(id, p)| {
                    format!(r#"{{"token":{},"logprob":{}}}"#, id, p)
                }).collect::<Vec<_>>().join(",");
                let body = format!(r#"{{"choices":[{{"message":{{"content":"{}"}},"logprobs":{{"top_logprobs":[{}]}}}}]}}"#,
                    text.replace('"', "\\\"").replace('\n', "\\n"), lp_json);
                send_json(stream, 200, &body);
            } else {
                let body = format!(r#"{{"choices":[{{"message":{{"content":"{}"}}}}]}}"#,
                    text.replace('"', "\\\"").replace('\n', "\\n"));
                send_json(stream, 200, &body);
            }
            return;
        }
    }
    send_json(stream, 200, r#"{"choices":[{"message":{"content":"Hello from DS4-Rust!"}}]}"#);
}

fn handle_chat_stream(s: &mut TcpStream, body: &str) {
    let prompt = parse_json_str(body, "prompt")
        .or_else(|| parse_json_str(body, "content"))
        .unwrap_or_default();
    let max_tokens = parse_json_int(body, "max_tokens").unwrap_or(256);
    let _temp = parse_json_f32(body, "temperature").unwrap_or(0.7);
    let _top_k = parse_json_int(body, "top_k").unwrap_or(40);
    let tools_schemas = extract_tools_schemas(body);

    if let Some(session) = crate::SESSION.get() {
        if let Ok(mut sess) = session.lock() {
            let full_prompt = apply_chat_template_with_tools(&prompt, tools_schemas.as_deref());
            let vocab_ptr: *const crate::model::Vocab = match sess.vocab.as_ref() {
                Some(v) => v as *const crate::model::Vocab,
                None => { send_json(s, 200, r#"{"choices":[{"message":{"content":"no vocab"}}]}"#); return; }
            };
            let tokens = unsafe { crate::tokenizer::bpe_tokenize(&*vocab_ptr, &full_prompt) };
            start_sse(s);
            let mut n = 0usize;
            sess.generate_stream(&tokens, max_tokens, |token| {
                let txt = unsafe { crate::tokenizer::token_decode(&*vocab_ptr, &[token]) };
                let json = format!(r#"{{"choices":[{{"index":{},"delta":{{"content":"{}"}}}}]}}"#,
                    n, txt.replace('"', "\\\"").replace('\n', "\\n"));
                sse_send(s, &json);
                n += 1;
            });
            sse_send(s, r#"{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#);
            sse_done(s);
            return;
        }
    }
    send_json(s, 200, r#"{"choices":[{"message":{"content":"Hello from DS4-Rust!"}}]}"#);
}

fn handle_completions(stream: &mut TcpStream, body: &str) {
    let prompt = parse_json_str(body, "prompt").unwrap_or_default();
    let max_tokens = parse_json_int(body, "max_tokens").unwrap_or(256);
    let _temp = parse_json_f32(body, "temperature").unwrap_or(0.7);
    let _top_k = parse_json_int(body, "top_k").unwrap_or(40);

    if let Some(session) = crate::SESSION.get() {
        if let Ok(mut sess) = session.lock() {
            let tokens = match sess.vocab.as_ref() {
                Some(vocab) => crate::tokenizer::bpe_tokenize(vocab, &prompt),
                None => vec![],
            };
            if !tokens.is_empty() {
                let result = sess.generate_with_stops(&tokens, max_tokens, &[], _temp, _top_k);
                let text = crate::tokenizer::token_decode(
                    sess.vocab.as_ref().unwrap(), &result);
                let body = format!(r#"{{"choices":[{{"text":"{}"}}]}}"#,
                    text.replace('"', "\\\"").replace('\n', "\\n"));
                send_json(stream, 200, &body);
                return;
            }
        }
    }
    send_json(stream, 200, r#"{"choices":[{"text":"Hello from DS4-Rust!"}]}"#);
}

fn handle_completions_stream(s: &mut TcpStream, body: &str) {
    let prompt = parse_json_str(body, "prompt").unwrap_or_default();
    let max_tokens = parse_json_int(body, "max_tokens").unwrap_or(256);
    let _temp = parse_json_f32(body, "temperature").unwrap_or(0.7);
    let _top_k = parse_json_int(body, "top_k").unwrap_or(40);

    if let Some(session) = crate::SESSION.get() {
        if let Ok(mut sess) = session.lock() {
            let vocab_ptr: *const crate::model::Vocab = match sess.vocab.as_ref() {
                Some(v) => v as *const crate::model::Vocab,
                None => { send_json(s, 200, r#"{"choices":[{"text":"no vocab"}]}"#); return; }
            };
            let tokens = unsafe { crate::tokenizer::bpe_tokenize(&*vocab_ptr, &prompt) };
            start_sse(s);
            let mut n = 0usize;
            sess.generate_stream(&tokens, max_tokens, |token| {
                let txt = unsafe { crate::tokenizer::token_decode(&*vocab_ptr, &[token]) };
                let json = format!(r#"{{"choices":[{{"index":{},"text":"{}"}}]}}"#,
                    n, txt.replace('"', "\\\"").replace('\n', "\\n"));
                sse_send(s, &json);
                n += 1;
            });
            sse_send(s, r#"{"choices":[{"index":0,"text":"","finish_reason":"stop"}]}"#);
            sse_done(s);
            return;
        }
    }
    send_json(s, 200, r#"{"choices":[{"text":"Hello from DS4-Rust!"}]}"#);
}

fn handle_cache_save(s: &mut TcpStream) {
    if let Some(session) = crate::SESSION.get() {
        if let Ok(sess) = session.lock() {
            match sess.save_snapshot() {
                Ok(data) => {
                    let b64 = base64_encode(&data);
                    let body = format!(r#"{{"snapshot":"{}","bytes":{}}}"#, b64, data.len());
                    send_json(s, 200, &body);
                }
                Err(e) => send_error(s, 500, e),
            }
            return;
        }
    }
    send_error(s, 500, "no session");
}

fn handle_cache_load(s: &mut TcpStream, body: &str) {
    if let Some(b64) = parse_json_str(body, "snapshot") {
        if let Some(session) = crate::SESSION.get() {
            if let Ok(mut sess) = session.lock() {
                match base64_decode(&b64) {
                    Some(data) => {
                        match sess.load_snapshot(&data) {
                            Ok(_) => send_json(s, 200, r#"{"status":"ok"}"#),
                            Err(e) => send_error(s, 500, e),
                        }
                    }
                    None => send_error(s, 400, "invalid base64"),
                }
                return;
            }
        }
    }
    send_error(s, 400, "missing snapshot");
}

fn base64_encode(data: &[u8]) -> String {
    const CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::new();
    for chunk in data.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = chunk.get(1).copied().unwrap_or(0) as u32;
        let b2 = chunk.get(2).copied().unwrap_or(0) as u32;
        let triple = (b0 << 16) | (b1 << 8) | b2;
        out.push(CHARS[((triple >> 18) & 0x3f) as usize] as char);
        out.push(CHARS[((triple >> 12) & 0x3f) as usize] as char);
        out.push(if chunk.len() > 1 { CHARS[((triple >> 6) & 0x3f) as usize] as char } else { '=' });
        out.push(if chunk.len() > 2 { CHARS[(triple & 0x3f) as usize] as char } else { '=' });
    }
    out
}

fn base64_decode(data: &str) -> Option<Vec<u8>> {
    const DECODE: [i8; 128] = {
        let mut table = [-1i8; 128];
        let chars = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut i = 0;
        while i < 64 {
            table[chars[i] as usize] = i as i8;
            i += 1;
        }
        table
    };
    let bytes = data.as_bytes();
    let mut out = Vec::with_capacity(bytes.len() / 4 * 3);
    let mut i = 0;
    while i + 4 <= bytes.len() {
        let mut vals = [0u8; 4];
        for j in 0..4 {
            let c = bytes[i + j];
            if c == b'=' { vals[j] = 0; }
            else if c as usize >= 128 { return None; }
            else { vals[j] = DECODE[c as usize] as u8; }
        }
        out.push((vals[0] << 2) | (vals[1] >> 4));
        if bytes[i + 2] != b'=' { out.push((vals[1] << 4) | (vals[2] >> 2)); }
        if bytes[i + 3] != b'=' { out.push((vals[2] << 6) | vals[3]); }
        i += 4;
    }
    Some(out)
}

fn handle_tokenize(s: &mut TcpStream, body: &str) {
    if let Some(text) = parse_json_str(body, "text") {
        if let Some(session) = crate::SESSION.get() {
            if let Ok(sess) = session.lock() {
                if let Some(ref vocab) = sess.vocab {
                    let tokens = crate::tokenizer::bpe_tokenize(vocab, &text);
                    let json = format!(r#"{{"tokens":{:?},"n_tokens":{}}}"#, tokens, tokens.len());
                    send_json(s, 200, &json);
                    return;
                }
            }
        }
        send_error(s, 400, "Tokenizer not loaded");
    }
}

pub fn serve(port: u16) -> Result<(), Box<dyn std::error::Error>> {
    let addr = format!("0.0.0.0:{}", port);
    let listener = TcpListener::bind(&addr)?;
    eprintln!("ds4-server: listening on http://{}", addr);
    for stream in listener.incoming() {
        match stream { Ok(s) => handle_client(s), Err(e) => eprintln!("error: {}", e) }
    }
    Ok(())
}
