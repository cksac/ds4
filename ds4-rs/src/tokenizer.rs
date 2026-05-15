use crate::gguf::GgufModel;
use crate::model::Vocab;
use std::collections::HashMap;

fn utf8_len(c0: u8) -> usize {
    if c0 < 0x80 { 1 } else if (c0 & 0xe0) == 0xc0 { 2 }
    else if (c0 & 0xf0) == 0xe0 { 3 }
    else if (c0 & 0xf8) == 0xf0 { 4 }
    else { 1 }
}

fn next_utf8(s: &[u8], pos: usize) -> usize {
    let n = utf8_len(s[pos]);
    pos + n.min(s.len() - pos).max(1)
}

fn codepoint_at(s: &[u8], pos: usize) -> u32 {
    let c0 = s[pos];
    let n = utf8_len(c0);
    if pos + n > s.len() { return c0 as u32; }
    match n {
        1 => c0 as u32,
        2 => ((c0 & 0x1f) as u32) << 6 | (s[pos+1] & 0x3f) as u32,
        3 => ((c0 & 0x0f) as u32) << 12 | ((s[pos+1] & 0x3f) as u32) << 6 | (s[pos+2] & 0x3f) as u32,
        _ => ((c0 & 0x07) as u32) << 18 | ((s[pos+1] & 0x3f) as u32) << 12 | ((s[pos+2] & 0x3f) as u32) << 6 | (s[pos+3] & 0x3f) as u32,
    }
}

fn ascii_alpha(c: u8) -> bool { c.is_ascii_alphabetic() }
fn ascii_digit(c: u8) -> bool { c.is_ascii_digit() }
fn ascii_newline(c: u8) -> bool { c == b'\n' || c == b'\r' }
fn ascii_space(c: u8) -> bool { c == b' ' || c == b'\t' || c == b'\n' || c == b'\r' || c == 0x0b || c == 0x0c }

fn joyai_ascii_punct_symbol(c: u8) -> bool {
    (c >= b'!' && c <= b'/') || (c >= b':' && c <= b'@')
    || (c >= b'[' && c <= b'`') || (c >= b'{' && c <= b'~')
}

fn utf8_is_cjk_hira_kata(cp: u32) -> bool {
    (cp >= 0x4e00 && cp <= 0x9fa5) || (cp >= 0x3040 && cp <= 0x309f) || (cp >= 0x30a0 && cp <= 0x30ff)
}

fn joyai_letter_like_at(s: &[u8], pos: usize) -> bool {
    let c = s[pos];
    if c < 128 { return ascii_alpha(c); }
    true
}

fn joyai_consume_letters(s: &[u8], mut pos: usize) -> usize {
    while pos < s.len() && joyai_letter_like_at(s, pos) {
        pos = next_utf8(s, pos);
    }
    pos
}

fn joyai_cjk_at(s: &[u8], pos: usize) -> bool {
    if s[pos] < 128 { return false; }
    let cp = codepoint_at(s, pos);
    utf8_is_cjk_hira_kata(cp)
}

fn gpt2_byte_to_cp(b: u8) -> u32 {
    if (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174) { b as u32 }
    else {
        let mut n = 0u32;
        for x in 0u32..256 {
            if (x >= 33 && x <= 126) || (x >= 161 && x <= 172) || (x >= 174) { continue; }
            if x as u8 == b { return 256 + n; }
            n += 1;
        }
        b as u32
    }
}

fn utf8_put(buf: &mut Vec<u8>, cp: u32) {
    if cp <= 0x7f { buf.push(cp as u8); }
    else if cp <= 0x7ff { buf.push(0xc0 | (cp >> 6) as u8); buf.push(0x80 | (cp & 0x3f) as u8); }
    else if cp <= 0xffff {
        buf.push(0xe0 | (cp >> 12) as u8); buf.push(0x80 | ((cp >> 6) & 0x3f) as u8);
        buf.push(0x80 | (cp & 0x3f) as u8);
    } else {
        buf.push(0xf0 | (cp >> 18) as u8); buf.push(0x80 | ((cp >> 12) & 0x3f) as u8);
        buf.push(0x80 | ((cp >> 6) & 0x3f) as u8); buf.push(0x80 | (cp & 0x3f) as u8);
    }
}

fn byte_encode(text: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(text.len() * 4);
    for &b in text {
        utf8_put(&mut out, gpt2_byte_to_cp(b));
    }
    out
}

pub fn vocab_load(model: &GgufModel) -> Vocab {
    let mut v = Vocab {
        tokens: Vec::new(),
        merges: Vec::new(),
        scores: Vec::new(),
        bos_id: 1, eos_id: 2, unk_id: 0, sep_id: 3, pad_id: 3,
    };

    if let Some(data) = model.find_kv("tokenizer.ggml.tokens") {
        // GGUF array: [item_type(4) | count(8) | ...items...]; strings: [len(8) | bytes]
        let mut i = 12usize; // skip array header
        while i + 8 <= data.len() {
            let len = u64::from_le_bytes(data[i..i+8].try_into().unwrap_or([0;8])) as usize;
            i += 8;
            if i + len <= data.len() {
                v.tokens.push(String::from_utf8_lossy(&data[i..i+len]).to_string());
                i += len;
            } else { break; }
        }
    }

    if let Some(data) = model.find_kv("tokenizer.ggml.scores") {
        // GGUF array: [item_type(4) | count(8) | f32...]; skip 12-byte header
        let mut i = 12usize;
        while i + 4 <= data.len() {
            v.scores.push(f32::from_le_bytes(data[i..i+4].try_into().unwrap_or([0;4])));
            i += 4;
        }
    }

    // Load BPE merges
    // GGUF array of strings: [item_type(4) | count(8) | ...]; each item: [len(8) | bytes]
    // Each merge string is "piece_a piece_b" – look up IDs by string (not integer parse).
    if let Some(data) = model.find_kv("tokenizer.ggml.merges") {
        let token_map: HashMap<String, i32> = v.tokens.iter().enumerate()
            .map(|(i, t)| (t.clone(), i as i32)).collect();
        let mut i = 12usize; // skip array header
        while i + 8 <= data.len() {
            let slen = u64::from_le_bytes(data[i..i+8].try_into().unwrap_or([0;8])) as usize;
            i += 8;
            if i + slen <= data.len() {
                let s = String::from_utf8_lossy(&data[i..i+slen]).to_string();
                if let Some(sp) = s.find(' ') {
                    let a_str = &s[..sp];
                    let b_str = s[sp+1..].trim_end();
                    if let (Some(&a), Some(&b)) = (token_map.get(a_str), token_map.get(b_str)) {
                        v.merges.push((a, b));
                    }
                }
                i += slen;
            } else { break; }
        }
        eprintln!("ds4: loaded {} BPE merges", v.merges.len());
    }

    if let Some(data) = model.find_kv("tokenizer.ggml.bos_token_id") {
        if data.len() >= 4 { v.bos_id = u32::from_le_bytes(data[..4].try_into().unwrap_or([1,0,0,0])) as i32; }
    }
    if let Some(data) = model.find_kv("tokenizer.ggml.eos_token_id") {
        if data.len() >= 4 { v.eos_id = u32::from_le_bytes(data[..4].try_into().unwrap_or([2,0,0,0])) as i32; }
    }

    v
}

fn joyai_pre_tokenize(text: &str) -> Vec<String> {
    let s = text.as_bytes();
    let len = s.len();
    let mut pos = 0;
    let mut pieces: Vec<String> = Vec::new();

    while pos < len {
        let start = pos;
        let c = s[pos];

        if ascii_digit(c) {
            let mut ndigits = 0;
            while pos < len && ascii_digit(s[pos]) && ndigits < 3 { pos += 1; ndigits += 1; }
        } else if joyai_cjk_at(s, pos) {
            loop {
                pos = next_utf8(s, pos);
                if pos >= len || !joyai_cjk_at(s, pos) { break; }
            }
        } else if joyai_ascii_punct_symbol(c) && pos + 1 < len && ascii_alpha(s[pos + 1]) {
            pos += 1;
            while pos < len && ascii_alpha(s[pos]) { pos += 1; }
        } else if joyai_letter_like_at(s, pos) {
            pos = joyai_consume_letters(s, pos);
        } else if !ascii_newline(c) && !joyai_ascii_punct_symbol(c) && pos + 1 < len
            && joyai_letter_like_at(s, pos + 1) {
            pos += 1;
            pos = joyai_consume_letters(s, pos);
        } else if c == b' ' && pos + 1 < len && joyai_ascii_punct_symbol(s[pos + 1]) {
            pos += 1;
            while pos < len && joyai_ascii_punct_symbol(s[pos]) { pos += 1; }
            while pos < len && ascii_newline(s[pos]) { pos += 1; }
        } else if joyai_ascii_punct_symbol(c) {
            while pos < len && joyai_ascii_punct_symbol(s[pos]) { pos += 1; }
            while pos < len && ascii_newline(s[pos]) { pos += 1; }
        } else if ascii_space(c) {
            let mut p = pos;
            let mut last_newline_end = 0;
            while p < len && ascii_space(s[p]) {
                if ascii_newline(s[p]) { last_newline_end = p + 1; }
                p += 1;
            }
            if last_newline_end > 0 {
                pos = last_newline_end;
            } else if p < len && p > pos + 1
                && (joyai_letter_like_at(s, p) || joyai_ascii_punct_symbol(s[p])) {
                pos = p - 1;
            } else {
                pos = p;
            }
        } else {
            pos = next_utf8(s, pos);
        }

        if pos == start { pos = next_utf8(s, pos); }
        pieces.push(String::from_utf8_lossy(&s[start..pos]).to_string());
    }
    pieces
}

// Build a rank table for BPE merges from vocab tokens
fn build_merge_ranks(vocab: &Vocab) -> HashMap<String, i32> {
    let mut ranks: HashMap<String, i32> = HashMap::new();
    for (i, &(a, b)) in vocab.merges.iter().enumerate() {
        let key = if a < vocab.tokens.len() as i32 && b < vocab.tokens.len() as i32 {
            let mut s = vocab.tokens[a as usize].clone();
            s.push_str(&vocab.tokens[b as usize]);
            s
        } else {
            continue;
        };
        ranks.entry(key).or_insert(i as i32);
    }
    ranks
}

// BPE encode a single pre-tokenized piece
fn bpe_encode_piece(vocab: &Vocab, text: &[u8], ranks: &HashMap<String, i32>) -> Vec<i32> {
    let encoded = byte_encode(text);
    if encoded.is_empty() { return vec![]; }

    // Split into byte-level UTF-8 symbols
    let mut syms: Vec<Vec<u8>> = Vec::new();
    let mut i = 0;
    while i < encoded.len() {
        let n = utf8_len(encoded[i]);
        let end = (i + n).min(encoded.len());
        syms.push(encoded[i..end].to_vec());
        i = end;
    }

    // Byte encoding turns each raw byte into 1-2 UTF-8 bytes.
    // The tokens in vocab use the byte-encoded form. We look up each symbol.
    // BPE merge loop
    loop {
        let mut best_i = -1i32;
        let mut best_rank = i32::MAX;

        for i in 0..syms.len().saturating_sub(1) {
            let mut pair = syms[i].clone();
            pair.extend_from_slice(&syms[i + 1]);
            let pair_str = String::from_utf8_lossy(&pair).to_string();
            if let Some(&rank) = ranks.get(&pair_str) {
                if rank < best_rank {
                    best_rank = rank;
                    best_i = i as i32;
                }
            }
        }

        if best_i < 0 { break; }

        // Merge
        let b = best_i as usize;
        let mut merged = syms[b].clone();
        merged.extend_from_slice(&syms[b + 1]);
        syms[b] = merged;
        syms.remove(b + 1);
    }

    // Look up each symbol in vocab
    let mut out = Vec::new();
    let token_map: HashMap<String, i32> = vocab.tokens.iter().enumerate()
        .map(|(i, t)| (t.clone(), i as i32)).collect();

    for sym in &syms {
        let s = String::from_utf8_lossy(sym).to_string();
        if let Some(&id) = token_map.get(&s) {
            out.push(id);
        } else {
            // Fallback: byte-level fallback
            for &b in text {
                let byte_str = String::from_utf8_lossy(&[b]).to_string();
                if let Some(&id) = token_map.get(&byte_str) {
                    out.push(id);
                }
            }
        }
    }
    out
}

// Returns true for tokens that must be matched verbatim (DeepSeek special tokens,
// angle-bracket tags like <think> / </think>, etc.).
fn is_special_token(t: &str) -> bool {
    // Full-width vertical bar ｜ = U+FF5C — used in <｜begin▁of▁sentence｜> etc.
    if t.contains('<') && (t.contains('\u{FF5C}') || t.ends_with('>')) && t.len() > 2 {
        return true;
    }
    false
}

pub fn bpe_tokenize(vocab: &Vocab, text: &str) -> Vec<i32> {
    let ranks = build_merge_ranks(vocab);

    // Collect special tokens sorted longest-first for greedy left-to-right matching.
    let mut special: Vec<(&str, i32)> = vocab.tokens.iter().enumerate()
        .filter(|(_, t)| is_special_token(t))
        .map(|(i, t)| (t.as_str(), i as i32))
        .collect();
    special.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

    let text_bytes = text.as_bytes();
    let tlen = text_bytes.len();
    let mut out = Vec::new();
    let mut pos = 0usize;

    while pos < tlen {
        // 1. Try to match a special token verbatim.
        let mut matched = false;
        for &(sp, sp_id) in &special {
            let sb = sp.as_bytes();
            if text_bytes[pos..].starts_with(sb) {
                out.push(sp_id);
                pos += sb.len();
                matched = true;
                break;
            }
        }
        if matched { continue; }

        // 2. Collect the next run of non-special text.
        let seg_start = pos;
        let mut seg_end = pos;
        'advance: while seg_end < tlen {
            for &(sp, _) in &special {
                if text_bytes[seg_end..].starts_with(sp.as_bytes()) {
                    break 'advance;
                }
            }
            let skip = utf8_len(text_bytes[seg_end]).min(tlen - seg_end).max(1);
            seg_end += skip;
        }

        if seg_end > seg_start {
            let segment = match std::str::from_utf8(&text_bytes[seg_start..seg_end]) {
                Ok(s) => s,
                Err(_) => { pos = seg_end; continue; }
            };
            for piece in &joyai_pre_tokenize(segment) {
                out.extend(bpe_encode_piece(vocab, piece.as_bytes(), &ranks));
            }
        }
        pos = seg_end;
    }

    out
}

pub fn token_decode(vocab: &Vocab, token_ids: &[i32]) -> String {
    let mut out = String::new();
    for &tid in token_ids {
        if tid == vocab.bos_id || tid == vocab.eos_id { continue; }
        if (tid as usize) < vocab.tokens.len() {
            out.push_str(&vocab.tokens[tid as usize]);
        } else {
            out.push('?');
        }
    }
    out
}
