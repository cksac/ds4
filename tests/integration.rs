use std::ffi::c_void;
use std::fs;
use std::path::Path;
use std::ptr;

use ds4_rust::{
    generate_rust, Backend, Engine, EngineOptions, GenerationOptions, ThinkMode,
};

#[repr(C)]
struct Ds4MetalTensor {
    _private: [u8; 0],
}

unsafe extern "C" {
    fn ds4_metal_tensor_alloc(bytes: u64) -> *mut Ds4MetalTensor;
    fn ds4_metal_tensor_free(tensor: *mut Ds4MetalTensor);
    fn ds4_metal_tensor_write(
        tensor: *mut Ds4MetalTensor,
        offset: u64,
        data: *const c_void,
        bytes: u64,
    ) -> i32;
    fn ds4_metal_tensor_read(
        tensor: *const Ds4MetalTensor,
        offset: u64,
        data: *mut c_void,
        bytes: u64,
    ) -> i32;
    fn ds4_metal_begin_commands() -> i32;
    fn ds4_metal_end_commands() -> i32;
    fn ds4_metal_synchronize() -> i32;
    fn ds4_metal_set_model_map(model_map: *const c_void, model_size: u64) -> i32;
    fn ds4_metal_set_quality(quality: bool) -> i32;
    fn ds4_metal_matmul_f16_tensor(
        out: *mut Ds4MetalTensor,
        model_map: *const c_void,
        model_size: u64,
        weight_offset: u64,
        in_dim: u64,
        out_dim: u64,
        x: *const Ds4MetalTensor,
        n_tok: u64,
    ) -> i32;
}

struct MetalTensor {
    ptr: *mut Ds4MetalTensor,
}

impl MetalTensor {
    fn alloc(bytes: u64) -> Self {
        let ptr = unsafe { ds4_metal_tensor_alloc(bytes) };
        assert!(!ptr.is_null(), "failed to allocate Metal tensor");
        Self { ptr }
    }
}

impl Drop for MetalTensor {
    fn drop(&mut self) {
        unsafe { ds4_metal_tensor_free(self.ptr) };
    }
}

struct AlignedBuffer {
    ptr: *mut c_void,
}

impl AlignedBuffer {
    fn new(bytes: usize, align: usize) -> Self {
        let mut ptr = ptr::null_mut();
        let rc = unsafe { libc::posix_memalign(&mut ptr, align, bytes) };
        assert_eq!(rc, 0, "posix_memalign failed with code {rc}");
        assert!(!ptr.is_null(), "posix_memalign returned null");
        Self { ptr }
    }
}

impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        unsafe { libc::free(self.ptr) };
    }
}

#[derive(Clone, Debug, Default)]
struct VectorStepTop {
    bytes: Vec<u8>,
    logprob: f32,
}

#[derive(Clone, Debug, Default)]
struct VectorStep {
    selected: Vec<u8>,
    top: Vec<VectorStepTop>,
}

#[derive(Clone, Debug, Default)]
struct VectorCase {
    id: String,
    prompt_path: String,
    ctx: i32,
    steps: Vec<VectorStep>,
}

fn test_model_path() -> String {
    std::env::var("DS4_TEST_MODEL").unwrap_or_else(|_| "ds4flash.gguf".to_owned())
}

fn open_engine() -> Engine {
    Engine::open(&EngineOptions {
        model_path: test_model_path(),
        mtp_path: None,
        backend: Backend::Metal,
        n_threads: 0,
        mtp_draft_tokens: 1,
        mtp_margin: 3.0,
        warm_weights: false,
        quality: false,
    })
    .expect("engine open")
}

fn round_up_u64(n: u64, align: u64) -> u64 {
    (n + align - 1) & !(align - 1)
}

fn page_size() -> u64 {
    let raw = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    assert!(raw > 0, "sysconf(_SC_PAGESIZE) failed");
    raw as u64
}

fn float_to_f16(f: f32) -> u16 {
    let bits = f.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mut mant = bits & 0x7f_ffff;

    if exp <= 0 {
        if exp < -10 {
            return sign;
        }
        mant |= 0x80_0000;
        let shift = (14 - exp) as u32;
        let mut half_mant = mant >> shift;
        if ((mant >> (shift - 1)) & 1) != 0 {
            half_mant += 1;
        }
        return sign | (half_mant as u16);
    }
    if exp >= 31 {
        return sign | 0x7c00;
    }

    let mut half = (sign as u32) | ((exp as u32) << 10) | (mant >> 13);
    if (mant & 0x1000) != 0 {
        half += 1;
    }
    half as u16
}

fn count_substr(haystack: &str, needle: &str) -> usize {
    if needle.is_empty() {
        return 0;
    }
    let mut count = 0usize;
    let mut start = 0usize;
    while let Some(offset) = haystack[start..].find(needle) {
        count += 1;
        start += offset + needle.len();
    }
    count
}

fn hex_digit(c: u8) -> Option<u8> {
    match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(10 + c - b'a'),
        b'A'..=b'F' => Some(10 + c - b'A'),
        _ => None,
    }
}

fn hex_to_bytes(hex: &str) -> Option<Vec<u8>> {
    let h = hex.trim();
    if h.len() % 2 != 0 {
        return None;
    }
    let bytes = h.as_bytes();
    let mut out = Vec::with_capacity(h.len() / 2);
    let mut i = 0usize;
    while i < bytes.len() {
        let hi = hex_digit(bytes[i])?;
        let lo = hex_digit(bytes[i + 1])?;
        out.push((hi << 4) | lo);
        i += 2;
    }
    Some(out)
}

fn parse_vector_cases(path: &str) -> Vec<VectorCase> {
    let content = fs::read_to_string(path).expect("read vector file");
    let mut lines = content.lines();
    let mut out = Vec::new();

    while let Some(raw) = lines.next() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if !line.starts_with("case ") {
            panic!("unexpected line before vector case: {line}");
        }

        let mut parts = line.split_whitespace();
        let _ = parts.next();
        let id = parts.next().expect("case id").to_owned();
        let ctx: i32 = parts.next().expect("case ctx").parse().expect("parse ctx");
        let nsteps: usize = parts
            .next()
            .expect("case nsteps")
            .parse()
            .expect("parse nsteps");
        let prompt_path = parts.next().expect("case prompt path").to_owned();
        let mut case = VectorCase {
            id,
            prompt_path,
            ctx,
            steps: vec![VectorStep::default(); nsteps],
        };

        let mut current_step: Option<usize> = None;
        for raw_case_line in lines.by_ref() {
            let case_line = raw_case_line.trim();
            if case_line.is_empty() || case_line.starts_with('#') {
                continue;
            }
            if case_line == "end" {
                break;
            }
            if case_line.starts_with("step ") {
                let mut step_parts = case_line.split_whitespace();
                let _ = step_parts.next();
                let step_idx: usize = step_parts
                    .next()
                    .expect("step index")
                    .parse()
                    .expect("parse step index");
                let selected_hex = step_parts.next().expect("step token hex");
                let ntop: usize = step_parts
                    .next()
                    .expect("step ntop")
                    .parse()
                    .expect("parse ntop");
                assert!(step_idx < case.steps.len(), "step index out of range");
                let selected = hex_to_bytes(selected_hex).expect("invalid step hex bytes");
                case.steps[step_idx].selected = selected;
                case.steps[step_idx].top = Vec::with_capacity(ntop);
                current_step = Some(step_idx);
                continue;
            }
            if case_line.starts_with("top ") {
                let step_idx = current_step.expect("top without step");
                let mut top_parts = case_line.split_whitespace();
                let _ = top_parts.next();
                let token_hex = top_parts.next().expect("top token hex");
                let logprob: f32 = top_parts
                    .next()
                    .expect("top logprob")
                    .parse()
                    .expect("parse top logprob");
                case.steps[step_idx].top.push(VectorStepTop {
                    bytes: hex_to_bytes(token_hex).expect("invalid top token hex bytes"),
                    logprob,
                });
                continue;
            }
            panic!("unexpected vector line: {case_line}");
        }

        out.push(case);
    }

    out
}

#[test]
fn metal_f16_matvec_fast_nr0_4() {
    let in_dim = 4096u64;
    let out_dim = 512u64;
    let weight_bytes = in_dim * out_dim * std::mem::size_of::<u16>() as u64;
    let page_size = page_size();
    let weight_alloc = round_up_u64(weight_bytes, page_size);
    let weights = AlignedBuffer::new(weight_alloc as usize, page_size as usize);

    unsafe {
        ptr::write_bytes(weights.ptr, 0, weight_alloc as usize);
    }
    let weights_slice = unsafe {
        std::slice::from_raw_parts_mut(weights.ptr as *mut u16, (weight_alloc / 2) as usize)
    };
    for o in 0..out_dim {
        for i in 0..in_dim {
            let w = (((o * 3 + i * 5) % 23) as i32 - 11) as f32 / 64.0;
            weights_slice[(o * in_dim + i) as usize] = float_to_f16(w);
        }
    }

    let x = MetalTensor::alloc(in_dim * 4);
    let out = MetalTensor::alloc(out_dim * 4);
    let mut x_host = vec![0.0f32; in_dim as usize];
    let mut out_host = vec![0.0f32; out_dim as usize];
    for (i, xval) in x_host.iter_mut().enumerate() {
        *xval = ((i as i32 % 31) - 15) as f32 / 32.0;
    }

    let write_ok = unsafe {
        ds4_metal_tensor_write(
            x.ptr,
            0,
            x_host.as_ptr() as *const c_void,
            (x_host.len() * std::mem::size_of::<f32>()) as u64,
        )
    };
    assert_ne!(write_ok, 0, "writing x tensor failed");

    let set_map_ok = unsafe { ds4_metal_set_model_map(weights.ptr as *const c_void, weight_alloc) };
    assert_ne!(set_map_ok, 0, "setting model map failed");
    unsafe {
        ds4_metal_set_quality(false);
    }

    let begin_ok = unsafe { ds4_metal_begin_commands() };
    assert_ne!(begin_ok, 0, "begin commands failed");
    let matmul_ok = unsafe {
        ds4_metal_matmul_f16_tensor(
            out.ptr,
            weights.ptr as *const c_void,
            weight_alloc,
            0,
            in_dim,
            out_dim,
            x.ptr as *const Ds4MetalTensor,
            1,
        )
    };
    assert_ne!(matmul_ok, 0, "metal matmul failed");
    let end_ok = unsafe { ds4_metal_end_commands() };
    assert_ne!(end_ok, 0, "end commands failed");
    let sync_ok = unsafe { ds4_metal_synchronize() };
    assert_ne!(sync_ok, 0, "metal synchronize failed");

    let read_ok = unsafe {
        ds4_metal_tensor_read(
            out.ptr as *const Ds4MetalTensor,
            0,
            out_host.as_mut_ptr() as *mut c_void,
            (out_host.len() * std::mem::size_of::<f32>()) as u64,
        )
    };
    assert_ne!(read_ok, 0, "reading output tensor failed");

    let mut max_abs = 0.0f32;
    for o in 0..out_dim {
        let mut reference = 0.0f32;
        for i in 0..in_dim {
            let w = (((o * 3 + i * 5) % 23) as i32 - 11) as f32 / 64.0;
            reference += w * x_host[i as usize];
        }
        let err = (out_host[o as usize] - reference).abs();
        if err > max_abs {
            max_abs = err;
        }
    }

    assert!(max_abs < 0.02, "max abs error too high: {max_abs}");
}

#[test]
#[ignore]
fn long_security_continuation() {
    let prompt_path = std::env::var("DS4_TEST_LONG_PROMPT")
        .unwrap_or_else(|_| "tests/long_context_security_prompt.txt".to_owned());
    let prompt_text = fs::read_to_string(&prompt_path).expect("read long context prompt");

    let engine = open_engine();
    let prompt = engine
        .tokenize_rendered_chat(&prompt_text)
        .expect("tokenize rendered chat");
    assert!(prompt.len() > 30000, "prompt token count too small: {}", prompt.len());

    let memory = Engine::context_memory_estimate(Backend::Metal, 100000);
    let mut session = engine.create_rust_session(100000, memory.raw_cap);
    let result = generate_rust(
        &engine,
        &mut session,
        &prompt,
        GenerationOptions {
            max_tokens: 700,
            temperature: 0.8,
            top_k: 40,
            top_p: 0.95,
            min_p: 0.05,
            seed: Some(12345),
        },
    )
    .expect("generation should succeed");

    assert!(result.completion_tokens > 0, "completion token count is zero");
    let text = result.text_lossy();
    assert!(text.contains("</think>"), "output must contain </think>");
    assert_eq!(
        count_substr(&text, "</think>"),
        1,
        "output must contain exactly one </think> marker"
    );
    assert_eq!(
        count_substr(&text, "The most critical security issue"),
        1,
        "expected phrase must appear exactly once"
    );
    assert!(
        text.contains("arbitrary file"),
        "output must contain 'arbitrary file'"
    );
}

#[test]
#[ignore]
fn official_logprob_vectors() {
    let vector_file = std::env::var("DS4_TEST_VECTOR_FILE")
        .unwrap_or_else(|_| "tests/test-vectors/official.vec".to_owned());
    let cases = parse_vector_cases(&vector_file);
    assert!(!cases.is_empty(), "no vector cases found");

    let engine = open_engine();
    for case in cases {
        let prompt_path = Path::new(&case.prompt_path);
        let prompt_text = fs::read_to_string(prompt_path)
            .unwrap_or_else(|e| panic!("failed to read prompt {}: {e}", case.prompt_path));
        let prompt = engine
            .encode_chat_prompt(Some(""), &prompt_text, ThinkMode::None)
            .unwrap_or_else(|e| panic!("failed to encode prompt for {}: {e}", case.id));

        let memory = Engine::context_memory_estimate(Backend::Metal, case.ctx);
        let mut session = engine.create_rust_session(case.ctx as u32, memory.raw_cap);
        session
            .sync(&prompt)
            .unwrap_or_else(|e| panic!("failed to sync case {}: {e}", case.id));

        for (i, step) in case.steps.iter().enumerate() {
            let token = session.argmax_token();
            assert!(token >= 0, "negative argmax token in case {} step {}", case.id, i);
            let token_bytes = engine.token_bytes(token);
            assert_eq!(
                token_bytes,
                step.selected,
                "selected token mismatch in case {} step {}",
                case.id,
                i
            );

            let scores = session.top_logprobs(20);
            for official_top in &step.top {
                let mut found = false;
                let mut local_lp = 0.0f32;
                for (local_token, lp) in &scores {
                    if engine.token_bytes(*local_token) == official_top.bytes {
                        found = true;
                        local_lp = *lp;
                        break;
                    }
                }
                assert!(
                    found,
                    "official top token missing locally in case {} step {}",
                    case.id,
                    i
                );
                let delta = (local_lp - official_top.logprob).abs();
                assert!(
                    delta <= 4.0,
                    "logprob delta too high in case {} step {}: local={} official={} delta={}",
                    case.id,
                    i,
                    local_lp,
                    official_top.logprob,
                    delta
                );
            }

            if i + 1 < case.steps.len() {
                session
                    .eval(token)
                    .unwrap_or_else(|e| panic!("failed to eval in case {} step {}: {e}", case.id, i));
            }
        }
    }
}