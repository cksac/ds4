//! FFI bindings to the C engine (ds4.h).

use std::ffi::{c_char, c_int, c_void};

// ── Opaque types ───────────────────────────────────────────────────────

#[repr(C)]
pub struct ds4_engine {
    _private: [u8; 0],
}

#[repr(C)]
pub struct ds4_session {
    _private: [u8; 0],
}

// ── Enums ──────────────────────────────────────────────────────────────

#[allow(non_camel_case_types)]
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ds4_backend {
    DS4_BACKEND_METAL = 0,
    DS4_BACKEND_CUDA = 1,
    DS4_BACKEND_CPU = 2,
}

#[allow(non_camel_case_types)]
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ds4_think_mode {
    DS4_THINK_NONE = 0,
    DS4_THINK_HIGH = 1,
    DS4_THINK_MAX = 2,
}

// ── Structs ────────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Debug, Clone)]
pub struct ds4_tokens {
    pub v: *mut c_int,
    pub len: c_int,
    pub cap: c_int,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ds4_token_score {
    pub id: c_int,
    pub logit: f32,
    pub logprob: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ds4_engine_options {
    pub model_path: *const c_char,
    pub mtp_path: *const c_char,
    pub backend: ds4_backend,
    pub n_threads: c_int,
    pub mtp_draft_tokens: c_int,
    pub mtp_margin: f32,
    pub directional_steering_file: *const c_char,
    pub directional_steering_attn: f32,
    pub directional_steering_ffn: f32,
    pub warm_weights: bool,
    pub quality: bool,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ds4_context_memory {
    pub total_bytes: u64,
    pub raw_bytes: u64,
    pub compressed_bytes: u64,
    pub scratch_bytes: u64,
    pub prefill_cap: u32,
    pub raw_cap: u32,
    pub comp_cap: u32,
}

// ── Engine functions ───────────────────────────────────────────────────

extern "C" {
    pub fn ds4_engine_open(out: *mut *mut ds4_engine, opt: *const ds4_engine_options) -> c_int;
    pub fn ds4_engine_close(e: *mut ds4_engine);
    pub fn ds4_engine_summary(e: *mut ds4_engine);
    pub fn ds4_backend_name(backend: ds4_backend) -> *const c_char;
    pub fn ds4_think_mode_enabled(mode: ds4_think_mode) -> bool;
    pub fn ds4_think_mode_name(mode: ds4_think_mode) -> *const c_char;
    pub fn ds4_think_max_min_context() -> u32;
    pub fn ds4_think_mode_for_context(mode: ds4_think_mode, ctx_size: c_int) -> ds4_think_mode;
    pub fn ds4_context_memory_estimate(backend: ds4_backend, ctx_size: c_int) -> ds4_context_memory;
    pub fn ds4_engine_generate_argmax(
        e: *mut ds4_engine,
        prompt: *const ds4_tokens,
        n_predict: c_int,
        ctx_size: c_int,
        emit: Option<unsafe extern "C" fn(*mut c_void, c_int)>,
        done: Option<unsafe extern "C" fn(*mut c_void)>,
        emit_ud: *mut c_void,
        progress: Option<unsafe extern "C" fn(*mut c_void, *const c_char, c_int, c_int)>,
        progress_ud: *mut c_void,
    ) -> c_int;
    pub fn ds4_engine_dump_tokens(e: *mut ds4_engine, tokens: *const ds4_tokens);
    pub fn ds4_dump_text_tokenization(
        model_path: *const c_char,
        text: *const c_char,
        fp: *mut c_void,
    ) -> c_int;

    // ── Tokenization ───────────────────────────────────────────────
    pub fn ds4_tokenize_text(e: *mut ds4_engine, text: *const c_char, out: *mut ds4_tokens);
    pub fn ds4_encode_chat_prompt(
        e: *mut ds4_engine,
        system: *const c_char,
        prompt: *const c_char,
        think_mode: ds4_think_mode,
        out: *mut ds4_tokens,
    );
    pub fn ds4_chat_begin(e: *mut ds4_engine, tokens: *mut ds4_tokens);
    pub fn ds4_chat_append_message(
        e: *mut ds4_engine,
        tokens: *mut ds4_tokens,
        role: *const c_char,
        content: *const c_char,
    );
    pub fn ds4_chat_append_assistant_prefix(
        e: *mut ds4_engine,
        tokens: *mut ds4_tokens,
        think_mode: ds4_think_mode,
    );
    pub fn ds4_token_text(e: *mut ds4_engine, token: c_int, len: *mut usize) -> *mut c_char;
    pub fn ds4_token_eos(e: *mut ds4_engine) -> c_int;

    // ── Tokens ─────────────────────────────────────────────────────
    pub fn ds4_tokens_push(tv: *mut ds4_tokens, token: c_int);
    pub fn ds4_tokens_free(tv: *mut ds4_tokens);

    // ── Session ────────────────────────────────────────────────────
    pub fn ds4_session_create(
        out: *mut *mut ds4_session,
        e: *mut ds4_engine,
        ctx_size: c_int,
    ) -> c_int;
    pub fn ds4_session_free(s: *mut ds4_session);
    pub fn ds4_session_sync(
        s: *mut ds4_session,
        prompt: *const ds4_tokens,
        err: *mut c_char,
        errlen: usize,
    ) -> c_int;
    pub fn ds4_session_argmax(s: *mut ds4_session) -> c_int;
    pub fn ds4_session_sample(
        s: *mut ds4_session,
        temperature: f32,
        top_k: c_int,
        top_p: f32,
        min_p: f32,
        rng: *mut u64,
    ) -> c_int;
    pub fn ds4_session_eval(
        s: *mut ds4_session,
        token: c_int,
        err: *mut c_char,
        errlen: usize,
    ) -> c_int;
    pub fn ds4_session_pos(s: *mut ds4_session) -> c_int;
    pub fn ds4_session_ctx(s: *mut ds4_session) -> c_int;
}
