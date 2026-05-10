mod gguf;
mod ffi;
mod metal_runtime;
#[cfg(target_os = "macos")]
mod metal_native;
#[cfg(target_os = "macos")]
mod metal_bridge;
#[cfg(target_os = "macos")]
mod metal_kernels;

use gguf::{
    validate_model_config_bytes, BoundTensor, Ds4LayerTensorBindings, Ds4MtpTensorBindings,
    Ds4TensorBindings, GgufMap, ModelSummary, TensorDirectory, TokenizerMetadata,
};
#[cfg(target_os = "macos")]
use objc2::rc::Retained;
#[cfg(target_os = "macos")]
use objc2::runtime::ProtocolObject;
#[cfg(target_os = "macos")]
use objc2_metal::MTLBuffer;
use std::ptr::NonNull;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, bail, Context, Result};

pub const RENDERED_CHAT_BOS: &str = "<\u{ff5c}begin\u{2581}of\u{2581}sentence\u{ff5c}>";

const RENDERED_CHAT_USER: &str = "<\u{ff5c}User\u{ff5c}>";
const RENDERED_CHAT_ASSISTANT: &str = "<\u{ff5c}Assistant\u{ff5c}>";
const THINK_START_MARKER: &str = "<think>";
const THINK_END_MARKER: &str = "</think>";
const DSML_MARKER: &str = "\u{ff5c}DSML\u{ff5c}";
const TOOL_PREFIX: &str = "Tool: ";
const REASONING_EFFORT_MAX_PREFIX: &str = concat!(
    "Reasoning Effort: Absolute maximum with no shortcuts permitted.\n",
    "You MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root cause, rigorously stress-testing your logic against all potential paths, edge cases, and adversarial scenarios.\n",
    "Explicitly write out your entire deliberation process, documenting every intermediate step, considered alternative, and rejected hypothesis to ensure absolutely no assumption is left unchecked.\n\n",
);
const THINK_MAX_MIN_CONTEXT: i32 = 393_216;
const DS4_N_LAYER: u32 = 43;
const DS4_N_HC: usize = 4;
const DS4_N_HC_SINKHORN_ITER: usize = 20;
const DS4_N_ROT: usize = 64;
const DS4_NEG_INF: f32 = -1.0e30;
const DS4_RMS_EPS: f32 = 1.0e-6;
const DS4_HC_EPS: f32 = 1.0e-6;
const DS4_ROPE_FREQ_BASE: f32 = 10_000.0;
const DS4_ROPE_SCALE_FACTOR: f32 = 16.0;
const DS4_ROPE_YARN_BETA_FAST: f32 = 32.0;
const DS4_ROPE_YARN_BETA_SLOW: f32 = 1.0;
const DS4_COMPRESS_ROPE_FREQ_BASE: f32 = 160_000.0;
const DS4_ROPE_ORIG_CTX: u64 = 65_536;
const DS4_N_EMBD: usize = 4096;
const DS4_N_HEAD: u32 = 64;
const DS4_N_HEAD_KV: u32 = 1;
const DS4_N_LORA_O: u32 = 1024;
const DS4_N_OUT_GROUP: u32 = 8;
const DS4_N_EXPERT: u32 = 256;
const DS4_N_INDEXER_HEAD: u32 = 64;
const DS4_N_HEAD_DIM: u64 = 512;
const DS4_N_INDEXER_HEAD_DIM: u64 = 128;
const DS4_N_INDEXER_TOP_K: u64 = 512;
const DS4_N_EXPERT_USED: usize = 6;
const DS4_EXPERT_WEIGHT_SCALE: f32 = 1.5;
const DS4_SWIGLU_CLAMP_EXP: f32 = 10.0;
const QK_K: usize = 256;
const DS4_TENSOR_Q2_K: u32 = 10;
const DS4_TENSOR_Q4_K: u32 = 12;
const DS4_TENSOR_IQ2_XXS: u32 = 16;
const DS4_N_SWA: u32 = 128;
const METAL_PREFILL_LONG_PROMPT_CAP: i32 = 2048;
const METAL_RAW_CAP_ALIGN: u64 = 256;
const METAL_RAW_CAP_MAX: u32 = 8192;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Backend {
    Metal,
    Cpu,
}

impl Backend {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Metal => "metal",
            Self::Cpu => "cpu",
        }
    }
}

// Safe wrapper for a Metal device tensor. The pointer is always non-null.
// On macOS, Rust owns the underlying MTLBuffer through objc2-metal and keeps
// a lightweight native view wrapper only for the existing kernel ABI.
#[cfg(target_os = "macos")]
#[derive(Debug)]
struct MetalTensor {
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    ptr: NonNull<ffi::ds4_metal_tensor>,
    offset: u64,
    bytes: u64,
}

#[cfg(target_os = "macos")]
impl Drop for MetalTensor {
    fn drop(&mut self) {
        metal_runtime::tensor_free(self.ptr.as_ptr())
    }
}

#[cfg(target_os = "macos")]
unsafe impl Send for MetalTensor {}

#[cfg(target_os = "macos")]
impl MetalTensor {
    fn alloc(bytes: u64) -> Option<Self> {
        let raw_buffer = metal_runtime::buffer_alloc(bytes);
        let buffer = unsafe {
            Retained::from_raw(raw_buffer.cast::<ProtocolObject<dyn MTLBuffer>>())
        }?;
        let raw_view = metal_runtime::tensor_bind_owned_buffer(
            Retained::as_ptr(&buffer) as *const _ as *mut _,
            bytes,
        );
        let ptr = NonNull::new(raw_view)?;
        Some(Self {
            buffer,
            ptr,
            offset: 0,
            bytes,
        })
    }

    fn view(base: &Self, offset: u64, bytes: u64) -> Option<Self> {
        if offset > base.bytes || bytes > base.bytes - offset {
            return None;
        }
        let buffer = base.buffer.clone();
        let raw_view = metal_runtime::tensor_wrap_buffer(
            Retained::as_ptr(&buffer) as *const _ as *mut _,
            base.offset + offset,
            bytes,
        );
        let ptr = NonNull::new(raw_view)?;
        Some(Self {
            buffer,
            ptr,
            offset: base.offset + offset,
            bytes,
        })
    }

    fn as_ptr(&self) -> *mut ffi::ds4_metal_tensor {
        self.ptr.as_ptr()
    }

    fn as_const_ptr(&self) -> *const ffi::ds4_metal_tensor {
        self.ptr.as_ptr() as *const _
    }

    pub(crate) fn metal_buffer(&self) -> &ProtocolObject<dyn MTLBuffer> {
        &self.buffer
    }

    pub(crate) fn metal_offset(&self) -> u64 {
        self.offset
    }

    pub(crate) fn metal_bytes(&self) -> u64 {
        self.bytes
    }

    fn read_f32(&self, out: &mut Vec<f32>) -> bool {
        let byte_len = out.len().saturating_mul(std::mem::size_of::<f32>());
        if byte_len == 0 {
            return true;
        }
        unsafe {
            let src = (self.buffer.contents().as_ptr() as *const u8).add(self.offset as usize);
            std::ptr::copy_nonoverlapping(src, out.as_mut_ptr() as *mut u8, byte_len);
        }
        true
    }

    fn write_f32(&self, data: &[f32]) -> bool {
        let byte_len = data.len().saturating_mul(std::mem::size_of::<f32>());
        if byte_len == 0 {
            return true;
        }
        unsafe {
            let dst = (self.buffer.contents().as_ptr() as *mut u8).add(self.offset as usize);
            std::ptr::copy_nonoverlapping(data.as_ptr() as *const u8, dst, byte_len);
        }
        true
    }
}

#[cfg(not(target_os = "macos"))]
#[derive(Debug)]
struct MetalTensor {
    ptr: NonNull<ffi::ds4_metal_tensor>,
}

#[cfg(not(target_os = "macos"))]
impl Drop for MetalTensor {
    fn drop(&mut self) {
        metal_runtime::tensor_free(self.ptr.as_ptr())
    }
}

#[cfg(not(target_os = "macos"))]
unsafe impl Send for MetalTensor {}

#[cfg(not(target_os = "macos"))]
impl MetalTensor {
    fn alloc(bytes: u64) -> Option<Self> {
        let ptr = metal_runtime::tensor_alloc(bytes);
        NonNull::new(ptr).map(|ptr| Self { ptr })
    }

    fn view(base: &Self, offset: u64, bytes: u64) -> Option<Self> {
        let ptr = metal_runtime::tensor_view(base.ptr.as_ptr(), offset, bytes);
        NonNull::new(ptr).map(|ptr| Self { ptr })
    }

    fn as_ptr(&self) -> *mut ffi::ds4_metal_tensor {
        self.ptr.as_ptr()
    }

    fn as_const_ptr(&self) -> *const ffi::ds4_metal_tensor {
        self.ptr.as_ptr() as *const _
    }

    fn read_f32(&self, out: &mut Vec<f32>) -> bool {
        let n = out.len();
        if n == 0 {
            return true;
        }
        metal_runtime::tensor_read(
            self.as_const_ptr(),
            0,
            out.as_mut_ptr() as *mut _,
            (n * 4) as u64,
        ) != 0
    }

    fn write_f32(&self, data: &[f32]) -> bool {
        if data.is_empty() {
            return true;
        }
        metal_runtime::tensor_write(
            self.as_ptr(),
            0,
            data.as_ptr() as *const _,
            (data.len() * 4) as u64,
        ) != 0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ThinkMode {
    None,
    High,
    Max,
}

impl ThinkMode {
    pub fn enabled(self) -> bool {
        matches!(self, Self::High | Self::Max)
    }

    pub fn for_context(self, ctx_size: i32) -> Self {
        if self == Self::Max && ctx_size < THINK_MAX_MIN_CONTEXT {
            Self::High
        } else {
            self
        }
    }
}

#[derive(Clone, Debug)]
pub struct EngineOptions {
    pub model_path: String,
    pub mtp_path: Option<String>,
    pub backend: Backend,
    pub n_threads: i32,
    pub mtp_draft_tokens: i32,
    pub mtp_margin: f32,
    pub warm_weights: bool,
    pub quality: bool,
}

impl Default for EngineOptions {
    fn default() -> Self {
        Self {
            model_path: "ds4flash.gguf".to_owned(),
            mtp_path: None,
            backend: Backend::Metal,
            n_threads: 0,
            mtp_draft_tokens: 1,
            mtp_margin: 3.0,
            warm_weights: false,
            quality: false,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ContextMemory {
    pub total_bytes: u64,
    pub raw_bytes: u64,
    pub compressed_bytes: u64,
    pub scratch_bytes: u64,
    pub prefill_cap: u32,
    pub raw_cap: u32,
    pub comp_cap: u32,
}

impl From<ffi::ds4_context_memory> for ContextMemory {
    fn from(value: ffi::ds4_context_memory) -> Self {
        Self {
            total_bytes: value.total_bytes,
            raw_bytes: value.raw_bytes,
            compressed_bytes: value.compressed_bytes,
            scratch_bytes: value.scratch_bytes,
            prefill_cap: value.prefill_cap,
            raw_cap: value.raw_cap,
            comp_cap: value.comp_cap,
        }
    }
}

#[derive(Debug)]
pub struct Tokens {
    raw: Vec<i32>,
}

unsafe impl Send for Tokens {}

impl Default for Tokens {
    fn default() -> Self {
        Self::new()
    }
}

impl Tokens {
    pub fn new() -> Self {
        Self { raw: Vec::new() }
    }

    pub fn len(&self) -> i32 {
        i32::try_from(self.raw.len()).unwrap_or(i32::MAX)
    }

    pub fn is_empty(&self) -> bool {
        self.raw.is_empty()
    }

    pub fn as_slice(&self) -> &[i32] {
        &self.raw
    }

    pub fn push(&mut self, token: i32) {
        self.raw.push(token)
    }

    pub fn extend_from(&mut self, other: &Tokens) {
        self.raw.extend_from_slice(other.as_slice())
    }
}

#[derive(Debug, Clone)]
pub struct HcPreOutput {
    pub residual_hc: Vec<f32>,
    pub sublayer_input: Vec<f32>,
    pub post: Vec<f32>,
    pub comb: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct AttentionProjectionOutput {
    pub pre: HcPreOutput,
    pub norm: Vec<f32>,
    pub q: Vec<f32>,
    pub qr_norm: Vec<f32>,
    pub kv: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct CompressedAttentionStepOutput {
    pub projection: AttentionProjectionOutput,
    pub new_comp: Option<Vec<f32>>,
    pub after_attn_hc: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct MtpDraftStepOutput {
    pub input_hc: Vec<f32>,
    pub hc_state: Vec<f32>,
    pub logits: Vec<f32>,
    pub top_token: i32,
}

#[derive(Debug, Clone)]
pub struct ExpertSelection {
    pub probs: Vec<f32>,
    pub selected: Vec<i32>,
    pub weights: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct FfnPreparationOutput {
    pub pre: HcPreOutput,
    pub norm: Vec<f32>,
    pub shared: Vec<f32>,
    pub selection: ExpertSelection,
}

#[derive(Debug, Clone)]
pub struct RustLayerCache {
    pub raw_kv: Vec<f32>,
    pub attn_comp_kv: Vec<f32>,
    pub attn_state_kv: Vec<f32>,
    pub attn_state_score: Vec<f32>,
    pub index_comp_kv: Vec<f32>,
    pub index_state_kv: Vec<f32>,
    pub index_state_score: Vec<f32>,
    pub cap_raw: u32,
    pub comp_cap: u32,
    pub compress_ratio: u32,
}

#[derive(Debug, Clone)]
pub struct RustKvCache {
    pub layers: Vec<RustLayerCache>,
}

// Per-layer Metal KV cache state for the device-side decode path.
#[derive(Debug)]
struct MetalLayerCache {
    raw_kv: MetalTensor,
    // Compressed-attention caches; present only for layers with ratio != 0.
    attn_comp_kv: Option<MetalTensor>,
    attn_state_kv: Option<MetalTensor>,
    attn_state_score: Option<MetalTensor>,
    // Ratio-4 indexer caches; present only for ratio == 4.
    index_comp_kv: Option<MetalTensor>,
    index_state_kv: Option<MetalTensor>,
    index_state_score: Option<MetalTensor>,
    // Running compressed-row counts, updated each decode step.
    n_comp: u32,
    n_index_comp: u32,
    compress_ratio: u32,
}

// Full Metal decode graph: scratch tensors plus per-layer KV caches.
// This mirrors ds4_metal_graph in ds4.c but covers only the single-token
// decode path; prefill and MTP state are not included here.
#[derive(Debug)]
struct MetalDecodeGraph {
    // Active HC state; `cur_hc` and `after_ffn_hc` are swapped every layer.
    cur_hc: MetalTensor,
    after_ffn_hc: MetalTensor,

    // HC mixer / split workspace (flat_hc is reused for both HC passes).
    flat_hc: MetalTensor,
    hc_mix: MetalTensor,
    // hc_split is the backing tensor; hc_pre/hc_post/hc_comb are views into it
    // at offsets 0, n_hc, and 2*n_hc respectively.
    hc_split: MetalTensor,

    // Attention scratch
    attn_cur: MetalTensor,
    attn_norm: MetalTensor,
    qr: MetalTensor,
    qr_norm: MetalTensor,
    q: MetalTensor,
    kv_raw: MetalTensor,
    kv: MetalTensor,

    // Compressor / indexer scratch
    comp_kv_cur: MetalTensor,
    comp_sc_cur: MetalTensor,
    indexer_q: MetalTensor,
    indexer_weights: MetalTensor,
    indexer_scores: MetalTensor,
    comp_selected: MetalTensor,

    // Attention output
    heads: MetalTensor,
    attn_low: MetalTensor,
    attn_out: MetalTensor,
    after_attn_hc: MetalTensor,

    // FFN scratch
    ffn_cur: MetalTensor,
    ffn_norm: MetalTensor,
    shared_gate: MetalTensor,
    shared_up: MetalTensor,
    shared_mid: MetalTensor,
    shared_out: MetalTensor,
    router_logits: MetalTensor,
    router_probs: MetalTensor,
    router_selected: MetalTensor,
    router_weights: MetalTensor,
    routed_gate: MetalTensor,
    routed_up: MetalTensor,
    routed_mid: MetalTensor,
    routed_down: MetalTensor,
    routed_out: MetalTensor,

    // Output head scratch
    output_pre: MetalTensor,
    output_weights: MetalTensor,
    output_embd: MetalTensor,
    output_norm: MetalTensor,
    logits: MetalTensor,

    // Per-layer KV caches
    layers: Vec<MetalLayerCache>,

    // Ring / sliding-window parameters
    raw_cap: u32,
    raw_window: u32,
    _comp_cap: u32,
    vocab_dim: u64,
}

unsafe impl Send for MetalDecodeGraph {}

impl MetalDecodeGraph {
    // Compute the number of raw KV rows currently visible for a single-token
    // decode step at position `pos`, mirroring metal_graph_raw_span_for_batch.
    fn raw_span(&self, pos: u32) -> u32 {
        let window = if self.raw_window != 0 { self.raw_window } else { DS4_N_SWA };
        let needed = (window as u64).saturating_add(1);
        let available = (pos as u64).saturating_add(1);
        needed.min(available).min(self.raw_cap as u64) as u32
    }

    // Ring index of the raw KV row for position `pos`.
    fn raw_row(&self, pos: u32) -> u32 {
        pos % self.raw_cap
    }

    // Starting ring index of the visible raw window.
    fn raw_start(&self, pos: u32, n_raw: u32) -> u32 {
        if n_raw == 0 || self.raw_cap == 0 {
            return 0;
        }
        (pos + 1 - n_raw) % self.raw_cap
    }
}

#[derive(Debug, Clone)]
struct RustOutputScratch {
    flat_hc: Vec<f32>,
    output_pre: Vec<f32>,
    output_weights: Vec<f32>,
    output_embd: Vec<f32>,
    output_norm: Vec<f32>,
    q8_xq: Vec<i8>,
    q8_xscale: Vec<f32>,
}

impl RustOutputScratch {
    fn new() -> Self {
        let q8_blocks = DS4_N_EMBD.div_ceil(32);
        Self {
            flat_hc: vec![0.0; DS4_N_HC * DS4_N_EMBD],
            output_pre: vec![0.0; DS4_N_HC],
            output_weights: vec![0.0; DS4_N_HC],
            output_embd: vec![0.0; DS4_N_EMBD],
            output_norm: vec![0.0; DS4_N_EMBD],
            q8_xq: vec![0; q8_blocks * 32],
            q8_xscale: vec![0.0; q8_blocks],
        }
    }
}

#[derive(Debug, Clone)]
struct RustAttentionDecodeScratch {
    q8_xq: Vec<i8>,
    q8_xscale: Vec<f32>,
    attn_norm: Vec<f32>,
    attn_qr: Vec<f32>,
    attn_qr_norm: Vec<f32>,
    attn_q: Vec<f32>,
    attn_kv: Vec<f32>,
    raw_cache_row: Vec<f32>,
    attn_comp_row: Vec<f32>,
    index_comp_row: Vec<f32>,
}

impl RustAttentionDecodeScratch {
    fn new() -> Self {
        let q8_blocks = DS4_N_EMBD.div_ceil(32);
        Self {
            q8_xq: vec![0; q8_blocks * 32],
            q8_xscale: vec![0.0; q8_blocks],
            attn_norm: Vec::new(),
            attn_qr: Vec::new(),
            attn_qr_norm: Vec::new(),
            attn_q: Vec::new(),
            attn_kv: Vec::new(),
            raw_cache_row: Vec::new(),
            attn_comp_row: Vec::new(),
            index_comp_row: Vec::new(),
        }
    }
}

#[derive(Debug, Clone)]
struct RustFfnDecodeScratch {
    q8_xq: Vec<i8>,
    q8_xscale: Vec<f32>,
    ffn_norm: Vec<f32>,
    shared_gate: Vec<f32>,
    shared_up: Vec<f32>,
    shared_mid: Vec<f32>,
    shared_out: Vec<f32>,
    routed_xq: Vec<Q8KBlock>,
    routed_mid_all: Vec<f32>,
    routed_midq: Vec<Q8KBlock>,
    routed_out: Vec<f32>,
    ffn_out: Vec<f32>,
}

impl RustFfnDecodeScratch {
    fn new() -> Self {
        let q8_blocks = DS4_N_EMBD.div_ceil(32);
        Self {
            q8_xq: vec![0; q8_blocks * 32],
            q8_xscale: vec![0.0; q8_blocks],
            ffn_norm: Vec::new(),
            shared_gate: Vec::new(),
            shared_up: Vec::new(),
            shared_mid: Vec::new(),
            shared_out: Vec::new(),
            routed_xq: Vec::new(),
            routed_mid_all: Vec::new(),
            routed_midq: Vec::new(),
            routed_out: Vec::new(),
            ffn_out: Vec::new(),
        }
    }
}

#[derive(Debug)]
pub struct RustSession<'a> {
    engine: &'a Engine,
    cache: RustKvCache,
    metal_graph: Option<Box<MetalDecodeGraph>>,
    tokens: Vec<i32>,
    position: u32,
    ctx_size: u32,
    raw_cap: u32,
    hc_state: Vec<f32>,
    logits: Vec<f32>,
    mtp_prefix_len: usize,
    mtp_hc_state: Vec<f32>,
    mtp_raw_kv: Vec<f32>,
    mtp_logits: Vec<f32>,
    mtp_draft_token: i32,
    mtp_draft_valid: bool,
    attention_scratch: RustAttentionDecodeScratch,
    ffn_scratch: RustFfnDecodeScratch,
    output_scratch: RustOutputScratch,
}

#[derive(Debug, Clone)]
struct RustDraftVerification {
    accepted: usize,
    cache: RustKvCache,
    tokens: Vec<i32>,
    position: u32,
    hc_state: Vec<f32>,
    logits: Vec<f32>,
    attention_scratch: RustAttentionDecodeScratch,
    ffn_scratch: RustFfnDecodeScratch,
    output_scratch: RustOutputScratch,
}

#[derive(Debug, Clone)]
struct RustMtpDraftFrontier {
    hc_state: Vec<f32>,
    raw_kv: Vec<f32>,
    logits: Vec<f32>,
    next_token: i32,
}

#[derive(Debug, Clone)]
struct RustBatch2Verification {
    row0_top: i32,
    row1_top: Option<i32>,
    prefix1: Option<RustDraftVerification>,
    verification: RustDraftVerification,
}

#[derive(Debug, Clone)]
struct RustDraftBatchVerifier<'a> {
    engine: &'a Engine,
    cache: RustKvCache,
    tokens: Vec<i32>,
    position: u32,
    hc_state: Vec<f32>,
    logits: Vec<f32>,
    attention_scratch: RustAttentionDecodeScratch,
    ffn_scratch: RustFfnDecodeScratch,
    output_scratch: RustOutputScratch,
}

impl<'a> RustDraftBatchVerifier<'a> {
    fn from_session(session: &RustSession<'a>) -> Self {
        Self {
            engine: session.engine,
            cache: session.cache.clone(),
            tokens: session.tokens.clone(),
            position: session.position,
            hc_state: session.hc_state.clone(),
            logits: session.logits.clone(),
            attention_scratch: session.attention_scratch.clone(),
            ffn_scratch: session.ffn_scratch.clone(),
            output_scratch: session.output_scratch.clone(),
        }
    }

    fn eval(&mut self, token: i32) -> Result<()> {
        self.hc_state = self.engine.eval_token_with_rust_backend_and_scratch(
            &mut self.cache,
            token,
            self.position,
            &mut self.attention_scratch,
            &mut self.ffn_scratch,
        )?;
        self.engine
            .output_logits_with_scratch(&self.hc_state, &mut self.output_scratch, &mut self.logits)?;
        self.tokens.push(token);
        self.position += 1;
        Ok(())
    }

    fn into_verification(self, accepted: usize) -> RustDraftVerification {
        RustDraftVerification {
            accepted,
            cache: self.cache,
            tokens: self.tokens,
            position: self.position,
            hc_state: self.hc_state,
            logits: self.logits,
            attention_scratch: self.attention_scratch,
            ffn_scratch: self.ffn_scratch,
            output_scratch: self.output_scratch,
        }
    }

    fn snapshot_verification(&self, accepted: usize) -> RustDraftVerification {
        self.clone().into_verification(accepted)
    }

    fn verify_exact_batch2(mut self, drafts: [i32; 2], eos_token: i32) -> Result<RustBatch2Verification> {
        let row0_top = sample_argmax(&self.logits);
        if row0_top != drafts[0] {
            return Ok(RustBatch2Verification {
                row0_top,
                row1_top: None,
                prefix1: None,
                verification: self.into_verification(0),
            });
        }

        self.eval(drafts[0])?;
        let prefix1 = self.snapshot_verification(1);
        if drafts[0] == eos_token {
            return Ok(RustBatch2Verification {
                row0_top,
                row1_top: None,
                prefix1: Some(prefix1.clone()),
                verification: prefix1,
            });
        }

        let row1_top = sample_argmax(&self.logits);
        if row1_top != drafts[1] {
            return Ok(RustBatch2Verification {
                row0_top,
                row1_top: Some(row1_top),
                prefix1: Some(prefix1.clone()),
                verification: prefix1,
            });
        }

        self.eval(drafts[1])?;
        Ok(RustBatch2Verification {
            row0_top,
            row1_top: Some(row1_top),
            prefix1: Some(prefix1),
            verification: self.into_verification(2),
        })
    }

    fn verify_exact_prefix(mut self, drafts: &[i32], eos_token: i32) -> Result<RustDraftVerification> {
        let mut accepted = 0usize;
        for token in drafts.iter().copied() {
            if sample_argmax(&self.logits) != token {
                break;
            }
            self.eval(token)?;
            accepted += 1;
            if token == eos_token {
                break;
            }
        }

        Ok(self.into_verification(accepted))
    }
}

#[derive(Clone, Debug)]
struct Q8KBlock {
    d: f32,
    qs: [i8; QK_K],
    bsums: [i16; QK_K / 16],
}

const KMASK_IQ2XS: [u8; 8] = [1, 2, 4, 8, 16, 32, 64, 128];

const KSIGNS_IQ2XS: [u8; 128] = [
    0, 129, 130, 3, 132, 5, 6, 135, 136, 9, 10, 139, 12, 141, 142, 15, 144, 17, 18, 147,
    20, 149, 150, 23, 24, 153, 154, 27, 156, 29, 30, 159, 160, 33, 34, 163, 36, 165, 166,
    39, 40, 169, 170, 43, 172, 45, 46, 175, 48, 177, 178, 51, 180, 53, 54, 183, 184, 57,
    58, 187, 60, 189, 190, 63, 192, 65, 66, 195, 68, 197, 198, 71, 72, 201, 202, 75, 204,
    77, 78, 207, 80, 209, 210, 83, 212, 85, 86, 215, 216, 89, 90, 219, 92, 221, 222, 95,
    96, 225, 226, 99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111, 240,
    113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
];

const IQ2XXS_GRID: [u64; 256] = [
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x08080808082b0808,
    0x08080808082b082b, 0x08080808082b2b08, 0x08080808082b2b2b, 0x0808080819080819,
    0x0808080819081908, 0x0808080819190808, 0x0808080819192b08, 0x08080808192b0819,
    0x08080808192b1908, 0x080808082b080808, 0x080808082b08082b, 0x080808082b082b2b,
    0x080808082b2b082b, 0x0808081908080819, 0x0808081908081908, 0x0808081908190808,
    0x0808081908191919, 0x0808081919080808, 0x080808192b081908, 0x080808192b192b08,
    0x0808082b08080808, 0x0808082b0808082b, 0x0808082b082b082b, 0x0808082b2b08082b,
    0x0808190808080819, 0x0808190808081908, 0x0808190808190808, 0x08081908082b0819,
    0x08081908082b1908, 0x0808190819080808, 0x080819081908082b, 0x0808190819082b08,
    0x08081908192b0808, 0x080819082b080819, 0x080819082b081908, 0x080819082b190808,
    0x080819082b2b1908, 0x0808191908080808, 0x080819190808082b, 0x0808191908082b08,
    0x08081919082b0808, 0x080819191908192b, 0x08081919192b2b19, 0x080819192b080808,
    0x080819192b190819, 0x0808192b08082b19, 0x0808192b08190808, 0x0808192b19080808,
    0x0808192b2b081908, 0x0808192b2b2b1908, 0x08082b0808080808, 0x08082b0808081919,
    0x08082b0808082b08, 0x08082b0808191908, 0x08082b08082b2b08, 0x08082b0819080819,
    0x08082b0819081908, 0x08082b0819190808, 0x08082b081919082b, 0x08082b082b082b08,
    0x08082b1908081908, 0x08082b1919080808, 0x08082b2b0808082b, 0x08082b2b08191908,
    0x0819080808080819, 0x0819080808081908, 0x0819080808190808, 0x08190808082b0819,
    0x0819080819080808, 0x08190808192b0808, 0x081908082b081908, 0x081908082b190808,
    0x081908082b191919, 0x0819081908080808, 0x0819081908082b08, 0x08190819082b0808,
    0x0819081919190808, 0x0819081919192b2b, 0x081908192b080808, 0x0819082b082b1908,
    0x0819082b19081919, 0x0819190808080808, 0x0819190808082b08, 0x08191908082b0808,
    0x08191908082b1919, 0x0819190819082b19, 0x081919082b080808, 0x0819191908192b08,
    0x08191919192b082b, 0x0819192b08080808, 0x0819192b0819192b, 0x08192b0808080819,
    0x08192b0808081908, 0x08192b0808190808, 0x08192b0819080808, 0x08192b082b080819,
    0x08192b1908080808, 0x08192b1908081919, 0x08192b192b2b0808, 0x08192b2b19190819,
    0x082b080808080808, 0x082b08080808082b, 0x082b080808082b2b, 0x082b080819081908,
    0x082b0808192b0819, 0x082b08082b080808, 0x082b08082b08082b, 0x082b0819082b2b19,
    0x082b081919082b08, 0x082b082b08080808, 0x082b082b0808082b, 0x082b190808080819,
    0x082b190808081908, 0x082b190808190808, 0x082b190819080808, 0x082b19081919192b,
    0x082b191908080808, 0x082b191919080819, 0x082b1919192b1908, 0x082b192b2b190808,
    0x082b2b0808082b08, 0x082b2b08082b0808, 0x082b2b082b191908, 0x082b2b2b19081908,
    0x1908080808080819, 0x1908080808081908, 0x1908080808190808, 0x1908080808192b08,
    0x19080808082b0819, 0x19080808082b1908, 0x1908080819080808, 0x1908080819082b08,
    0x190808081919192b, 0x19080808192b0808, 0x190808082b080819, 0x190808082b081908,
    0x190808082b190808, 0x1908081908080808, 0x19080819082b0808, 0x19080819192b0819,
    0x190808192b080808, 0x190808192b081919, 0x1908082b08080819, 0x1908082b08190808,
    0x1908082b19082b08, 0x1908082b1919192b, 0x1908082b192b2b08, 0x1908190808080808,
    0x1908190808082b08, 0x19081908082b0808, 0x190819082b080808, 0x190819082b192b19,
    0x190819190819082b, 0x19081919082b1908, 0x1908192b08080808, 0x19082b0808080819,
    0x19082b0808081908, 0x19082b0808190808, 0x19082b0819080808, 0x19082b0819081919,
    0x19082b1908080808, 0x19082b1919192b08, 0x19082b19192b0819, 0x19082b192b08082b,
    0x19082b2b19081919, 0x19082b2b2b190808, 0x1919080808080808, 0x1919080808082b08,
    0x1919080808190819, 0x1919080808192b19, 0x19190808082b0808, 0x191908082b080808,
    0x191908082b082b08, 0x1919081908081908, 0x191908191908082b, 0x191908192b2b1908,
    0x1919082b2b190819, 0x191919082b190808, 0x191919082b19082b, 0x1919191908082b2b,
    0x1919192b08080819, 0x1919192b19191908, 0x19192b0808080808, 0x19192b0808190819,
    0x19192b0808192b19, 0x19192b08192b1908, 0x19192b1919080808, 0x19192b2b08082b08,
    0x192b080808081908, 0x192b080808190808, 0x192b080819080808, 0x192b0808192b2b08,
    0x192b081908080808, 0x192b081919191919, 0x192b082b08192b08, 0x192b082b192b0808,
    0x192b190808080808, 0x192b190808081919, 0x192b191908190808, 0x192b19190819082b,
    0x192b19192b081908, 0x192b2b081908082b, 0x2b08080808080808, 0x2b0808080808082b,
    0x2b08080808082b2b, 0x2b08080819080819, 0x2b0808082b08082b, 0x2b08081908081908,
    0x2b08081908192b08, 0x2b08081919080808, 0x2b08082b08190819, 0x2b08190808080819,
    0x2b08190808081908, 0x2b08190808190808, 0x2b08190808191919, 0x2b08190819080808,
    0x2b081908192b0808, 0x2b08191908080808, 0x2b0819191908192b, 0x2b0819192b191908,
    0x2b08192b08082b19, 0x2b08192b19080808, 0x2b08192b192b0808, 0x2b082b080808082b,
    0x2b082b1908081908, 0x2b082b2b08190819, 0x2b19080808081908, 0x2b19080808190808,
    0x2b190808082b1908, 0x2b19080819080808, 0x2b1908082b2b0819, 0x2b1908190819192b,
    0x2b1908192b080808, 0x2b19082b19081919, 0x2b19190808080808, 0x2b191908082b082b,
    0x2b19190819081908, 0x2b19191919190819, 0x2b192b082b080819, 0x2b192b19082b0808,
    0x2b2b08080808082b, 0x2b2b080819190808, 0x2b2b08082b081919, 0x2b2b081908082b19,
    0x2b2b082b08080808, 0x2b2b190808192b08, 0x2b2b2b0819190808, 0x2b2b2b1908081908,
];

#[derive(Debug)]
pub struct Engine {
    backend: Backend,
    model_map: GgufMap,
    #[allow(dead_code)]
    mtp_map: Option<GgufMap>,
    #[allow(dead_code)]
    tensor_directory: TensorDirectory,
    #[allow(dead_code)]
    tensor_bindings: Ds4TensorBindings,
    #[allow(dead_code)]
    mtp_tensor_bindings: Option<Ds4MtpTensorBindings>,
    tokenizer: TokenizerMetadata,
    special_tokens: SpecialTokens,
    eos_token: i32,
    mtp_draft_tokens: i32,
    mtp_margin: f32,
    quality: bool,
}

unsafe impl Send for Engine {}

impl Engine {
    fn layer_binding(&self, layer: u32) -> Result<&Ds4LayerTensorBindings> {
        self.tensor_bindings
            .layers
            .get(layer as usize)
            .with_context(|| format!("layer index {} is outside the DS4 layer range", layer))
    }

    fn mtp_runtime(&self) -> Result<(&Ds4MtpTensorBindings, &[u8])> {
        let mtp = self.mtp_tensor_bindings.as_ref().context("MTP model is not loaded")?;
        let mtp_map = self.mtp_map.as_ref().context("MTP model is not loaded")?;
        Ok((mtp, mtp_map.as_bytes()))
    }

    fn hc_pre_from_state(
        &self,
        fn_tensor: &BoundTensor,
        scale_tensor: &BoundTensor,
        base_tensor: &BoundTensor,
        residual_hc: &[f32],
    ) -> Result<HcPreOutput> {
        hc_pre_from_tensors(
            self.model_map.as_bytes(),
            fn_tensor,
            scale_tensor,
            base_tensor,
            residual_hc,
        )
    }

    pub fn open(options: &EngineOptions) -> Result<Self> {
        let shared_mapping = options.backend == Backend::Metal;
        let model_map = GgufMap::open(options.model_path.as_str(), shared_mapping)?;
        let tensor_directory = TensorDirectory::from_bytes(model_map.as_bytes())?;
        validate_model_config_bytes(model_map.as_bytes())?;
        let tensor_bindings = tensor_directory.bind_ds4_tensors()?;
        let tokenizer = TokenizerMetadata::from_bytes(model_map.as_bytes())?;
        let mtp_map = options
            .mtp_path
            .as_deref()
            .map(|path| GgufMap::open(path, shared_mapping))
            .transpose()?;
        let mtp_tensor_bindings = mtp_map
            .as_ref()
            .map(|map| -> Result<_> {
                let tensor_directory = TensorDirectory::from_bytes(map.as_bytes())?;
                tensor_directory.bind_ds4_mtp_tensors()
            })
            .transpose()?;
        if options.backend == Backend::Metal {
            let model_range_offset = tensor_directory.tensor_data_pos();
            let model_range_size = model_map
                .len_u64()
                .checked_sub(model_range_offset)
                .context("model tensor data offset exceeds mapped model size")?;
            let mut model_ranges = vec![metal_runtime::ModelMapRange {
                model_map: model_map.as_ptr(),
                model_size: model_map.len_u64(),
                map_offset: model_range_offset,
                map_size: model_range_size,
            }];
            if let Some(mtp_map) = mtp_map.as_ref() {
                let mtp_dir = TensorDirectory::from_bytes(mtp_map.as_bytes())?;
                let mtp_range_offset = mtp_dir.tensor_data_pos();
                let mtp_range_size = mtp_map
                    .len_u64()
                    .checked_sub(mtp_range_offset)
                    .context("MTP tensor data offset exceeds mapped model size")?;
                model_ranges.push(metal_runtime::ModelMapRange {
                    model_map: mtp_map.as_ptr(),
                    model_size: mtp_map.len_u64(),
                    map_offset: mtp_range_offset,
                    map_size: mtp_range_size,
                });
            }
            metal_runtime::initialize(options.quality, &model_ranges)?;
        }
        let special_tokens = SpecialTokens::bootstrap(&tokenizer)?;
        let eos_token = special_tokens.eos();

        Ok(Self {
            backend: options.backend,
            model_map,
            mtp_map,
            tensor_directory,
            tensor_bindings,
            mtp_tensor_bindings,
            tokenizer,
            special_tokens,
            eos_token,
            mtp_draft_tokens: options.mtp_draft_tokens,
            mtp_margin: options.mtp_margin,
            quality: options.quality,
        })
    }

    pub fn summary(&self) -> Result<()> {
        ModelSummary::from_bytes(self.model_map.as_bytes())?.print();
        Ok(())
    }

    pub fn backend(&self) -> Backend {
        self.backend
    }

    pub fn dump_tokens(&self, tokens: &Tokens) {
        eprintln!("{:?}", tokens.as_slice());
        for token in tokens.as_slice() {
            let bytes = self.token_bytes(*token);
            let text = String::from_utf8_lossy(&bytes);
            eprintln!("{:>6}  {}", token, text);
        }
    }

    pub fn context_memory_estimate(backend: Backend, ctx_size: i32) -> ContextMemory {
        context_memory_estimate(backend, ctx_size)
    }

    pub fn token_eos(&self) -> i32 {
        self.eos_token
    }

    pub fn mtp_draft_tokens(&self) -> i32 {
        self.mtp_draft_tokens
    }

    pub fn has_rust_mtp_drafter(&self) -> bool {
        self.mtp_tensor_bindings.is_some() && self.mtp_map.is_some() && self.mtp_draft_tokens() > 1
    }

    pub fn token_bytes(&self, token: i32) -> Vec<u8> {
        self.tokenizer.token_bytes(token)
    }

    pub fn token_embedding_row(&self, token: i32) -> Result<Vec<f32>> {
        let token = u64::try_from(token).context("token id is negative")?;
        self.tensor_bindings
            .token_embd
            .read_f16_row(self.model_map.as_bytes(), token)
    }

    pub fn token_input_hc(&self, token: i32) -> Result<Vec<f32>> {
        let embedding = self.token_embedding_row(token)?;
        Ok(hc_from_plain_embedding(&embedding, DS4_N_HC))
    }

    pub fn mtp_input_hc(&self, prev_hc: &[f32], token: i32) -> Result<Vec<f32>> {
        let (mtp, mtp_bytes) = self.mtp_runtime()?;
        let embedding = self.token_embedding_row(token)?;
        let enorm = rms_norm_weight(
            &embedding,
            &mtp.enorm.read_f32_values(mtp_bytes)?,
            DS4_RMS_EPS,
        )?;
        let eproj = matvec_q8_0_tensor(&mtp.e_proj, mtp_bytes, &enorm)?;
        let eproj_hc = hc_from_plain_embedding(&eproj, DS4_N_HC);
        let hnorm_hc = rms_norm_weight_rows(
            prev_hc,
            &mtp.hnorm.read_f32_values(mtp_bytes)?,
            DS4_RMS_EPS,
        )?;
        let hproj_hc = matvec_q8_0_rows_tensor(&mtp.h_proj, mtp_bytes, &hnorm_hc)?;
        if eproj_hc.len() != hproj_hc.len() {
            bail!(
                "MTP input HC branch lengths differ: repeated embed {} vs HC projection {}",
                eproj_hc.len(),
                hproj_hc.len()
            )
        }

        Ok(eproj_hc
            .into_iter()
            .zip(hproj_hc)
            .map(|(embed, hc)| embed + hc)
            .collect())
    }

    pub fn layer_attention_pre(&self, layer: u32, residual_hc: &[f32]) -> Result<HcPreOutput> {
        let binding = self.layer_binding(layer)?;
        self.hc_pre_from_state(
            &binding.hc_attn_fn,
            &binding.hc_attn_scale,
            &binding.hc_attn_base,
            residual_hc,
        )
    }

    pub fn layer_attention_pre_from_token(&self, layer: u32, token: i32) -> Result<HcPreOutput> {
        let residual_hc = self.token_input_hc(token)?;
        self.layer_attention_pre(layer, &residual_hc)
    }

    pub fn layer_ffn_pre(&self, layer: u32, residual_hc: &[f32]) -> Result<HcPreOutput> {
        let binding = self.layer_binding(layer)?;
        self.hc_pre_from_state(
            &binding.hc_ffn_fn,
            &binding.hc_ffn_scale,
            &binding.hc_ffn_base,
            residual_hc,
        )
    }

    pub fn hc_post(&self, block_out: &[f32], pre: &HcPreOutput) -> Result<Vec<f32>> {
        hc_post(block_out, &pre.residual_hc, &pre.post, &pre.comb)
    }

    pub fn layer_attention_norm(&self, layer: u32, x: &[f32]) -> Result<Vec<f32>> {
        let binding = self.layer_binding(layer)?;
        let weight = binding.attn_norm.read_f32_values(self.model_map.as_bytes())?;
        rms_norm_weight(x, &weight, DS4_RMS_EPS)
    }

    fn layer_attention_norm_with_scratch(
        &self,
        layer: u32,
        x: &[f32],
        scratch: &mut RustAttentionDecodeScratch,
    ) -> Result<()> {
        let binding = self.layer_binding(layer)?;
        rms_norm_no_weight_into(x, DS4_RMS_EPS, &mut scratch.attn_norm)?;
        mul_f32_tensor_inplace(&mut scratch.attn_norm, &binding.attn_norm, self.model_map.as_bytes())
    }

    pub fn layer_q_projection(&self, layer: u32, norm: &[f32]) -> Result<(Vec<f32>, Vec<f32>)> {
        let binding = self.layer_binding(layer)?;
        let qr = matvec_q8_0_tensor(&binding.attn_q_a, self.model_map.as_bytes(), norm)?;
        let q_a_norm = binding.attn_q_a_norm.read_f32_values(self.model_map.as_bytes())?;
        let qr_norm = rms_norm_weight(&qr, &q_a_norm, DS4_RMS_EPS)?;
        let mut q = matvec_q8_0_tensor(&binding.attn_q_b, self.model_map.as_bytes(), &qr_norm)?;

        let n_head = binding.attn_sinks.read_f32_values(self.model_map.as_bytes())?.len();
        if n_head == 0 || q.len() % n_head != 0 {
            bail!("layer {} Q projection output does not divide into heads", layer)
        }
        let head_dim = q.len() / n_head;
        head_rms_norm_inplace(&mut q, n_head, head_dim, DS4_RMS_EPS)?;

        Ok((q, qr_norm))
    }

    fn layer_q_projection_with_scratch(&self, layer: u32, scratch: &mut RustAttentionDecodeScratch) -> Result<()> {
        let binding = self.layer_binding(layer)?;
        let gguf_bytes = self.model_map.as_bytes();
        matvec_q8_0_tensor_into(
            &binding.attn_q_a,
            gguf_bytes,
            &scratch.attn_norm,
            &mut scratch.q8_xq,
            &mut scratch.q8_xscale,
            &mut scratch.attn_qr,
        )?;
        rms_norm_no_weight_into(&scratch.attn_qr, DS4_RMS_EPS, &mut scratch.attn_qr_norm)?;
        mul_f32_tensor_inplace(&mut scratch.attn_qr_norm, &binding.attn_q_a_norm, gguf_bytes)?;
        matvec_q8_0_tensor_into(
            &binding.attn_q_b,
            gguf_bytes,
            &scratch.attn_qr_norm,
            &mut scratch.q8_xq,
            &mut scratch.q8_xscale,
            &mut scratch.attn_q,
        )?;

        let n_head = binding.attn_sinks.read_f32_values(gguf_bytes)?.len();
        if n_head == 0 || scratch.attn_q.len() % n_head != 0 {
            bail!("layer {} Q projection output does not divide into heads", layer)
        }
        let head_dim = scratch.attn_q.len() / n_head;
        head_rms_norm_inplace(&mut scratch.attn_q, n_head, head_dim, DS4_RMS_EPS)
    }

    pub fn layer_kv_projection(&self, layer: u32, norm: &[f32]) -> Result<Vec<f32>> {
        let binding = self.layer_binding(layer)?;
        let raw = matvec_q8_0_tensor(&binding.attn_kv, self.model_map.as_bytes(), norm)?;
        let kv_norm = binding.attn_kv_a_norm.read_f32_values(self.model_map.as_bytes())?;
        rms_norm_weight(&raw, &kv_norm, DS4_RMS_EPS)
    }

    fn layer_kv_projection_with_scratch(&self, layer: u32, scratch: &mut RustAttentionDecodeScratch) -> Result<()> {
        let binding = self.layer_binding(layer)?;
        let gguf_bytes = self.model_map.as_bytes();
        matvec_q8_0_tensor_into(
            &binding.attn_kv,
            gguf_bytes,
            &scratch.attn_norm,
            &mut scratch.q8_xq,
            &mut scratch.q8_xscale,
            &mut scratch.attn_qr,
        )?;
        rms_norm_no_weight_into(&scratch.attn_qr, DS4_RMS_EPS, &mut scratch.attn_kv)?;
        mul_f32_tensor_inplace(&mut scratch.attn_kv, &binding.attn_kv_a_norm, gguf_bytes)
    }

    fn layer_attention_qkv_with_scratch(
        &self,
        layer: u32,
        residual_hc: &[f32],
        pos: u32,
        scratch: &mut RustAttentionDecodeScratch,
    ) -> Result<HcPreOutput> {
        let pre = self.layer_attention_pre(layer, residual_hc)?;
        self.layer_attention_norm_with_scratch(layer, &pre.sublayer_input, scratch)?;
        self.layer_q_projection_with_scratch(layer, scratch)?;
        self.layer_kv_projection_with_scratch(layer, scratch)?;
        self.layer_apply_q_rope(layer, pos, &mut scratch.attn_q)?;
        self.layer_apply_kv_rope_and_fp8(layer, pos, &mut scratch.attn_kv)?;
        Ok(pre)
    }

    pub fn layer_attention_projections(
        &self,
        layer: u32,
        residual_hc: &[f32],
    ) -> Result<AttentionProjectionOutput> {
        let pre = self.layer_attention_pre(layer, residual_hc)?;
        let norm = self.layer_attention_norm(layer, &pre.sublayer_input)?;
        let (q, qr_norm) = self.layer_q_projection(layer, &norm)?;
        let kv = self.layer_kv_projection(layer, &norm)?;

        Ok(AttentionProjectionOutput {
            pre,
            norm,
            q,
            qr_norm,
            kv,
        })
    }

    fn layer_apply_q_rope_impl(&self, layer: u32, pos: u32, q: &mut [f32], inverse: bool) -> Result<()> {
        let binding = self.layer_binding(layer)?;
        let n_head = binding.attn_sinks.read_f32_values(self.model_map.as_bytes())?.len();
        if n_head == 0 || q.len() % n_head != 0 {
            bail!("layer {} Q tensor does not divide into heads", layer)
        }
        rope_tail_layer_inplace(q, n_head, q.len() / n_head, DS4_N_ROT, pos, layer, inverse)
    }

    pub fn layer_apply_q_rope(&self, layer: u32, pos: u32, q: &mut [f32]) -> Result<()> {
        self.layer_apply_q_rope_impl(layer, pos, q, false)
    }

    pub fn layer_unapply_q_rope(&self, layer: u32, pos: u32, heads: &mut [f32]) -> Result<()> {
        self.layer_apply_q_rope_impl(layer, pos, heads, true)
    }

    pub fn layer_apply_kv_rope_and_fp8(&self, layer: u32, pos: u32, kv: &mut [f32]) -> Result<()> {
        rope_tail_layer_inplace(kv, 1, kv.len(), DS4_N_ROT, pos, layer, false)?;
        dsv4_fp8_kv_quantize_row_inplace(kv, kv.len(), DS4_N_ROT)
    }

    pub fn layer_attention_qkv(
        &self,
        layer: u32,
        residual_hc: &[f32],
        pos: u32,
    ) -> Result<AttentionProjectionOutput> {
        let mut out = self.layer_attention_projections(layer, residual_hc)?;
        self.layer_apply_q_rope(layer, pos, &mut out.q)?;
        self.layer_apply_kv_rope_and_fp8(layer, pos, &mut out.kv)?;
        Ok(out)
    }

    pub fn layer_attention_qkv_from_token(
        &self,
        layer: u32,
        token: i32,
        pos: u32,
    ) -> Result<AttentionProjectionOutput> {
        let residual_hc = self.token_input_hc(token)?;
        self.layer_attention_qkv(layer, &residual_hc, pos)
    }

    pub fn layer_attention_rows_raw(&self, layer: u32, q: &[f32], kv_rows: &[f32]) -> Result<Vec<f32>> {
        let binding = self.layer_binding(layer)?;
        let sinks = binding.attn_sinks.read_f32_values(self.model_map.as_bytes())?;
        attention_rows_with_sinks(q, kv_rows, &sinks)
    }

    pub fn layer_attention_mixed(
        &self,
        layer: u32,
        q: &[f32],
        raw_kv_rows: &[f32],
        comp_kv_rows: &[f32],
        comp_allowed: Option<&[bool]>,
    ) -> Result<Vec<f32>> {
        let binding = self.layer_binding(layer)?;
        let sinks = binding.attn_sinks.read_f32_values(self.model_map.as_bytes())?;
        attention_rows_mixed_with_sinks(q, raw_kv_rows, comp_kv_rows, &sinks, comp_allowed)
    }

    pub fn layer_grouped_out(&self, layer: u32, heads: &[f32]) -> Result<Vec<f32>> {
        let binding = self.layer_binding(layer)?;
        let group_dim = usize::try_from(binding.attn_output_a.descriptor.dims[0])
            .context("attention output A group width does not fit in usize")?;
        let total_low = usize::try_from(binding.attn_output_a.descriptor.dims[1])
            .context("attention output A rank does not fit in usize")?;
        if heads.len() % group_dim != 0 {
            bail!("attention head vector length {} is not divisible by group width {}", heads.len(), group_dim)
        }
        let n_groups = heads.len() / group_dim;
        if total_low % n_groups != 0 {
            bail!("attention output A rank {} is not divisible by group count {}", total_low, n_groups)
        }
        let rank = total_low / n_groups;

        let low = matvec_q8_0_grouped_tensor(
            &binding.attn_output_a,
            self.model_map.as_bytes(),
            heads,
            n_groups,
            group_dim,
            rank,
        )?;
        matvec_q8_0_tensor(&binding.attn_output_b, self.model_map.as_bytes(), &low)
    }

    pub fn layer_attention_raw_uncompressed(
        &self,
        layer: u32,
        residual_hc: &[f32],
        prior_kv_rows: &[f32],
        pos: u32,
    ) -> Result<Vec<f32>> {
        if layer_compress_ratio(layer) != 0 {
            bail!("layer {} is compressed; raw uncompressed attention helper only applies to ratio-0 layers", layer)
        }

        let projection = self.layer_attention_qkv(layer, residual_hc, pos)?;
        if !prior_kv_rows.is_empty() && prior_kv_rows.len() % projection.kv.len() != 0 {
            bail!(
                "prior KV rows length {} is not a multiple of KV width {}",
                prior_kv_rows.len(),
                projection.kv.len()
            )
        }

        let mut all_kv = Vec::with_capacity(prior_kv_rows.len() + projection.kv.len());
        all_kv.extend_from_slice(prior_kv_rows);
        all_kv.extend_from_slice(&projection.kv);

        let mut heads = self.layer_attention_rows_raw(layer, &projection.q, &all_kv)?;
        self.layer_unapply_q_rope(layer, pos, &mut heads)?;
        let attn_out = self.layer_grouped_out(layer, &heads)?;
        self.hc_post(&attn_out, &projection.pre)
    }

    pub fn layer_attention_raw_uncompressed_from_token(
        &self,
        layer: u32,
        token: i32,
        prior_kv_rows: &[f32],
        pos: u32,
    ) -> Result<Vec<f32>> {
        let residual_hc = self.token_input_hc(token)?;
        self.layer_attention_raw_uncompressed(layer, &residual_hc, prior_kv_rows, pos)
    }

    pub fn layer_compressor_decode_one(
        &self,
        layer: u32,
        attn_norm: &[f32],
        state_kv: &mut [f32],
        state_score: &mut [f32],
        pos: u32,
    ) -> Result<Option<Vec<f32>>> {
        let binding = self.layer_binding(layer)?;
        let ratio = usize::try_from(binding.compress_ratio).context("compress ratio does not fit in usize")?;
        if ratio == 0 {
            bail!("layer {} does not use compressed attention", layer)
        }

        let ape = binding
            .attn_compressor_ape
            .as_ref()
            .context("missing attention compressor APE tensor")?;
        let wkv = binding
            .attn_compressor_kv
            .as_ref()
            .context("missing attention compressor KV tensor")?;
        let wgate = binding
            .attn_compressor_gate
            .as_ref()
            .context("missing attention compressor gate tensor")?;
        let norm = binding
            .attn_compressor_norm
            .as_ref()
            .context("missing attention compressor norm tensor")?;
        let head_dim = usize::try_from(norm.descriptor.dims[0]).context("compressor norm width does not fit in usize")?;

        compressor_decode_one_tensor(
            self.model_map.as_bytes(),
            layer,
            attn_norm,
            wkv,
            wgate,
            ape,
            norm,
            state_kv,
            state_score,
            head_dim,
            ratio,
            pos,
        )
    }

    pub fn layer_indexer_compressor_decode_one(
        &self,
        layer: u32,
        attn_norm: &[f32],
        state_kv: &mut [f32],
        state_score: &mut [f32],
        pos: u32,
    ) -> Result<Option<Vec<f32>>> {
        let binding = self.layer_binding(layer)?;
        if binding.compress_ratio != 4 {
            bail!("layer {} does not use ratio-4 indexer compression", layer)
        }

        let ape = binding
            .indexer_compressor_ape
            .as_ref()
            .context("missing indexer compressor APE tensor")?;
        let wkv = binding
            .indexer_compressor_kv
            .as_ref()
            .context("missing indexer compressor KV tensor")?;
        let wgate = binding
            .indexer_compressor_gate
            .as_ref()
            .context("missing indexer compressor gate tensor")?;
        let norm = binding
            .indexer_compressor_norm
            .as_ref()
            .context("missing indexer compressor norm tensor")?;
        let head_dim = usize::try_from(norm.descriptor.dims[0]).context("indexer compressor norm width does not fit in usize")?;

        compressor_decode_one_tensor(
            self.model_map.as_bytes(),
            layer,
            attn_norm,
            wkv,
            wgate,
            ape,
            norm,
            state_kv,
            state_score,
            head_dim,
            4,
            pos,
        )
    }

    pub fn layer_indexer_allowed_decode_one(
        &self,
        layer: u32,
        attn_norm: &[f32],
        qr_norm: &[f32],
        index_comp_rows: &[f32],
        pos: u32,
    ) -> Result<Vec<bool>> {
        let binding = self.layer_binding(layer)?;
        if binding.compress_ratio != 4 {
            bail!("layer {} does not use ratio-4 indexer selection", layer)
        }

        let index_q_b = binding
            .indexer_attn_q_b
            .as_ref()
            .context("missing indexer Q projection tensor")?;
        let index_proj = binding
            .indexer_proj
            .as_ref()
            .context("missing indexer projection tensor")?;

        let head_dim = usize::try_from(DS4_N_INDEXER_HEAD_DIM).context("indexer head dim does not fit in usize")?;
        if index_comp_rows.len() % head_dim != 0 {
            bail!(
                "indexer compressed row length {} is not divisible by indexer head width {}",
                index_comp_rows.len(),
                head_dim
            )
        }

        let n_comp = index_comp_rows.len() / head_dim;
        if n_comp == 0 {
            return Ok(Vec::new());
        }

        let top_k = usize::min(usize::try_from(DS4_N_INDEXER_TOP_K).unwrap_or(usize::MAX), n_comp);
        if top_k == n_comp {
            return Ok(vec![true; n_comp]);
        }

        let mut q = matvec_any_tensor(index_q_b, self.model_map.as_bytes(), qr_norm)?;
        let n_head = q.len() / head_dim;
        rope_tail_layer_inplace(&mut q, n_head, head_dim, DS4_N_ROT, pos, layer, false)?;

        let mut weights = matvec_any_tensor(index_proj, self.model_map.as_bytes(), attn_norm)?;
        let scale = 1.0 / ((head_dim * n_head) as f32).sqrt();
        for weight in &mut weights {
            *weight *= scale;
        }

        let mut scores = vec![0.0f32; n_comp];
        for row in 0..n_comp {
            let kv = &index_comp_rows[row * head_dim..(row + 1) * head_dim];
            let mut score = 0.0f32;
            for head in 0..n_head {
                let qh = &q[head * head_dim..(head + 1) * head_dim];
                let dot = dot_f32(kv, qh)?.max(0.0);
                score += dot * weights[head];
            }
            scores[row] = score;
        }

        Ok(select_top_k_mask(&scores, top_k))
    }

    pub fn layer_attention_compressed_decode_step(
        &self,
        layer: u32,
        residual_hc: &[f32],
        prior_raw_kv_rows: &[f32],
        prior_comp_kv_rows: &[f32],
        state_kv: &mut [f32],
        state_score: &mut [f32],
        comp_allowed: Option<&[bool]>,
        pos: u32,
    ) -> Result<CompressedAttentionStepOutput> {
        if layer_compress_ratio(layer) == 0 {
            bail!("layer {} is not compressed", layer)
        }

        let projection = self.layer_attention_qkv(layer, residual_hc, pos)?;
        if !prior_raw_kv_rows.is_empty() && prior_raw_kv_rows.len() % projection.kv.len() != 0 {
            bail!(
                "prior raw KV rows length {} is not a multiple of KV width {}",
                prior_raw_kv_rows.len(),
                projection.kv.len()
            )
        }
        if !prior_comp_kv_rows.is_empty() && prior_comp_kv_rows.len() % projection.kv.len() != 0 {
            bail!(
                "prior compressed KV rows length {} is not a multiple of KV width {}",
                prior_comp_kv_rows.len(),
                projection.kv.len()
            )
        }

        let mut raw_kv_rows = Vec::with_capacity(prior_raw_kv_rows.len() + projection.kv.len());
        raw_kv_rows.extend_from_slice(prior_raw_kv_rows);
        raw_kv_rows.extend(f16_round_slice(&projection.kv));

        let new_comp = self.layer_compressor_decode_one(layer, &projection.norm, state_kv, state_score, pos)?;
        let mut comp_kv_rows = Vec::with_capacity(prior_comp_kv_rows.len() + new_comp.as_ref().map_or(0, Vec::len));
        comp_kv_rows.extend_from_slice(prior_comp_kv_rows);
        if let Some(comp) = new_comp.as_ref() {
            comp_kv_rows.extend(f16_round_slice(comp));
        }

        let mut heads = self.layer_attention_mixed(layer, &projection.q, &raw_kv_rows, &comp_kv_rows, comp_allowed)?;
        self.layer_unapply_q_rope(layer, pos, &mut heads)?;
        let attn_out = self.layer_grouped_out(layer, &heads)?;
        let after_attn_hc = self.hc_post(&attn_out, &projection.pre)?;

        Ok(CompressedAttentionStepOutput {
            projection,
            new_comp,
            after_attn_hc,
        })
    }

    pub fn layer_attention_compressed_decode_step_from_token(
        &self,
        layer: u32,
        token: i32,
        prior_raw_kv_rows: &[f32],
        prior_comp_kv_rows: &[f32],
        state_kv: &mut [f32],
        state_score: &mut [f32],
        comp_allowed: Option<&[bool]>,
        pos: u32,
    ) -> Result<CompressedAttentionStepOutput> {
        let residual_hc = self.token_input_hc(token)?;
        self.layer_attention_compressed_decode_step(
            layer,
            &residual_hc,
            prior_raw_kv_rows,
            prior_comp_kv_rows,
            state_kv,
            state_score,
            comp_allowed,
            pos,
        )
    }

    pub fn layer_attention_projections_from_token(
        &self,
        layer: u32,
        token: i32,
    ) -> Result<AttentionProjectionOutput> {
        let residual_hc = self.token_input_hc(token)?;
        self.layer_attention_projections(layer, &residual_hc)
    }

    pub fn layer_shared_ffn(&self, layer: u32, norm: &[f32]) -> Result<Vec<f32>> {
        let binding = self.layer_binding(layer)?;
        let gate = matvec_q8_0_tensor(&binding.ffn_gate_shexp, self.model_map.as_bytes(), norm)?;
        let up = matvec_q8_0_tensor(&binding.ffn_up_shexp, self.model_map.as_bytes(), norm)?;
        let mid = swiglu(&gate, &up)?;
        matvec_q8_0_tensor(&binding.ffn_down_shexp, self.model_map.as_bytes(), &mid)
    }

    fn layer_shared_ffn_with_scratch(&self, layer: u32, scratch: &mut RustFfnDecodeScratch) -> Result<()> {
        let binding = self.layer_binding(layer)?;
        let gguf_bytes = self.model_map.as_bytes();
        matvec_q8_0_tensor_into(
            &binding.ffn_gate_shexp,
            gguf_bytes,
            &scratch.ffn_norm,
            &mut scratch.q8_xq,
            &mut scratch.q8_xscale,
            &mut scratch.shared_gate,
        )?;
        matvec_q8_0_tensor_into(
            &binding.ffn_up_shexp,
            gguf_bytes,
            &scratch.ffn_norm,
            &mut scratch.q8_xq,
            &mut scratch.q8_xscale,
            &mut scratch.shared_up,
        )?;
        swiglu_into(&scratch.shared_gate, &scratch.shared_up, &mut scratch.shared_mid)?;
        matvec_q8_0_tensor_into(
            &binding.ffn_down_shexp,
            gguf_bytes,
            &scratch.shared_mid,
            &mut scratch.q8_xq,
            &mut scratch.q8_xscale,
            &mut scratch.shared_out,
        )
    }

    pub fn layer_router_probs(&self, layer: u32, norm: &[f32]) -> Result<Vec<f32>> {
        let binding = self.layer_binding(layer)?;
        let logits = matvec_f16_tensor(&binding.ffn_gate_inp, self.model_map.as_bytes(), norm)?;
        Ok(logits
            .into_iter()
            .map(|logit| softplus_stable(logit).sqrt())
            .collect())
    }

    pub fn layer_select_experts(&self, layer: u32, norm: &[f32], token: i32) -> Result<ExpertSelection> {
        let binding = self.layer_binding(layer)?;
        let probs = self.layer_router_probs(layer, norm)?;
        if probs.is_empty() {
            bail!("layer {} router produced no expert probabilities", layer)
        }

        let (selected, weights) = if let Some(table) = binding.ffn_gate_tid2eid.as_ref() {
            let selected = read_i32_tensor_row(table, self.model_map.as_bytes(), token)?;
            let weights = hash_router_weights_from_probs(&probs, &selected)?;
            (selected, weights)
        } else {
            let mut selection = probs.clone();
            if let Some(bias) = binding.ffn_exp_probs_b.as_ref() {
                let bias_values = bias.read_f32_values(self.model_map.as_bytes())?;
                if bias_values.len() != selection.len() {
                    bail!(
                        "expert bias length {} does not match router probability length {}",
                        bias_values.len(),
                        selection.len()
                    )
                }
                for (score, bias) in selection.iter_mut().zip(bias_values.iter().copied()) {
                    *score += bias;
                }
            }

            let selected = select_top_k_desc_indices(&selection, usize::min(DS4_N_EXPERT_USED, selection.len()))
                .into_iter()
                .map(|index| i32::try_from(index).unwrap_or_default())
                .collect::<Vec<_>>();
            let weights = hash_router_weights_from_probs(&probs, &selected)?;
            (selected, weights)
        };

        Ok(ExpertSelection {
            probs,
            selected,
            weights,
        })
    }

    pub fn layer_ffn_prepare(
        &self,
        layer: u32,
        residual_hc: &[f32],
        token: i32,
    ) -> Result<FfnPreparationOutput> {
        let pre = self.layer_ffn_pre(layer, residual_hc)?;
        let binding = self.layer_binding(layer)?;
        let norm_weight = binding.ffn_norm.read_f32_values(self.model_map.as_bytes())?;
        let norm = rms_norm_weight(&pre.sublayer_input, &norm_weight, DS4_RMS_EPS)?;
        let shared = self.layer_shared_ffn(layer, &norm)?;
        let selection = self.layer_select_experts(layer, &norm, token)?;

        Ok(FfnPreparationOutput {
            pre,
            norm,
            shared,
            selection,
        })
    }

    pub fn layer_ffn_prepare_from_token(&self, layer: u32, token: i32) -> Result<FfnPreparationOutput> {
        let residual_hc = self.token_input_hc(token)?;
        self.layer_ffn_prepare(layer, &residual_hc, token)
    }

    fn layer_ffn_norm_with_scratch(
        &self,
        layer: u32,
        residual_hc: &[f32],
        scratch: &mut RustFfnDecodeScratch,
    ) -> Result<HcPreOutput> {
        let pre = self.layer_ffn_pre(layer, residual_hc)?;
        let binding = self.layer_binding(layer)?;
        rms_norm_no_weight_into(&pre.sublayer_input, DS4_RMS_EPS, &mut scratch.ffn_norm)?;
        mul_f32_tensor_inplace(&mut scratch.ffn_norm, &binding.ffn_norm, self.model_map.as_bytes())?;
        Ok(pre)
    }

    pub fn layer_routed_expert_down(
        &self,
        layer: u32,
        selected: &[i32],
        mids: &[f32],
    ) -> Result<Vec<f32>> {
        let binding = self.layer_binding(layer)?;
        let down = &binding.ffn_down_exps;
        if down.descriptor.tensor_type != DS4_TENSOR_Q2_K || down.descriptor.ndim != 3 {
            bail!("{} is not a 3D Q2_K expert tensor", down.name)
        }
        if selected.is_empty() {
            bail!("routed expert down projection requires at least one selected expert")
        }

        let in_dim = usize::try_from(down.descriptor.dims[0]).context("expert down input width does not fit in usize")?;
        let out_dim = usize::try_from(down.descriptor.dims[1]).context("expert down output width does not fit in usize")?;
        if mids.len() != selected.len() * in_dim {
            bail!(
                "mid activation length {} does not match {} selected experts with width {}",
                mids.len(),
                selected.len(),
                in_dim
            )
        }

        let mut quantized = Vec::with_capacity(selected.len());
        for slot in 0..selected.len() {
            let start = slot * in_dim;
            let end = start + in_dim;
            quantized.push(quantize_row_q8_k(&mids[start..end])?);
        }

        let mut out = vec![0.0f32; out_dim];
        for (slot, expert) in selected.iter().copied().enumerate() {
            let (base, row_bytes, _, tensor_out_dim) = expert_tensor_bytes(
                down,
                self.model_map.as_bytes(),
                u32::try_from(expert).context("selected expert index is negative")?,
            )?;
            if tensor_out_dim != out_dim {
                bail!("expert {} output width {} does not match expected {}", expert, tensor_out_dim, out_dim)
            }
            for row in 0..out_dim {
                let start = row * row_bytes;
                let end = start + row_bytes;
                out[row] += dot_q2_k_q8_k_row(&base[start..end], &quantized[slot], in_dim)?;
            }
        }

        Ok(out)
    }

    fn layer_routed_expert_down_with_scratch(
        &self,
        layer: u32,
        selected: &[i32],
        scratch: &mut RustFfnDecodeScratch,
    ) -> Result<()> {
        let binding = self.layer_binding(layer)?;
        let down = &binding.ffn_down_exps;
        if down.descriptor.tensor_type != DS4_TENSOR_Q2_K || down.descriptor.ndim != 3 {
            bail!("{} is not a 3D Q2_K expert tensor", down.name)
        }
        if selected.is_empty() {
            bail!("routed expert down projection requires at least one selected expert")
        }

        let in_dim = usize::try_from(down.descriptor.dims[0]).context("expert down input width does not fit in usize")?;
        let out_dim = usize::try_from(down.descriptor.dims[1]).context("expert down output width does not fit in usize")?;
        if scratch.routed_mid_all.len() != selected.len() * in_dim {
            bail!(
                "mid activation length {} does not match {} selected experts with width {}",
                scratch.routed_mid_all.len(),
                selected.len(),
                in_dim
            )
        }

        let blocks_per_mid = in_dim / QK_K;
        scratch.routed_midq.clear();
        scratch.routed_midq.resize(selected.len() * blocks_per_mid, zero_q8_k_block());
        for slot in 0..selected.len() {
            let start = slot * in_dim;
            let end = start + in_dim;
            let q_start = slot * blocks_per_mid;
            let q_end = q_start + blocks_per_mid;
            quantize_row_q8_k_into_slice(&scratch.routed_mid_all[start..end], &mut scratch.routed_midq[q_start..q_end])?;
        }

        scratch.routed_out.clear();
        scratch.routed_out.resize(out_dim, 0.0);
        for (slot, expert) in selected.iter().copied().enumerate() {
            let (base, row_bytes, _, tensor_out_dim) = expert_tensor_bytes(
                down,
                self.model_map.as_bytes(),
                u32::try_from(expert).context("selected expert index is negative")?,
            )?;
            if tensor_out_dim != out_dim {
                bail!("expert {} output width {} does not match expected {}", expert, tensor_out_dim, out_dim)
            }
            let q_start = slot * blocks_per_mid;
            let q_end = q_start + blocks_per_mid;
            let quantized = &scratch.routed_midq[q_start..q_end];
            for row in 0..out_dim {
                let start = row * row_bytes;
                let end = start + row_bytes;
                scratch.routed_out[row] += dot_q2_k_q8_k_row(&base[start..end], quantized, in_dim)?;
            }
        }

        Ok(())
    }

    pub fn layer_routed_expert_mid(
        &self,
        layer: u32,
        norm: &[f32],
        selected: &[i32],
        expert_weight: &[f32],
    ) -> Result<Vec<f32>> {
        if selected.len() != expert_weight.len() {
            bail!(
                "selected expert count {} does not match weight count {}",
                selected.len(),
                expert_weight.len()
            )
        }
        if selected.is_empty() {
            bail!("routed expert mid projection requires at least one selected expert")
        }

        let binding = self.layer_binding(layer)?;
        let gate = &binding.ffn_gate_exps;
        let up = &binding.ffn_up_exps;
        if gate.descriptor.tensor_type != DS4_TENSOR_IQ2_XXS || gate.descriptor.ndim != 3 {
            bail!("{} is not a 3D IQ2_XXS expert tensor", gate.name)
        }
        if up.descriptor.tensor_type != DS4_TENSOR_IQ2_XXS || up.descriptor.ndim != 3 {
            bail!("{} is not a 3D IQ2_XXS expert tensor", up.name)
        }

        let xq = quantize_row_q8_k(norm)?;
        let in_dim = usize::try_from(gate.descriptor.dims[0]).context("expert gate input width does not fit in usize")?;
        let out_dim = usize::try_from(gate.descriptor.dims[1]).context("expert gate output width does not fit in usize")?;
        let mut mids = vec![0.0f32; selected.len() * out_dim];

        for (slot, expert) in selected.iter().copied().enumerate() {
            let expert_index = u32::try_from(expert).context("selected expert index is negative")?;
            let (gate_base, gate_row_bytes, gate_in_dim, gate_out_dim) =
                expert_tensor_bytes(gate, self.model_map.as_bytes(), expert_index)?;
            let (up_base, up_row_bytes, up_in_dim, up_out_dim) =
                expert_tensor_bytes(up, self.model_map.as_bytes(), expert_index)?;
            if gate_in_dim != up_in_dim || gate_out_dim != up_out_dim {
                bail!("expert {} gate/up tensor shapes do not match", expert)
            }
            if gate_in_dim != in_dim || gate_out_dim != out_dim {
                bail!("expert {} gate/up tensor shape does not match expected routed expert layout", expert)
            }

            for row in 0..out_dim {
                let gate_start = row * gate_row_bytes;
                let gate_end = gate_start + gate_row_bytes;
                let up_start = row * up_row_bytes;
                let up_end = up_start + up_row_bytes;
                let mut gate_value = dot_iq2_xxs_q8_k_row(&gate_base[gate_start..gate_end], &xq, in_dim)?;
                let mut up_value = dot_iq2_xxs_q8_k_row(&up_base[up_start..up_end], &xq, in_dim)?;
                if DS4_SWIGLU_CLAMP_EXP > 1.0e-6 {
                    gate_value = gate_value.min(DS4_SWIGLU_CLAMP_EXP);
                    up_value = up_value.clamp(-DS4_SWIGLU_CLAMP_EXP, DS4_SWIGLU_CLAMP_EXP);
                }
                mids[slot * out_dim + row] = silu(gate_value) * up_value * expert_weight[slot];
            }
        }

        Ok(mids)
    }

    fn layer_routed_expert_mid_with_scratch(
        &self,
        layer: u32,
        selected: &[i32],
        expert_weight: &[f32],
        scratch: &mut RustFfnDecodeScratch,
    ) -> Result<()> {
        if selected.len() != expert_weight.len() {
            bail!(
                "selected expert count {} does not match weight count {}",
                selected.len(),
                expert_weight.len()
            )
        }
        if selected.is_empty() {
            bail!("routed expert mid projection requires at least one selected expert")
        }

        let binding = self.layer_binding(layer)?;
        let gate = &binding.ffn_gate_exps;
        let up = &binding.ffn_up_exps;
        if gate.descriptor.tensor_type != DS4_TENSOR_IQ2_XXS || gate.descriptor.ndim != 3 {
            bail!("{} is not a 3D IQ2_XXS expert tensor", gate.name)
        }
        if up.descriptor.tensor_type != DS4_TENSOR_IQ2_XXS || up.descriptor.ndim != 3 {
            bail!("{} is not a 3D IQ2_XXS expert tensor", up.name)
        }

        let in_dim = usize::try_from(gate.descriptor.dims[0]).context("expert gate input width does not fit in usize")?;
        let out_dim = usize::try_from(gate.descriptor.dims[1]).context("expert gate output width does not fit in usize")?;
        let blocks = scratch.ffn_norm.len() / QK_K;
        scratch.routed_xq.clear();
        scratch.routed_xq.resize(blocks, zero_q8_k_block());
        quantize_row_q8_k_into_slice(&scratch.ffn_norm, &mut scratch.routed_xq)?;

        scratch.routed_mid_all.clear();
        scratch.routed_mid_all.resize(selected.len() * out_dim, 0.0);
        for (slot, expert) in selected.iter().copied().enumerate() {
            let expert_index = u32::try_from(expert).context("selected expert index is negative")?;
            let (gate_base, gate_row_bytes, gate_in_dim, gate_out_dim) =
                expert_tensor_bytes(gate, self.model_map.as_bytes(), expert_index)?;
            let (up_base, up_row_bytes, up_in_dim, up_out_dim) =
                expert_tensor_bytes(up, self.model_map.as_bytes(), expert_index)?;
            if gate_in_dim != up_in_dim || gate_out_dim != up_out_dim {
                bail!("expert {} gate/up tensor shapes do not match", expert)
            }
            if gate_in_dim != in_dim || gate_out_dim != out_dim {
                bail!("expert {} gate/up tensor shape does not match expected routed expert layout", expert)
            }

            for row in 0..out_dim {
                let gate_start = row * gate_row_bytes;
                let gate_end = gate_start + gate_row_bytes;
                let up_start = row * up_row_bytes;
                let up_end = up_start + up_row_bytes;
                let mut gate_value = dot_iq2_xxs_q8_k_row(&gate_base[gate_start..gate_end], &scratch.routed_xq, in_dim)?;
                let mut up_value = dot_iq2_xxs_q8_k_row(&up_base[up_start..up_end], &scratch.routed_xq, in_dim)?;
                if DS4_SWIGLU_CLAMP_EXP > 1.0e-6 {
                    gate_value = gate_value.min(DS4_SWIGLU_CLAMP_EXP);
                    up_value = up_value.clamp(-DS4_SWIGLU_CLAMP_EXP, DS4_SWIGLU_CLAMP_EXP);
                }
                scratch.routed_mid_all[slot * out_dim + row] = silu(gate_value) * up_value * expert_weight[slot];
            }
        }

        Ok(())
    }

    pub fn layer_routed_moe(&self, layer: u32, norm: &[f32], token: i32) -> Result<Vec<f32>> {
        let selection = self.layer_select_experts(layer, norm, token)?;
        let mids = self.layer_routed_expert_mid(layer, norm, &selection.selected, &selection.weights)?;
        self.layer_routed_expert_down(layer, &selection.selected, &mids)
    }

    pub fn layer_ffn(&self, layer: u32, residual_hc: &[f32], token: i32) -> Result<Vec<f32>> {
        let prepared = self.layer_ffn_prepare(layer, residual_hc, token)?;
        let routed = self.layer_routed_expert_down(
            layer,
            &prepared.selection.selected,
            &self.layer_routed_expert_mid(
                layer,
                &prepared.norm,
                &prepared.selection.selected,
                &prepared.selection.weights,
            )?,
        )?;
        if routed.len() != prepared.shared.len() {
            bail!(
                "routed MoE width {} does not match shared expert width {}",
                routed.len(),
                prepared.shared.len()
            )
        }

        let ffn_out = routed
            .iter()
            .zip(prepared.shared.iter())
            .map(|(routed, shared)| routed + shared)
            .collect::<Vec<_>>();
        self.hc_post(&ffn_out, &prepared.pre)
    }

    fn layer_ffn_with_scratch(
        &self,
        layer: u32,
        residual_hc: &[f32],
        token: i32,
        scratch: &mut RustFfnDecodeScratch,
    ) -> Result<Vec<f32>> {
        let pre = self.layer_ffn_norm_with_scratch(layer, residual_hc, scratch)?;
        self.layer_shared_ffn_with_scratch(layer, scratch)?;
        let selection = self.layer_select_experts(layer, &scratch.ffn_norm, token)?;
        self.layer_routed_expert_mid_with_scratch(layer, &selection.selected, &selection.weights, scratch)?;
        self.layer_routed_expert_down_with_scratch(layer, &selection.selected, scratch)?;
        if scratch.routed_out.len() != scratch.shared_out.len() {
            bail!(
                "routed MoE width {} does not match shared expert width {}",
                scratch.routed_out.len(),
                scratch.shared_out.len()
            )
        }

        scratch.ffn_out.clear();
        scratch.ffn_out.reserve(scratch.routed_out.len());
        scratch.ffn_out.extend(
            scratch
                .routed_out
                .iter()
                .zip(scratch.shared_out.iter())
                .map(|(routed, shared)| routed + shared),
        );
        self.hc_post(&scratch.ffn_out, &pre)
    }

    pub fn layer_ffn_from_token(&self, layer: u32, token: i32) -> Result<Vec<f32>> {
        let residual_hc = self.token_input_hc(token)?;
        self.layer_ffn(layer, &residual_hc, token)
    }

    pub fn layer_forward_self(
        &self,
        layer: u32,
        residual_hc: &[f32],
        token: i32,
        pos: u32,
    ) -> Result<Vec<f32>> {
        let projection = self.layer_attention_qkv(layer, residual_hc, pos)?;
        let mut heads = self.layer_attention_rows_raw(layer, &projection.q, &projection.kv)?;
        self.layer_unapply_q_rope(layer, pos, &mut heads)?;
        let attn_out = self.layer_grouped_out(layer, &heads)?;
        let after_attn_hc = self.hc_post(&attn_out, &projection.pre)?;
        self.layer_ffn(layer, &after_attn_hc, token)
    }

    pub fn forward_first_token_hc(&self, token: i32) -> Result<Vec<f32>> {
        let mut cur = self.token_input_hc(token)?;
        for layer in 0..DS4_N_LAYER {
            cur = self.layer_forward_self(layer, &cur, token, 0)?;
        }
        Ok(cur)
    }

    pub fn forward_first_token_logits(&self, token: i32) -> Result<Vec<f32>> {
        let hc = self.forward_first_token_hc(token)?;
        self.output_logits(&hc)
    }

    pub fn new_rust_kv_cache(&self, ctx_size: u32, raw_cap: u32) -> RustKvCache {
        let cap_raw = if raw_cap == 0 { DS4_N_SWA.min(ctx_size) } else { raw_cap.min(ctx_size).max(1) };
        let mut layers = Vec::with_capacity(DS4_N_LAYER as usize);
        for layer in 0..DS4_N_LAYER {
            let ratio = layer_compress_ratio(layer);
            let mut cache = RustLayerCache {
                raw_kv: Vec::new(),
                attn_comp_kv: Vec::new(),
                attn_state_kv: Vec::new(),
                attn_state_score: Vec::new(),
                index_comp_kv: Vec::new(),
                index_state_kv: Vec::new(),
                index_state_score: Vec::new(),
                cap_raw,
                comp_cap: 0,
                compress_ratio: ratio,
            };

            if ratio != 0 {
                let coff = if ratio == 4 { 2usize } else { 1usize };
                let comp_cap = ctx_size / ratio + 2;
                let attn_width = coff * usize::try_from(DS4_N_HEAD_DIM).unwrap_or_default();
                let attn_rows = coff * usize::try_from(ratio).unwrap_or_default();
                cache.comp_cap = comp_cap;
                cache.attn_state_kv = vec![0.0; attn_width * attn_rows];
                cache.attn_state_score = vec![DS4_NEG_INF; attn_width * attn_rows];
                if ratio == 4 {
                    let index_width = coff * usize::try_from(DS4_N_INDEXER_HEAD_DIM).unwrap_or_default();
                    let index_rows = coff * usize::try_from(ratio).unwrap_or_default();
                    cache.index_state_kv = vec![0.0; index_width * index_rows];
                    cache.index_state_score = vec![DS4_NEG_INF; index_width * index_rows];
                }
            }

            layers.push(cache);
        }
        RustKvCache { layers }
    }

    // Allocate a Metal decode graph for a session of `ctx_size` tokens.
    // `raw_cap` follows the same semantics as new_rust_kv_cache.
    // Returns None if any Metal allocation fails.
    pub(crate) fn new_metal_decode_graph(&self, ctx_size: u32, raw_cap: u32) -> Option<Box<MetalDecodeGraph>> {
        let cap_raw = if raw_cap == 0 { DS4_N_SWA.min(ctx_size) } else { raw_cap.min(ctx_size).max(1) };
        let raw_window = DS4_N_SWA.min(ctx_size).max(1);
        let min_ratio = 4u32; // smallest non-zero compress_ratio used in the model
        let comp_cap = ctx_size / min_ratio + 2;
        let comp_cap = comp_cap.max(2);

        // Derive tensor dimensions from the layer-0 binding (consistent across layers).
        let binding = self.layer_binding(0).ok()?;
        let q_rank = binding.attn_q_a.descriptor.dims[1];
        let shared_dim = binding.ffn_gate_shexp.descriptor.dims[1];
        let routed_mid_dim = binding.ffn_gate_exps.descriptor.dims[1];
        let vocab_dim = self.tensor_bindings.output.descriptor.dims[1];

        const F32: u64 = 4;
        let hc_dim = DS4_N_HC as u64 * DS4_N_EMBD as u64;
        let mix_hc = 2 * DS4_N_HC as u64 + DS4_N_HC as u64 * DS4_N_HC as u64;
        let q_dim = DS4_N_HEAD as u64 * DS4_N_HEAD_DIM;
        let low_dim = DS4_N_OUT_GROUP as u64 * DS4_N_LORA_O as u64;
        let comp_width_max = 2 * DS4_N_HEAD_DIM.max(DS4_N_INDEXER_HEAD_DIM);
        let indexer_q_dim = DS4_N_INDEXER_HEAD as u64 * DS4_N_INDEXER_HEAD_DIM;

        macro_rules! alloc {
            ($bytes:expr) => {
                MetalTensor::alloc($bytes)?
            };
        }

        let cur_hc        = alloc!(hc_dim * F32);
        let after_ffn_hc  = alloc!(hc_dim * F32);
        let flat_hc       = alloc!(hc_dim * F32);
        let hc_mix        = alloc!(mix_hc * F32);
        let hc_split      = alloc!(mix_hc * F32);
        let attn_cur      = alloc!(DS4_N_EMBD as u64 * F32);
        let attn_norm     = alloc!(DS4_N_EMBD as u64 * F32);
        let qr            = alloc!(q_rank * F32);
        let qr_norm       = alloc!(q_rank * F32);
        let q             = alloc!(q_dim * F32);
        let kv_raw        = alloc!(DS4_N_HEAD_DIM * F32);
        let kv            = alloc!(DS4_N_HEAD_DIM * F32);
        let comp_kv_cur   = alloc!(comp_width_max * F32);
        let comp_sc_cur   = alloc!(comp_width_max * F32);
        let indexer_q     = alloc!(indexer_q_dim * F32);
        let indexer_weights = alloc!(DS4_N_INDEXER_HEAD as u64 * F32);
        let indexer_scores  = alloc!(comp_cap as u64 * F32);
        let comp_selected   = alloc!(DS4_N_INDEXER_TOP_K.max(1) * 4); // u32
        let heads         = alloc!(q_dim * F32);
        let attn_low      = alloc!(low_dim * F32);
        let attn_out      = alloc!(DS4_N_EMBD as u64 * F32);
        let after_attn_hc = alloc!(hc_dim * F32);
        let ffn_cur       = alloc!(DS4_N_EMBD as u64 * F32);
        let ffn_norm      = alloc!(DS4_N_EMBD as u64 * F32);
        let shared_gate   = alloc!(shared_dim * F32);
        let shared_up     = alloc!(shared_dim * F32);
        let shared_mid    = alloc!(shared_dim * F32);
        let shared_out    = alloc!(DS4_N_EMBD as u64 * F32);
        let router_logits = alloc!(DS4_N_EXPERT as u64 * F32);
        let router_probs  = alloc!(DS4_N_EXPERT as u64 * F32);
        let router_selected = alloc!(DS4_N_EXPERT_USED as u64 * 4); // i32
        let router_weights  = alloc!(DS4_N_EXPERT_USED as u64 * F32);
        let routed_gate   = alloc!(DS4_N_EXPERT_USED as u64 * routed_mid_dim * F32);
        let routed_up     = alloc!(DS4_N_EXPERT_USED as u64 * routed_mid_dim * F32);
        let routed_mid    = alloc!(DS4_N_EXPERT_USED as u64 * routed_mid_dim * F32);
        let routed_down   = alloc!(DS4_N_EXPERT_USED as u64 * DS4_N_EMBD as u64 * F32);
        let routed_out    = alloc!(DS4_N_EMBD as u64 * F32);
        let output_pre    = alloc!(DS4_N_HC as u64 * F32);
        let output_weights = alloc!(DS4_N_HC as u64 * F32);
        let output_embd   = alloc!(DS4_N_EMBD as u64 * F32);
        let output_norm   = alloc!(DS4_N_EMBD as u64 * F32);
        let logits        = alloc!(vocab_dim * F32);

        // Per-layer KV caches.
        let mut layers: Vec<MetalLayerCache> = Vec::with_capacity(DS4_N_LAYER as usize);
        for il in 0..DS4_N_LAYER {
            let ratio = layer_compress_ratio(il);
            let raw_kv = alloc!(cap_raw as u64 * DS4_N_HEAD_DIM * F32);

            let (attn_comp_kv, attn_state_kv, attn_state_score,
                 index_comp_kv, index_state_kv, index_state_score) = if ratio != 0 {
                let coff = if ratio == 4 { 2u64 } else { 1u64 };
                let attn_width = coff * DS4_N_HEAD_DIM;
                let attn_rows = coff * ratio as u64;
                let comp_kv   = alloc!(comp_cap as u64 * DS4_N_HEAD_DIM * F32);
                let st_kv     = alloc!(attn_width * attn_rows * F32);
                let st_sc     = alloc!(attn_width * attn_rows * F32);
                // Initialize compressor state tensors (kv = 0, score = -inf).
                st_kv.write_f32(&vec![0.0f32; (attn_width * attn_rows) as usize]);
                st_sc.write_f32(&vec![DS4_NEG_INF; (attn_width * attn_rows) as usize]);

                let (idx_comp, idx_st_kv, idx_st_sc) = if ratio == 4 {
                    let index_width = coff * DS4_N_INDEXER_HEAD_DIM;
                    let index_rows = coff * ratio as u64;
                    let ic   = alloc!(comp_cap as u64 * DS4_N_INDEXER_HEAD_DIM * F32);
                    let isk  = alloc!(index_width * index_rows * F32);
                    let iss  = alloc!(index_width * index_rows * F32);
                    isk.write_f32(&vec![0.0f32; (index_width * index_rows) as usize]);
                    iss.write_f32(&vec![DS4_NEG_INF; (index_width * index_rows) as usize]);
                    (Some(ic), Some(isk), Some(iss))
                } else {
                    (None, None, None)
                };

                (Some(comp_kv), Some(st_kv), Some(st_sc), idx_comp, idx_st_kv, idx_st_sc)
            } else {
                (None, None, None, None, None, None)
            };

            layers.push(MetalLayerCache {
                raw_kv,
                attn_comp_kv,
                attn_state_kv,
                attn_state_score,
                index_comp_kv,
                index_state_kv,
                index_state_score,
                n_comp: 0,
                n_index_comp: 0,
                compress_ratio: ratio,
            });
        }

        Some(Box::new(MetalDecodeGraph {
            cur_hc,
            after_ffn_hc,
            flat_hc,
            hc_mix,
            hc_split,
            attn_cur,
            attn_norm,
            qr,
            qr_norm,
            q,
            kv_raw,
            kv,
            comp_kv_cur,
            comp_sc_cur,
            indexer_q,
            indexer_weights,
            indexer_scores,
            comp_selected,
            heads,
            attn_low,
            attn_out,
            after_attn_hc,
            ffn_cur,
            ffn_norm,
            shared_gate,
            shared_up,
            shared_mid,
            shared_out,
            router_logits,
            router_probs,
            router_selected,
            router_weights,
            routed_gate,
            routed_up,
            routed_mid,
            routed_down,
            routed_out,
            output_pre,
            output_weights,
            output_embd,
            output_norm,
            logits,
            layers,
            raw_cap: cap_raw,
            raw_window,
            _comp_cap: comp_cap,
            vocab_dim,
        }))
    }

    pub fn create_rust_session(&self, ctx_size: u32, raw_cap: u32) -> RustSession<'_> {
        RustSession::new(self, ctx_size, raw_cap)
    }

    pub fn eval_token_with_rust_cache(&self, cache: &mut RustKvCache, token: i32, pos: u32) -> Result<Vec<f32>> {
        let mut scratch = RustAttentionDecodeScratch::new();
        let mut ffn_scratch = RustFfnDecodeScratch::new();
        self.eval_token_with_rust_backend_and_scratch(cache, token, pos, &mut scratch, &mut ffn_scratch)
    }

    fn eval_token_with_rust_backend_and_scratch(
        &self,
        cache: &mut RustKvCache,
        token: i32,
        pos: u32,
        scratch: &mut RustAttentionDecodeScratch,
        ffn_scratch: &mut RustFfnDecodeScratch,
    ) -> Result<Vec<f32>> {
        match self.backend {
            Backend::Cpu => self.eval_token_with_rust_cpu_backend_and_scratch(
                cache,
                token,
                pos,
                scratch,
                ffn_scratch,
            ),
            Backend::Metal => self.eval_token_with_rust_metal_backend_and_scratch(
                cache,
                token,
                pos,
                scratch,
                ffn_scratch,
            ),
        }
    }

    fn eval_token_with_rust_cpu_backend_and_scratch(
        &self,
        cache: &mut RustKvCache,
        token: i32,
        pos: u32,
        scratch: &mut RustAttentionDecodeScratch,
        ffn_scratch: &mut RustFfnDecodeScratch,
    ) -> Result<Vec<f32>> {
        self.eval_token_with_rust_cache_and_scratch(cache, token, pos, scratch, ffn_scratch)
    }

    fn eval_token_with_rust_metal_backend_and_scratch(
        &self,
        cache: &mut RustKvCache,
        token: i32,
        pos: u32,
        scratch: &mut RustAttentionDecodeScratch,
        ffn_scratch: &mut RustFfnDecodeScratch,
    ) -> Result<Vec<f32>> {
        // CPU fallback used when no MetalDecodeGraph is attached to the session.
        self.eval_token_with_rust_cache_and_scratch(cache, token, pos, scratch, ffn_scratch)
    }

    // Evaluate one token through all 43 layers on the GPU using the Metal
    // kernel library, then read back the final HC state from the device.
    // This is the real Metal execution path: compute stays on the GPU for the
    // full layer stack; only the logits vector and the final HC state leave
    // device memory.
    pub(crate) fn eval_token_with_metal_graph(
        &self,
        metal: &mut MetalDecodeGraph,
        token: i32,
        pos: u32,
    ) -> Result<()> {
        let model_map = self.model_map.as_ptr();
        let model_size = self.model_map.len_u64();
        let token_u32 = u32::try_from(token).context("token id is negative")?;
        let weights = &self.tensor_bindings;

        let raw_row = metal.raw_row(pos);
        let n_raw   = metal.raw_span(pos);

        // ---- Begin Metal command encoding -----------------------------------
        metal_runtime::begin_commands()?;

        // 1. Embed the input token into the initial HC state.
        let n_vocab = weights.token_embd.descriptor.dims[1] as u32;
        let ok = unsafe {
            crate::metal_kernels::embed_token_hc_tensor(
                metal.cur_hc.as_ptr(),
                model_map,
                model_size,
                weights.token_embd.descriptor.abs_offset,
                n_vocab,
                token_u32,
                DS4_N_EMBD as u32,
                DS4_N_HC as u32,
            )
        };
        if ok == 0 { bail!("ds4_metal_embed_token_hc_tensor failed") }

        // 2. Run the 43 transformer layers.
        for il in 0..DS4_N_LAYER {
            self.metal_encode_decode_layer(metal, il, pos, raw_row, n_raw, token_u32)?;
            // Swap cur_hc and after_ffn_hc so the next layer sees the updated state.
            std::mem::swap(&mut metal.cur_hc, &mut metal.after_ffn_hc);
        }

        // 3. Encode the output head (HC collapse → norm → vocab projection).
        self.metal_encode_output_head(metal)?;

        // ---- End command encoding and run on GPU ----------------------------
        metal_runtime::end_commands()?;
        metal_runtime::synchronize()?;

        Ok(())
    }

    // Read logits out of the Metal graph into a CPU Vec.
    pub(crate) fn metal_graph_read_logits(&self, metal: &MetalDecodeGraph, out: &mut Vec<f32>) -> Result<()> {
        out.resize(metal.vocab_dim as usize, 0.0);
        if !metal.logits.read_f32(out) {
            bail!("ds4_metal_tensor_read failed for logits")
        }
        Ok(())
    }

    // Run one transformer layer on the GPU.  Mirrors metal_graph_encode_decode_layer in ds4.c.
    fn metal_encode_decode_layer(
        &self,
        metal: &mut MetalDecodeGraph,
        il: u32,
        pos: u32,
        raw_row: u32,
        n_raw: u32,
        token: u32,
    ) -> Result<()> {
        let model_map  = self.model_map.as_ptr();
        let model_size = self.model_map.len_u64();
        let binding    = self.layer_binding(il)?;

        // Compute raw_start before taking the mutable layer_cache borrow.
        let raw_start = metal.raw_start(pos, n_raw);

        let layer_cache = &mut metal.layers[il as usize];

        let ratio      = layer_cache.compress_ratio;
        let compressed = ratio != 0;
        let coff       = if ratio == 4 { 2u32 } else { 1u32 };
        let freq_base  = layer_rope_freq_base(il);
        let freq_scale = layer_rope_freq_scale(il);
        let ext_factor = if compressed && DS4_ROPE_SCALE_FACTOR > 1.0 { 1.0f32 } else { 0.0f32 };
        let attn_factor = if ext_factor != 0.0 && freq_scale > 0.0 {
            1.0f32 / (1.0 + 0.1 * (1.0 / freq_scale).ln())
        } else {
            1.0f32
        };
        let n_ctx_orig  = if compressed { DS4_ROPE_ORIG_CTX as u32 } else { 0u32 };

        let q_rank = binding.attn_q_a.descriptor.dims[1] as u32;
        let hc_dim = DS4_N_HC as u32 * DS4_N_EMBD as u32;
        let mix_hc = 2 * DS4_N_HC as u32 + DS4_N_HC as u32 * DS4_N_HC as u32;
        let q_dim  = DS4_N_HEAD * DS4_N_HEAD_DIM as u32;

        macro_rules! ok {
            ($call:expr, $name:expr) => {{
                let rc = $call;
                if rc == 0 { bail!("{} failed at layer {}", $name, il) }
            }};
        }

        // ---- Attention HC pre: flat norm → HC mixer → split+weighted sum+norm
        ok!(metal_runtime::rms_norm_plain_tensor(
            &metal.flat_hc, &metal.cur_hc, hc_dim, DS4_RMS_EPS), "rms_norm_plain");

        ok!(unsafe { crate::metal_kernels::matmul_f16_tensor(
            metal.hc_mix.as_ptr(), model_map, model_size,
            binding.hc_attn_fn.descriptor.abs_offset,
            hc_dim as u64, mix_hc as u64,
            metal.flat_hc.as_const_ptr(), 1) }, "hc_attn_fn matmul");

        ok!(unsafe { crate::metal_kernels::hc_split_weighted_sum_norm_tensor(
            metal.attn_cur.as_ptr(), metal.attn_norm.as_ptr(), metal.hc_split.as_ptr(),
            metal.hc_mix.as_const_ptr(), metal.cur_hc.as_const_ptr(),
            model_map, model_size,
            binding.hc_attn_scale.descriptor.abs_offset,
            binding.hc_attn_base.descriptor.abs_offset,
            binding.attn_norm.descriptor.abs_offset,
            DS4_N_EMBD as u32, DS4_N_HC as u32, DS4_N_HC_SINKHORN_ITER as u32,
            DS4_HC_EPS, DS4_RMS_EPS) }, "hc_split_weighted_sum_norm");

        // ---- Q low-rank projection and KV projection (fused norm pass) -----
        ok!(unsafe { crate::metal_kernels::matmul_q8_0_tensor(
            metal.qr.as_ptr(), model_map, model_size,
            binding.attn_q_a.descriptor.abs_offset,
            DS4_N_EMBD as u64, q_rank as u64,
            metal.attn_norm.as_const_ptr(), 1) }, "attn_q_a matmul");

        ok!(unsafe { crate::metal_kernels::matmul_q8_0_tensor(
            metal.kv_raw.as_ptr(), model_map, model_size,
            binding.attn_kv.descriptor.abs_offset,
            DS4_N_EMBD as u64, DS4_N_HEAD_DIM,
            metal.attn_norm.as_const_ptr(), 1) }, "attn_kv matmul");

        ok!(unsafe { crate::metal_kernels::dsv4_qkv_rms_norm_rows_tensor(
            metal.qr_norm.as_ptr(), metal.qr.as_const_ptr(),
            model_map, model_size,
            binding.attn_q_a_norm.descriptor.abs_offset, q_rank,
            metal.kv.as_ptr(), metal.kv_raw.as_const_ptr(),
            binding.attn_kv_a_norm.descriptor.abs_offset, DS4_N_HEAD_DIM as u32,
            1, DS4_RMS_EPS) }, "qkv_rms_norm");

        // ---- Full Q projection and per-head RMSNorm -------------------------
        ok!(unsafe { crate::metal_kernels::matmul_q8_0_tensor(
            metal.q.as_ptr(), model_map, model_size,
            binding.attn_q_b.descriptor.abs_offset,
            q_rank as u64, q_dim as u64,
            metal.qr_norm.as_const_ptr(), 1) }, "attn_q_b matmul");

        ok!(unsafe { crate::metal_kernels::head_rms_norm_tensor(
            metal.q.as_ptr(), 1, DS4_N_HEAD, DS4_N_HEAD_DIM as u32, DS4_RMS_EPS) },
            "head_rms_norm Q");

        // ---- Q RoPE ---------------------------------------------------------
        ok!(unsafe { crate::metal_kernels::rope_tail_tensor(
            metal.q.as_ptr(), 1, DS4_N_HEAD, DS4_N_HEAD_DIM as u32,
            DS4_N_ROT as u32, pos, n_ctx_orig, false,
            freq_base, freq_scale, ext_factor, attn_factor,
            DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW) }, "rope Q");

        // ---- KV RoPE + FP8 quantize + store raw cache ----------------------
        ok!(unsafe { crate::metal_kernels::rope_tail_tensor(
            metal.kv.as_ptr(), 1, DS4_N_HEAD_KV, DS4_N_HEAD_DIM as u32,
            DS4_N_ROT as u32, pos, n_ctx_orig, false,
            freq_base, freq_scale, ext_factor, attn_factor,
            DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW) }, "rope KV");

        ok!(unsafe { crate::metal_kernels::kv_fp8_store_raw_tensor(
            metal.kv.as_ptr(), layer_cache.raw_kv.as_ptr(),
            metal.raw_cap, raw_row, DS4_N_HEAD_DIM as u32, DS4_N_ROT as u32) },
            "kv_fp8_store_raw");

        // ---- Compressor / indexer update for compressed layers --------------
        let (n_comp, comp_selected_ptr) = if compressed {
            let comp_width  = coff as u64 * DS4_N_HEAD_DIM;
            let emit = ((pos + 1) % ratio) == 0;

            // Compressor projection from attn_norm.
            let ape_b = binding.attn_compressor_ape.as_ref()
                .ok_or_else(|| anyhow!("missing attn_compressor_ape at layer {}", il))?;
            let kv_b  = binding.attn_compressor_kv.as_ref()
                .ok_or_else(|| anyhow!("missing attn_compressor_kv at layer {}", il))?;
            let gate_b = binding.attn_compressor_gate.as_ref()
                .ok_or_else(|| anyhow!("missing attn_compressor_gate at layer {}", il))?;
            let norm_b = binding.attn_compressor_norm.as_ref()
                .ok_or_else(|| anyhow!("missing attn_compressor_norm at layer {}", il))?;

            ok!(unsafe { crate::metal_kernels::matmul_f16_pair_tensor(
                metal.comp_kv_cur.as_ptr(), metal.comp_sc_cur.as_ptr(),
                model_map, model_size,
                kv_b.descriptor.abs_offset, gate_b.descriptor.abs_offset,
                DS4_N_EMBD as u64, comp_width,
                metal.attn_norm.as_const_ptr(), 1) }, "compressor proj");

            let attn_state_kv    = layer_cache.attn_state_kv.as_ref()
                .ok_or_else(|| anyhow!("missing attn_state_kv at layer {}", il))?;
            let attn_state_score = layer_cache.attn_state_score.as_ref()
                .ok_or_else(|| anyhow!("missing attn_state_score at layer {}", il))?;
            let attn_comp_kv     = layer_cache.attn_comp_kv.as_ref()
                .ok_or_else(|| anyhow!("missing attn_comp_kv at layer {}", il))?;

            let comp_row = layer_cache.n_comp;
            ok!(unsafe { crate::metal_kernels::compressor_update_tensor(
                metal.comp_kv_cur.as_const_ptr(), metal.comp_sc_cur.as_const_ptr(),
                attn_state_kv.as_ptr(), attn_state_score.as_ptr(),
                attn_comp_kv.as_ptr(),
                model_map, model_size,
                ape_b.descriptor.abs_offset, ape_b.descriptor.tensor_type,
                norm_b.descriptor.abs_offset, norm_b.descriptor.tensor_type,
                DS4_N_HEAD_DIM as u32, ratio, pos, comp_row,
                DS4_N_ROT as u32, n_ctx_orig,
                freq_base, freq_scale, ext_factor, attn_factor,
                DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW,
                DS4_RMS_EPS) }, "compressor_update attn");

            if emit {
                // FP8-quantize the newly appended compressed row.
                let row_offset = comp_row as u64 * DS4_N_HEAD_DIM * 4;
                let row_bytes  = DS4_N_HEAD_DIM * 4;
                let row_view = MetalTensor::view(attn_comp_kv, row_offset, row_bytes)
                    .ok_or_else(|| anyhow!("failed to create comp row view at layer {}", il))?;
                ok!(unsafe { crate::metal_kernels::dsv4_fp8_kv_quantize_tensor(
                    row_view.as_ptr(), 1, DS4_N_HEAD_DIM as u32, DS4_N_ROT as u32) },
                    "fp8_kv_quantize comp row");
                layer_cache.n_comp += 1;
            }

            // Indexer compressor for ratio-4 layers.
            let mut comp_sel_ptr: *const ffi::ds4_metal_tensor = std::ptr::null();
            let n_comp = layer_cache.n_comp;

            if ratio == 4 {
                let ape_ib  = binding.indexer_compressor_ape.as_ref()
                    .ok_or_else(|| anyhow!("missing indexer_compressor_ape at layer {}", il))?;
                let kv_ib   = binding.indexer_compressor_kv.as_ref()
                    .ok_or_else(|| anyhow!("missing indexer_compressor_kv at layer {}", il))?;
                let gate_ib = binding.indexer_compressor_gate.as_ref()
                    .ok_or_else(|| anyhow!("missing indexer_compressor_gate at layer {}", il))?;
                let norm_ib = binding.indexer_compressor_norm.as_ref()
                    .ok_or_else(|| anyhow!("missing indexer_compressor_norm at layer {}", il))?;

                let index_width = coff as u64 * DS4_N_INDEXER_HEAD_DIM;

                ok!(unsafe { crate::metal_kernels::matmul_f16_pair_tensor(
                    metal.comp_kv_cur.as_ptr(), metal.comp_sc_cur.as_ptr(),
                    model_map, model_size,
                    kv_ib.descriptor.abs_offset, gate_ib.descriptor.abs_offset,
                    DS4_N_EMBD as u64, index_width,
                    metal.attn_norm.as_const_ptr(), 1) }, "indexer compressor proj");

                let index_state_kv    = layer_cache.index_state_kv.as_ref()
                    .ok_or_else(|| anyhow!("missing index_state_kv at layer {}", il))?;
                let index_state_score = layer_cache.index_state_score.as_ref()
                    .ok_or_else(|| anyhow!("missing index_state_score at layer {}", il))?;
                let index_comp_kv     = layer_cache.index_comp_kv.as_ref()
                    .ok_or_else(|| anyhow!("missing index_comp_kv at layer {}", il))?;

                let index_row = layer_cache.n_index_comp;
                ok!(unsafe { crate::metal_kernels::compressor_update_tensor(
                    metal.comp_kv_cur.as_const_ptr(), metal.comp_sc_cur.as_const_ptr(),
                    index_state_kv.as_ptr(), index_state_score.as_ptr(),
                    index_comp_kv.as_ptr(),
                    model_map, model_size,
                    ape_ib.descriptor.abs_offset, ape_ib.descriptor.tensor_type,
                    norm_ib.descriptor.abs_offset, norm_ib.descriptor.tensor_type,
                    DS4_N_INDEXER_HEAD_DIM as u32, ratio, pos, index_row,
                    DS4_N_ROT as u32, n_ctx_orig,
                    freq_base, freq_scale, ext_factor, attn_factor,
                    DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW,
                    DS4_RMS_EPS) }, "compressor_update indexer");

                if emit { layer_cache.n_index_comp += 1; }

                let decode_top_k = DS4_N_INDEXER_TOP_K as u32;
                let n_index_comp = layer_cache.n_index_comp;
                if n_comp > decode_top_k {
                    // Run the indexer to select top-k compressed rows.
                    let indexer_q_b = binding.indexer_attn_q_b.as_ref()
                        .ok_or_else(|| anyhow!("missing indexer_attn_q_b at layer {}", il))?;
                    let indexer_proj_b = binding.indexer_proj.as_ref()
                        .ok_or_else(|| anyhow!("missing indexer_proj at layer {}", il))?;
                    let indexer_q_dim = DS4_N_INDEXER_HEAD as u64 * DS4_N_INDEXER_HEAD_DIM;

                    ok!(unsafe { crate::metal_kernels::matmul_f16_tensor(
                        metal.indexer_q.as_ptr(), model_map, model_size,
                        indexer_q_b.descriptor.abs_offset,
                        q_rank as u64, indexer_q_dim,
                        metal.qr_norm.as_const_ptr(), 1) }, "indexer Q proj");

                    ok!(unsafe { crate::metal_kernels::rope_tail_tensor(
                        metal.indexer_q.as_ptr(), 1,
                        DS4_N_INDEXER_HEAD, DS4_N_INDEXER_HEAD_DIM as u32,
                        DS4_N_ROT as u32, pos, n_ctx_orig, false,
                        freq_base, freq_scale, ext_factor, attn_factor,
                        DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW) }, "rope indexer Q");

                    ok!(unsafe { crate::metal_kernels::matmul_f16_tensor(
                        metal.indexer_weights.as_ptr(), model_map, model_size,
                        indexer_proj_b.descriptor.abs_offset,
                        DS4_N_EMBD as u64, DS4_N_INDEXER_HEAD as u64,
                        metal.attn_norm.as_const_ptr(), 1) }, "indexer weights proj");

                    let index_scale = 1.0f32
                        / ((DS4_N_INDEXER_HEAD_DIM as f32 * DS4_N_INDEXER_HEAD as f32).sqrt());

                    ok!(unsafe { crate::metal_kernels::indexer_score_one_tensor(
                        metal.indexer_scores.as_ptr(),
                        metal.indexer_q.as_const_ptr(),
                        metal.indexer_weights.as_const_ptr(),
                        index_comp_kv.as_const_ptr(),
                        n_index_comp, DS4_N_INDEXER_HEAD,
                        DS4_N_INDEXER_HEAD_DIM as u32, index_scale) }, "indexer_score");

                    let effective_top_k = decode_top_k.min(n_index_comp);
                    ok!(unsafe { crate::metal_kernels::indexer_topk_tensor(
                        metal.comp_selected.as_ptr(),
                        metal.indexer_scores.as_const_ptr(),
                        n_index_comp, 1, effective_top_k) }, "indexer_topk");

                    comp_sel_ptr = metal.comp_selected.as_const_ptr();
                }
            }

            (n_comp, comp_sel_ptr)
        } else {
            (0u32, std::ptr::null())
        };

        // ---- Attention compute -----------------------------------------------
        // raw_start was computed above before the mutable layer_cache borrow.
        let attn_sinks_offset = binding.attn_sinks.descriptor.abs_offset;

        if !comp_selected_ptr.is_null() {
            // Ratio-4 indexed mixed attention.
            let attn_comp_kv = layer_cache.attn_comp_kv.as_ref().unwrap();
            let n_selected = (DS4_N_INDEXER_TOP_K as u32).min(layer_cache.n_index_comp);
            ok!(unsafe { crate::metal_kernels::attention_indexed_mixed_batch_heads_tensor(
                metal.heads.as_ptr(), model_map, model_size, attn_sinks_offset,
                metal.q.as_const_ptr(), layer_cache.raw_kv.as_const_ptr(),
                attn_comp_kv.as_const_ptr(), comp_selected_ptr,
                1, pos, n_raw, metal.raw_cap, raw_start,
                n_comp, n_selected,
                metal.raw_window, ratio, DS4_N_HEAD, DS4_N_HEAD_DIM as u32) },
                "attention_indexed_mixed");
        } else {
            // Raw-only or raw+compressed attention.
            let comp_kv_ptr: *const ffi::ds4_metal_tensor = if n_comp > 0 {
                layer_cache.attn_comp_kv.as_ref().unwrap().as_const_ptr()
            } else {
                std::ptr::null()
            };
            ok!(unsafe { crate::metal_kernels::attention_decode_heads_tensor(
                metal.heads.as_ptr(), model_map, model_size, attn_sinks_offset,
                metal.q.as_const_ptr(), layer_cache.raw_kv.as_const_ptr(),
                n_raw, metal.raw_cap, raw_start,
                comp_kv_ptr, n_comp,
                std::ptr::null(), 0,
                DS4_N_HEAD, DS4_N_HEAD_DIM as u32) },
                "attention_decode_heads");
        }

        // ---- Inverse RoPE on attention output --------------------------------
        ok!(unsafe { crate::metal_kernels::rope_tail_tensor(
            metal.heads.as_ptr(), 1, DS4_N_HEAD, DS4_N_HEAD_DIM as u32,
            DS4_N_ROT as u32, pos, n_ctx_orig, true,
            freq_base, freq_scale, ext_factor, attn_factor,
            DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW) }, "inverse rope heads");

        // ---- Attention output projection + HC expand (fused) -----------------
        let group_dim = DS4_N_HEAD_DIM * (DS4_N_HEAD / DS4_N_OUT_GROUP) as u64;
        let rank      = DS4_N_LORA_O as u64;
        ok!(unsafe { crate::metal_kernels::attention_output_low_q8_tensor(
            metal.attn_low.as_ptr(), model_map, model_size,
            binding.attn_output_a.descriptor.abs_offset,
            group_dim, rank, DS4_N_OUT_GROUP,
            metal.heads.as_const_ptr()) }, "attn_output_low");

        ok!(unsafe { crate::metal_kernels::matmul_q8_0_hc_expand_tensor(
            metal.after_attn_hc.as_ptr(), metal.attn_out.as_ptr(),
            model_map, model_size,
            binding.attn_output_b.descriptor.abs_offset,
            DS4_N_OUT_GROUP as u64 * rank, DS4_N_EMBD as u64,
            metal.attn_low.as_const_ptr(),
            metal.cur_hc.as_const_ptr(), metal.hc_split.as_const_ptr(),
            DS4_N_EMBD as u32, DS4_N_HC as u32) }, "attn_output_hc_expand");

        // ---- FFN HC pre: flat norm → HC mixer → split+weighted sum+norm ------
        let shared_dim = binding.ffn_gate_shexp.descriptor.dims[1];
        let expert_in_dim  = binding.ffn_gate_exps.descriptor.dims[0] as u32;
        let expert_mid_dim = binding.ffn_gate_exps.descriptor.dims[1] as u32;
        let down_in_dim    = binding.ffn_down_exps.descriptor.dims[0] as u32;
        let routed_out_dim = binding.ffn_down_exps.descriptor.dims[1] as u32;

        ok!(metal_runtime::rms_norm_plain_tensor(
            &metal.flat_hc, &metal.after_attn_hc, hc_dim, DS4_RMS_EPS),
            "rms_norm_plain FFN");

        ok!(unsafe { crate::metal_kernels::matmul_f16_tensor(
            metal.hc_mix.as_ptr(), model_map, model_size,
            binding.hc_ffn_fn.descriptor.abs_offset,
            hc_dim as u64, mix_hc as u64,
            metal.flat_hc.as_const_ptr(), 1) }, "hc_ffn_fn matmul");

        ok!(unsafe { crate::metal_kernels::hc_split_weighted_sum_norm_tensor(
            metal.ffn_cur.as_ptr(), metal.ffn_norm.as_ptr(), metal.hc_split.as_ptr(),
            metal.hc_mix.as_const_ptr(), metal.after_attn_hc.as_const_ptr(),
            model_map, model_size,
            binding.hc_ffn_scale.descriptor.abs_offset,
            binding.hc_ffn_base.descriptor.abs_offset,
            binding.ffn_norm.descriptor.abs_offset,
            DS4_N_EMBD as u32, DS4_N_HC as u32, DS4_N_HC_SINKHORN_ITER as u32,
            DS4_HC_EPS, DS4_RMS_EPS) }, "hc_split_weighted_sum_norm FFN");

        // ---- Router + routed MoE --------------------------------------------
        ok!(unsafe { crate::metal_kernels::matmul_f16_tensor(
            metal.router_logits.as_ptr(), model_map, model_size,
            binding.ffn_gate_inp.descriptor.abs_offset,
            DS4_N_EMBD as u64, DS4_N_EXPERT as u64,
            metal.ffn_norm.as_const_ptr(), 1) }, "router logits");

        let bias_offset = binding.ffn_exp_probs_b.as_ref()
            .map(|b| b.descriptor.abs_offset).unwrap_or(0);
        let hash_offset = binding.ffn_gate_tid2eid.as_ref()
            .map(|b| b.descriptor.abs_offset).unwrap_or(0);
        let hash_rows   = binding.ffn_gate_tid2eid.as_ref()
            .map(|b| b.descriptor.dims[1] as u32).unwrap_or(0);
        let has_bias    = binding.ffn_exp_probs_b.is_some();
        let hash_mode   = binding.ffn_gate_tid2eid.is_some();

        ok!(unsafe { crate::metal_kernels::router_select_tensor(
            metal.router_selected.as_ptr(), metal.router_weights.as_ptr(),
            metal.router_probs.as_ptr(),
            model_map, model_size,
            bias_offset, hash_offset, hash_rows,
            token, 0, 0, has_bias, hash_mode,
            metal.router_logits.as_const_ptr()) }, "router_select");

        // Expert byte layout for the routed MoE kernel.
        let gate_row_bytes = metal_expert_row_bytes(
            binding.ffn_gate_exps.descriptor.tensor_type,
            binding.ffn_gate_exps.descriptor.dims[0]);
        let down_row_bytes = metal_expert_row_bytes(
            binding.ffn_down_exps.descriptor.tensor_type,
            binding.ffn_down_exps.descriptor.dims[0]);
        let gate_expert_bytes = expert_mid_dim as u64 * gate_row_bytes;
        let down_expert_bytes = routed_out_dim as u64 * down_row_bytes;

        ok!(unsafe { crate::metal_kernels::routed_moe_one_tensor(
            metal.routed_out.as_ptr(),
            metal.routed_gate.as_ptr(), metal.routed_up.as_ptr(),
            metal.routed_mid.as_ptr(),
            metal.routed_down.as_ptr(),
            model_map, model_size,
            binding.ffn_gate_exps.descriptor.abs_offset,
            binding.ffn_up_exps.descriptor.abs_offset,
            binding.ffn_down_exps.descriptor.abs_offset,
            binding.ffn_gate_exps.descriptor.tensor_type,
            binding.ffn_down_exps.descriptor.tensor_type,
            gate_expert_bytes, gate_row_bytes,
            down_expert_bytes, down_row_bytes,
            expert_in_dim, down_in_dim, routed_out_dim,
            metal.router_selected.as_const_ptr(),
            metal.router_weights.as_const_ptr(),
            DS4_N_EXPERT_USED as u32, DS4_SWIGLU_CLAMP_EXP,
            metal.ffn_norm.as_const_ptr()) }, "routed_moe_one");

        // ---- Shared expert + fused HC expand --------------------------------
        ok!(unsafe { crate::metal_kernels::shared_gate_up_swiglu_q8_0_tensor(
            metal.shared_gate.as_ptr(), metal.shared_up.as_ptr(), metal.shared_mid.as_ptr(),
            model_map, model_size,
            binding.ffn_gate_shexp.descriptor.abs_offset,
            binding.ffn_up_shexp.descriptor.abs_offset,
            DS4_N_EMBD as u64, shared_dim,
            metal.ffn_norm.as_const_ptr()) }, "shared_gate_up_swiglu");

        // Fused shared down projection + routed add + HC expand.
        ok!(unsafe { crate::metal_kernels::shared_down_hc_expand_q8_0_tensor(
            metal.after_ffn_hc.as_ptr(), metal.shared_out.as_ptr(),
            model_map, model_size,
            binding.ffn_down_shexp.descriptor.abs_offset,
            shared_dim, DS4_N_EMBD as u64,
            metal.shared_mid.as_const_ptr(),
            metal.routed_out.as_const_ptr(),
            metal.after_attn_hc.as_const_ptr(), metal.hc_split.as_const_ptr(),
            DS4_N_EMBD as u32, DS4_N_HC as u32) }, "shared_down_hc_expand");

        // Silence unused-variable warning when there are no assertions after the last use.
        let _ = n_comp;

        Ok(())
    }

    // Encode the output head onto the current Metal command buffer.
    // Reads from metal.cur_hc (which holds the final layer's after_ffn_hc after
    // all layer swaps) and writes logits to metal.logits.
    fn metal_encode_output_head(&self, metal: &MetalDecodeGraph) -> Result<()> {
        let model_map  = self.model_map.as_ptr();
        let model_size = self.model_map.len_u64();
        let weights    = &self.tensor_bindings;
        let hc_dim     = DS4_N_HC as u32 * DS4_N_EMBD as u32;

        macro_rules! ok {
            ($call:expr, $name:expr) => {{
                let rc = $call;
                if rc == 0 { bail!("output head: {} failed", $name) }
            }};
        }

        ok!(metal_runtime::rms_norm_plain_tensor(
            &metal.flat_hc, &metal.cur_hc, hc_dim, DS4_RMS_EPS),
            "rms_norm_plain HC");

        ok!(unsafe { crate::metal_kernels::matmul_f16_tensor(
            metal.output_pre.as_ptr(), model_map, model_size,
            weights.output_hc_fn.descriptor.abs_offset,
            hc_dim as u64, DS4_N_HC as u64,
            metal.flat_hc.as_const_ptr(), 1) }, "output_hc_fn matmul");

        ok!(unsafe { crate::metal_kernels::output_hc_weights_tensor(
            metal.output_weights.as_ptr(), metal.output_pre.as_const_ptr(),
            model_map, model_size,
            weights.output_hc_scale.descriptor.abs_offset,
            weights.output_hc_base.descriptor.abs_offset,
            DS4_N_HC as u32, DS4_HC_EPS) }, "output_hc_weights");

        ok!(unsafe { crate::metal_kernels::hc_weighted_sum_tensor(
            metal.output_embd.as_ptr(), metal.cur_hc.as_const_ptr(),
            metal.output_weights.as_const_ptr(),
            DS4_N_EMBD as u32, DS4_N_HC as u32) }, "hc_weighted_sum");

        ok!(unsafe { crate::metal_kernels::rms_norm_weight_tensor(
            metal.output_norm.as_ptr(), metal.output_embd.as_const_ptr(),
            model_map, model_size,
            weights.output_norm.descriptor.abs_offset,
            DS4_N_EMBD as u32, DS4_RMS_EPS) }, "output_norm");

        ok!(unsafe { crate::metal_kernels::matmul_q8_0_tensor(
            metal.logits.as_ptr(), model_map, model_size,
            weights.output.descriptor.abs_offset,
            DS4_N_EMBD as u64, metal.vocab_dim,
            metal.output_norm.as_const_ptr(), 1) }, "vocab matmul");

        Ok(())
    }

    fn eval_token_with_rust_cache_and_scratch(
        &self,
        cache: &mut RustKvCache,
        token: i32,
        pos: u32,
        scratch: &mut RustAttentionDecodeScratch,
        ffn_scratch: &mut RustFfnDecodeScratch,
    ) -> Result<Vec<f32>> {
        if cache.layers.len() != DS4_N_LAYER as usize {
            bail!("Rust KV cache layer count {} does not match model layer count {}", cache.layers.len(), DS4_N_LAYER)
        }

        let mut cur = self.token_input_hc(token)?;
        for layer in 0..DS4_N_LAYER {
            let layer_cache = &mut cache.layers[layer as usize];
            let pre = self.layer_attention_qkv_with_scratch(layer, &cur, pos, scratch)?;
            f16_round_into(&scratch.attn_kv, &mut scratch.raw_cache_row);
            push_raw_cache_row(
                &mut layer_cache.raw_kv,
                usize::try_from(layer_cache.cap_raw).unwrap_or_default(),
                &scratch.raw_cache_row,
            );

            let mut comp_allowed = None;
            if layer_cache.compress_ratio != 0 {
                if let Some(comp) = self.layer_compressor_decode_one(
                    layer,
                    &scratch.attn_norm,
                    &mut layer_cache.attn_state_kv,
                    &mut layer_cache.attn_state_score,
                    pos,
                )? {
                    f16_round_into(&comp, &mut scratch.attn_comp_row);
                    push_comp_cache_row(
                        &mut layer_cache.attn_comp_kv,
                        usize::try_from(layer_cache.comp_cap).unwrap_or_default(),
                        &scratch.attn_comp_row,
                    )?;
                }

                if layer_cache.compress_ratio == 4 {
                    if let Some(index_comp) = self.layer_indexer_compressor_decode_one(
                        layer,
                        &scratch.attn_norm,
                        &mut layer_cache.index_state_kv,
                        &mut layer_cache.index_state_score,
                        pos,
                    )? {
                        f16_round_into(&index_comp, &mut scratch.index_comp_row);
                        push_comp_cache_row(
                            &mut layer_cache.index_comp_kv,
                            usize::try_from(layer_cache.comp_cap).unwrap_or_default(),
                            &scratch.index_comp_row,
                        )?;
                    }

                    let mask = self.layer_indexer_allowed_decode_one(
                        layer,
                        &scratch.attn_norm,
                        &scratch.attn_qr_norm,
                        &layer_cache.index_comp_kv,
                        pos,
                    )?;
                    comp_allowed = Some(mask);
                }
            }

            let mut heads = if layer_cache.compress_ratio == 0 {
                self.layer_attention_rows_raw(layer, &scratch.attn_q, &layer_cache.raw_kv)?
            } else {
                self.layer_attention_mixed(
                    layer,
                    &scratch.attn_q,
                    &layer_cache.raw_kv,
                    &layer_cache.attn_comp_kv,
                    comp_allowed.as_deref(),
                )?
            };
            self.layer_unapply_q_rope(layer, pos, &mut heads)?;
            let attn_out = self.layer_grouped_out(layer, &heads)?;
            let after_attn_hc = self.hc_post(&attn_out, &pre)?;
            cur = self.layer_ffn_with_scratch(layer, &after_attn_hc, token, ffn_scratch)?;
        }
        Ok(cur)
    }

    pub fn eval_token_logits_with_rust_cache(&self, cache: &mut RustKvCache, token: i32, pos: u32) -> Result<Vec<f32>> {
        let hc = self.eval_token_with_rust_cache(cache, token, pos)?;
        self.output_logits(&hc)
    }

    pub fn prefill_tokens_with_rust_cache(
        &self,
        cache: &mut RustKvCache,
        tokens: &[i32],
        pos0: u32,
    ) -> Result<Vec<f32>> {
        let mut hc = Vec::new();
        for (index, token) in tokens.iter().copied().enumerate() {
            hc = self.eval_token_with_rust_cache(cache, token, pos0 + index as u32)?;
        }
        Ok(hc)
    }

    pub fn output_hc_head(&self, hc_state: &[f32]) -> Result<Vec<f32>> {
        let flat = rms_norm_no_weight(hc_state, DS4_RMS_EPS)?;
        let pre = matvec_f16_tensor(
            &self.tensor_bindings.output_hc_fn,
            self.model_map.as_bytes(),
            &flat,
        )?;
        let scale = self
            .tensor_bindings
            .output_hc_scale
            .read_f32_values(self.model_map.as_bytes())?;
        let base = self
            .tensor_bindings
            .output_hc_base
            .read_f32_values(self.model_map.as_bytes())?;
        if scale.len() != 1 {
            bail!("output_hc_scale must contain exactly one value")
        }
        if base.len() != pre.len() {
            bail!("output_hc_base width does not match output_hc_fn output")
        }

        let weights = pre
            .iter()
            .zip(base.iter())
            .map(|(value, base)| sigmoid_stable(*value * scale[0] + *base) + DS4_HC_EPS)
            .collect::<Vec<_>>();
        hc_weighted_sum(hc_state, &weights)
    }

    pub fn output_hc_head_rows(&self, hc_rows: &[f32]) -> Result<Vec<f32>> {
        let row_width = DS4_N_HC * DS4_N_EMBD;
        let flat = rms_norm_no_weight_rows(hc_rows, row_width, DS4_RMS_EPS)?;
        let pre = matvec_f16_rows_tensor(
            &self.tensor_bindings.output_hc_fn,
            self.model_map.as_bytes(),
            &flat,
        )?;
        let scale = read_single_f32_tensor(&self.tensor_bindings.output_hc_scale, self.model_map.as_bytes())?;
        let mut weights = Vec::new();
        fill_sigmoid_weights_rows_from_base_tensor(
            &pre,
            scale,
            &self.tensor_bindings.output_hc_base,
            self.model_map.as_bytes(),
            &mut weights,
        )?;
        hc_weighted_sum_rows(hc_rows, &weights, DS4_N_HC)
    }

    pub fn mtp_output_hc_head(&self, hc_state: &[f32]) -> Result<Vec<f32>> {
        let (mtp, mtp_bytes) = self.mtp_runtime()?;
        let flat = rms_norm_no_weight(hc_state, DS4_RMS_EPS)?;
        let pre = matvec_f16_tensor(&mtp.hc_head_fn, mtp_bytes, &flat)?;
        let scale = read_single_f32_tensor(&mtp.hc_head_scale, mtp_bytes)?;
        let mut weights = Vec::new();
        fill_sigmoid_weights_from_base_tensor(&pre, scale, &mtp.hc_head_base, mtp_bytes, &mut weights)?;
        hc_weighted_sum(hc_state, &weights)
    }

    pub fn output_head_input(&self, hc_state: &[f32]) -> Result<Vec<f32>> {
        let embd = self.output_hc_head(hc_state)?;
        let weight = self
            .tensor_bindings
            .output_norm
            .read_f32_values(self.model_map.as_bytes())?;
        rms_norm_weight(&embd, &weight, DS4_RMS_EPS)
    }

    pub fn output_head_input_rows(&self, hc_rows: &[f32]) -> Result<Vec<f32>> {
        let embd_rows = self.output_hc_head_rows(hc_rows)?;
        let weight = self
            .tensor_bindings
            .output_norm
            .read_f32_values(self.model_map.as_bytes())?;
        rms_norm_weight_rows(&embd_rows, &weight, DS4_RMS_EPS)
    }

    pub fn mtp_output_head_input(&self, hc_state: &[f32]) -> Result<Vec<f32>> {
        let (mtp, mtp_bytes) = self.mtp_runtime()?;
        let embd = self.mtp_output_hc_head(hc_state)?;
        rms_norm_weight(&embd, &mtp.norm.read_f32_values(mtp_bytes)?, DS4_RMS_EPS)
    }

    pub fn output_logits(&self, hc_state: &[f32]) -> Result<Vec<f32>> {
        let norm = self.output_head_input(hc_state)?;
        matvec_q8_0_tensor(&self.tensor_bindings.output, self.model_map.as_bytes(), &norm)
    }

    pub fn output_logits_rows(&self, hc_rows: &[f32]) -> Result<Vec<f32>> {
        let norm_rows = self.output_head_input_rows(hc_rows)?;
        matvec_q8_0_rows_tensor(&self.tensor_bindings.output, self.model_map.as_bytes(), &norm_rows)
    }

    pub fn mtp_output_logits(&self, hc_state: &[f32]) -> Result<Vec<f32>> {
        let norm = self.mtp_output_head_input(hc_state)?;
        matvec_q8_0_tensor(&self.tensor_bindings.output, self.model_map.as_bytes(), &norm)
    }

    pub fn mtp_draft_step(
        &self,
        prev_hc: &[f32],
        token: i32,
        pos: u32,
        raw_kv_rows: &mut Vec<f32>,
        raw_cap: u32,
    ) -> Result<MtpDraftStepOutput> {
        let (mtp, mtp_bytes) = self.mtp_runtime()?;
        let binding = &mtp.block;
        let input_hc = self.mtp_input_hc(prev_hc, token)?;

        let attn_pre = hc_pre_from_tensors(
            mtp_bytes,
            &binding.hc_attn_fn,
            &binding.hc_attn_scale,
            &binding.hc_attn_base,
            &input_hc,
        )?;
        let attn_norm = rms_norm_weight(
            &attn_pre.sublayer_input,
            &binding.attn_norm.read_f32_values(mtp_bytes)?,
            DS4_RMS_EPS,
        )?;
        let qr = matvec_q8_0_tensor(&binding.attn_q_a, mtp_bytes, &attn_norm)?;
        let qr_norm = rms_norm_weight(
            &qr,
            &binding.attn_q_a_norm.read_f32_values(mtp_bytes)?,
            DS4_RMS_EPS,
        )?;
        let mut q = matvec_q8_0_tensor(&binding.attn_q_b, mtp_bytes, &qr_norm)?;
        let sinks = binding.attn_sinks.read_f32_values(mtp_bytes)?;
        let n_head = sinks.len();
        if n_head == 0 || q.len() % n_head != 0 {
            bail!("MTP draft Q projection width {} does not divide into {} heads", q.len(), n_head)
        }
        let head_dim = q.len() / n_head;
        head_rms_norm_inplace(&mut q, n_head, head_dim, DS4_RMS_EPS)?;

        let raw = matvec_q8_0_tensor(&binding.attn_kv, mtp_bytes, &attn_norm)?;
        let mut kv = rms_norm_weight(
            &raw,
            &binding.attn_kv_a_norm.read_f32_values(mtp_bytes)?,
            DS4_RMS_EPS,
        )?;
        rope_tail_layer_inplace(&mut q, n_head, head_dim, DS4_N_ROT, pos, 0, false)?;
        let kv_width = kv.len();
        rope_tail_layer_inplace(&mut kv, 1, kv_width, DS4_N_ROT, pos, 0, false)?;
        dsv4_fp8_kv_quantize_row_inplace(&mut kv, kv_width, DS4_N_ROT)?;

        let rounded_kv = f16_round_slice(&kv);
        let cap_raw = if raw_cap == 0 {
            usize::try_from(DS4_N_SWA).unwrap_or_default()
        } else {
            usize::try_from(raw_cap).unwrap_or_default()
        };
        push_raw_cache_row(raw_kv_rows, cap_raw, &rounded_kv);

        let mut heads = attention_rows_with_sinks(&q, raw_kv_rows, &sinks)?;
        rope_tail_layer_inplace(&mut heads, n_head, head_dim, DS4_N_ROT, pos, 0, true)?;

        let group_dim = usize::try_from(binding.attn_output_a.descriptor.dims[0])
            .context("MTP attention output A group width does not fit in usize")?;
        let total_low = usize::try_from(binding.attn_output_a.descriptor.dims[1])
            .context("MTP attention output A rank does not fit in usize")?;
        if heads.len() % group_dim != 0 {
            bail!(
                "MTP attention head vector length {} is not divisible by group width {}",
                heads.len(),
                group_dim
            )
        }
        let n_groups = heads.len() / group_dim;
        if total_low % n_groups != 0 {
            bail!(
                "MTP attention output A rank {} is not divisible by group count {}",
                total_low,
                n_groups
            )
        }
        let rank = total_low / n_groups;
        let low = matvec_q8_0_grouped_tensor(
            &binding.attn_output_a,
            mtp_bytes,
            &heads,
            n_groups,
            group_dim,
            rank,
        )?;
        let attn_out = matvec_q8_0_tensor(&binding.attn_output_b, mtp_bytes, &low)?;
        let after_attn_hc = hc_post(&attn_out, &attn_pre.residual_hc, &attn_pre.post, &attn_pre.comb)?;

        let ffn_pre = hc_pre_from_tensors(
            mtp_bytes,
            &binding.hc_ffn_fn,
            &binding.hc_ffn_scale,
            &binding.hc_ffn_base,
            &after_attn_hc,
        )?;
        let ffn_norm = rms_norm_weight(
            &ffn_pre.sublayer_input,
            &binding.ffn_norm.read_f32_values(mtp_bytes)?,
            DS4_RMS_EPS,
        )?;

        let shared_gate = matvec_q8_0_tensor(&binding.ffn_gate_shexp, mtp_bytes, &ffn_norm)?;
        let shared_up = matvec_q8_0_tensor(&binding.ffn_up_shexp, mtp_bytes, &ffn_norm)?;
        let shared_mid = swiglu(&shared_gate, &shared_up)?;
        let shared = matvec_q8_0_tensor(&binding.ffn_down_shexp, mtp_bytes, &shared_mid)?;

        let probs = matvec_f16_tensor(&binding.ffn_gate_inp, mtp_bytes, &ffn_norm)?
            .into_iter()
            .map(|logit| softplus_stable(logit).sqrt())
            .collect::<Vec<_>>();
        if probs.is_empty() {
            bail!("MTP expert router produced no probabilities")
        }
        let mut selection = probs.clone();
        if let Some(bias) = binding.ffn_exp_probs_b.as_ref() {
            let bias_values = bias.read_f32_values(mtp_bytes)?;
            if bias_values.len() != selection.len() {
                bail!(
                    "MTP expert bias length {} does not match router probability length {}",
                    bias_values.len(),
                    selection.len()
                )
            }
            for (score, bias) in selection.iter_mut().zip(bias_values.iter().copied()) {
                *score += bias;
            }
        }
        let selected = select_top_k_desc_indices(&selection, usize::min(DS4_N_EXPERT_USED, selection.len()))
            .into_iter()
            .map(|index| i32::try_from(index).unwrap_or_default())
            .collect::<Vec<_>>();
        let expert_weight = hash_router_weights_from_probs(&probs, &selected)?;

        let gate = &binding.ffn_gate_exps;
        let up = &binding.ffn_up_exps;
        let down = &binding.ffn_down_exps;
        if gate.descriptor.tensor_type != DS4_TENSOR_IQ2_XXS || gate.descriptor.ndim != 3 {
            bail!("{} is not a 3D IQ2_XXS expert tensor", gate.name)
        }
        if up.descriptor.tensor_type != DS4_TENSOR_IQ2_XXS || up.descriptor.ndim != 3 {
            bail!("{} is not a 3D IQ2_XXS expert tensor", up.name)
        }
        if down.descriptor.tensor_type != DS4_TENSOR_Q2_K || down.descriptor.ndim != 3 {
            bail!("{} is not a 3D Q2_K expert tensor", down.name)
        }

        let xq = quantize_row_q8_k(&ffn_norm)?;
        let in_dim = usize::try_from(gate.descriptor.dims[0]).context("MTP expert gate input width does not fit in usize")?;
        let mid_dim = usize::try_from(gate.descriptor.dims[1]).context("MTP expert gate output width does not fit in usize")?;
        let out_dim = usize::try_from(down.descriptor.dims[1]).context("MTP expert down output width does not fit in usize")?;
        let mut mids = vec![0.0f32; selected.len() * mid_dim];
        for (slot, expert) in selected.iter().copied().enumerate() {
            let expert_index = u32::try_from(expert).context("selected MTP expert index is negative")?;
            let (gate_base, gate_row_bytes, gate_in_dim, gate_out_dim) =
                expert_tensor_bytes(gate, mtp_bytes, expert_index)?;
            let (up_base, up_row_bytes, up_in_dim, up_out_dim) =
                expert_tensor_bytes(up, mtp_bytes, expert_index)?;
            if gate_in_dim != up_in_dim || gate_out_dim != up_out_dim {
                bail!("MTP expert {} gate/up tensor shapes do not match", expert)
            }
            if gate_in_dim != in_dim || gate_out_dim != mid_dim {
                bail!("MTP expert {} gate/up tensor shape does not match expected layout", expert)
            }

            for row in 0..mid_dim {
                let gate_start = row * gate_row_bytes;
                let gate_end = gate_start + gate_row_bytes;
                let up_start = row * up_row_bytes;
                let up_end = up_start + up_row_bytes;
                let mut gate_value = dot_iq2_xxs_q8_k_row(&gate_base[gate_start..gate_end], &xq, in_dim)?;
                let mut up_value = dot_iq2_xxs_q8_k_row(&up_base[up_start..up_end], &xq, in_dim)?;
                if DS4_SWIGLU_CLAMP_EXP > 1.0e-6 {
                    gate_value = gate_value.min(DS4_SWIGLU_CLAMP_EXP);
                    up_value = up_value.clamp(-DS4_SWIGLU_CLAMP_EXP, DS4_SWIGLU_CLAMP_EXP);
                }
                mids[slot * mid_dim + row] = silu(gate_value) * up_value * expert_weight[slot];
            }
        }

        let mut routed = vec![0.0f32; out_dim];
        for (slot, expert) in selected.iter().copied().enumerate() {
            let expert_index = u32::try_from(expert).context("selected MTP expert index is negative")?;
            let (down_base, down_row_bytes, down_in_dim, down_out_dim) =
                expert_tensor_bytes(down, mtp_bytes, expert_index)?;
            if down_in_dim != mid_dim || down_out_dim != out_dim {
                bail!("MTP expert {} down tensor shape does not match expected layout", expert)
            }
            let start = slot * mid_dim;
            let end = start + mid_dim;
            let quantized = quantize_row_q8_k(&mids[start..end])?;
            for row in 0..out_dim {
                let row_start = row * down_row_bytes;
                let row_end = row_start + down_row_bytes;
                routed[row] += dot_q2_k_q8_k_row(&down_base[row_start..row_end], &quantized, mid_dim)?;
            }
        }
        if routed.len() != shared.len() {
            bail!(
                "MTP routed MoE width {} does not match shared expert width {}",
                routed.len(),
                shared.len()
            )
        }

        let ffn_out = routed
            .iter()
            .zip(shared.iter())
            .map(|(routed, shared)| routed + shared)
            .collect::<Vec<_>>();
        let hc_state = hc_post(&ffn_out, &ffn_pre.residual_hc, &ffn_pre.post, &ffn_pre.comb)?;
        let logits = self.mtp_output_logits(&hc_state)?;
        let top_token = sample_argmax(&logits);

        Ok(MtpDraftStepOutput {
            input_hc,
            hc_state,
            logits,
            top_token,
        })
    }

    fn output_logits_with_scratch(
        &self,
        hc_state: &[f32],
        scratch: &mut RustOutputScratch,
        logits: &mut Vec<f32>,
    ) -> Result<()> {
        let gguf_bytes = self.model_map.as_bytes();
        rms_norm_no_weight_into(hc_state, DS4_RMS_EPS, &mut scratch.flat_hc)?;
        matvec_f16_tensor_into(
            &self.tensor_bindings.output_hc_fn,
            gguf_bytes,
            &scratch.flat_hc,
            &mut scratch.output_pre,
        )?;

        let scale = read_single_f32_tensor(&self.tensor_bindings.output_hc_scale, gguf_bytes)?;
        fill_sigmoid_weights_from_base_tensor(
            &scratch.output_pre,
            scale,
            &self.tensor_bindings.output_hc_base,
            gguf_bytes,
            &mut scratch.output_weights,
        )?;
        hc_weighted_sum_into(hc_state, &scratch.output_weights, &mut scratch.output_embd)?;
        rms_norm_no_weight_into(&scratch.output_embd, DS4_RMS_EPS, &mut scratch.output_norm)?;
        mul_f32_tensor_inplace(&mut scratch.output_norm, &self.tensor_bindings.output_norm, gguf_bytes)?;
        matvec_q8_0_tensor_into(
            &self.tensor_bindings.output,
            gguf_bytes,
            &scratch.output_norm,
            &mut scratch.q8_xq,
            &mut scratch.q8_xscale,
            logits,
        )
    }

    pub fn tokenize_text(&self, text: &str) -> Result<Tokens> {
        if text.as_bytes().contains(&0) {
            bail!("prompt contains an embedded NUL byte")
        }

        let mut out = Tokens::new();
        for token in self.tokenizer.tokenize_text(text) {
            out.push(token);
        }
        Ok(out)
    }

    pub fn tokenize_rendered_chat(&self, text: &str) -> Result<Tokens> {
        if text.as_bytes().contains(&0) {
            bail!("rendered chat contains an embedded NUL byte")
        }

        let mut out = Tokens::new();
        let bytes = text.as_bytes();
        let mut span_start = 0usize;
        let mut pos = 0usize;
        while pos < bytes.len() {
            if let Some((matched_len, token)) = self.special_tokens.matching_token(&bytes[pos..]) {
                if span_start < pos {
                    let span = std::str::from_utf8(&bytes[span_start..pos])
                        .context("rendered chat split on a non-UTF-8 boundary")?;
                    let span_tokens = self.tokenize_text(span)?;
                    out.extend_from(&span_tokens);
                }
                out.push(token);
                pos += matched_len;
                span_start = pos;
            } else {
                pos += 1;
            }
        }

        if span_start < bytes.len() {
            let span = std::str::from_utf8(&bytes[span_start..])
                .context("rendered chat tail is not valid UTF-8")?;
            let span_tokens = self.tokenize_text(span)?;
            out.extend_from(&span_tokens);
        }

        Ok(out)
    }

    pub fn encode_chat_prompt(&self, system: Option<&str>, prompt: &str, think_mode: ThinkMode) -> Result<Tokens> {
        let rendered = render_one_shot_prompt(system, prompt, think_mode);
        self.tokenize_rendered_chat(&rendered)
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        if self.backend == Backend::Metal {
            metal_runtime::cleanup()
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RustSessionSyncPlan {
    Keep,
    Extend { suffix_start: usize },
    Rebuild { prefix_len: usize },
}

fn shared_prefix_len(lhs: &[i32], rhs: &[i32]) -> usize {
    lhs.iter()
        .zip(rhs.iter())
        .take_while(|(lhs, rhs)| lhs == rhs)
        .count()
}

fn plan_rust_session_sync(existing_tokens: &[i32], prompt_tokens: &[i32]) -> RustSessionSyncPlan {
    let prefix_len = shared_prefix_len(existing_tokens, prompt_tokens);
    if prefix_len == existing_tokens.len() {
        if prefix_len == prompt_tokens.len() {
            RustSessionSyncPlan::Keep
        } else {
            RustSessionSyncPlan::Extend {
                suffix_start: prefix_len,
            }
        }
    } else {
        RustSessionSyncPlan::Rebuild { prefix_len }
    }
}

fn rust_session_spec_accept_limit(mtp_draft_tokens: i32, max_tokens: i32, accepted_cap: usize) -> usize {
    if max_tokens <= 0 || accepted_cap == 0 {
        return 0;
    }

    let max_accept = usize::try_from(max_tokens).unwrap_or(usize::MAX);
    let draft_accept = usize::try_from(mtp_draft_tokens.max(0))
        .unwrap_or(usize::MAX)
        .saturating_add(1);
    accepted_cap.min(max_accept).min(draft_accept)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RustSpecVerifyPlan {
    Exact { draft_tokens: usize },
    ExactBatch2,
    MarginSkipSingle,
}

fn rust_mtp_strict(engine: &Engine) -> bool {
    engine.quality || std::env::var_os("DS4_MTP_STRICT").is_some()
}

fn rust_mtp_margin_threshold(engine: &Engine) -> f32 {
    let default_threshold = engine.mtp_margin;
    let Some(value) = std::env::var_os("DS4_MTP_MIN_MARGIN") else {
        return default_threshold;
    };
    value
        .into_string()
        .ok()
        .and_then(|value| value.parse::<f32>().ok())
        .filter(|value| *value >= 0.0)
        .unwrap_or(default_threshold)
}

fn logits_top2_margin(logits: &[f32]) -> f32 {
    let mut best = f32::NEG_INFINITY;
    let mut second = f32::NEG_INFINITY;
    for value in logits.iter().copied() {
        if value > best {
            second = best;
            best = value;
        } else if value > second {
            second = value;
        }
    }
    best - second
}

fn plan_rust_spec_verify(
    draft_count: usize,
    strict_mtp: bool,
    margin_threshold: f32,
    mtp_margin: Option<f32>,
) -> RustSpecVerifyPlan {
    if !strict_mtp
        && draft_count == 2
        && margin_threshold > 0.0
        && mtp_margin.is_some_and(|margin| margin < margin_threshold)
    {
        RustSpecVerifyPlan::MarginSkipSingle
    } else if strict_mtp && draft_count == 2 {
        RustSpecVerifyPlan::ExactBatch2
    } else {
        RustSpecVerifyPlan::Exact {
            draft_tokens: draft_count,
        }
    }
}

impl<'a> RustSession<'a> {
    pub fn new(engine: &'a Engine, ctx_size: u32, raw_cap: u32) -> Self {
        let mtp_hc_state = if engine.has_rust_mtp_drafter() {
            vec![0.0; DS4_N_HC * DS4_N_EMBD]
        } else {
            Vec::new()
        };
        let metal_graph = if engine.backend == Backend::Metal {
            engine.new_metal_decode_graph(ctx_size, raw_cap)
        } else {
            None
        };
        Self {
            engine,
            cache: engine.new_rust_kv_cache(ctx_size, raw_cap),
            metal_graph,
            tokens: Vec::new(),
            position: 0,
            ctx_size,
            raw_cap,
            hc_state: Vec::new(),
            logits: Vec::new(),
            mtp_prefix_len: 0,
            mtp_hc_state,
            mtp_raw_kv: Vec::new(),
            mtp_logits: Vec::new(),
            mtp_draft_token: 0,
            mtp_draft_valid: false,
            attention_scratch: RustAttentionDecodeScratch::new(),
            ffn_scratch: RustFfnDecodeScratch::new(),
            output_scratch: RustOutputScratch::new(),
        }
    }

    fn reset(&mut self) {
        self.cache = self.engine.new_rust_kv_cache(self.ctx_size, self.raw_cap);
        // Reset Metal decode graph compressor counters and state tensors.
        if let Some(ref mut mg) = self.metal_graph {
            for layer in mg.layers.iter_mut() {
                layer.n_comp = 0;
                layer.n_index_comp = 0;
                let ratio = layer.compress_ratio;
                if ratio != 0 {
                    let coff = if ratio == 4 { 2u64 } else { 1u64 };
                    if let (Some(sk), Some(ss)) = (layer.attn_state_kv.as_ref(), layer.attn_state_score.as_ref()) {
                        let n = (coff * DS4_N_HEAD_DIM * coff * ratio as u64) as usize;
                        sk.write_f32(&vec![0.0f32; n]);
                        ss.write_f32(&vec![DS4_NEG_INF; n]);
                    }
                    if ratio == 4 {
                        if let (Some(sk), Some(ss)) = (layer.index_state_kv.as_ref(), layer.index_state_score.as_ref()) {
                            let n = (coff * DS4_N_INDEXER_HEAD_DIM * coff * ratio as u64) as usize;
                            sk.write_f32(&vec![0.0f32; n]);
                            ss.write_f32(&vec![DS4_NEG_INF; n]);
                        }
                    }
                }
            }
        }
        self.tokens.clear();
        self.position = 0;
        self.hc_state.clear();
        self.logits.clear();
        self.mtp_prefix_len = 0;
        self.mtp_raw_kv.clear();
        self.mtp_logits.clear();
        self.mtp_draft_token = 0;
        self.mtp_draft_valid = false;
        if self.engine.has_rust_mtp_drafter() {
            self.mtp_hc_state.clear();
            self.mtp_hc_state.resize(DS4_N_HC * DS4_N_EMBD, 0.0);
        } else {
            self.mtp_hc_state.clear();
        }
    }

    fn ensure_mtp_prefix_state(&mut self) -> Result<bool> {
        if !self.engine.has_rust_mtp_drafter() {
            return Ok(false);
        }
        if self.mtp_hc_state.len() != DS4_N_HC * DS4_N_EMBD {
            self.mtp_hc_state.clear();
            self.mtp_hc_state.resize(DS4_N_HC * DS4_N_EMBD, 0.0);
        }

        while self.mtp_prefix_len < self.tokens.len() {
            let token = self.tokens[self.mtp_prefix_len];
            let pos = u32::try_from(self.mtp_prefix_len).context("MTP prefix length does not fit in u32")?;
            let draft = self.engine.mtp_draft_step(
                &self.mtp_hc_state,
                token,
                pos,
                &mut self.mtp_raw_kv,
                self.raw_cap,
            )?;
            self.mtp_hc_state = draft.hc_state;
            self.mtp_logits = draft.logits;
            self.mtp_draft_token = draft.top_token;
            self.mtp_draft_valid = true;
            self.mtp_prefix_len += 1;
        }

        Ok(self.mtp_draft_valid)
    }

    fn verify_draft_suffix_exact(&self, drafts: &[i32], eos_token: i32) -> Result<RustDraftVerification> {
        RustDraftBatchVerifier::from_session(self).verify_exact_prefix(drafts, eos_token)
    }

    fn commit_verified_draft(&mut self, verified: RustDraftVerification) {
        self.cache = verified.cache;
        self.tokens = verified.tokens;
        self.position = verified.position;
        self.hc_state = verified.hc_state;
        self.logits = verified.logits;
        self.attention_scratch = verified.attention_scratch;
        self.ffn_scratch = verified.ffn_scratch;
        self.output_scratch = verified.output_scratch;
    }

    pub fn common_prefix(&self, prompt: &Tokens) -> i32 {
        shared_prefix_len(&self.tokens, prompt.as_slice()) as i32
    }

    pub fn sync(&mut self, prompt: &Tokens) -> Result<()> {
        let prompt_tokens = prompt.as_slice();
        match plan_rust_session_sync(&self.tokens, prompt_tokens) {
            RustSessionSyncPlan::Keep => Ok(()),
            RustSessionSyncPlan::Extend { suffix_start } => self.prefill(&prompt_tokens[suffix_start..]),
            RustSessionSyncPlan::Rebuild { prefix_len } => {
                self.rewind(prefix_len as i32)?;
                self.prefill(&prompt_tokens[prefix_len..])
            }
        }
    }

    pub fn invalidate(&mut self) {
        self.reset();
    }

    pub fn rewind(&mut self, pos: i32) -> Result<()> {
        let current_len = self.tokens.len();
        let max_pos = i32::try_from(current_len).unwrap_or(i32::MAX);
        let target = pos.clamp(0, max_pos) as usize;
        if target == current_len {
            return Ok(());
        }

        let prefix = self.tokens[..target].to_vec();
        self.reset();
        if prefix.is_empty() {
            Ok(())
        } else {
            self.prefill(&prefix)
        }
    }

    pub fn prefill(&mut self, tokens: &[i32]) -> Result<()> {
        for token in tokens.iter().copied() {
            self.eval(token)?;
        }
        Ok(())
    }

    pub fn eval(&mut self, token: i32) -> Result<()> {
        if let Some(ref mut mg) = self.metal_graph {
            self.engine.eval_token_with_metal_graph(mg, token, self.position)?;
            self.engine.metal_graph_read_logits(mg, &mut self.logits)?;
        } else {
            self.hc_state = self
                .engine
                .eval_token_with_rust_backend_and_scratch(
                    &mut self.cache,
                    token,
                    self.position,
                    &mut self.attention_scratch,
                    &mut self.ffn_scratch,
                )?;
            self.engine
                .output_logits_with_scratch(&self.hc_state, &mut self.output_scratch, &mut self.logits)?;
        }
        self.tokens.push(token);
        self.position += 1;
        self.mtp_draft_valid = false;
        Ok(())
    }

    pub fn eval_speculative_argmax(
        &mut self,
        first_token: i32,
        max_tokens: i32,
        eos_token: i32,
        accepted: &mut [i32],
    ) -> Result<usize> {
        let accept_limit = rust_session_spec_accept_limit(self.engine.mtp_draft_tokens(), max_tokens, accepted.len());
        if accept_limit == 0 {
            return Ok(0);
        }

        // RustSession does not have the MTP drafter yet, but deterministic decode
        // can still use the multi-token accepted-buffer path by chaining exact
        // target-model argmax steps from the current logits.
        self.eval(first_token)?;
        accepted[0] = first_token;
        let mut n_accept = 1usize;
        if first_token == eos_token || n_accept >= accept_limit {
            return Ok(n_accept);
        }

        if !self.ensure_mtp_prefix_state()? || !self.mtp_draft_valid {
            while n_accept < accept_limit {
                let token = sample_argmax(&self.logits);
                if token == eos_token {
                    break;
                }
                self.eval(token)?;
                accepted[n_accept] = token;
                n_accept += 1;
            }
            return Ok(n_accept);
        }

        let mut drafts = Vec::with_capacity(accept_limit.saturating_sub(n_accept));
        let mut draft_frontiers = Vec::with_capacity(accept_limit.saturating_sub(n_accept));
        let mut mtp_hc_state = self.mtp_hc_state.clone();
        let mut mtp_raw_kv = self.mtp_raw_kv.clone();
        let mut draft_token = self.mtp_draft_token;
        while drafts.len() < accept_limit.saturating_sub(n_accept) {
            let token = draft_token;
            let draft_index = u32::try_from(drafts.len())
                .context("draft index does not fit in u32")?;
            let pos = self
                .position
                .checked_add(draft_index)
                .context("draft position overflowed u32")?;
            let draft = self
                .engine
                .mtp_draft_step(&mtp_hc_state, token, pos, &mut mtp_raw_kv, self.raw_cap)?;
            drafts.push(token);
            mtp_hc_state = draft.hc_state;
            draft_frontiers.push(RustMtpDraftFrontier {
                hc_state: mtp_hc_state.clone(),
                raw_kv: mtp_raw_kv.clone(),
                logits: draft.logits.clone(),
                next_token: draft.top_token,
            });
            draft_token = draft.top_token;
            if token == eos_token {
                break;
            }
        }

        let verify_plan = plan_rust_spec_verify(
            drafts.len(),
            rust_mtp_strict(self.engine),
            rust_mtp_margin_threshold(self.engine),
            draft_frontiers.first().map(|frontier| logits_top2_margin(&frontier.logits)),
        );
        let verify_count = match verify_plan {
            RustSpecVerifyPlan::Exact { draft_tokens } => draft_tokens,
            RustSpecVerifyPlan::ExactBatch2 => 2,
            RustSpecVerifyPlan::MarginSkipSingle => 1,
        };
        let verified = match verify_plan {
            RustSpecVerifyPlan::ExactBatch2 => {
                let RustBatch2Verification {
                    row0_top,
                    row1_top,
                    prefix1,
                    verification,
                } = RustDraftBatchVerifier::from_session(self)
                    .verify_exact_batch2([drafts[0], drafts[1]], eos_token)?;
                let _ = row0_top;
                let _ = row1_top;
                if verification.accepted == 1 {
                    prefix1.unwrap_or(verification)
                } else {
                    verification
                }
            }
            _ => self.verify_draft_suffix_exact(&drafts[..verify_count], eos_token)?,
        };
        for token in drafts.iter().copied().take(verified.accepted) {
            accepted[n_accept] = token;
            n_accept += 1;
        }
        if verified.accepted > 0 {
            self.commit_verified_draft(verified);
            if let Some(frontier) = draft_frontiers.get(self.tokens.len().saturating_sub(self.mtp_prefix_len + 1)) {
                self.mtp_hc_state = frontier.hc_state.clone();
                self.mtp_raw_kv = frontier.raw_kv.clone();
                self.mtp_logits = frontier.logits.clone();
                self.mtp_draft_token = frontier.next_token;
                self.mtp_draft_valid = true;
                self.mtp_prefix_len = self.tokens.len();
            } else {
                self.mtp_draft_valid = false;
                self.ensure_mtp_prefix_state()?;
            }
        }

        Ok(n_accept)
    }

    pub fn ctx(&self) -> i32 {
        self.ctx_size as i32
    }

    pub fn pos(&self) -> i32 {
        self.position as i32
    }

    pub fn sample(&self, temperature: f32, top_k: i32, top_p: f32, min_p: f32, rng: &mut u64) -> i32 {
        sample_top_p_min_p(&self.logits, temperature, top_k, top_p, min_p, rng)
    }

    pub fn position(&self) -> u32 {
        self.position
    }

    pub fn tokens(&self) -> &[i32] {
        &self.tokens
    }

    pub fn hc_state(&self) -> &[f32] {
        &self.hc_state
    }

    pub fn logits(&self) -> &[f32] {
        &self.logits
    }

    pub fn argmax_token(&self) -> i32 {
        if self.logits.is_empty() {
            -1
        } else {
            sample_argmax(&self.logits)
        }
    }

    pub fn top_logprobs(&self, n: usize) -> Vec<(i32, f32)> {
        if self.logits.is_empty() || n == 0 {
            return Vec::new();
        }
        let max_l = self
            .logits
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        if !max_l.is_finite() {
            return Vec::new();
        }
        let sum_exp: f32 = self
            .logits
            .iter()
            .copied()
            .filter(|v| v.is_finite())
            .map(|l| (l - max_l).exp())
            .sum();
        if sum_exp <= 0.0 || !sum_exp.is_finite() {
            return Vec::new();
        }
        let log_sum_exp = sum_exp.ln() + max_l;
        let mut pairs: Vec<(i32, f32)> = self
            .logits
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(i, l)| {
                if l.is_finite() {
                    Some((i as i32, l - log_sum_exp))
                } else {
                    None
                }
            })
            .collect();
        pairs.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        pairs.truncate(n);
        pairs
    }

    pub fn cache(&self) -> &RustKvCache {
        &self.cache
    }

    pub fn cache_mut(&mut self) -> &mut RustKvCache {
        &mut self.cache
    }
}

#[derive(Clone, Debug)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Clone, Copy, Debug)]
pub struct GenerationOptions {
    pub max_tokens: i32,
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,
    pub min_p: f32,
    pub seed: Option<u64>,
}

impl Default for GenerationOptions {
    fn default() -> Self {
        Self {
            max_tokens: 50_000,
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            seed: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct GenerationResult {
    pub bytes: Vec<u8>,
    pub prompt_tokens: i32,
    pub completion_tokens: i32,
    pub prefill_time: Duration,
    pub decode_time: Duration,
}

impl GenerationResult {
    pub fn text_lossy(&self) -> String {
        String::from_utf8_lossy(&self.bytes).into_owned()
    }
}

pub fn is_rendered_chat_prompt(prompt: &str) -> bool {
    prompt.starts_with(RENDERED_CHAT_BOS)
}

pub fn build_chat_prompt(
    engine: &Engine,
    system: Option<&str>,
    messages: &[ChatMessage],
    think_mode: ThinkMode,
) -> Result<Tokens> {
    let rendered = render_chat_messages(system, messages, think_mode);
    engine.tokenize_rendered_chat(&rendered)
}

pub fn build_chat_generation_prompt(
    engine: &Engine,
    system: Option<&str>,
    messages: &[ChatMessage],
    think_mode: ThinkMode,
) -> Result<Tokens> {
    let rendered = render_chat_generation_prompt(system, messages, think_mode);
    engine.tokenize_rendered_chat(&rendered)
}

pub trait GenerationSessionLike {
    fn sync_prompt(&mut self, prompt: &Tokens) -> Result<()>;
    fn ctx(&self) -> i32;
    fn pos(&self) -> i32;
    fn sample_token(&mut self, temperature: f32, top_k: i32, top_p: f32, min_p: f32, rng: &mut u64) -> i32;
    fn eval_generated(
        &mut self,
        token: i32,
        use_mtp: bool,
        remaining: i32,
        eos: i32,
        accepted_buf: &mut [i32; 17],
    ) -> Result<usize>;
    fn clear_progress(&mut self) {}
}

impl<'a> GenerationSessionLike for RustSession<'a> {
    fn sync_prompt(&mut self, prompt: &Tokens) -> Result<()> {
        RustSession::sync(self, prompt)
    }

    fn ctx(&self) -> i32 {
        RustSession::ctx(self)
    }

    fn pos(&self) -> i32 {
        RustSession::pos(self)
    }

    fn sample_token(&mut self, temperature: f32, top_k: i32, top_p: f32, min_p: f32, rng: &mut u64) -> i32 {
        RustSession::sample(self, temperature, top_k, top_p, min_p, rng)
    }

    fn eval_generated(
        &mut self,
        token: i32,
        use_mtp: bool,
        remaining: i32,
        eos: i32,
        accepted_buf: &mut [i32; 17],
    ) -> Result<usize> {
        if use_mtp {
            RustSession::eval_speculative_argmax(self, token, remaining, eos, accepted_buf)
        } else {
            RustSession::eval(self, token)?;
            accepted_buf[0] = token;
            Ok(1)
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct SampleCandidate {
    id: i32,
    logit: f32,
    prob: f32,
}

fn sample_rng_next(state: &mut u64) -> u64 {
    let mut x = *state;
    if x == 0 {
        x = 0x9e37_79b9_7f4a_7c15;
    }
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    x.wrapping_mul(0x2545_f491_4f6c_dd1d)
}

fn sample_rng_f32(state: &mut u64) -> f32 {
    let x = sample_rng_next(state);
    ((x >> 40) & 0x00ff_ffff) as f32 / 16_777_216.0
}

fn sample_argmax(logits: &[f32]) -> i32 {
    let mut best = 0usize;
    let mut best_value = f32::NEG_INFINITY;
    for (index, value) in logits.iter().copied().enumerate() {
        if value > best_value {
            best = index;
            best_value = value;
        }
    }
    best as i32
}

fn sample_full_vocab(logits: &[f32], temperature: f32, top_p: f32, min_p: f32, rng: &mut u64) -> i32 {
    let mut max_logit = DS4_NEG_INF;
    let mut best = 0i32;
    let mut finite = 0usize;
    for (index, value) in logits.iter().copied().enumerate() {
        if !value.is_finite() {
            continue;
        }
        finite += 1;
        if value > max_logit {
            max_logit = value;
            best = index as i32;
        }
    }
    if finite == 0 {
        return sample_argmax(logits);
    }

    if top_p >= 1.0 {
        let min_rel = min_p.max(0.0);
        let mut sum = 0.0f32;
        for value in logits.iter().copied() {
            if !value.is_finite() {
                continue;
            }
            let prob = ((value - max_logit) / temperature).exp();
            if prob < min_rel {
                continue;
            }
            sum += prob;
        }
        if sum <= 0.0 || !sum.is_finite() {
            return best;
        }
        let mut r = sample_rng_f32(rng) * sum;
        for (index, value) in logits.iter().copied().enumerate() {
            if !value.is_finite() {
                continue;
            }
            let prob = ((value - max_logit) / temperature).exp();
            if prob < min_rel {
                continue;
            }
            r -= prob;
            if r <= 0.0 {
                return index as i32;
            }
        }
        return best;
    }

    let mut candidates = logits
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, value)| value.is_finite())
        .map(|(index, logit)| SampleCandidate {
            id: index as i32,
            logit,
            prob: ((logit - max_logit) / temperature).exp(),
        })
        .collect::<Vec<_>>();
    let sum = candidates.iter().map(|candidate| candidate.prob).sum::<f32>();
    if sum <= 0.0 || !sum.is_finite() {
        return best;
    }

    candidates.sort_by(|lhs, rhs| rhs.logit.total_cmp(&lhs.logit));
    let min_prob = (candidates[0].prob / sum) * min_p.max(0.0);
    let mut filtered_sum = 0.0f32;
    let mut filtered = 0usize;
    for (index, candidate) in candidates.iter().enumerate() {
        let prob = candidate.prob / sum;
        if index > 0 && prob < min_prob {
            break;
        }
        filtered_sum += candidate.prob;
        filtered += 1;
        if filtered_sum / sum >= top_p {
            break;
        }
    }
    if filtered == 0 {
        return best;
    }

    let mut r = sample_rng_f32(rng) * filtered_sum;
    for candidate in candidates.iter().take(filtered) {
        r -= candidate.prob;
        if r <= 0.0 {
            return candidate.id;
        }
    }
    candidates[filtered - 1].id
}

fn sample_top_p_min_p(logits: &[f32], temperature: f32, top_k: i32, top_p: f32, min_p: f32, rng: &mut u64) -> i32 {
    if temperature <= 0.0 {
        return sample_argmax(logits);
    }
    let top_p = if top_p <= 0.0 || top_p > 1.0 { 1.0 } else { top_p };
    let min_p = min_p.max(0.0);
    if top_k <= 0 {
        return sample_full_vocab(logits, temperature, top_p, min_p, rng);
    }

    let capped_top_k = usize::min(top_k.clamp(0, 1024) as usize, logits.len());
    if capped_top_k == 0 {
        return sample_argmax(logits);
    }

    let mut ids = vec![0i32; capped_top_k];
    let mut vals = vec![0.0f32; capped_top_k];
    let mut n = 0usize;
    for (index, value) in logits.iter().copied().enumerate() {
        if !value.is_finite() {
            continue;
        }
        if n == capped_top_k && value <= vals[n - 1] {
            continue;
        }
        let mut j = if n < capped_top_k { n } else { n - 1 };
        if n < capped_top_k {
            n += 1;
        }
        while j > 0 && vals[j - 1] < value {
            vals[j] = vals[j - 1];
            ids[j] = ids[j - 1];
            j -= 1;
        }
        vals[j] = value;
        ids[j] = index as i32;
    }
    if n == 0 {
        return sample_argmax(logits);
    }

    let max_logit = vals[0];
    let mut probs = vec![0.0f32; n];
    let mut sum = 0.0f32;
    for index in 0..n {
        probs[index] = ((vals[index] - max_logit) / temperature).exp();
        sum += probs[index];
    }
    if sum <= 0.0 || !sum.is_finite() {
        return ids[0];
    }

    let min_prob = (probs[0] / sum) * min_p;
    let mut filtered_sum = 0.0f32;
    let mut filtered = 0usize;
    for index in 0..n {
        let prob = probs[index] / sum;
        if index > 0 && prob < min_prob {
            break;
        }
        filtered_sum += probs[index];
        filtered += 1;
        if filtered_sum / sum >= top_p {
            break;
        }
    }
    if filtered == 0 {
        return ids[0];
    }

    let mut r = sample_rng_f32(rng) * filtered_sum;
    for index in 0..filtered {
        r -= probs[index];
        if r <= 0.0 {
            return ids[index];
        }
    }
    ids[filtered - 1]
}

pub fn generate_with_session<S: GenerationSessionLike>(
    engine: &Engine,
    session: &mut S,
    prompt: &Tokens,
    options: GenerationOptions,
) -> Result<GenerationResult> {
    let prefill_start = Instant::now();
    session.sync_prompt(prompt)?;
    session.clear_progress();
    let prefill_time = prefill_start.elapsed();

    let room = session.ctx() - session.pos();
    let max_tokens = if room <= 1 {
        0
    } else {
        options.max_tokens.min(room - 1)
    };

    let mut rng = options.seed.unwrap_or_else(default_seed);
    let eos = engine.token_eos();
    let use_mtp = options.temperature <= 0.0
        && engine.mtp_draft_tokens() > 1
        && std::env::var_os("DS4_MTP_SPEC_DISABLE").is_none();

    let decode_start = Instant::now();
    let mut generated = 0i32;
    let mut bytes = Vec::new();
    while generated < max_tokens {
        let token = session.sample_token(
            options.temperature,
            options.top_k,
            options.top_p,
            options.min_p,
            &mut rng,
        );
        if token == eos {
            break;
        }

        let mut accepted_buf = [0i32; 17];
        let accepted = session.eval_generated(token, use_mtp, max_tokens - generated, eos, &mut accepted_buf)?;

        let mut stop = false;
        for accepted_token in accepted_buf.iter().copied().take(accepted) {
            if accepted_token == eos {
                stop = true;
                break;
            }
            bytes.extend_from_slice(&engine.token_bytes(accepted_token));
            generated += 1;
            if generated >= max_tokens {
                break;
            }
        }
        if stop {
            break;
        }
    }

    Ok(GenerationResult {
        bytes,
        prompt_tokens: prompt.len(),
        completion_tokens: generated,
        prefill_time,
        decode_time: decode_start.elapsed(),
    })
}

pub fn generate_rust(
    engine: &Engine,
    session: &mut RustSession<'_>,
    prompt: &Tokens,
    options: GenerationOptions,
) -> Result<GenerationResult> {
    generate_with_session(engine, session, prompt, options)
}

fn default_seed() -> u64 {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_else(|_| Duration::from_secs(0))
        .as_nanos() as u64;
    now ^ (std::process::id() as u64).rotate_left(32)
}

pub fn mib_string(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

pub fn ensure_supported_role(role: &str) -> Result<()> {
    match role {
        "system" | "developer" | "user" | "assistant" => Ok(()),
        _ => Err(anyhow!("unsupported role: {role}")),
    }
}

fn render_one_shot_prompt(system: Option<&str>, prompt: &str, think_mode: ThinkMode) -> String {
    let mut rendered = render_chat_prefix(system, think_mode);
    rendered.push_str(RENDERED_CHAT_USER);
    rendered.push_str(prompt);
    rendered.push_str(RENDERED_CHAT_ASSISTANT);
    rendered.push_str(if think_mode.enabled() {
        THINK_START_MARKER
    } else {
        THINK_END_MARKER
    });
    rendered
}

fn render_chat_messages(system: Option<&str>, messages: &[ChatMessage], think_mode: ThinkMode) -> String {
    let mut rendered = render_chat_prefix(system, think_mode);
    for message in messages {
        render_chat_message(&mut rendered, &message.role, &message.content);
    }
    rendered
}

fn render_chat_generation_prompt(
    system: Option<&str>,
    messages: &[ChatMessage],
    think_mode: ThinkMode,
) -> String {
    let mut rendered = render_chat_messages(system, messages, think_mode);
    rendered.push_str(RENDERED_CHAT_ASSISTANT);
    rendered.push_str(if think_mode.enabled() {
        THINK_START_MARKER
    } else {
        THINK_END_MARKER
    });
    rendered
}

fn render_chat_prefix(system: Option<&str>, think_mode: ThinkMode) -> String {
    let mut rendered = String::from(RENDERED_CHAT_BOS);
    if think_mode == ThinkMode::Max {
        rendered.push_str(REASONING_EFFORT_MAX_PREFIX);
    }
    if let Some(system) = system {
        if !system.is_empty() {
            rendered.push_str(system);
        }
    }
    rendered
}

fn render_chat_message(rendered: &mut String, role: &str, content: &str) {
    let role = if role.is_empty() { "user" } else { role };
    match role {
        "system" | "developer" => rendered.push_str(content),
        "assistant" => {
            rendered.push_str(RENDERED_CHAT_ASSISTANT);
            if !content.starts_with(THINK_START_MARKER) && !content.starts_with(THINK_END_MARKER) {
                rendered.push_str(THINK_END_MARKER);
            }
            rendered.push_str(content);
        }
        "tool" | "function" => {
            rendered.push_str(RENDERED_CHAT_USER);
            rendered.push_str(TOOL_PREFIX);
            rendered.push_str(content);
        }
        _ => {
            rendered.push_str(RENDERED_CHAT_USER);
            rendered.push_str(content);
        }
    }
}

fn context_memory_estimate(backend: Backend, ctx_size: i32) -> ContextMemory {
    let ctx = if ctx_size > 0 { ctx_size as u32 } else { 1 };
    let mut memory = ContextMemory {
        total_bytes: 0,
        raw_bytes: 0,
        compressed_bytes: 0,
        scratch_bytes: 0,
        prefill_cap: 0,
        raw_cap: 0,
        comp_cap: 0,
    };

    if backend == Backend::Metal {
        memory.prefill_cap = metal_graph_prefill_cap_for_prompt(ctx as i32);
        memory.raw_cap = metal_graph_raw_cap_for_context(ctx as i32, memory.prefill_cap);

        let min_ratio = (0..DS4_N_LAYER)
            .map(layer_compress_ratio)
            .filter(|ratio| *ratio != 0)
            .min()
            .unwrap_or(ctx.max(1));
        memory.comp_cap = ctx / min_ratio + 2;
        if memory.comp_cap < 2 {
            memory.comp_cap = 2;
        }

        memory.raw_bytes = DS4_N_LAYER as u64 * memory.raw_cap as u64 * DS4_N_HEAD_DIM * std::mem::size_of::<f32>() as u64;
        for layer in 0..DS4_N_LAYER {
            let ratio = layer_compress_ratio(layer);
            if ratio == 0 {
                continue;
            }
            memory.compressed_bytes += memory.comp_cap as u64 * DS4_N_HEAD_DIM * std::mem::size_of::<f32>() as u64;
            if ratio == 4 {
                memory.compressed_bytes +=
                    memory.comp_cap as u64 * DS4_N_INDEXER_HEAD_DIM * std::mem::size_of::<f32>() as u64;
            }
        }
        memory.scratch_bytes =
            2 * memory.comp_cap as u64 * memory.prefill_cap as u64 * std::mem::size_of::<f32>() as u64;
    } else {
        memory.raw_cap = default_raw_cap(ctx);
        memory.raw_bytes = DS4_N_LAYER as u64 * memory.raw_cap as u64 * DS4_N_HEAD_DIM * std::mem::size_of::<f32>() as u64;
        for layer in 0..DS4_N_LAYER {
            let ratio = layer_compress_ratio(layer);
            if ratio == 0 {
                continue;
            }
            let comp_cap = ctx / ratio + 2;
            if ratio == 4 {
                memory.comp_cap = comp_cap;
            }
            memory.compressed_bytes += comp_cap as u64 * DS4_N_HEAD_DIM * std::mem::size_of::<f32>() as u64;
            if ratio == 4 {
                memory.compressed_bytes += comp_cap as u64 * DS4_N_INDEXER_HEAD_DIM * std::mem::size_of::<f32>() as u64;
            }
        }
        if memory.comp_cap == 0 {
            memory.comp_cap = ctx / 4 + 2;
        }
        memory.scratch_bytes =
            (memory.raw_cap as u64 + memory.comp_cap as u64) * std::mem::size_of::<f32>() as u64
            + memory.comp_cap as u64 * std::mem::size_of::<f32>() as u64
            + memory.comp_cap as u64 * std::mem::size_of::<bool>() as u64;
    }

    memory.total_bytes = memory.raw_bytes + memory.compressed_bytes + memory.scratch_bytes;
    memory
}

fn layer_compress_ratio(layer: u32) -> u32 {
    if layer < 2 {
        0
    } else if layer & 1 == 0 {
        4
    } else {
        128
    }
}

fn default_raw_cap(ctx_size: u32) -> u32 {
    let mut raw_cap = DS4_N_SWA.min(ctx_size);
    if raw_cap == 0 {
        raw_cap = 1;
    }
    raw_cap
}

/// Bytes per row for a quantized routed-expert tensor, matching
/// `routed_expert_row_bytes` in ds4.c.  `dim0` is the row width (e.g. 4096).
fn metal_expert_row_bytes(tensor_type: u32, dim0: u64) -> u64 {
    const QK_K: u64 = 256;
    let block_bytes: u64 = match tensor_type {
        DS4_TENSOR_IQ2_XXS => 66,
        DS4_TENSOR_Q2_K    => 84,
        DS4_TENSOR_Q4_K    => 144,
        _                  => 84, // safe fallback; real code panics in C
    };
    (dim0 / QK_K) * block_bytes
}

fn metal_graph_raw_cap_for_context(ctx_size: i32, prefill_cap: u32) -> u32 {
    let ctx_size = ctx_size.max(1) as u32;
    let raw_window = default_raw_cap(ctx_size);

    let mut wanted = raw_window as u64 + prefill_cap as u64;
    if wanted > ctx_size as u64 {
        wanted = ctx_size as u64;
    }
    if wanted == 0 {
        wanted = 1;
    }
    wanted = align_up_u64(wanted, METAL_RAW_CAP_ALIGN);
    if wanted > METAL_RAW_CAP_MAX as u64 {
        wanted = METAL_RAW_CAP_MAX as u64;
    }
    let mut raw_cap = wanted as u32;
    if raw_cap < raw_window {
        raw_cap = raw_window;
    }

    if let Some(override_cap) = parse_env_u32("DS4_METAL_GRAPH_RAW_CAP") {
        raw_cap = override_cap;
        if raw_cap > ctx_size {
            raw_cap = ctx_size;
        }
        if raw_cap > METAL_RAW_CAP_MAX {
            raw_cap = METAL_RAW_CAP_MAX;
        }
        if raw_cap < raw_window {
            raw_cap = raw_window;
        }
    }

    raw_cap
}

fn metal_graph_prefill_cap_for_prompt(prompt_len: i32) -> u32 {
    if prompt_len <= 0 {
        return 1;
    }

    let mut cap = prompt_len as u32;
    if let Some(chunk) = std::env::var_os("DS4_METAL_PREFILL_CHUNK") {
        let value = chunk.to_string_lossy();
        if let Ok(parsed) = value.parse::<i32>() {
            if parsed <= 0 {
                return cap;
            }
            cap = parsed as u32;
        }
    } else if prompt_len > METAL_PREFILL_LONG_PROMPT_CAP {
        cap = METAL_PREFILL_LONG_PROMPT_CAP as u32;
    }

    if cap == 0 {
        cap = 1;
    }
    if cap > prompt_len as u32 {
        cap = prompt_len as u32;
    }
    cap
}

fn parse_env_u32(name: &str) -> Option<u32> {
    let raw = std::env::var(name).ok()?;
    let parsed = raw.parse::<u32>().ok()?;
    if parsed == 0 {
        None
    } else {
        Some(parsed)
    }
}

fn align_up_u64(value: u64, align: u64) -> u64 {
    if align == 0 {
        value
    } else {
        value.div_ceil(align) * align
    }
}

fn hc_from_plain_embedding(embedding: &[f32], n_hc: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(embedding.len().saturating_mul(n_hc));
    for _ in 0..n_hc {
        out.extend_from_slice(embedding);
    }
    out
}

fn hc_pre_from_tensors(
    gguf_bytes: &[u8],
    fn_tensor: &BoundTensor,
    scale_tensor: &BoundTensor,
    base_tensor: &BoundTensor,
    residual_hc: &[f32],
) -> Result<HcPreOutput> {
    let flat = rms_norm_no_weight(residual_hc, DS4_RMS_EPS)?;
    let mix = matvec_f16_tensor(fn_tensor, gguf_bytes, &flat)?;
    let scale = scale_tensor.read_f32_values(gguf_bytes)?;
    let base = base_tensor.read_f32_values(gguf_bytes)?;
    let split = hc_split_sinkhorn(&mix, &scale, &base, DS4_N_HC, DS4_N_HC_SINKHORN_ITER, 1.0e-6)?;
    let sublayer_input = hc_weighted_sum(residual_hc, &split.pre)?;

    Ok(HcPreOutput {
        residual_hc: residual_hc.to_vec(),
        sublayer_input,
        post: split.post,
        comb: split.comb,
    })
}

fn rms_norm_no_weight(x: &[f32], eps: f32) -> Result<Vec<f32>> {
    if x.is_empty() {
        bail!("cannot RMS-normalize an empty vector")
    }

    let sum_squares = x
        .iter()
        .map(|value| {
            let value = f64::from(*value);
            value * value
        })
        .sum::<f64>();
    let scale = 1.0f32 / ((sum_squares / x.len() as f64) as f32 + eps).sqrt();
    Ok(x.iter().map(|value| *value * scale).collect())
}

fn rms_norm_no_weight_rows(rows: &[f32], row_width: usize, eps: f32) -> Result<Vec<f32>> {
    if row_width == 0 {
        bail!("cannot RMS-normalize rows with zero width")
    }
    if rows.len() % row_width != 0 {
        bail!(
            "rowwise RMSNorm input length {} is not divisible by row width {}",
            rows.len(),
            row_width
        )
    }

    let mut out = Vec::with_capacity(rows.len());
    for row in rows.chunks(row_width) {
        out.extend(rms_norm_no_weight(row, eps)?);
    }
    Ok(out)
}

fn rms_norm_no_weight_into(x: &[f32], eps: f32, out: &mut Vec<f32>) -> Result<()> {
    if x.is_empty() {
        bail!("cannot RMS-normalize an empty vector")
    }

    let sum_squares = x
        .iter()
        .map(|value| {
            let value = f64::from(*value);
            value * value
        })
        .sum::<f64>();
    let scale = 1.0f32 / ((sum_squares / x.len() as f64) as f32 + eps).sqrt();
    out.clear();
    out.resize(x.len(), 0.0);
    for (dst, value) in out.iter_mut().zip(x.iter().copied()) {
        *dst = value * scale;
    }
    Ok(())
}

fn rms_norm_weight(x: &[f32], weight: &[f32], eps: f32) -> Result<Vec<f32>> {
    if x.len() != weight.len() {
        bail!(
            "RMSNorm input length {} does not match weight length {}",
            x.len(),
            weight.len()
        )
    }

    let norm = rms_norm_no_weight(x, eps)?;
    Ok(norm
        .into_iter()
        .zip(weight.iter().copied())
        .map(|(value, weight)| value * weight)
        .collect())
}

fn rms_norm_weight_rows(rows: &[f32], weight: &[f32], eps: f32) -> Result<Vec<f32>> {
    if weight.is_empty() {
        bail!("cannot RMS-normalize HC rows with an empty weight vector")
    }
    if rows.len() % weight.len() != 0 {
        bail!(
            "HC row length {} is not divisible by weight length {}",
            rows.len(),
            weight.len()
        )
    }

    let mut out = Vec::with_capacity(rows.len());
    for row in rows.chunks(weight.len()) {
        out.extend(rms_norm_weight(row, weight, eps)?);
    }
    Ok(out)
}

fn matvec_f16_tensor(weight: &BoundTensor, gguf_bytes: &[u8], x: &[f32]) -> Result<Vec<f32>> {
    if weight.descriptor.ndim != 2 {
        bail!("{} is not a 2D tensor", weight.name)
    }

    let in_dim = usize::try_from(weight.descriptor.dims[0]).context("tensor input width does not fit in usize")?;
    let out_dim = usize::try_from(weight.descriptor.dims[1]).context("tensor output width does not fit in usize")?;
    if x.len() != in_dim {
        bail!(
            "{} expects {} inputs, got {}",
            weight.name,
            in_dim,
            x.len()
        )
    }

    let mut out = Vec::with_capacity(out_dim);
    for row in 0..out_dim {
        let row_values = weight.read_f16_row(gguf_bytes, row as u64)?;
        out.push(row_values.iter().zip(x.iter()).map(|(w, x)| *w * *x).sum());
    }
    Ok(out)
}

fn matvec_f16_rows_tensor(weight: &BoundTensor, gguf_bytes: &[u8], x_rows: &[f32]) -> Result<Vec<f32>> {
    if weight.descriptor.ndim != 2 {
        bail!("{} is not a 2D tensor", weight.name)
    }

    let in_dim = usize::try_from(weight.descriptor.dims[0]).context("tensor input width does not fit in usize")?;
    if x_rows.len() % in_dim != 0 {
        bail!(
            "rowwise F16 input length {} is not divisible by tensor input width {}",
            x_rows.len(),
            in_dim
        )
    }

    let out_dim = usize::try_from(weight.descriptor.dims[1]).context("tensor output width does not fit in usize")?;
    let mut out = Vec::with_capacity((x_rows.len() / in_dim) * out_dim);
    for row in x_rows.chunks(in_dim) {
        out.extend(matvec_f16_tensor(weight, gguf_bytes, row)?);
    }
    Ok(out)
}

fn matvec_f16_tensor_into(weight: &BoundTensor, gguf_bytes: &[u8], x: &[f32], out: &mut Vec<f32>) -> Result<()> {
    if weight.descriptor.tensor_type != 1 || weight.descriptor.ndim != 2 {
        bail!("{} is not a 2D F16 tensor", weight.name)
    }

    let in_dim = usize::try_from(weight.descriptor.dims[0]).context("tensor input width does not fit in usize")?;
    let out_dim = usize::try_from(weight.descriptor.dims[1]).context("tensor output width does not fit in usize")?;
    if x.len() != in_dim {
        bail!("{} expects {} inputs, got {}", weight.name, in_dim, x.len())
    }

    let data = weight.data(gguf_bytes)?;
    let row_bytes = in_dim
        .checked_mul(2)
        .context("tensor row byte count overflow")?;
    let needed = row_bytes
        .checked_mul(out_dim)
        .context("tensor byte size overflow")?;
    if data.len() < needed {
        bail!("{} does not contain enough F16 row data", weight.name)
    }

    out.clear();
    out.resize(out_dim, 0.0);
    for (row, dst) in out.iter_mut().enumerate() {
        let start = row * row_bytes;
        let row_data = &data[start..start + row_bytes];
        *dst = row_data
            .chunks_exact(2)
            .zip(x.iter().copied())
            .map(|(chunk, input)| f16_to_f32_bits(u16::from_le_bytes([chunk[0], chunk[1]])) * input)
            .sum();
    }
    Ok(())
}

fn quantize_q8_0_activation(x: &[f32]) -> (Vec<i8>, Vec<f32>) {
    let blocks = x.len().div_ceil(32);
    let mut xq = vec![0i8; blocks * 32];
    let mut scale = vec![0.0f32; blocks];

    for block in 0..blocks {
        let start = block * 32;
        let end = (start + 32).min(x.len());
        let slice = &x[start..end];
        let amax = slice
            .iter()
            .copied()
            .map(f32::abs)
            .fold(0.0f32, f32::max);
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        scale[block] = d;

        for (index, value) in slice.iter().copied().enumerate() {
            let quantized = (value * id).round().clamp(-128.0, 127.0) as i32;
            xq[start + index] = quantized as i8;
        }
    }

    (xq, scale)
}

fn quantize_q8_0_activation_into(x: &[f32], xq: &mut Vec<i8>, scale: &mut Vec<f32>) {
    let blocks = x.len().div_ceil(32);
    xq.clear();
    xq.resize(blocks * 32, 0);
    scale.clear();
    scale.resize(blocks, 0.0);

    for block in 0..blocks {
        let start = block * 32;
        let end = (start + 32).min(x.len());
        let slice = &x[start..end];
        let amax = slice
            .iter()
            .copied()
            .map(f32::abs)
            .fold(0.0f32, f32::max);
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        scale[block] = d;

        for (index, value) in slice.iter().copied().enumerate() {
            let quantized = (value * id).round().clamp(-128.0, 127.0) as i32;
            xq[start + index] = quantized as i8;
        }
    }
}

fn matvec_q8_0_tensor(weight: &BoundTensor, gguf_bytes: &[u8], x: &[f32]) -> Result<Vec<f32>> {
    if weight.descriptor.tensor_type != 8 || weight.descriptor.ndim != 2 {
        bail!("{} is not a 2D Q8_0 tensor", weight.name)
    }

    let in_dim = usize::try_from(weight.descriptor.dims[0]).context("Q8_0 tensor input width does not fit in usize")?;
    let out_dim = usize::try_from(weight.descriptor.dims[1]).context("Q8_0 tensor output width does not fit in usize")?;
    if x.len() != in_dim {
        bail!(
            "{} expects {} inputs, got {}",
            weight.name,
            in_dim,
            x.len()
        )
    }

    let blocks = in_dim.div_ceil(32);
    let row_bytes = blocks * 34;
    let data = weight.data(gguf_bytes)?;
    let needed = row_bytes
        .checked_mul(out_dim)
        .context("Q8_0 tensor byte size overflow")?;
    if data.len() < needed {
        bail!("{} does not contain enough Q8_0 row data", weight.name)
    }

    let (xq, xscale) = quantize_q8_0_activation(x);
    let mut out = Vec::with_capacity(out_dim);
    for row in 0..out_dim {
        let start = row * row_bytes;
        out.push(dot_q8_0_row(&data[start..start + row_bytes], &xq, &xscale, in_dim)?);
    }
    Ok(out)
}

fn matvec_q8_0_tensor_into(
    weight: &BoundTensor,
    gguf_bytes: &[u8],
    x: &[f32],
    xq: &mut Vec<i8>,
    xscale: &mut Vec<f32>,
    out: &mut Vec<f32>,
) -> Result<()> {
    if weight.descriptor.tensor_type != 8 || weight.descriptor.ndim != 2 {
        bail!("{} is not a 2D Q8_0 tensor", weight.name)
    }

    let in_dim = usize::try_from(weight.descriptor.dims[0]).context("Q8_0 tensor input width does not fit in usize")?;
    let out_dim = usize::try_from(weight.descriptor.dims[1]).context("Q8_0 tensor output width does not fit in usize")?;
    if x.len() != in_dim {
        bail!("{} expects {} inputs, got {}", weight.name, in_dim, x.len())
    }

    let blocks = in_dim.div_ceil(32);
    let row_bytes = blocks * 34;
    let data = weight.data(gguf_bytes)?;
    let needed = row_bytes
        .checked_mul(out_dim)
        .context("Q8_0 tensor byte size overflow")?;
    if data.len() < needed {
        bail!("{} does not contain enough Q8_0 row data", weight.name)
    }

    quantize_q8_0_activation_into(x, xq, xscale);
    out.clear();
    out.resize(out_dim, 0.0);
    for (row, dst) in out.iter_mut().enumerate() {
        let start = row * row_bytes;
        *dst = dot_q8_0_row(&data[start..start + row_bytes], xq, xscale, in_dim)?;
    }
    Ok(())
}

fn matvec_q8_0_rows_tensor(weight: &BoundTensor, gguf_bytes: &[u8], x_rows: &[f32]) -> Result<Vec<f32>> {
    if weight.descriptor.tensor_type != 8 || weight.descriptor.ndim != 2 {
        bail!("{} is not a 2D Q8_0 tensor", weight.name)
    }

    let in_dim = usize::try_from(weight.descriptor.dims[0]).context("Q8_0 tensor input width does not fit in usize")?;
    if x_rows.len() % in_dim != 0 {
        bail!(
            "rowwise Q8_0 input length {} is not divisible by tensor input width {}",
            x_rows.len(),
            in_dim
        )
    }

    let out_dim = usize::try_from(weight.descriptor.dims[1]).context("Q8_0 tensor output width does not fit in usize")?;
    let mut out = Vec::with_capacity((x_rows.len() / in_dim) * out_dim);
    for row in x_rows.chunks(in_dim) {
        out.extend(matvec_q8_0_tensor(weight, gguf_bytes, row)?);
    }
    Ok(out)
}

fn matvec_q8_0_grouped_tensor(
    weight: &BoundTensor,
    gguf_bytes: &[u8],
    x: &[f32],
    n_groups: usize,
    group_dim: usize,
    rank: usize,
) -> Result<Vec<f32>> {
    if weight.descriptor.tensor_type != 8 || weight.descriptor.ndim != 2 {
        bail!("{} is not a 2D Q8_0 tensor", weight.name)
    }
    if usize::try_from(weight.descriptor.dims[0]).ok() != Some(group_dim)
        || usize::try_from(weight.descriptor.dims[1]).ok() != Some(n_groups * rank)
    {
        bail!("{} does not match the grouped Q8_0 layout", weight.name)
    }
    if x.len() != n_groups * group_dim {
        bail!(
            "grouped Q8_0 input length {} does not match {} groups of width {}",
            x.len(),
            n_groups,
            group_dim
        )
    }

    let blocks = group_dim.div_ceil(32);
    let row_bytes = blocks * 34;
    let data = weight.data(gguf_bytes)?;
    let needed = row_bytes
        .checked_mul(n_groups * rank)
        .context("grouped Q8_0 tensor byte size overflow")?;
    if data.len() < needed {
        bail!("{} does not contain enough grouped Q8_0 row data", weight.name)
    }

    let mut out = vec![0.0f32; n_groups * rank];
    for group in 0..n_groups {
        let start = group * group_dim;
        let end = start + group_dim;
        let (xq, xscale) = quantize_q8_0_activation(&x[start..end]);
        for row_in_group in 0..rank {
            let index = group * rank + row_in_group;
            let row_start = index * row_bytes;
            out[index] = dot_q8_0_row(&data[row_start..row_start + row_bytes], &xq, &xscale, group_dim)?;
        }
    }

    Ok(out)
}

fn dot_q8_0_row(row: &[u8], xq: &[i8], xscale: &[f32], in_dim: usize) -> Result<f32> {
    let blocks = in_dim.div_ceil(32);
    let needed = blocks.checked_mul(34).context("Q8_0 row byte count overflow")?;
    if row.len() < needed {
        bail!("Q8_0 row is too short for {} blocks", blocks)
    }
    if xq.len() < blocks * 32 || xscale.len() < blocks {
        bail!("Q8_0 activation buffers are too small")
    }

    let mut acc = 0.0f32;
    for block in 0..blocks {
        let row_offset = block * 34;
        let scale_bits = u16::from_le_bytes([row[row_offset], row[row_offset + 1]]);
        let qs = &row[row_offset + 2..row_offset + 34];
        let i0 = block * 32;
        let n = (in_dim - i0).min(32);
        let dot = qs[..n]
            .iter()
            .zip(xq[i0..i0 + n].iter())
            .map(|(lhs, rhs)| i32::from(*lhs as i8) * i32::from(*rhs))
            .sum::<i32>();
        acc += f16_to_f32_bits(scale_bits) * xscale[block] * dot as f32;
    }
    Ok(acc)
}

fn f16_to_f32_bits(bits: u16) -> f32 {
    let sign = (u32::from(bits & 0x8000)) << 16;
    let exp = ((bits >> 10) & 0x1f) as i32;
    let mut mant = u32::from(bits & 0x03ff);

    let value = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            let mut exp = 1i32;
            while (mant & 0x0400) == 0 {
                mant <<= 1;
                exp -= 1;
            }
            mant &= 0x03ff;
            sign | (((exp + 127 - 15) as u32) << 23) | (mant << 13)
        }
    } else if exp == 31 {
        sign | 0x7f80_0000 | (mant << 13)
    } else {
        sign | (((exp + 127 - 15) as u32) << 23) | (mant << 13)
    };

    f32::from_bits(value)
}

#[derive(Debug)]
struct HcSplitOutput {
    pre: Vec<f32>,
    post: Vec<f32>,
    comb: Vec<f32>,
}

fn hc_split_sinkhorn(
    mix: &[f32],
    scale: &[f32],
    base: &[f32],
    n_hc: usize,
    iters: usize,
    eps: f32,
) -> Result<HcSplitOutput> {
    if scale.len() != 3 {
        bail!("HC split scale must contain 3 values, got {}", scale.len())
    }

    let needed = 2 * n_hc + n_hc * n_hc;
    if mix.len() != needed {
        bail!("HC split mix length {} does not match expected {}", mix.len(), needed)
    }
    if base.len() != needed {
        bail!("HC split base length {} does not match expected {}", base.len(), needed)
    }
    if n_hc == 0 {
        bail!("HC split requires at least one HC stream")
    }

    let pre_scale = scale[0];
    let post_scale = scale[1];
    let comb_scale = scale[2];

    let mut pre = vec![0.0f32; n_hc];
    for index in 0..n_hc {
        let z = mix[index] * pre_scale + base[index];
        pre[index] = sigmoid_stable(z) + eps;
    }

    let mut post = vec![0.0f32; n_hc];
    for index in 0..n_hc {
        let off = n_hc + index;
        let z = mix[off] * post_scale + base[off];
        post[index] = 2.0 / (1.0 + (-z).exp());
    }

    let mut comb = vec![0.0f32; n_hc * n_hc];
    for dst in 0..n_hc {
        let mut row_max = f32::NEG_INFINITY;
        for src in 0..n_hc {
            let index = src + dst * n_hc;
            let off = 2 * n_hc + index;
            let value = mix[off] * comb_scale + base[off];
            comb[index] = value;
            row_max = row_max.max(value);
        }

        let mut row_sum = 0.0f32;
        for src in 0..n_hc {
            let index = src + dst * n_hc;
            let value = (comb[index] - row_max).exp();
            comb[index] = value;
            row_sum += value;
        }

        let inv = 1.0 / row_sum;
        for src in 0..n_hc {
            let index = src + dst * n_hc;
            comb[index] = comb[index] * inv + eps;
        }
    }

    normalize_hc_columns(&mut comb, n_hc, eps);

    for _ in 1..iters {
        normalize_hc_rows(&mut comb, n_hc, eps);
        normalize_hc_columns(&mut comb, n_hc, eps);
    }

    Ok(HcSplitOutput { pre, post, comb })
}

fn normalize_hc_rows(comb: &mut [f32], n_hc: usize, eps: f32) {
    for dst in 0..n_hc {
        let mut sum = 0.0f32;
        for src in 0..n_hc {
            sum += comb[src + dst * n_hc];
        }

        let inv = 1.0 / (sum + eps);
        for src in 0..n_hc {
            comb[src + dst * n_hc] *= inv;
        }
    }
}

fn normalize_hc_columns(comb: &mut [f32], n_hc: usize, eps: f32) {
    for src in 0..n_hc {
        let mut sum = 0.0f32;
        for dst in 0..n_hc {
            sum += comb[src + dst * n_hc];
        }

        let inv = 1.0 / (sum + eps);
        for dst in 0..n_hc {
            comb[src + dst * n_hc] *= inv;
        }
    }
}

fn sigmoid_stable(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    }
}

fn hc_weighted_sum(hc_state: &[f32], weights: &[f32]) -> Result<Vec<f32>> {
    if weights.is_empty() {
        bail!("cannot HC-reduce with no weights")
    }
    if hc_state.len() % weights.len() != 0 {
        bail!(
            "HC state length {} is not divisible by weight count {}",
            hc_state.len(),
            weights.len()
        )
    }

    let embd = hc_state.len() / weights.len();
    let mut out = vec![0.0f32; embd];
    for (index, weight) in weights.iter().copied().enumerate() {
        let start = index * embd;
        let end = start + embd;
        for (out_value, input) in out.iter_mut().zip(hc_state[start..end].iter().copied()) {
            *out_value += input * weight;
        }
    }
    Ok(out)
}

fn hc_weighted_sum_rows(hc_rows: &[f32], weights_rows: &[f32], n_hc: usize) -> Result<Vec<f32>> {
    if n_hc == 0 {
        bail!("cannot HC-reduce rows with zero HC streams")
    }
    if weights_rows.is_empty() {
        bail!("cannot HC-reduce rows with no weights")
    }
    if weights_rows.len() % n_hc != 0 {
        bail!(
            "rowwise HC weight length {} is not divisible by HC stream count {}",
            weights_rows.len(),
            n_hc
        )
    }

    let n_rows = weights_rows.len() / n_hc;
    if hc_rows.len() % n_rows != 0 {
        bail!(
            "rowwise HC state length {} is not divisible by row count {}",
            hc_rows.len(),
            n_rows
        )
    }
    let hc_width = hc_rows.len() / n_rows;
    if hc_width % n_hc != 0 {
        bail!(
            "rowwise HC width {} is not divisible by HC stream count {}",
            hc_width,
            n_hc
        )
    }

    let embd = hc_width / n_hc;
    let mut out = vec![0.0f32; n_rows * embd];
    for row in 0..n_rows {
        let hc_row = &hc_rows[row * hc_width..(row + 1) * hc_width];
        let weights = &weights_rows[row * n_hc..(row + 1) * n_hc];
        let out_row = &mut out[row * embd..(row + 1) * embd];
        for (index, weight) in weights.iter().copied().enumerate() {
            let start = index * embd;
            let end = start + embd;
            for (out_value, input) in out_row.iter_mut().zip(hc_row[start..end].iter().copied()) {
                *out_value += input * weight;
            }
        }
    }
    Ok(out)
}

fn hc_weighted_sum_into(hc_state: &[f32], weights: &[f32], out: &mut Vec<f32>) -> Result<()> {
    if weights.is_empty() {
        bail!("cannot HC-reduce with no weights")
    }
    if hc_state.len() % weights.len() != 0 {
        bail!(
            "HC state length {} is not divisible by weight count {}",
            hc_state.len(),
            weights.len()
        )
    }

    let embd = hc_state.len() / weights.len();
    out.clear();
    out.resize(embd, 0.0);
    for (index, weight) in weights.iter().copied().enumerate() {
        let start = index * embd;
        let end = start + embd;
        for (out_value, input) in out.iter_mut().zip(hc_state[start..end].iter().copied()) {
            *out_value += input * weight;
        }
    }
    Ok(())
}

fn read_single_f32_tensor(tensor: &BoundTensor, gguf_bytes: &[u8]) -> Result<f32> {
    let data = tensor.data(gguf_bytes)?;
    if data.len() != 4 {
        bail!("{} must contain exactly one F32 value", tensor.name)
    }
    Ok(f32::from_le_bytes([data[0], data[1], data[2], data[3]]))
}

fn fill_sigmoid_weights_from_base_tensor(
    pre: &[f32],
    scale: f32,
    base_tensor: &BoundTensor,
    gguf_bytes: &[u8],
    out: &mut Vec<f32>,
) -> Result<()> {
    let data = base_tensor.data(gguf_bytes)?;
    let expected = pre
        .len()
        .checked_mul(4)
        .context("output HC base byte count overflow")?;
    if data.len() != expected {
        bail!(
            "{} byte length {} does not match output HC width {}",
            base_tensor.name,
            data.len(),
            pre.len()
        )
    }

    out.clear();
    out.resize(pre.len(), 0.0);
    for ((dst, pre_value), chunk) in out.iter_mut().zip(pre.iter().copied()).zip(data.chunks_exact(4)) {
        let base = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        *dst = sigmoid_stable(pre_value * scale + base) + DS4_HC_EPS;
    }
    Ok(())
}

fn mul_f32_tensor_inplace(values: &mut [f32], tensor: &BoundTensor, gguf_bytes: &[u8]) -> Result<()> {
    let data = tensor.data(gguf_bytes)?;
    let expected = values
        .len()
        .checked_mul(4)
        .context("F32 tensor byte count overflow")?;
    if data.len() != expected {
        bail!(
            "{} byte length {} does not match expected F32 width {}",
            tensor.name,
            data.len(),
            values.len()
        )
    }

    for (value, chunk) in values.iter_mut().zip(data.chunks_exact(4)) {
        *value *= f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
    Ok(())
}

fn hc_post(block_out: &[f32], residual_hc: &[f32], post: &[f32], comb: &[f32]) -> Result<Vec<f32>> {
    if post.is_empty() {
        bail!("cannot HC-post with no HC streams")
    }
    if comb.len() != post.len() * post.len() {
        bail!(
            "HC combine matrix length {} does not match HC count {}",
            comb.len(),
            post.len()
        )
    }
    if residual_hc.len() % post.len() != 0 {
        bail!(
            "residual HC length {} is not divisible by HC count {}",
            residual_hc.len(),
            post.len()
        )
    }

    let n_hc = post.len();
    let n_embd = residual_hc.len() / n_hc;
    if block_out.len() != n_embd {
        bail!(
            "block output length {} does not match embedding width {}",
            block_out.len(),
            n_embd
        )
    }

    let mut out_hc = vec![0.0f32; residual_hc.len()];
    for dst in 0..n_hc {
        for dim in 0..n_embd {
            let mut acc = block_out[dim] * post[dst];
            for src in 0..n_hc {
                acc += comb[dst + src * n_hc] * residual_hc[src * n_embd + dim];
            }
            out_hc[dst * n_embd + dim] = acc;
        }
    }
    Ok(out_hc)
}

fn attention_rows_mixed_with_sinks(
    q: &[f32],
    raw_kv_rows: &[f32],
    comp_kv_rows: &[f32],
    sinks: &[f32],
    comp_allowed: Option<&[bool]>,
) -> Result<Vec<f32>> {
    if sinks.is_empty() {
        bail!("attention sinks cannot be empty")
    }
    if q.len() % sinks.len() != 0 {
        bail!("query width {} is not divisible by head count {}", q.len(), sinks.len())
    }

    let n_head = sinks.len();
    let head_dim = q.len() / n_head;
    if raw_kv_rows.len() % head_dim != 0 {
        bail!("raw KV row buffer length {} is not divisible by head width {}", raw_kv_rows.len(), head_dim)
    }
    if comp_kv_rows.len() % head_dim != 0 {
        bail!("compressed KV row buffer length {} is not divisible by head width {}", comp_kv_rows.len(), head_dim)
    }

    let n_raw = raw_kv_rows.len() / head_dim;
    let n_comp = comp_kv_rows.len() / head_dim;
    if let Some(mask) = comp_allowed {
        if mask.len() != n_comp {
            bail!("compressed attention mask length {} does not match compressed row count {}", mask.len(), n_comp)
        }
    }

    let kq_scale = 1.0 / (head_dim as f32).sqrt();
    let mut out_heads = vec![0.0f32; q.len()];
    let mut scores = vec![0.0f32; n_raw + n_comp];

    for head in 0..n_head {
        let qh = &q[head * head_dim..(head + 1) * head_dim];
        let mut max_score = sinks[head];
        let mut index = 0;

        for row in 0..n_raw {
            let kv = &raw_kv_rows[row * head_dim..(row + 1) * head_dim];
            scores[index] = dot_f32(qh, kv)? * kq_scale;
            max_score = max_score.max(scores[index]);
            index += 1;
        }
        for row in 0..n_comp {
            if comp_allowed.is_some_and(|mask| !mask[row]) {
                scores[index] = DS4_NEG_INF;
            } else {
                let kv = &comp_kv_rows[row * head_dim..(row + 1) * head_dim];
                scores[index] = dot_f32(qh, kv)? * kq_scale;
                max_score = max_score.max(scores[index]);
            }
            index += 1;
        }

        let oh = &mut out_heads[head * head_dim..(head + 1) * head_dim];
        let mut denom = (sinks[head] - max_score).exp();
        index = 0;
        for row in 0..n_raw {
            let weight = (scores[index] - max_score).exp();
            let kv = &raw_kv_rows[row * head_dim..(row + 1) * head_dim];
            denom += weight;
            axpy_f32(oh, kv, weight)?;
            index += 1;
        }
        for row in 0..n_comp {
            if scores[index] <= DS4_NEG_INF * 0.5 {
                index += 1;
                continue;
            }
            let weight = (scores[index] - max_score).exp();
            let kv = &comp_kv_rows[row * head_dim..(row + 1) * head_dim];
            denom += weight;
            axpy_f32(oh, kv, weight)?;
            index += 1;
        }

        scale_f32(oh, 1.0 / denom);
    }

    Ok(out_heads)
}

fn compressor_pool_decode_state(
    state_kv: &[f32],
    state_score: &[f32],
    head_dim: usize,
    compress_ratio: usize,
) -> Result<Vec<f32>> {
    if head_dim == 0 || compress_ratio == 0 {
        bail!("compressor pooling requires non-zero dimensions")
    }

    let coff = if compress_ratio == 4 { 2 } else { 1 };
    let width = coff * head_dim;
    let rows = coff * compress_ratio;
    let expected_len = width * rows;
    if state_kv.len() != expected_len || state_score.len() != expected_len {
        bail!(
            "compressor state length mismatch: expected {}, got {} and {}",
            expected_len,
            state_kv.len(),
            state_score.len()
        )
    }

    let mut out = vec![0.0f32; head_dim];
    for j in 0..head_dim {
        let mut max_score = DS4_NEG_INF;
        if compress_ratio == 4 {
            for row in 0..compress_ratio {
                let sp = state_score[row * width + j];
                let sc = state_score[(compress_ratio + row) * width + head_dim + j];
                max_score = max_score.max(sp).max(sc);
            }
        } else {
            for row in 0..compress_ratio {
                max_score = max_score.max(state_score[row * width + j]);
            }
        }

        if max_score <= DS4_NEG_INF * 0.5 {
            continue;
        }

        let mut denom = 0.0f32;
        let mut sum = 0.0f32;
        if compress_ratio == 4 {
            for row in 0..compress_ratio {
                let wp = (state_score[row * width + j] - max_score).exp();
                let wc = (state_score[(compress_ratio + row) * width + head_dim + j] - max_score).exp();
                denom += wp + wc;
                sum += wp * state_kv[row * width + j];
                sum += wc * state_kv[(compress_ratio + row) * width + head_dim + j];
            }
        } else {
            for row in 0..compress_ratio {
                let weight = (state_score[row * width + j] - max_score).exp();
                denom += weight;
                sum += weight * state_kv[row * width + j];
            }
        }

        out[j] = if denom > 0.0 { sum / denom } else { 0.0 };
    }

    Ok(out)
}

fn compressor_decode_one_tensor(
    gguf_bytes: &[u8],
    layer: u32,
    x: &[f32],
    wkv: &BoundTensor,
    wgate: &BoundTensor,
    ape: &BoundTensor,
    norm: &BoundTensor,
    state_kv: &mut [f32],
    state_score: &mut [f32],
    head_dim: usize,
    compress_ratio: usize,
    pos: u32,
) -> Result<Option<Vec<f32>>> {
    let coff = if compress_ratio == 4 { 2 } else { 1 };
    let width = coff * head_dim;
    let rows = coff * compress_ratio;
    let expected_len = width * rows;
    if state_kv.len() != expected_len || state_score.len() != expected_len {
        bail!(
            "compressor state length mismatch: expected {}, got {} and {}",
            expected_len,
            state_kv.len(),
            state_score.len()
        )
    }

    let pos_mod = usize::try_from(pos).context("position does not fit in usize")? % compress_ratio;
    let row = if compress_ratio == 4 { compress_ratio + pos_mod } else { pos_mod };
    let should_compress = (usize::try_from(pos).context("position does not fit in usize")? + 1) % compress_ratio == 0;

    let kv_cur = matvec_any_tensor(wkv, gguf_bytes, x)?;
    let mut sc_cur = matvec_any_tensor(wgate, gguf_bytes, x)?;
    if kv_cur.len() != width || sc_cur.len() != width {
        bail!("compressor projections do not match expected width {}", width)
    }
    let ape_row = ape.read_f16_row(gguf_bytes, pos_mod as u64)?;
    if ape_row.len() != width {
        bail!("compressor APE row width {} does not match expected {}", ape_row.len(), width)
    }
    for (score, ape_bias) in sc_cur.iter_mut().zip(ape_row.iter().copied()) {
        *score += ape_bias;
    }

    let row_start = row * width;
    state_kv[row_start..row_start + width].copy_from_slice(&kv_cur);
    state_score[row_start..row_start + width].copy_from_slice(&sc_cur);

    if !should_compress {
        return Ok(None);
    }

    let pooled = compressor_pool_decode_state(state_kv, state_score, head_dim, compress_ratio)?;
    let norm_weight = norm.read_f32_values(gguf_bytes)?;
    let mut out = rms_norm_weight(&pooled, &norm_weight, DS4_RMS_EPS)?;
    let comp_pos = pos + 1 - compress_ratio as u32;
    rope_tail_layer_inplace(&mut out, 1, head_dim, DS4_N_ROT, comp_pos, layer, false)?;
    if head_dim == usize::try_from(DS4_N_HEAD_DIM).unwrap_or_default() {
        dsv4_fp8_kv_quantize_row_inplace(&mut out, head_dim, DS4_N_ROT)?;
    }

    if compress_ratio == 4 {
        for row in 0..compress_ratio {
            let src = (compress_ratio + row) * width;
            let dst = row * width;
            let tmp_kv = state_kv[src..src + width].to_vec();
            let tmp_score = state_score[src..src + width].to_vec();
            state_kv[dst..dst + width].copy_from_slice(&tmp_kv);
            state_score[dst..dst + width].copy_from_slice(&tmp_score);
        }
        for row in 0..compress_ratio {
            let src = row * width;
            let dst = (compress_ratio + row) * width;
            let tmp_kv = state_kv[src..src + width].to_vec();
            let tmp_score = state_score[src..src + width].to_vec();
            state_kv[dst..dst + width].copy_from_slice(&tmp_kv);
            state_score[dst..dst + width].copy_from_slice(&tmp_score);
        }
    }

    Ok(Some(out))
}

fn select_top_k_mask(scores: &[f32], top_k: usize) -> Vec<bool> {
    let mut allowed = vec![false; scores.len()];
    if top_k >= scores.len() {
        allowed.fill(true);
        return allowed;
    }

    for _ in 0..top_k {
        let mut best_index = None;
        let mut best_score = DS4_NEG_INF;
        for (index, score) in scores.iter().copied().enumerate() {
            if !allowed[index] && score > best_score {
                best_score = score;
                best_index = Some(index);
            }
        }
        if let Some(index) = best_index {
            allowed[index] = true;
        }
    }

    allowed
}

fn select_top_k_desc_indices(scores: &[f32], top_k: usize) -> Vec<usize> {
    let mut selected = Vec::new();
    for (index, score) in scores.iter().copied().enumerate() {
        let limit = top_k.min(selected.len() + 1);
        let mut inserted = false;
        for slot in 0..limit {
            if slot == selected.len() || score > scores[selected[slot]] {
                selected.insert(slot, index);
                inserted = true;
                break;
            }
        }
        if !inserted && selected.len() < top_k {
            selected.push(index);
        }
        if selected.len() > top_k {
            selected.truncate(top_k);
        }
    }
    selected.truncate(top_k.min(scores.len()));
    selected
}

fn hash_router_weights_from_probs(probs: &[f32], selected: &[i32]) -> Result<Vec<f32>> {
    let mut weights = Vec::with_capacity(selected.len());
    let mut sum = 0.0f32;
    for &expert in selected {
        let index = usize::try_from(expert).context("selected expert index is negative")?;
        let weight = *probs
            .get(index)
            .with_context(|| format!("selected expert {} is outside router range {}", expert, probs.len()))?;
        weights.push(weight);
        sum += weight;
    }

    if sum < 6.103_515_6e-5 {
        sum = 6.103_515_6e-5;
    }
    for weight in &mut weights {
        *weight = *weight / sum * DS4_EXPERT_WEIGHT_SCALE;
    }
    Ok(weights)
}

fn matvec_any_tensor(weight: &BoundTensor, gguf_bytes: &[u8], x: &[f32]) -> Result<Vec<f32>> {
    match weight.descriptor.tensor_type {
        1 => matvec_f16_tensor(weight, gguf_bytes, x),
        8 => matvec_q8_0_tensor(weight, gguf_bytes, x),
        other => bail!("{} has unsupported matvec tensor type {}", weight.name, other),
    }
}

fn fill_sigmoid_weights_rows_from_base_tensor(
    pre_rows: &[f32],
    scale: f32,
    base_tensor: &BoundTensor,
    gguf_bytes: &[u8],
    out: &mut Vec<f32>,
) -> Result<()> {
    let base = base_tensor.read_f32_values(gguf_bytes)?;
    if base.is_empty() {
        bail!("{} must contain at least one base value", base_tensor.name)
    }
    if pre_rows.len() % base.len() != 0 {
        bail!(
            "rowwise output HC width {} is not divisible by base width {}",
            pre_rows.len(),
            base.len()
        )
    }

    out.clear();
    out.reserve(pre_rows.len());
    for row in pre_rows.chunks(base.len()) {
        for (pre_value, base_value) in row.iter().copied().zip(base.iter().copied()) {
            out.push(sigmoid_stable(pre_value * scale + base_value) + DS4_HC_EPS);
        }
    }
    Ok(())
}

fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x007f_ffff;

    if exp == 0xff {
        return sign | if mant == 0 { 0x7c00 } else { 0x7e00 };
    }

    let exp16 = exp - 127 + 15;
    if exp16 >= 0x1f {
        return sign | 0x7c00;
    }
    if exp16 <= 0 {
        if exp16 < -10 {
            return sign;
        }

        let mant32 = mant | 0x0080_0000;
        let shift = (14 - exp16) as u32;
        let mut mant16 = (mant32 >> shift) as u16;
        let round_bit = 1u32 << (shift - 1);
        let remainder = mant32 & ((round_bit << 1) - 1);
        if remainder > round_bit || (remainder == round_bit && (mant16 & 1) != 0) {
            mant16 = mant16.wrapping_add(1);
        }
        return sign | mant16;
    }

    let mut exp_bits = (exp16 as u16) << 10;
    let mut mant16 = (mant >> 13) as u16;
    let remainder = mant & 0x1fff;
    if remainder > 0x1000 || (remainder == 0x1000 && (mant16 & 1) != 0) {
        mant16 = mant16.wrapping_add(1);
        if mant16 == 0x0400 {
            mant16 = 0;
            exp_bits = exp_bits.wrapping_add(0x0400);
            if exp_bits >= 0x7c00 {
                return sign | 0x7c00;
            }
        }
    }

    sign | exp_bits | mant16
}

fn f16_round_slice(values: &[f32]) -> Vec<f32> {
    let mut out = Vec::new();
    f16_round_into(values, &mut out);
    out
}

fn f16_round_into(values: &[f32], out: &mut Vec<f32>) {
    out.clear();
    out.resize(values.len(), 0.0);
    for (dst, value) in out.iter_mut().zip(values.iter().copied()) {
        *dst = f16_to_f32_bits(f32_to_f16_bits(value));
    }
}

fn push_raw_cache_row(rows: &mut Vec<f32>, cap_rows: usize, row: &[f32]) {
    if cap_rows == 0 || row.is_empty() {
        return;
    }
    let row_dim = row.len();
    let n_rows = rows.len() / row_dim;
    if n_rows < cap_rows {
        rows.extend_from_slice(row);
        return;
    }

    rows.copy_within(row_dim.., 0);
    rows.truncate(rows.len() - row_dim);
    rows.extend_from_slice(row);
}

fn push_comp_cache_row(rows: &mut Vec<f32>, cap_rows: usize, row: &[f32]) -> Result<()> {
    if row.is_empty() {
        return Ok(());
    }
    let row_dim = row.len();
    let n_rows = rows.len() / row_dim;
    if n_rows >= cap_rows {
        bail!("compressed KV cache capacity {} exceeded", cap_rows)
    }
    rows.extend_from_slice(row);
    Ok(())
}

fn read_i32_tensor_row(weight: &BoundTensor, gguf_bytes: &[u8], row_index: i32) -> Result<Vec<i32>> {
    if weight.descriptor.tensor_type != 26 || weight.descriptor.ndim != 2 {
        bail!("{} is not a 2D I32 tensor", weight.name)
    }
    let width = usize::try_from(weight.descriptor.dims[0]).context("tensor width does not fit in usize")?;
    let rows = usize::try_from(weight.descriptor.dims[1]).context("tensor row count does not fit in usize")?;
    let index = usize::try_from(row_index).context("token index is negative")?;
    if index >= rows {
        bail!("tensor {} row {} is outside {} rows", weight.name, row_index, rows)
    }

    let data = weight.data(gguf_bytes)?;
    let row_bytes = width.checked_mul(4).context("I32 tensor row byte count overflow")?;
    let start = index.checked_mul(row_bytes).context("I32 tensor row offset overflow")?;
    let end = start.checked_add(row_bytes).context("I32 tensor row range overflow")?;
    let row = data
        .get(start..end)
        .with_context(|| format!("tensor {} row {} points outside mapped data", weight.name, row_index))?;

    Ok(row
        .chunks_exact(4)
        .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

fn expert_tensor_bytes<'a>(
    weight: &BoundTensor,
    gguf_bytes: &'a [u8],
    expert: u32,
) -> Result<(&'a [u8], usize, usize, usize)> {
    if weight.descriptor.ndim != 3 {
        bail!("{} is not a 3D expert tensor", weight.name)
    }
    let in_dim = usize::try_from(weight.descriptor.dims[0]).context("expert input width does not fit in usize")?;
    let out_dim = usize::try_from(weight.descriptor.dims[1]).context("expert output width does not fit in usize")?;
    let n_expert = usize::try_from(weight.descriptor.dims[2]).context("expert count does not fit in usize")?;
    let expert_index = usize::try_from(expert).context("expert index does not fit in usize")?;
    if expert_index >= n_expert {
        bail!("expert {} is outside tensor {} with {} experts", expert, weight.name, n_expert)
    }

    let (block_elems, block_bytes) = match weight.descriptor.tensor_type {
        DS4_TENSOR_Q2_K => (QK_K, 84usize),
        DS4_TENSOR_IQ2_XXS => (QK_K, 66usize),
        other => bail!("{} has unsupported expert tensor type {}", weight.name, other),
    };
    let blocks = in_dim.div_ceil(block_elems);
    let row_bytes = blocks.checked_mul(block_bytes).context("expert row byte count overflow")?;
    let expert_bytes = out_dim.checked_mul(row_bytes).context("expert tensor byte count overflow")?;
    let start = expert_index.checked_mul(expert_bytes).context("expert tensor offset overflow")?;
    let end = start.checked_add(expert_bytes).context("expert tensor range overflow")?;
    let data = weight.data(gguf_bytes)?;
    let base = data
        .get(start..end)
        .with_context(|| format!("expert {} points outside tensor {} data", expert, weight.name))?;
    Ok((base, row_bytes, in_dim, out_dim))
}

fn zero_q8_k_block() -> Q8KBlock {
    Q8KBlock {
        d: 0.0,
        qs: [0; QK_K],
        bsums: [0; QK_K / 16],
    }
}

fn quantize_row_q8_k_into_slice(x: &[f32], out: &mut [Q8KBlock]) -> Result<()> {
    if x.len() % QK_K != 0 {
        bail!("Q8_K quantization length {} is not QK_K aligned", x.len())
    }
    if out.len() != x.len() / QK_K {
        bail!(
            "Q8_K output block count {} does not match input length {}",
            out.len(),
            x.len()
        )
    }

    for (dst, chunk) in out.iter_mut().zip(x.chunks_exact(QK_K)) {
        let mut max_value = 0.0f32;
        let mut amax = 0.0f32;
        for &value in chunk {
            let abs = value.abs();
            if abs > amax {
                amax = abs;
                max_value = value;
            }
        }

        if amax == 0.0 {
            *dst = zero_q8_k_block();
            continue;
        }

        let iscale = -127.0 / max_value;
        let mut qs = [0i8; QK_K];
        for (index, value) in chunk.iter().copied().enumerate() {
            let mut quantized = (iscale * value).round() as i32;
            quantized = quantized.clamp(-128, 127);
            qs[index] = quantized as i8;
        }

        let mut bsums = [0i16; QK_K / 16];
        for (block_index, sum) in bsums.iter_mut().enumerate() {
            *sum = qs[block_index * 16..(block_index + 1) * 16]
                .iter()
                .map(|value| i16::from(*value))
                .sum();
        }

        *dst = Q8KBlock {
            d: 1.0 / iscale,
            qs,
            bsums,
        };
    }

    Ok(())
}

fn quantize_row_q8_k(x: &[f32]) -> Result<Vec<Q8KBlock>> {
    if x.len() % QK_K != 0 {
        bail!("Q8_K quantization length {} is not QK_K aligned", x.len())
    }

    let mut out = vec![zero_q8_k_block(); x.len() / QK_K];
    quantize_row_q8_k_into_slice(x, &mut out)?;

    Ok(out)
}

fn dot_q2_16(q2: &[u8], q8: &[i8], shift: usize) -> i32 {
    q2.iter()
        .zip(q8.iter())
        .map(|(q2, q8)| i32::from(*q8) * i32::from((q2 >> shift) & 3))
        .sum()
}

fn iq2xxs_signed_values(grid_index: u8, sign_index: u8) -> [i8; 8] {
    let grid = IQ2XXS_GRID[grid_index as usize].to_le_bytes();
    let signs = KSIGNS_IQ2XS[sign_index as usize];
    let mut out = [0i8; 8];
    for index in 0..8 {
        let value = grid[index] as i8;
        out[index] = if (signs & KMASK_IQ2XS[index]) != 0 { -value } else { value };
    }
    out
}

fn dot_iq2_pair_16(grid0: &[i8; 8], grid1: &[i8; 8], q8: &[i8]) -> i32 {
    let mut sum = 0i32;
    for index in 0..8 {
        sum += i32::from(grid0[index]) * i32::from(q8[index]);
    }
    for index in 0..8 {
        sum += i32::from(grid1[index]) * i32::from(q8[8 + index]);
    }
    sum
}

fn dot_iq2_xxs_q8_k_row(row: &[u8], xq: &[Q8KBlock], in_dim: usize) -> Result<f32> {
    if row.len() != xq.len() * 66 {
        bail!("IQ2_XXS row byte length {} does not match {} Q8_K blocks", row.len(), xq.len())
    }
    if in_dim != xq.len() * QK_K {
        bail!("Q8_K block count {} does not match input width {}", xq.len(), in_dim)
    }

    let mut sum = 0.0f32;
    for (block_index, block) in xq.iter().enumerate() {
        let start = block_index * 66;
        let chunk = &row[start..start + 66];
        let d = f16_to_f32_bits(u16::from_le_bytes([chunk[0], chunk[1]])) * block.d;
        let mut q2 = &chunk[2..];
        let mut bsum = 0i32;
        let mut q8_offset = 0usize;

        for _ in 0..(QK_K / 32) {
            let _aux0 = u32::from_le_bytes([q2[0], q2[1], q2[2], q2[3]]);
            let aux1 = u32::from_le_bytes([q2[4], q2[5], q2[6], q2[7]]);
            let aux8 = [q2[0], q2[1], q2[2], q2[3], q2[4], q2[5], q2[6], q2[7]];
            q2 = &q2[8..];

            let ls = 2 * ((aux1 >> 28) as i32) + 1;
            let mut sumi = 0i32;
            for pair in 0..2 {
                let l = pair * 2;
                let sign_idx0 = ((aux1 >> (7 * l)) & 127) as u8;
                let sign_idx1 = ((aux1 >> (7 * (l + 1))) & 127) as u8;
                let grid0 = iq2xxs_signed_values(aux8[l], sign_idx0);
                let grid1 = iq2xxs_signed_values(aux8[l + 1], sign_idx1);
                let q8_start = q8_offset;
                sumi += dot_iq2_pair_16(&grid0, &grid1, &block.qs[q8_start..q8_start + 16]);
                q8_offset += 16;
            }
            bsum += sumi * ls;
        }

        sum += d * bsum as f32 * 0.125;
    }

    Ok(sum)
}

fn dot_q2_k_q8_k_row(row: &[u8], xq: &[Q8KBlock], in_dim: usize) -> Result<f32> {
    if row.len() != xq.len() * 84 {
        bail!("Q2_K row byte length {} does not match {} Q8_K blocks", row.len(), xq.len())
    }
    if in_dim != xq.len() * QK_K {
        bail!("Q8_K block count {} does not match input width {}", xq.len(), in_dim)
    }

    let mut sum = 0.0f32;
    for (block_index, block) in xq.iter().enumerate() {
        let start = block_index * 84;
        let chunk = &row[start..start + 84];
        let scales = &chunk[..16];
        let q2 = &chunk[16..80];
        let d = f16_to_f32_bits(u16::from_le_bytes([chunk[80], chunk[81]]));
        let dmin = f16_to_f32_bits(u16::from_le_bytes([chunk[82], chunk[83]]));

        let summs = block
            .bsums
            .iter()
            .zip(scales.iter())
            .map(|(bsum, scale)| i32::from(*bsum) * i32::from(scale >> 4))
            .sum::<i32>();

        let mut isum = 0i32;
        let mut scale_index = 0usize;
        for (chunk_index, q2_chunk) in q2.chunks_exact(32).enumerate() {
            let mut shift = 0usize;
            for pair in 0..4 {
            let q8_start = chunk_index * 128 + pair * 32;
                let scale0 = i32::from(scales[scale_index] & 0x0f);
                scale_index += 1;
                isum += scale0 * dot_q2_16(&q2_chunk[..16], &block.qs[q8_start..q8_start + 16], shift);

                let scale1 = i32::from(scales[scale_index] & 0x0f);
                scale_index += 1;
                isum += scale1 * dot_q2_16(&q2_chunk[16..], &block.qs[q8_start + 16..q8_start + 32], shift);

                shift += 2;
            }
        }

        sum += block.d * d * isum as f32 - block.d * dmin * summs as f32;
    }

    Ok(sum)
}

fn dot_f32(a: &[f32], b: &[f32]) -> Result<f32> {
    if a.len() != b.len() {
        bail!("dot product inputs have different lengths: {} vs {}", a.len(), b.len())
    }
    Ok(a.iter().zip(b.iter()).map(|(lhs, rhs)| *lhs * *rhs).sum())
}

fn axpy_f32(y: &mut [f32], x: &[f32], scale: f32) -> Result<()> {
    if y.len() != x.len() {
        bail!("AXPY inputs have different lengths: {} vs {}", y.len(), x.len())
    }
    for (out, input) in y.iter_mut().zip(x.iter().copied()) {
        *out += scale * input;
    }
    Ok(())
}

fn scale_f32(x: &mut [f32], scale: f32) {
    for value in x {
        *value *= scale;
    }
}

fn softplus_stable(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        x.exp()
    } else {
        x.exp().ln_1p()
    }
}

fn silu(x: f32) -> f32 {
    x * sigmoid_stable(x)
}

fn swiglu(gate: &[f32], up: &[f32]) -> Result<Vec<f32>> {
    if gate.len() != up.len() {
        bail!("SwiGLU gate/up lengths differ: {} vs {}", gate.len(), up.len())
    }
    Ok(gate
        .iter()
        .zip(up.iter())
        .map(|(gate, up)| silu(*gate) * *up)
        .collect())
}

fn swiglu_into(gate: &[f32], up: &[f32], out: &mut Vec<f32>) -> Result<()> {
    if gate.len() != up.len() {
        bail!("SwiGLU gate/up lengths differ: {} vs {}", gate.len(), up.len())
    }
    out.clear();
    out.resize(gate.len(), 0.0);
    for ((dst, gate), up) in out.iter_mut().zip(gate.iter().copied()).zip(up.iter().copied()) {
        *dst = silu(gate) * up;
    }
    Ok(())
}

fn attention_rows_with_sinks(q: &[f32], kv_rows: &[f32], sinks: &[f32]) -> Result<Vec<f32>> {
    if sinks.is_empty() {
        bail!("attention sinks cannot be empty")
    }
    if q.len() % sinks.len() != 0 {
        bail!("query width {} is not divisible by head count {}", q.len(), sinks.len())
    }

    let n_head = sinks.len();
    let head_dim = q.len() / n_head;
    if kv_rows.len() % head_dim != 0 {
        bail!("KV row buffer length {} is not divisible by head width {}", kv_rows.len(), head_dim)
    }
    let n_kv = kv_rows.len() / head_dim;
    let kq_scale = 1.0 / (head_dim as f32).sqrt();
    let mut out_heads = vec![0.0f32; q.len()];
    let mut scores = vec![0.0f32; n_kv];

    for head in 0..n_head {
        let qh = &q[head * head_dim..(head + 1) * head_dim];
        let mut max_score = sinks[head];
        for row in 0..n_kv {
            let kv = &kv_rows[row * head_dim..(row + 1) * head_dim];
            scores[row] = dot_f32(qh, kv)? * kq_scale;
            max_score = max_score.max(scores[row]);
        }

        let oh = &mut out_heads[head * head_dim..(head + 1) * head_dim];
        let mut denom = (sinks[head] - max_score).exp();
        for row in 0..n_kv {
            let weight = (scores[row] - max_score).exp();
            let kv = &kv_rows[row * head_dim..(row + 1) * head_dim];
            denom += weight;
            axpy_f32(oh, kv, weight)?;
        }

        scale_f32(oh, 1.0 / denom);
    }

    Ok(out_heads)
}

fn head_rms_norm_inplace(x: &mut [f32], n_head: usize, head_dim: usize, eps: f32) -> Result<()> {
    if n_head == 0 || head_dim == 0 {
        bail!("head RMSNorm requires non-zero head count and head size")
    }
    if x.len() != n_head * head_dim {
        bail!(
            "head RMSNorm input length {} does not match shape {}x{}",
            x.len(),
            n_head,
            head_dim
        )
    }

    for chunk in x.chunks_exact_mut(head_dim) {
        let sum_squares = chunk
            .iter()
            .map(|value| {
                let value = f64::from(*value);
                value * value
            })
            .sum::<f64>();
        let scale = 1.0f32 / ((sum_squares / head_dim as f64) as f32 + eps).sqrt();
        for value in chunk {
            *value *= scale;
        }
    }

    Ok(())
}

fn rope_yarn_ramp(low: f32, high: f32, index: usize) -> f32 {
    let y = ((index / 2) as f32 - low) / (high - low).max(0.001);
    1.0 - y.clamp(0.0, 1.0)
}

fn rope_yarn_corr_dim(n_dims: usize, n_ctx_orig: u64, n_rot: f32, base: f32) -> f32 {
    n_dims as f32
        * ((n_ctx_orig as f32) / (n_rot * 2.0 * std::f32::consts::PI)).ln()
        / (2.0 * base.ln())
}

fn rope_yarn_corr_dims(
    n_dims: usize,
    n_ctx_orig: u64,
    freq_base: f32,
    beta_fast: f32,
    beta_slow: f32,
) -> [f32; 2] {
    let start = rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_base).floor();
    let end = rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_base).ceil();
    [start.max(0.0), end.min((n_dims - 1) as f32)]
}

fn layer_rope_freq_base(layer: u32) -> f32 {
    if layer_compress_ratio(layer) != 0 && DS4_COMPRESS_ROPE_FREQ_BASE > 0.0 {
        DS4_COMPRESS_ROPE_FREQ_BASE
    } else {
        DS4_ROPE_FREQ_BASE
    }
}

fn layer_rope_freq_scale(layer: u32) -> f32 {
    if layer_compress_ratio(layer) == 0 || DS4_ROPE_SCALE_FACTOR <= 0.0 {
        1.0
    } else {
        1.0 / DS4_ROPE_SCALE_FACTOR
    }
}

fn rope_tail_ext_inplace(
    x: &mut [f32],
    n_head: usize,
    head_dim: usize,
    n_rot: usize,
    pos: u32,
    n_ctx_orig: u64,
    freq_base: f32,
    freq_scale: f32,
    ext_factor: f32,
    attn_factor: f32,
    beta_fast: f32,
    beta_slow: f32,
    inverse: bool,
) -> Result<()> {
    if n_head == 0 || head_dim == 0 || n_rot == 0 || n_rot > head_dim || n_rot % 2 != 0 {
        bail!("invalid RoPE dimensions")
    }
    if x.len() != n_head * head_dim {
        bail!(
            "RoPE input length {} does not match shape {}x{}",
            x.len(),
            n_head,
            head_dim
        )
    }

    let n_nope = head_dim - n_rot;
    let theta_scale = freq_base.powf(-2.0 / n_rot as f32);
    let sin_sign = if inverse { -1.0 } else { 1.0 };
    let corr_dims = if ext_factor != 0.0 {
        rope_yarn_corr_dims(n_rot, n_ctx_orig, freq_base, beta_fast, beta_slow)
    } else {
        [0.0, 0.0]
    };

    for head in 0..n_head {
        let tail = &mut x[head * head_dim + n_nope..(head + 1) * head_dim];
        let mut theta_extrap = pos as f32;

        for index in (0..n_rot).step_by(2) {
            let theta_interp = freq_scale * theta_extrap;
            let mut theta = theta_interp;
            let mut mscale = attn_factor;

            if ext_factor != 0.0 {
                let ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], index) * ext_factor;
                theta = theta_interp * (1.0 - ramp_mix) + theta_extrap * ramp_mix;
                mscale *= 1.0 + 0.1 * (1.0 / freq_scale).ln();
            }

            let cos_theta = theta.cos() * mscale;
            let sin_theta = sin_sign * theta.sin() * mscale;
            let x0 = tail[index];
            let x1 = tail[index + 1];
            tail[index] = x0 * cos_theta - x1 * sin_theta;
            tail[index + 1] = x0 * sin_theta + x1 * cos_theta;
            theta_extrap *= theta_scale;
        }
    }

    Ok(())
}

fn rope_tail_layer_inplace(
    x: &mut [f32],
    n_head: usize,
    head_dim: usize,
    n_rot: usize,
    pos: u32,
    layer: u32,
    inverse: bool,
) -> Result<()> {
    let compressed = layer_compress_ratio(layer) != 0;
    let freq_base = layer_rope_freq_base(layer);
    let freq_scale = layer_rope_freq_scale(layer);
    let ext_factor = if compressed && DS4_ROPE_SCALE_FACTOR > 1.0 { 1.0 } else { 0.0 };
    let mut attn_factor = 1.0f32;
    if ext_factor != 0.0 && freq_scale > 0.0 {
        attn_factor /= 1.0 + 0.1 * (1.0 / freq_scale).ln();
    }

    rope_tail_ext_inplace(
        x,
        n_head,
        head_dim,
        n_rot,
        pos,
        if compressed { DS4_ROPE_ORIG_CTX } else { 0 },
        freq_base,
        freq_scale,
        ext_factor,
        attn_factor,
        DS4_ROPE_YARN_BETA_FAST,
        DS4_ROPE_YARN_BETA_SLOW,
        inverse,
    )
}

fn dsv4_e4m3fn_value(index: i32) -> f32 {
    const EXP_SCALE: [f32; 16] = [
        0.0, 0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0,
        64.0, 128.0, 256.0,
    ];

    let exp = ((index >> 3) & 0x0f) as usize;
    let mant = (index & 0x07) as f32;
    if exp == 0 {
        mant * 0.001953125
    } else {
        (1.0 + mant * 0.125) * EXP_SCALE[exp]
    }
}

fn dsv4_e4m3fn_dequant(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs().min(448.0);

    let mut low = 0;
    let mut high = 126;
    while low < high {
        let mid = (low + high + 1) >> 1;
        if dsv4_e4m3fn_value(mid) <= ax {
            low = mid;
        } else {
            high = mid - 1;
        }
    }

    let mut best = low;
    if best < 126 {
        let best_diff = (ax - dsv4_e4m3fn_value(best)).abs();
        let next_diff = (ax - dsv4_e4m3fn_value(best + 1)).abs();
        if next_diff < best_diff || (next_diff == best_diff && ((best + 1) & 1) == 0 && (best & 1) != 0) {
            best += 1;
        }
    }

    sign * dsv4_e4m3fn_value(best)
}

fn dsv4_fp8_kv_quantize_row_inplace(x: &mut [f32], head_dim: usize, n_rot: usize) -> Result<()> {
    if head_dim > x.len() || n_rot > head_dim {
        bail!("invalid KV quantization dimensions")
    }

    let n_nope = head_dim - n_rot;
    if n_nope % 64 != 0 {
        bail!("non-RoPE KV width {} is not 64-aligned", n_nope)
    }

    for offset in (0..n_nope).step_by(64) {
        let block = &mut x[offset..offset + 64];
        let mut amax = block.iter().copied().map(f32::abs).fold(0.0f32, f32::max);
        if amax < 1.0e-4 {
            amax = 1.0e-4;
        }
        let scale = 2.0f32.powf((amax / 448.0).log2().ceil());

        for value in block {
            let clamped = (*value / scale).clamp(-448.0, 448.0);
            *value = dsv4_e4m3fn_dequant(clamped) * scale;
        }
    }

    Ok(())
}

#[derive(Debug, Clone, Copy)]
struct SpecialToken {
    text: &'static str,
    token: i32,
}

#[derive(Debug, Clone)]
struct SpecialTokens {
    values: [SpecialToken; 7],
}

impl SpecialTokens {
    fn bootstrap(tokenizer: &TokenizerMetadata) -> Result<Self> {
        Ok(Self {
            values: [
                SpecialToken::load(tokenizer, RENDERED_CHAT_BOS)?,
                SpecialToken::load(tokenizer, "<\u{ff5c}end\u{2581}of\u{2581}sentence\u{ff5c}>")?,
                SpecialToken::load(tokenizer, RENDERED_CHAT_USER)?,
                SpecialToken::load(tokenizer, RENDERED_CHAT_ASSISTANT)?,
                SpecialToken::load(tokenizer, THINK_START_MARKER)?,
                SpecialToken::load(tokenizer, THINK_END_MARKER)?,
                SpecialToken::load(tokenizer, DSML_MARKER)?,
            ],
        })
    }

    fn matching_token(&self, bytes: &[u8]) -> Option<(usize, i32)> {
        self.values.iter().find_map(|entry| {
            bytes.starts_with(entry.text.as_bytes())
                .then_some((entry.text.len(), entry.token))
        })
    }

    fn eos(&self) -> i32 {
        self.values[1].token
    }
}

impl SpecialToken {
    fn load(tokenizer: &TokenizerMetadata, text: &'static str) -> Result<Self> {
        Ok(Self {
            text,
            token: tokenizer.lookup_required(text)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn think_max_downgrades_below_context_floor() {
        assert_eq!(ThinkMode::Max.for_context(32_768), ThinkMode::High);
        assert_eq!(ThinkMode::Max.for_context(393_216), ThinkMode::Max);
    }

    #[test]
    fn renders_one_shot_prompt_like_c_helper() {
        let rendered = render_one_shot_prompt(Some("sys"), "hello", ThinkMode::High);
        assert_eq!(
            rendered,
            format!(
                "{bos}sys{user}hello{assistant}{think}",
                bos = RENDERED_CHAT_BOS,
                user = RENDERED_CHAT_USER,
                assistant = RENDERED_CHAT_ASSISTANT,
                think = THINK_START_MARKER,
            )
        );
    }

    #[test]
    fn renders_assistant_messages_with_implicit_think_close() {
        let mut rendered = String::new();
        render_chat_message(&mut rendered, "assistant", "done");
        assert_eq!(
            rendered,
            format!("{assistant}{think_end}done", assistant = RENDERED_CHAT_ASSISTANT, think_end = THINK_END_MARKER)
        );
    }

    #[test]
    fn builds_initial_hc_state_from_plain_embedding() {
        assert_eq!(
            hc_from_plain_embedding(&[1.0, -2.0], 3),
            vec![1.0, -2.0, 1.0, -2.0, 1.0, -2.0]
        );
    }

    #[test]
    fn hc_weighted_sum_reduces_hc_streams() {
        assert_eq!(
            hc_weighted_sum(&[1.0, 2.0, 10.0, 20.0], &[0.25, 0.75]).unwrap(),
            vec![7.75, 15.5]
        );
    }

    #[test]
    fn hc_split_sinkhorn_balances_rows_and_columns() {
        let split = hc_split_sinkhorn(
            &[0.1, -0.2, 0.3, 0.4, 0.5, -0.3, 0.2, 0.8],
            &[1.0, 0.5, 0.25],
            &[0.0; 8],
            2,
            20,
            1.0e-6,
        )
        .unwrap();

        assert_eq!(split.pre.len(), 2);
        assert_eq!(split.post.len(), 2);
        assert_eq!(split.comb.len(), 4);
        assert!(split.pre.iter().all(|value| *value > 0.0));
        assert!(split.post.iter().all(|value| *value >= 0.0 && *value <= 2.0));

        let row0 = split.comb[0] + split.comb[1];
        let row1 = split.comb[2] + split.comb[3];
        let col0 = split.comb[0] + split.comb[2];
        let col1 = split.comb[1] + split.comb[3];

        assert!((row0 - 1.0).abs() < 1.0e-4);
        assert!((row1 - 1.0).abs() < 1.0e-4);
        assert!((col0 - 1.0).abs() < 1.0e-4);
        assert!((col1 - 1.0).abs() < 1.0e-4);
    }

    #[test]
    fn hc_post_mixes_block_and_residual_streams() {
        let out = hc_post(
            &[1.0, 2.0],
            &[10.0, 20.0, 100.0, 200.0],
            &[0.5, 2.0],
            &[0.1, 0.3, 0.2, 0.4],
        )
        .unwrap();

        assert_eq!(out, vec![21.5, 43.0, 45.0, 90.0]);
    }

    #[test]
    fn head_rms_norm_normalizes_each_head_independently() {
        let mut values = vec![3.0, 4.0, 0.0, 5.0];
        head_rms_norm_inplace(&mut values, 2, 2, 0.0).unwrap();

        assert!((values[0] - 0.84852815).abs() < 1.0e-6);
        assert!((values[1] - 1.1313709).abs() < 1.0e-6);
        assert!((values[2] - 0.0).abs() < 1.0e-6);
        assert!((values[3] - 1.4142135).abs() < 1.0e-6);
    }

    #[test]
    fn rope_tail_inverse_round_trips_uncompressed_layer() {
        let mut values = vec![1.0, 2.0, 3.0, 4.0];
        let original = values.clone();

        rope_tail_layer_inplace(&mut values, 1, 4, 4, 7, 0, false).unwrap();
        rope_tail_layer_inplace(&mut values, 1, 4, 4, 7, 0, true).unwrap();

        for (got, want) in values.iter().zip(original.iter()) {
            assert!((got - want).abs() < 1.0e-5);
        }
    }

    #[test]
    fn kv_fp8_quantization_leaves_rope_tail_unchanged() {
        let mut values = (0..128).map(|index| index as f32 / 10.0).collect::<Vec<_>>();
        let tail_before = values[64..].to_vec();

        dsv4_fp8_kv_quantize_row_inplace(&mut values, 128, 64).unwrap();

        assert_eq!(&values[64..], tail_before.as_slice());
    }

    #[test]
    fn attention_rows_with_sinks_weights_value_rows() {
        let out = attention_rows_with_sinks(
            &[1.0, 0.0],
            &[1.0, 0.0, 0.0, 1.0],
            &[0.0],
        )
        .unwrap();

        assert!(out[0] > out[1]);
        assert!(out[0] < 1.0);
        assert!(out[1] > 0.0);
    }

    #[test]
    fn attention_rows_mixed_mask_skips_disallowed_comp_rows() {
        let out = attention_rows_mixed_with_sinks(
            &[1.0, 0.0],
            &[0.0, 1.0],
            &[10.0, 0.0],
            &[0.0],
            Some(&[false]),
        )
        .unwrap();

        assert!(out[0].abs() < 1.0e-6);
        assert!((out[1] - 0.5).abs() < 1.0e-6);
    }

    #[test]
    fn compressor_pool_ratio4_uses_both_lanes() {
        let head_dim = 2;
        let ratio = 4;
        let width = 2 * head_dim;
        let rows = 2 * ratio;
        let mut state_kv = vec![0.0f32; width * rows];
        let mut state_score = vec![DS4_NEG_INF; width * rows];

        state_kv[0] = 1.0;
        state_score[0] = 0.0;
        state_kv[1] = 2.0;
        state_score[1] = 0.0;
        state_kv[width + 0] = 3.0;
        state_score[width + 0] = 0.0;
        state_kv[width + 1] = 4.0;
        state_score[width + 1] = 0.0;
        state_kv[(ratio * width) + head_dim] = 5.0;
        state_score[(ratio * width) + head_dim] = 0.0;
        state_kv[(ratio * width) + head_dim + 1] = 6.0;
        state_score[(ratio * width) + head_dim + 1] = 0.0;
        state_kv[((ratio + 1) * width) + head_dim] = 7.0;
        state_score[((ratio + 1) * width) + head_dim] = 0.0;
        state_kv[((ratio + 1) * width) + head_dim + 1] = 8.0;
        state_score[((ratio + 1) * width) + head_dim + 1] = 0.0;

        let pooled = compressor_pool_decode_state(&state_kv, &state_score, head_dim, ratio).unwrap();

        assert!((pooled[0] - 4.0).abs() < 1.0e-6);
        assert!((pooled[1] - 5.0).abs() < 1.0e-6);
    }

    #[test]
    fn select_top_k_mask_marks_highest_scores() {
        let allowed = select_top_k_mask(&[1.0, 3.0, 2.0, -4.0], 2);

        assert_eq!(allowed, vec![false, true, true, false]);
    }

    #[test]
    fn select_top_k_desc_indices_orders_descending() {
        let selected = select_top_k_desc_indices(&[1.0, 3.0, 2.0, -4.0], 3);

        assert_eq!(selected, vec![1, 2, 0]);
    }

    #[test]
    fn hash_router_weights_normalize_selected_probs() {
        let weights = hash_router_weights_from_probs(&[1.0, 2.0, 3.0], &[2, 0]).unwrap();

        assert!((weights[0] - 1.125).abs() < 1.0e-6);
        assert!((weights[1] - 0.375).abs() < 1.0e-6);
    }

    #[test]
    fn swiglu_multiplies_silu_gate_by_up() {
        let out = swiglu(&[0.0, 1.0], &[2.0, 3.0]).unwrap();

        assert!(out[0].abs() < 1.0e-6);
        assert!((out[1] - 2.1931758).abs() < 1.0e-6);
    }

    #[test]
    fn swiglu_into_matches_allocating() {
        let expected = swiglu(&[0.0, 1.0], &[2.0, 3.0]).unwrap();
        let mut out = Vec::new();

        swiglu_into(&[0.0, 1.0], &[2.0, 3.0], &mut out).unwrap();

        assert_eq!(out.len(), expected.len());
        for (lhs, rhs) in out.iter().zip(expected.iter()) {
            assert!((lhs - rhs).abs() < 1.0e-6);
        }
    }

    #[test]
    fn quantize_row_q8_k_zero_block_stays_zero() {
        let blocks = quantize_row_q8_k(&vec![0.0f32; QK_K]).unwrap();

        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].d, 0.0);
        assert!(blocks[0].qs.iter().all(|value| *value == 0));
        assert!(blocks[0].bsums.iter().all(|value| *value == 0));
    }

    #[test]
    fn quantize_row_q8_k_into_slice_matches_allocating() {
        let input = (0..QK_K).map(|index| index as f32 * 0.125 - 8.0).collect::<Vec<_>>();
        let expected = quantize_row_q8_k(&input).unwrap();
        let mut out = vec![zero_q8_k_block(); input.len() / QK_K];

        quantize_row_q8_k_into_slice(&input, &mut out).unwrap();

        assert_eq!(out.len(), expected.len());
        for (lhs, rhs) in out.iter().zip(expected.iter()) {
            assert!((lhs.d - rhs.d).abs() < 1.0e-6);
            assert_eq!(lhs.qs, rhs.qs);
            assert_eq!(lhs.bsums, rhs.bsums);
        }
    }

    #[test]
    fn dot_q2_k_q8_k_row_accumulates_scaled_values() {
        let mut row = vec![0u8; 84];
        for scale in &mut row[..16] {
            *scale = 1;
        }
        for value in &mut row[16..80] {
            *value = 1;
        }
        row[80..82].copy_from_slice(&0x3c00u16.to_le_bytes());
        row[82..84].copy_from_slice(&0u16.to_le_bytes());

        let xq = vec![Q8KBlock {
            d: 1.0,
            qs: [1; QK_K],
            bsums: [16; QK_K / 16],
        }];

        let out = dot_q2_k_q8_k_row(&row, &xq, QK_K).unwrap();
        assert!((out - 64.0).abs() < 1.0e-6);
    }

    #[test]
    fn dot_iq2_xxs_q8_k_row_accumulates_signed_grid_values() {
        let mut row = vec![0u8; 66];
        row[0..2].copy_from_slice(&0x3c00u16.to_le_bytes());

        let xq = vec![Q8KBlock {
            d: 1.0,
            qs: [1; QK_K],
            bsums: [16; QK_K / 16],
        }];

        let out = dot_iq2_xxs_q8_k_row(&row, &xq, QK_K).unwrap();
        assert!((out - 256.0).abs() < 1.0e-6);
    }

    #[test]
    fn push_raw_cache_row_slides_when_full() {
        let mut rows = vec![1.0, 2.0, 3.0, 4.0];
        push_raw_cache_row(&mut rows, 2, &[5.0, 6.0]);

        assert_eq!(rows, vec![3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn sample_top_p_min_p_uses_argmax_at_zero_temperature() {
        let mut rng = 1u64;
        let token = sample_top_p_min_p(&[1.0, 3.0, 2.0], 0.0, 0, 1.0, 0.0, &mut rng);

        assert_eq!(token, 1);
    }

    #[test]
    fn f16_round_into_matches_allocating() {
        let expected = f16_round_slice(&[0.1, -1.75, 3.5]);
        let mut out = Vec::new();

        f16_round_into(&[0.1, -1.75, 3.5], &mut out);

        assert_eq!(out, expected);
    }

    #[test]
    fn plan_rust_session_sync_keeps_identical_prompt() {
        let plan = plan_rust_session_sync(&[1, 2, 3], &[1, 2, 3]);

        assert_eq!(plan, RustSessionSyncPlan::Keep);
    }

    #[test]
    fn plan_rust_session_sync_extends_matching_prefix() {
        let plan = plan_rust_session_sync(&[1, 2, 3], &[1, 2, 3, 4, 5]);

        assert_eq!(plan, RustSessionSyncPlan::Extend { suffix_start: 3 });
    }

    #[test]
    fn plan_rust_session_sync_rebuilds_shorter_prompt() {
        let plan = plan_rust_session_sync(&[1, 2, 3], &[1, 2]);

        assert_eq!(plan, RustSessionSyncPlan::Rebuild { prefix_len: 2 });
    }

    #[test]
    fn plan_rust_session_sync_rebuilds_divergent_suffix() {
        let plan = plan_rust_session_sync(&[1, 2, 3], &[1, 9, 3]);

        assert_eq!(plan, RustSessionSyncPlan::Rebuild { prefix_len: 1 });
    }

    #[test]
    fn rust_session_spec_accept_limit_includes_first_token() {
        let limit = rust_session_spec_accept_limit(2, 10, 17);

        assert_eq!(limit, 3);
    }

    #[test]
    fn rust_session_spec_accept_limit_respects_remaining_tokens() {
        let limit = rust_session_spec_accept_limit(8, 2, 17);

        assert_eq!(limit, 2);
    }

    #[test]
    fn rust_session_spec_accept_limit_respects_accept_buffer() {
        let limit = rust_session_spec_accept_limit(8, 20, 4);

        assert_eq!(limit, 4);
    }

    #[test]
    fn plan_rust_spec_verify_uses_margin_skip_for_low_margin_two_draft_non_strict() {
        assert_eq!(
            plan_rust_spec_verify(2, false, 3.0, Some(2.5)),
            RustSpecVerifyPlan::MarginSkipSingle
        );
    }

    #[test]
    fn plan_rust_spec_verify_keeps_exact_for_strict_two_draft() {
        assert_eq!(
            plan_rust_spec_verify(2, true, 3.0, Some(0.5)),
            RustSpecVerifyPlan::ExactBatch2
        );
    }

    #[test]
    fn plan_rust_spec_verify_keeps_exact_outside_low_margin_two_draft_case() {
        assert_eq!(
            plan_rust_spec_verify(3, false, 3.0, Some(0.5)),
            RustSpecVerifyPlan::Exact { draft_tokens: 3 }
        );
        assert_eq!(
            plan_rust_spec_verify(2, false, 3.0, Some(3.5)),
            RustSpecVerifyPlan::Exact { draft_tokens: 2 }
        );
    }

    #[test]
    fn rms_norm_weight_scales_normalized_values() {
        let out = rms_norm_weight(&[3.0, 4.0], &[2.0, 0.5], 0.0).unwrap();
        assert!((out[0] - 1.6970563).abs() < 1.0e-6);
        assert!((out[1] - 0.56568545).abs() < 1.0e-6);
    }

    #[test]
    fn rms_norm_weight_rows_matches_rowwise_allocating() {
        let rows = [3.0, 4.0, 0.0, 5.0];
        let weight = [2.0, 0.5];
        let expected = [
            rms_norm_weight(&rows[0..2], &weight, 0.0).unwrap(),
            rms_norm_weight(&rows[2..4], &weight, 0.0).unwrap(),
        ]
        .concat();

        let out = rms_norm_weight_rows(&rows, &weight, 0.0).unwrap();

        assert_eq!(out.len(), expected.len());
        for (lhs, rhs) in out.iter().zip(expected.iter()) {
            assert!((lhs - rhs).abs() < 1.0e-6);
        }
    }

    #[test]
    fn dot_q8_0_row_accumulates_scaled_products() {
        let mut row = vec![0u8; 34];
        row[0..2].copy_from_slice(&0x3c00u16.to_le_bytes());
        row[2] = 1u8;
        row[3] = 2u8;
        row[4] = 3u8;

        let mut xq = vec![0i8; 32];
        xq[0] = 1;
        xq[1] = 2;
        xq[2] = 3;

        let out = dot_q8_0_row(&row, &xq, &[1.0], 3).unwrap();
        assert!((out - 14.0).abs() < 1.0e-6);
    }

    #[test]
    fn rms_norm_no_weight_into_matches_allocating() {
        let mut out = Vec::new();
        let expected = rms_norm_no_weight(&[3.0, 4.0], DS4_RMS_EPS).unwrap();

        rms_norm_no_weight_into(&[3.0, 4.0], DS4_RMS_EPS, &mut out).unwrap();

        assert_eq!(out.len(), expected.len());
        for (lhs, rhs) in out.iter().zip(expected.iter()) {
            assert!((lhs - rhs).abs() < 1.0e-6);
        }
    }

    #[test]
    fn hc_weighted_sum_into_matches_allocating() {
        let mut out = Vec::new();
        let expected = hc_weighted_sum(&[1.0, 2.0, 3.0, 4.0], &[0.25, 0.75]).unwrap();

        hc_weighted_sum_into(&[1.0, 2.0, 3.0, 4.0], &[0.25, 0.75], &mut out).unwrap();

        assert_eq!(out, expected);
    }

    #[test]
    fn hc_weighted_sum_rows_matches_rowwise_allocating() {
        let hc_rows = [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];
        let weights_rows = [0.25, 0.75, 0.6, 0.4];
        let expected = [
            hc_weighted_sum(&hc_rows[0..4], &weights_rows[0..2]).unwrap(),
            hc_weighted_sum(&hc_rows[4..8], &weights_rows[2..4]).unwrap(),
        ]
        .concat();

        let out = hc_weighted_sum_rows(&hc_rows, &weights_rows, 2).unwrap();

        assert_eq!(out.len(), expected.len());
        for (lhs, rhs) in out.iter().zip(expected.iter()) {
            assert!((lhs - rhs).abs() < 1.0e-6);
        }
    }

    #[test]
    fn rms_norm_no_weight_rows_matches_rowwise_allocating() {
        let rows = [3.0, 4.0, 0.0, 5.0];
        let expected = [
            rms_norm_no_weight(&rows[0..2], DS4_RMS_EPS).unwrap(),
            rms_norm_no_weight(&rows[2..4], DS4_RMS_EPS).unwrap(),
        ]
        .concat();

        let out = rms_norm_no_weight_rows(&rows, 2, DS4_RMS_EPS).unwrap();

        assert_eq!(out.len(), expected.len());
        for (lhs, rhs) in out.iter().zip(expected.iter()) {
            assert!((lhs - rhs).abs() < 1.0e-6);
        }
    }

    #[test]
    fn matvec_f16_tensor_into_matches_allocating() {
        let tensor = BoundTensor {
            name: "test_f16".to_owned(),
            descriptor: crate::gguf::TensorDescriptor {
                ndim: 2,
                dims: vec![2, 2],
                tensor_type: 1,
                rel_offset: 0,
                abs_offset: 0,
                elements: 4,
                bytes: 8,
            },
        };
        let mut gguf_bytes = Vec::new();
        for value in [1.0f32, 2.0, 3.0, 4.0] {
            gguf_bytes.extend_from_slice(&f32_to_f16_bits(value).to_le_bytes());
        }

        let expected = matvec_f16_tensor(&tensor, &gguf_bytes, &[1.5, -0.5]).unwrap();
        let mut out = Vec::new();
        matvec_f16_tensor_into(&tensor, &gguf_bytes, &[1.5, -0.5], &mut out).unwrap();

        assert_eq!(out.len(), expected.len());
        for (lhs, rhs) in out.iter().zip(expected.iter()) {
            assert!((lhs - rhs).abs() < 1.0e-6);
        }
    }

    #[test]
    fn matvec_f16_rows_tensor_matches_rowwise_allocating() {
        let tensor = BoundTensor {
            name: "test_f16_rows".to_owned(),
            descriptor: crate::gguf::TensorDescriptor {
                ndim: 2,
                dims: vec![2, 2],
                tensor_type: 1,
                rel_offset: 0,
                abs_offset: 0,
                elements: 4,
                bytes: 8,
            },
        };
        let mut gguf_bytes = Vec::new();
        for value in [1.0f32, 2.0, 3.0, 4.0] {
            gguf_bytes.extend_from_slice(&f32_to_f16_bits(value).to_le_bytes());
        }

        let row_a = [1.5f32, -0.5];
        let row_b = [-2.0f32, 0.25];
        let expected = [
            matvec_f16_tensor(&tensor, &gguf_bytes, &row_a).unwrap(),
            matvec_f16_tensor(&tensor, &gguf_bytes, &row_b).unwrap(),
        ]
        .concat();
        let out = matvec_f16_rows_tensor(&tensor, &gguf_bytes, &[row_a[0], row_a[1], row_b[0], row_b[1]]).unwrap();

        assert_eq!(out.len(), expected.len());
        for (lhs, rhs) in out.iter().zip(expected.iter()) {
            assert!((lhs - rhs).abs() < 1.0e-6);
        }
    }

    #[test]
    fn matvec_q8_0_tensor_into_matches_allocating() {
        let tensor = BoundTensor {
            name: "test_q8_0".to_owned(),
            descriptor: crate::gguf::TensorDescriptor {
                ndim: 2,
                dims: vec![32, 2],
                tensor_type: 8,
                rel_offset: 0,
                abs_offset: 0,
                elements: 64,
                bytes: 68,
            },
        };
        let mut gguf_bytes = vec![0u8; 68];
        gguf_bytes[0..2].copy_from_slice(&0x3c00u16.to_le_bytes());
        for index in 0..32 {
            gguf_bytes[2 + index] = (index as i8 + 1) as u8;
        }
        gguf_bytes[34..36].copy_from_slice(&0x3c00u16.to_le_bytes());
        for index in 0..32 {
            gguf_bytes[36 + index] = (32 - index) as u8;
        }

        let x = (0..32).map(|value| value as f32 * 0.25 - 2.0).collect::<Vec<_>>();
        let expected = matvec_q8_0_tensor(&tensor, &gguf_bytes, &x).unwrap();
        let mut xq = Vec::new();
        let mut xscale = Vec::new();
        let mut out = Vec::new();
        matvec_q8_0_tensor_into(&tensor, &gguf_bytes, &x, &mut xq, &mut xscale, &mut out).unwrap();

        assert_eq!(out.len(), expected.len());
        for (lhs, rhs) in out.iter().zip(expected.iter()) {
            assert!((lhs - rhs).abs() < 1.0e-6);
        }
    }

    #[test]
    fn matvec_q8_0_rows_tensor_matches_rowwise_allocating() {
        let tensor = BoundTensor {
            name: "test_q8_0_rows".to_owned(),
            descriptor: crate::gguf::TensorDescriptor {
                ndim: 2,
                dims: vec![32, 2],
                tensor_type: 8,
                rel_offset: 0,
                abs_offset: 0,
                elements: 64,
                bytes: 68,
            },
        };
        let mut gguf_bytes = vec![0u8; 68];
        gguf_bytes[0..2].copy_from_slice(&0x3c00u16.to_le_bytes());
        for index in 0..32 {
            gguf_bytes[2 + index] = (index as i8 + 1) as u8;
        }
        gguf_bytes[34..36].copy_from_slice(&0x3c00u16.to_le_bytes());
        for index in 0..32 {
            gguf_bytes[36 + index] = (32 - index) as u8;
        }

        let row_a = (0..32).map(|value| value as f32 * 0.25 - 2.0).collect::<Vec<_>>();
        let row_b = (0..32).map(|value| 1.5 - value as f32 * 0.125).collect::<Vec<_>>();
        let mut x_rows = row_a.clone();
        x_rows.extend_from_slice(&row_b);
        let expected = [
            matvec_q8_0_tensor(&tensor, &gguf_bytes, &row_a).unwrap(),
            matvec_q8_0_tensor(&tensor, &gguf_bytes, &row_b).unwrap(),
        ]
        .concat();

        let out = matvec_q8_0_rows_tensor(&tensor, &gguf_bytes, &x_rows).unwrap();

        assert_eq!(out.len(), expected.len());
        for (lhs, rhs) in out.iter().zip(expected.iter()) {
            assert!((lhs - rhs).abs() < 1.0e-6);
        }
    }
}