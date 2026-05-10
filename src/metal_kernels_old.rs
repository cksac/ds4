// Rust Metal kernel wrappers - replaces ds4_metal.m kernel dispatch functions
//
// Strategy: Gradual port from C to Rust
// 1. Initially: all functions use C FFI as fallback
// 2. As each kernel is ported: replace FFI call with Rust Metal implementation
// 3. Finally: remove C dependency entirely
//
// Return value: i32 where 0 = failure, non-zero = success (matches C convention)

use libc::c_void;
use crate::ffi;

// ============================================================================
// EMBEDDING KERNELS
// ============================================================================

/// Embed a single token with head-control (HC) expansion
pub fn embed_token_hc_tensor(
    out_hc: *mut ffi::ds4_metal_tensor,
    model_map: *const c_void,
    model_size: u64,
    weight_offset: u64,
    n_vocab: u32,
    token: u32,
    n_embd: u32,
    n_hc: u32,
) -> i32 {
    // TODO: Implement in Rust using Metal
    // For now, use C FFI fallback
    unsafe {
        ffi::ds4_metal_embed_token_hc_tensor(
            out_hc, model_map, model_size, weight_offset,
            n_vocab, token, n_embd, n_hc
        )
    }
}

/// Embed multiple tokens with HC expansion (batched)
pub fn embed_tokens_hc_tensor(
    out_hc: *mut ffi::ds4_metal_tensor,
    tokens: *const ffi::ds4_metal_tensor,
    model_map: *const c_void,
    model_size: u64,
    weight_offset: u64,
    n_vocab: u32,
    n_tokens: u32,
    n_embd: u32,
    n_hc: u32,
) -> i32 {
    // TODO: Implement in Rust using Metal
    // For now, use C FFI fallback
    unsafe {
        ffi::ds4_metal_embed_tokens_hc_tensor(
            out_hc, tokens, model_map, model_size, weight_offset,
            n_vocab, n_tokens, n_embd, n_hc
        )
    }
}

// ============================================================================
// MATRIX MULTIPLICATION KERNELS  
// ============================================================================

/// Q8_0 quantized matrix multiplication
pub fn matmul_q8_0_tensor(
    out: *mut ffi::ds4_metal_tensor,
    model_map: *const c_void,
    model_size: u64,
    weight_offset: u64,
    in_dim: u64,
    out_dim: u64,
    x: *const ffi::ds4_metal_tensor,
    n_tok: u64,
) -> i32 {
    // TODO: Implement in Rust using Metal
    // For now, use C FFI fallback
    unsafe {
        ffi::ds4_metal_matmul_q8_0_tensor(
            out, model_map, model_size, weight_offset,
            in_dim, out_dim, x, n_tok
        )
    }
}

/// F16 weight matrix multiplication
pub fn matmul_f16_tensor(
    out: *mut ffi::ds4_metal_tensor,
    model_map: *const c_void,
    model_size: u64,
    weight_offset: u64,
    in_dim: u64,
    out_dim: u64,
    x: *const ffi::ds4_metal_tensor,
    n_tok: u64,
) -> i32 {
    // TODO: Implement in Rust using Metal
    // For now, use C FFI fallback
    unsafe {
        ffi::ds4_metal_matmul_f16_tensor(
            out, model_map, model_size, weight_offset,
            in_dim, out_dim, x, n_tok
        )
    }
}

/// F16 dual matrix multiplication
pub fn matmul_f16_pair_tensor(
    out_a: *mut ffi::ds4_metal_tensor,
    out_b: *mut ffi::ds4_metal_tensor,
    model_map: *const c_void,
    model_size: u64,
    weight_a_offset: u64,
    weight_b_offset: u64,
    in_dim: u64,
    out_dim: u64,
    x: *const ffi::ds4_metal_tensor,
    n_tok: u64,
) -> i32 {
    // TODO: Implement in Rust using Metal
    // For now, use C FFI fallback
    unsafe {
        ffi::ds4_metal_matmul_f16_pair_tensor(
            out_a, out_b, model_map, model_size,
            weight_a_offset, weight_b_offset, in_dim, out_dim,
            x, n_tok
        )
    }
}

/// F32 weight matrix multiplication
pub fn matmul_f32_tensor(
    out: *mut ds4_metal_tensor,
    model_map: *const c_void,
    model_size: u64,
    weight_offset: u64,
    in_dim: u64,
    out_dim: u64,
    x: *const ds4_metal_tensor,
    n_tok: u64,
) -> Result<()> {
    // TODO: Port from C implementation
    Err(anyhow!("matmul_f32_tensor not yet ported"))
}

/// Q8_0 matmul with head-control expansion
pub fn matmul_q8_0_hc_expand_tensor(
    out_hc: *mut ds4_metal_tensor,
    block_out: *mut ds4_metal_tensor,
    model_map: *const c_void,
    model_size: u64,
    weight_offset: u64,
    in_dim: u64,
    out_dim: u64,
    x: *const ds4_metal_tensor,
    residual_hc: *const ds4_metal_tensor,
    split: *const ds4_metal_tensor,
    n_embd: u32,
    n_hc: u32,
) -> Result<()> {
    // TODO: Port from C implementation
    Err(anyhow!("matmul_q8_0_hc_expand_tensor not yet ported"))
}

// ============================================================================
// NORMALIZATION KERNELS
// ============================================================================

/// RMS norm without weights
pub fn rms_norm_plain_rows_tensor(
    out: *mut ds4_metal_tensor,
    x: *const ds4_metal_tensor,
    n: u32,
    rows: u32,
    eps: f32,
) -> Result<()> {
    // TODO: Port from C implementation
    Err(anyhow!("rms_norm_plain_rows_tensor not yet ported"))
}

/// Weighted RMS norm
pub fn rms_norm_weight_tensor(
    out: *mut ds4_metal_tensor,
    x: *const ds4_metal_tensor,
    model_map: *const c_void,
    model_size: u64,
    weight_offset: u64,
    n: u32,
    eps: f32,
) -> Result<()> {
    // TODO: Port from C implementation
    Err(anyhow!("rms_norm_weight_tensor not yet ported"))
}

/// Weighted RMS norm (multiple rows)
pub fn rms_norm_weight_rows_tensor(
    out: *mut ds4_metal_tensor,
    x: *const ds4_metal_tensor,
    model_map: *const c_void,
    model_size: u64,
    weight_offset: u64,
    n: u32,
    rows: u32,
    eps: f32,
) -> Result<()> {
    // TODO: Port from C implementation
    Err(anyhow!("rms_norm_weight_rows_tensor not yet ported"))
}

/// Head-wise RMS norm
pub fn head_rms_norm_tensor(
    x: *mut ds4_metal_tensor,
    n_tok: u32,
    n_head: u32,
    head_dim: u32,
    eps: f32,
) -> Result<()> {
    // TODO: Port from C implementation
    Err(anyhow!("head_rms_norm_tensor not yet ported"))
}

/// Fused Q/KV RMS norm
pub fn dsv4_qkv_rms_norm_rows_tensor(
    q_out: *mut ds4_metal_tensor,
    q: *const ds4_metal_tensor,
    model_map: *const c_void,
    model_size: u64,
    q_weight_offset: u64,
    q_n: u32,
    kv_out: *mut ds4_metal_tensor,
    kv: *const ds4_metal_tensor,
    kv_weight_offset: u64,
    kv_n: u32,
    rows: u32,
    eps: f32,
) -> Result<()> {
    // TODO: Port from C implementation
    Err(anyhow!("dsv4_qkv_rms_norm_rows_tensor not yet ported"))
}

// ============================================================================
// POSITION ENCODING KERNELS
// ============================================================================

/// Rotary Position Embedding (RoPE) for tail dimensions
pub fn rope_tail_tensor(
    x: *mut ds4_metal_tensor,
    n_tok: u32,
    n_head: u32,
    head_dim: u32,
    n_rot: u32,
    pos0: u32,
    n_ctx_orig: u32,
    inverse: bool,
    freq_base: f32,
    freq_scale: f32,
    ext_factor: f32,
    attn_factor: f32,
    beta_fast: f32,
    beta_slow: f32,
) -> Result<()> {
    // TODO: Port from C implementation
    Err(anyhow!("rope_tail_tensor not yet ported"))
}

// ============================================================================
// ACTIVATION KERNELS
// ============================================================================

/// Element-wise addition
pub fn add_tensor(
    out: *mut ds4_metal_tensor,
    a: *const ds4_metal_tensor,
    b: *const ds4_metal_tensor,
    n: u32,
) -> Result<()> {
    // TODO: Port from C implementation
    Err(anyhow!("add_tensor not yet ported"))
}

/// SwiGLU activation
pub fn swiglu_tensor(
    out: *mut ds4_metal_tensor,
    gate: *const ds4_metal_tensor,
    up: *const ds4_metal_tensor,
    n: u32,
    clamp: f32,
    weight: f32,
) -> Result<()> {
    // TODO: Port from C implementation
    Err(anyhow!("swiglu_tensor not yet ported"))
}

/// Shared expert fused gate+up+SwiGLU
pub fn shared_gate_up_swiglu_q8_0_tensor(
    gate: *mut ds4_metal_tensor,
    up: *mut ds4_metal_tensor,
    mid: *mut ds4_metal_tensor,
    model_map: *const c_void,
    model_size: u64,
    gate_offset: u64,
    up_offset: u64,
    in_dim: u64,
    out_dim: u64,
    x: *const ds4_metal_tensor,
) -> Result<()> {
    // TODO: Port from C implementation
    Err(anyhow!("shared_gate_up_swiglu_q8_0_tensor not yet ported"))
}

/// Shared expert down projection with HC expand
pub fn shared_down_hc_expand_q8_0_tensor(
    out_hc: *mut ds4_metal_tensor,
    shared_out: *mut ds4_metal_tensor,
    model_map: *const c_void,
    model_size: u64,
    weight_offset: u64,
    in_dim: u64,
    out_dim: u64,
    shared_mid: *const ds4_metal_tensor,
    routed_out: *const ds4_metal_tensor,
    residual_hc: *const ds4_metal_tensor,
    split: *const ds4_metal_tensor,
    n_embd: u32,
    n_hc: u32,
) -> Result<()> {
    // TODO: Port from C implementation
    Err(anyhow!("shared_down_hc_expand_q8_0_tensor not yet ported"))
}

// ============================================================================
// ATTENTION KERNELS
// ============================================================================

/// Decode attention with heads
pub fn attention_decode_heads_tensor(
    heads: *mut ds4_metal_tensor,
    model_map: *const c_void,
    model_size: u64,
    sinks_offset: u64,
    q: *const ds4_metal_tensor,
    raw_kv: *const ds4_metal_tensor,
    n_raw: u32,
    raw_cap: u32,
    raw_start: u32,
    comp_kv: *const ds4_metal_tensor,
    n_comp: u32,
    comp_mask: *const ds4_metal_tensor,
    use_mask: u32,
    n_head: u32,
    head_dim: u32,
) -> Result<()> {
    // TODO: Port from C implementation
    Err(anyhow!("attention_decode_heads_tensor not yet ported"))
}

/// Prefill raw attention
pub fn attention_prefill_raw_heads_tensor(
    heads: *mut ds4_metal_tensor,
    model_map: *const c_void,
    model_size: u64,
    sinks_offset: u64,
    q: *const ds4_metal_tensor,
    raw_kv: *const ds4_metal_tensor,
    n_tokens: u32,
    window: u32,
    n_head: u32,
    head_dim: u32,
) -> Result<()> {
    // TODO: Port from C implementation
    Err(anyhow!("attention_prefill_raw_heads_tensor not yet ported"))
}

/// Decode mixed attention (raw + compressed)
pub fn attention_decode_mixed_batch_heads_tensor(
    heads: *mut ds4_metal_tensor,
    model_map: *const c_void,
    model_size: u64,
    sinks_offset: u64,
    q: *const ds4_metal_tensor,
    raw_kv: *const ds4_metal_tensor,
    comp_kv: *const ds4_metal_tensor,
    comp_mask: *const ds4_metal_tensor,
    use_comp_mask: u32,
    n_tokens: u32,
    pos0: u32,
    n_raw: u32,
    raw_cap: u32,
    raw_start: u32,
    n_comp: u32,
    window: u32,
    ratio: u32,
    n_head: u32,
    head_dim: u32,
) -> Result<()> {
    // TODO: Port from C implementation
    Err(anyhow!("attention_decode_mixed_batch_heads_tensor not yet ported"))
}

/// Indexed attention
pub fn attention_indexed_mixed_batch_heads_tensor(
    heads: *mut ds4_metal_tensor,
    model_map: *const c_void,
    model_size: u64,
    sinks_offset: u64,
    q: *const ds4_metal_tensor,
    raw_kv: *const ds4_metal_tensor,
    comp_kv: *const ds4_metal_tensor,
    topk: *const ds4_metal_tensor,
    n_tokens: u32,
    pos0: u32,
    n_raw: u32,
    raw_cap: u32,
    raw_start: u32,
    n_comp: u32,
    top_k: u32,
    window: u32,
    ratio: u32,
    n_head: u32,
    head_dim: u32,
) -> Result<()> {
    // TODO: Port from C implementation
    Err(anyhow!("attention_indexed_mixed_batch_heads_tensor not yet ported"))
}

/// Static prefill mixed attention
pub fn attention_prefill_static_mixed_heads_tensor(
    heads: *mut ds4_metal_tensor,
    model_map: *const c_void,
    model_size: u64,
    sinks_offset: u64,
    q: *const ds4_metal_tensor,
    raw_kv: *const ds4_metal_tensor,
    comp_kv: *const ds4_metal_tensor,
    n_tokens: u32,
    n_comp: u32,
    window: u32,
    ratio: u32,
    n_head: u32,
    head_dim: u32,
) -> Result<()> {
    // TODO: Port from C implementation
    Err(anyhow!("attention_prefill_static_mixed_heads_tensor not yet ported"))
}

/// Masked prefill attention
pub fn attention_prefill_masked_mixed_heads_tensor(
    heads: *mut ds4_metal_tensor,
    model_map: *const c_void,
    model_size: u64,
    sinks_offset: u64,
    q: *const ds4_metal_tensor,
    raw_kv: *const ds4_metal_tensor,
    comp_kv: *const ds4_metal_tensor,
    comp_mask: *const ds4_metal_tensor,
    n_tokens: u32,
    n_comp: u32,
    window: u32,
    ratio: u32,
    n_head: u32,
    head_dim: u32,
) -> Result<()> {
    // TODO: Port from C implementation
    Err(anyhow!("attention_prefill_masked_mixed_heads_tensor not yet ported"))
}

/// Batch decode raw attention
pub fn attention_decode_raw_batch_heads_tensor(
    heads: *mut ds4_metal_tensor,
    model_map: *const c_void,
    model_size: u64,
    sinks_offset: u64,
    q: *const ds4_metal_tensor,
    raw_kv: *const ds4_metal_tensor,
    n_tokens: u32,
    pos0: u32,
    n_raw: u32,
    raw_cap: u32,
    raw_start: u32,
    window: u32,
    n_head: u32,
    head_dim: u32,
) -> Result<()> {
    // TODO: Port from C implementation
    Err(anyhow!("attention_decode_raw_batch_heads_tensor not yet ported"))
}

/// Attention output projection
pub fn attention_output_q8_batch_tensor(
    out: *mut ds4_metal_tensor,
    low: *mut ds4_metal_tensor,
    group_tmp: *mut ds4_metal_tensor,
    low_tmp: *mut ds4_metal_tensor,
    model_map: *const c_void,
    model_size: u64,
    out_a_offset: u64,
    out_b_offset: u64,
    group_dim: u64,
    rank: u64,
    n_groups: u32,
    out_dim: u64,
    heads: *const ds4_metal_tensor,
    n_tokens: u32,
) -> Result<()> {
    // TODO: Port from C implementation
    Err(anyhow!("attention_output_q8_batch_tensor not yet ported"))
}

/// Attention output low-rank projection
pub fn attention_output_low_q8_tensor(
    low: *mut ds4_metal_tensor,
    model_map: *const c_void,
    model_size: u64,
    out_a_offset: u64,
    group_dim: u64,
    rank: u64,
    n_groups: u32,
    heads: *const ds4_metal_tensor,
) -> Result<()> {
    // TODO: Port from C implementation
    Err(anyhow!("attention_output_low_q8_tensor not yet ported"))
}

// ============================================================================
// HEAD-CONTROL (HC) KERNELS
// ============================================================================

/// HC split with Sinkhorn
pub fn hc_split_sinkhorn_tensor(
    out: *mut ds4_metal_tensor,
    mix: *const ds4_metal_tensor,
    model_map: *const c_void,
    model_size: u64,
    scale_offset: u64,
    base_offset: u64,
    n_hc: u32,
    sinkhorn_iters: u32,
    eps: f32,
) -> Result<()> {
    // TODO: Port from C implementation
    Err(anyhow!("hc_split_sinkhorn_tensor not yet ported"))
}

/// HC weighted sum
pub fn hc_weighted_sum_tensor(
    out: *mut ds4_metal_tensor,
    residual_hc: *const ds4_metal_tensor,
    weights: *const ds4_metal_tensor,
    n_embd: u32,
    n_hc: u32,
) -> Result<()> {
    // TODO: Port from C implementation
    Err(anyhow!("hc_weighted_sum_tensor not yet ported"))
}

/// HC split weighted sum
pub fn hc_split_weighted_sum_tensor(
    out: *mut ds4_metal_tensor,
    split: *mut ds4_metal_tensor,
    mix: *const ds4_metal_tensor,
    residual_hc: *const ds4_metal_tensor,
    model_map: *const c_void,
    model_size: u64,
    scale_offset: u64,
    base_offset: u64,
    n_embd: u32,
    n_hc: u32,
    sinkhorn_iters: u32,
    eps: f32,
) -> Result<()> {
    // TODO: Port from C implementation
    Err(anyhow!("hc_split_weighted_sum_tensor not yet ported"))
}

/// HC split weighted sum with norm
pub fn hc_split_weighted_sum_norm_tensor(
    out: *mut ds4_metal_tensor,
    norm_out: *mut ds4_metal_tensor,
    split: *mut ds4_metal_tensor,
    mix: *const ds4_metal_tensor,
    residual_hc: *const ds4_metal_tensor,
    model_map: *const c_void,
    model_size: u64,
    scale_offset: u64,
    base_offset: u64,
    norm_weight_offset: u64,
    n_embd: u32,
    n_hc: u32,
    sinkhorn_iters: u32,
    eps: f32,
    norm_eps: f32,
) -> Result<()> {
    // TODO: Port from C implementation
    Err(anyhow!("hc_split_weighted_sum_norm_tensor not yet ported"))
}

/// HC expand from block output
pub fn hc_expand_tensor(
    out_hc: *mut ds4_metal_tensor,
    block_out: *const ds4_metal_tensor,
    residual_hc: *const ds4_metal_tensor,
    post: *const ds4_metal_tensor,
    comb: *const ds4_metal_tensor,
    n_embd: u32,
    n_hc: u32,
) -> Result<()> {
    // TODO: Port from C implementation
    Err(anyhow!("hc_expand_tensor not yet ported"))
}

/// HC expand with split
pub fn hc_expand_split_tensor(
    out_hc: *mut ds4_metal_tensor,
    block_out: *const ds4_metal_tensor,
    residual_hc: *const ds4_metal_tensor,
    split: *const ds4_metal_tensor,
    n_embd: u32,
    n_hc: u32,
) -> Result<()> {
    // TODO: Port from C implementation
    Err(anyhow!("hc_expand_split_tensor not yet ported"))
}

/// HC expand with split add
pub fn hc_expand_add_split_tensor(
    out_hc: *mut ds4_metal_tensor,
    block_out: *const ds4_metal_tensor,
    block_add: *const ds4_metal_tensor,
    residual_hc: *const ds4_metal_tensor,
    split: *const ds4_metal_tensor,
    n_embd: u32,
    n_hc: u32,
) -> Result<()> {
    // TODO: Port from C implementation
    Err(anyhow!("hc_expand_add_split_tensor not yet ported"))
}

/// Output HC weights
pub fn output_hc_weights_tensor(
    out: *mut ds4_metal_tensor,
    pre: *const ds4_metal_tensor,
    model_map: *const c_void,
    model_size: u64,
    scale_offset: u64,
    base_offset: u64,
    n_hc: u32,
    eps: f32,
) -> Result<()> {
    // TODO: Port from C implementation
    Err(anyhow!("output_hc_weights_tensor not yet ported"))
}

// ============================================================================
// KV CACHE & COMPRESSION KERNELS  
// ============================================================================

/// Store raw KV pair
pub fn store_raw_kv_tensor(
    raw_cache: *mut ds4_metal_tensor,
    kv: *const ds4_metal_tensor,
    raw_cap: u32,
    row: u32,
    head_dim: u32,
) -> Result<()> {
    // TODO: Port from C implementation
    Err(anyhow!("store_raw_kv_tensor not yet ported"))
}

/// FP8 quantize and store KV
pub fn kv_fp8_store_raw_tensor(
    kv: *mut ds4_metal_tensor,
    raw_cache: *mut ds4_metal_tensor,
    raw_cap: u32,
    row: u32,
    head_dim: u32,
    n_rot: u32,
) -> Result<()> {
    // TODO: Port from C implementation
    Err(anyhow!("kv_fp8_store_raw_tensor not yet ported"))
}

/// Store batch raw KV
pub fn store_raw_kv_batch_tensor(
    raw_cache: *mut ds4_metal_tensor,
    kv: *const ds4_metal_tensor,
    raw_cap: u32,
    pos0: u32,
    n_tokens: u32,
    head_dim: u32,
) -> Result<()> {
    // TODO: Port from C implementation
    Err(anyhow!("store_raw_kv_batch_tensor not yet ported"))
}

/// FP8 KV quantize
pub fn dsv4_fp8_kv_quantize_tensor(
    x: *mut ds4_metal_tensor,
    n_tok: u32,
    head_dim: u32,
    n_rot: u32,
) -> Result<()> {
    // TODO: Port from C implementation
    Err(anyhow!("dsv4_fp8_kv_quantize_tensor not yet ported"))
}

/// Compressor update
pub fn compressor_update_tensor(
    kv_cur: *const ds4_metal_tensor,
    sc_cur: *const ds4_metal_tensor,
    state_kv: *mut ds4_metal_tensor,
    state_score: *mut ds4_metal_tensor,
    comp_cache: *mut ds4_metal_tensor,
    model_map: *const c_void,
    model_size: u64,
    ape_offset: u64,
    ape_type: u32,
    norm_offset: u64,
    norm_type: u32,
    head_dim: u32,
    ratio: u32,
    pos: u32,
    comp_row: u32,
    n_rot: u32,
    n_ctx_orig: u32,
    freq_base: f32,
    freq_scale: f32,
    ext_factor: f32,
    attn_factor: f32,
    beta_fast: f32,
    beta_slow: f32,
    rms_eps: f32,
) -> Result<()> {
    // TODO: Port from C implementation
    Err(anyhow!("compressor_update_tensor not yet ported"))
}

/// Compressor prefill
pub fn compressor_prefill_tensor(
    comp_cache: *mut ds4_metal_tensor,
    state_kv: *mut ds4_metal_tensor,
    state_score: *mut ds4_metal_tensor,
    kv: *const ds4_metal_tensor,
    sc: *const ds4_metal_tensor,
    model_map: *const c_void,
    model_size: u64,
    ape_offset: u64,
    ape_type: u32,
    norm_offset: u64,
    norm_type: u32,
    head_dim: u32,
    ratio: u32,
    pos0: u32,
    n_tokens: u32,
    n_rot: u32,
    n_ctx_orig: u32,
    quantize_fp8: i32,
    freq_base: f32,
    freq_scale: f32,
    ext_factor: f32,
    attn_factor: f32,
    beta_fast: f32,
    beta_slow: f32,
    rms_eps: f32,
) -> Result<()> {
    // TODO: Port from C implementation
    Err(anyhow!("compressor_prefill_tensor not yet ported"))
}

/// Compressor store batch
pub fn compressor_store_batch_tensor(
    kv: *const ds4_metal_tensor,
    sc: *const ds4_metal_tensor,
    state_kv: *mut ds4_metal_tensor,
    state_score: *mut ds4_metal_tensor,
    model_map: *const c_void,
    model_size: u64,
    ape_offset: u64,
    ape_type: u32,
    head_dim: u32,
    ratio: u32,
    pos0: u32,
    n_tokens: u32,
) -> Result<()> {
    // TODO: Port from C implementation
    Err(anyhow!("compressor_store_batch_tensor not yet ported"))
}

// ============================================================================
// MoE KERNELS
// ============================================================================

/// Router select for single token
pub fn router_select_tensor(
    selected: *mut ds4_metal_tensor,
    weights: *mut ds4_metal_tensor,
    probs: *mut ds4_metal_tensor,
    model_map: *const c_void,
    model_size: u64,
    bias_offset: u64,
    hash_offset: u64,
    hash_rows: u32,
    token: u32,
    n_expert_groups: u32,
    n_group_used: u32,
    has_bias: bool,
    hash_mode: bool,
    logits: *const ds4_metal_tensor,
) -> Result<()> {
    // TODO: Port from C implementation
    Err(anyhow!("router_select_tensor not yet ported"))
}

/// Router select batch
pub fn router_select_batch_tensor(
    selected: *mut ds4_metal_tensor,
    weights: *mut ds4_metal_tensor,
    probs: *mut ds4_metal_tensor,
    model_map: *const c_void,
    model_size: u64,
    bias_offset: u64,
    hash_offset: u64,
    hash_rows: u32,
    n_expert_groups: u32,
    n_group_used: u32,
    has_bias: i32,
    hash_mode: i32,
    logits: *const ds4_metal_tensor,
    tokens: *const ds4_metal_tensor,
    n_tokens: u32,
) -> Result<()> {
    // TODO: Port from C implementation
    Err(anyhow!("router_select_batch_tensor not yet ported"))
}

/// Routed MoE single token
pub fn routed_moe_one_tensor(
    out: *mut ds4_metal_tensor,
    gate: *mut ds4_metal_tensor,
    up: *mut ds4_metal_tensor,
    mid: *mut ds4_metal_tensor,
    experts: *mut ds4_metal_tensor,
    model_map: *const c_void,
    model_size: u64,
    gate_offset: u64,
    up_offset: u64,
    down_offset: u64,
    gate_type: u32,
    down_type: u32,
    gate_expert_bytes: u64,
    gate_row_bytes: u64,
    down_expert_bytes: u64,
    down_row_bytes: u64,
    expert_in_dim: u32,
    expert_mid_dim: u32,
    out_dim: u32,
    selected: *const ds4_metal_tensor,
    weights: *const ds4_metal_tensor,
    n_expert: u32,
    clamp: f32,
    x: *const ds4_metal_tensor,
) -> Result<()> {
    // TODO: Port from C implementation
    Err(anyhow!("routed_moe_one_tensor not yet ported"))
}

/// Routed MoE batch
pub fn routed_moe_batch_tensor(
    out: *mut ds4_metal_tensor,
    gate: *mut ds4_metal_tensor,
    up: *mut ds4_metal_tensor,
    mid: *mut ds4_metal_tensor,
    experts: *mut ds4_metal_tensor,
    model_map: *const c_void,
    model_size: u64,
    gate_offset: u64,
    up_offset: u64,
    down_offset: u64,
    gate_type: u32,
    down_type: u32,
    gate_expert_bytes: u64,
    gate_row_bytes: u64,
    down_expert_bytes: u64,
    down_row_bytes: u64,
    expert_in_dim: u32,
    expert_mid_dim: u32,
    out_dim: u32,
    selected: *const ds4_metal_tensor,
    weights: *const ds4_metal_tensor,
    n_expert: u32,
    clamp: f32,
    x: *const ds4_metal_tensor,
    n_tokens: u32,
) -> Result<()> {
    // TODO: Port from C implementation
    Err(anyhow!("routed_moe_batch_tensor not yet ported"))
}

// ============================================================================
// INDEXER KERNELS
// ============================================================================

/// Indexer score single
pub fn indexer_score_one_tensor(
    scores: *mut ds4_metal_tensor,
    q: *const ds4_metal_tensor,
    weights: *const ds4_metal_tensor,
    index_comp: *const ds4_metal_tensor,
    n_comp: u32,
    n_head: u32,
    head_dim: u32,
    scale: f32,
) -> Result<()> {
    // TODO: Port from C implementation
    Err(anyhow!("indexer_score_one_tensor not yet ported"))
}

/// Indexer scores prefill
pub fn indexer_scores_prefill_tensor(
    scores: *mut ds4_metal_tensor,
    q: *const ds4_metal_tensor,
    weights: *const ds4_metal_tensor,
    index_comp: *const ds4_metal_tensor,
    n_comp: u32,
    n_tokens: u32,
    n_head: u32,
    head_dim: u32,
    ratio: u32,
    scale: f32,
) -> Result<()> {
    // TODO: Port from C implementation
    Err(anyhow!("indexer_scores_prefill_tensor not yet ported"))
}

/// Indexer scores decode batch
pub fn indexer_scores_decode_batch_tensor(
    scores: *mut ds4_metal_tensor,
    q: *const ds4_metal_tensor,
    weights: *const ds4_metal_tensor,
    index_comp: *const ds4_metal_tensor,
    n_comp: u32,
    n_tokens: u32,
    pos0: u32,
    n_head: u32,
    head_dim: u32,
    ratio: u32,
    scale: f32,
) -> Result<()> {
    // TODO: Port from C implementation
    Err(anyhow!("indexer_scores_decode_batch_tensor not yet ported"))
}

/// Indexer topk
pub fn indexer_topk_tensor(
    selected: *mut ds4_metal_tensor,
    scores: *const ds4_metal_tensor,
    n_comp: u32,
    n_tokens: u32,
    top_k: u32,
) -> Result<()> {
    // TODO: Port from C implementation
    Err(anyhow!("indexer_topk_tensor not yet ported"))
}

/// TopK mask
pub fn dsv4_topk_mask_tensor(
    mask: *mut ds4_metal_tensor,
    topk: *const ds4_metal_tensor,
    n_comp: u32,
    n_tokens: u32,
    top_k: u32,
) -> Result<()> {
    // TODO: Port from C implementation
    Err(anyhow!("dsv4_topk_mask_tensor not yet ported"))
}

// ============================================================================
// UTILITY KERNELS
// ============================================================================

/// Repeat row across HC dimensions
pub fn repeat_hc_tensor(
    out: *mut ds4_metal_tensor,
    row: *const ds4_metal_tensor,
    n_embd: u32,
    n_hc: u32,
) -> Result<()> {
    // TODO: Port from C implementation
    Err(anyhow!("repeat_hc_tensor not yet ported"))
}

/// HC weighted sum split
pub fn hc_weighted_sum_split_tensor(
    out: *mut ds4_metal_tensor,
    residual_hc: *const ds4_metal_tensor,
    split: *const ds4_metal_tensor,
    n_embd: u32,
    n_hc: u32,
) -> Result<()> {
    // TODO: Port from C implementation
    Err(anyhow!("hc_weighted_sum_split_tensor not yet ported"))
}

#[cfg(test)]
mod tests {
    use super::*;
}
