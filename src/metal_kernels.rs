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

// Macro to reduce boilerplate for simple FFI wrapper functions
macro_rules! ffi_wrapper {
    ($name:ident, $ffi_name:path, $($param:ident : $param_type:ty),*) => {
        pub unsafe fn $name($($param: $param_type),*) -> i32 {
            // TODO: Implement in Rust using Metal (currently using C fallback)
            unsafe { $ffi_name($($param),*) }
        }
    };
}

// ============================================================================
// EMBEDDING KERNELS
// ============================================================================

pub unsafe fn embed_token_hc_tensor(
    out_hc: *mut ffi::ds4_metal_tensor,
    model_map: *const c_void,
    model_size: u64,
    weight_offset: u64,
    n_vocab: u32,
    token: u32,
    n_embd: u32,
    n_hc: u32,
) -> i32 {
    if out_hc.is_null() || model_map.is_null() || n_vocab == 0 || token >= n_vocab || n_embd == 0 || n_hc == 0 {
        return 0;
    }

    let src_row_bytes = (n_embd as u64).saturating_mul(2);
    let weight_bytes = (n_vocab as u64).saturating_mul(src_row_bytes);
    if weight_offset > model_size || weight_bytes > model_size.saturating_sub(weight_offset) {
        return 0;
    }

    let row_offset = weight_offset.saturating_add((token as u64).saturating_mul(src_row_bytes));
    if row_offset > model_size || src_row_bytes > model_size.saturating_sub(row_offset) {
        return 0;
    }

    let out_f32_len = (n_embd as usize).saturating_mul(n_hc as usize);
    let out_bytes = (out_f32_len as u64).saturating_mul(4);

    let src = unsafe {
        std::slice::from_raw_parts(
            (model_map as *const u8).add(row_offset as usize),
            src_row_bytes as usize,
        )
    };

    let mut row = vec![0f32; n_embd as usize];
    for (i, v) in row.iter_mut().enumerate() {
        let b0 = src[i * 2];
        let b1 = src[i * 2 + 1];
        *v = f16_to_f32_bits(u16::from_le_bytes([b0, b1]));
    }

    let mut out = vec![0f32; out_f32_len];
    for h in 0..(n_hc as usize) {
        let start = h * (n_embd as usize);
        let end = start + (n_embd as usize);
        out[start..end].copy_from_slice(&row);
    }

    unsafe {
        ffi::ds4_metal_tensor_write(
            out_hc,
            0,
            out.as_ptr() as *const c_void,
            out_bytes,
        )
    }
}

fn f16_to_f32_bits(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let frac = (bits & 0x03ff) as u32;

    let out = if exp == 0 {
        if frac == 0 {
            sign
        } else {
            let mut frac_norm = frac;
            let mut exp_shift = -1i32;
            while (frac_norm & 0x0400) == 0 {
                frac_norm <<= 1;
                exp_shift -= 1;
            }
            frac_norm &= 0x03ff;
            let exp32 = (127 - 15 + 1 + exp_shift) as u32;
            sign | (exp32 << 23) | (frac_norm << 13)
        }
    } else if exp == 0x1f {
        sign | 0x7f80_0000 | (frac << 13)
    } else {
        let exp32 = exp + (127 - 15);
        sign | (exp32 << 23) | (frac << 13)
    };

    f32::from_bits(out)
}

pub unsafe fn embed_tokens_hc_tensor(
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
    if out_hc.is_null() || tokens.is_null() || model_map.is_null() || n_vocab == 0 || n_tokens == 0 || n_embd == 0 || n_hc == 0 {
        return 0;
    }

    let src_row_bytes = (n_embd as u64).saturating_mul(2);
    let weight_bytes = (n_vocab as u64).saturating_mul(src_row_bytes);
    if weight_offset > model_size || weight_bytes > model_size.saturating_sub(weight_offset) {
        return 0;
    }

    let token_count = n_tokens as usize;
    let mut token_ids = vec![0i32; token_count];
    let read_ok = unsafe {
        ffi::ds4_metal_tensor_read(
            tokens,
            0,
            token_ids.as_mut_ptr() as *mut c_void,
            (token_count as u64).saturating_mul(4),
        )
    };
    if read_ok == 0 {
        return 0;
    }

    let embd_len = n_embd as usize;
    let hc_len = n_hc as usize;
    let one_token_out = embd_len.saturating_mul(hc_len);
    let mut out = vec![0f32; one_token_out.saturating_mul(token_count)];
    let model_ptr = model_map as *const u8;

    for (tok_idx, tok) in token_ids.iter().enumerate() {
        if *tok < 0 || (*tok as u32) >= n_vocab {
            return 0;
        }
        let row_offset = weight_offset.saturating_add((*tok as u64).saturating_mul(src_row_bytes));
        if row_offset > model_size || src_row_bytes > model_size.saturating_sub(row_offset) {
            return 0;
        }

        let src = unsafe {
            std::slice::from_raw_parts(
                model_ptr.add(row_offset as usize),
                src_row_bytes as usize,
            )
        };

        let token_base = tok_idx * one_token_out;
        for h in 0..hc_len {
            let dst_start = token_base + h * embd_len;
            let dst = &mut out[dst_start..dst_start + embd_len];
            for (i, v) in dst.iter_mut().enumerate() {
                let b0 = src[i * 2];
                let b1 = src[i * 2 + 1];
                *v = f16_to_f32_bits(u16::from_le_bytes([b0, b1]));
            }
        }
    }

    unsafe {
        ffi::ds4_metal_tensor_write(
            out_hc,
            0,
            out.as_ptr() as *const c_void,
            (out.len() as u64).saturating_mul(4),
        )
    }
}

// ============================================================================
// MATRIX MULTIPLICATION KERNELS
// ============================================================================

ffi_wrapper!(
    matmul_q8_0_tensor,
    ffi::ds4_metal_matmul_q8_0_tensor,
    out: *mut ffi::ds4_metal_tensor,
    model_map: *const c_void,
    model_size: u64,
    weight_offset: u64,
    in_dim: u64,
    out_dim: u64,
    x: *const ffi::ds4_metal_tensor,
    n_tok: u64
);

ffi_wrapper!(
    matmul_f16_tensor,
    ffi::ds4_metal_matmul_f16_tensor,
    out: *mut ffi::ds4_metal_tensor,
    model_map: *const c_void,
    model_size: u64,
    weight_offset: u64,
    in_dim: u64,
    out_dim: u64,
    x: *const ffi::ds4_metal_tensor,
    n_tok: u64
);

ffi_wrapper!(
    matmul_f16_pair_tensor,
    ffi::ds4_metal_matmul_f16_pair_tensor,
    out_a: *mut ffi::ds4_metal_tensor,
    out_b: *mut ffi::ds4_metal_tensor,
    model_map: *const c_void,
    model_size: u64,
    weight_a_offset: u64,
    weight_b_offset: u64,
    in_dim: u64,
    out_dim: u64,
    x: *const ffi::ds4_metal_tensor,
    n_tok: u64
);

ffi_wrapper!(
    matmul_f32_tensor,
    ffi::ds4_metal_matmul_f32_tensor,
    out: *mut ffi::ds4_metal_tensor,
    model_map: *const c_void,
    model_size: u64,
    weight_offset: u64,
    in_dim: u64,
    out_dim: u64,
    x: *const ffi::ds4_metal_tensor,
    n_tok: u64
);

ffi_wrapper!(
    matmul_q8_0_hc_expand_tensor,
    ffi::ds4_metal_matmul_q8_0_hc_expand_tensor,
    out_hc: *mut ffi::ds4_metal_tensor,
    block_out: *mut ffi::ds4_metal_tensor,
    model_map: *const c_void,
    model_size: u64,
    weight_offset: u64,
    in_dim: u64,
    out_dim: u64,
    x: *const ffi::ds4_metal_tensor,
    residual_hc: *const ffi::ds4_metal_tensor,
    split: *const ffi::ds4_metal_tensor,
    n_embd: u32,
    n_hc: u32
);

// ============================================================================
// NORMALIZATION KERNELS
// ============================================================================

ffi_wrapper!(
    rms_norm_plain_rows_tensor,
    ffi::ds4_metal_rms_norm_plain_rows_tensor,
    out: *mut ffi::ds4_metal_tensor,
    x: *const ffi::ds4_metal_tensor,
    n: u32,
    rows: u32,
    eps: std::os::raw::c_float
);

ffi_wrapper!(
    rms_norm_weight_tensor,
    ffi::ds4_metal_rms_norm_weight_tensor,
    out: *mut ffi::ds4_metal_tensor,
    x: *const ffi::ds4_metal_tensor,
    model_map: *const c_void,
    model_size: u64,
    weight_offset: u64,
    n: u32,
    eps: std::os::raw::c_float
);

ffi_wrapper!(
    rms_norm_weight_rows_tensor,
    ffi::ds4_metal_rms_norm_weight_rows_tensor,
    out: *mut ffi::ds4_metal_tensor,
    x: *const ffi::ds4_metal_tensor,
    model_map: *const c_void,
    model_size: u64,
    weight_offset: u64,
    n: u32,
    rows: u32,
    eps: std::os::raw::c_float
);

ffi_wrapper!(
    dsv4_qkv_rms_norm_rows_tensor,
    ffi::ds4_metal_dsv4_qkv_rms_norm_rows_tensor,
    q_out: *mut ffi::ds4_metal_tensor,
    q: *const ffi::ds4_metal_tensor,
    model_map: *const c_void,
    model_size: u64,
    q_weight_offset: u64,
    q_n: u32,
    kv_out: *mut ffi::ds4_metal_tensor,
    kv: *const ffi::ds4_metal_tensor,
    kv_weight_offset: u64,
    kv_n: u32,
    rows: u32,
    eps: std::os::raw::c_float
);

ffi_wrapper!(
    head_rms_norm_tensor,
    ffi::ds4_metal_head_rms_norm_tensor,
    x: *mut ffi::ds4_metal_tensor,
    n_tok: u32,
    n_head: u32,
    head_dim: u32,
    eps: std::os::raw::c_float
);

// ============================================================================
// POSITIONAL ENCODING & ROPE KERNELS
// ============================================================================

ffi_wrapper!(
    rope_tail_tensor,
    ffi::ds4_metal_rope_tail_tensor,
    x: *mut ffi::ds4_metal_tensor,
    n_tok: u32,
    n_head: u32,
    head_dim: u32,
    n_rot: u32,
    pos0: u32,
    n_ctx_orig: u32,
    inverse: bool,
    freq_base: std::os::raw::c_float,
    freq_scale: std::os::raw::c_float,
    ext_factor: std::os::raw::c_float,
    attn_factor: std::os::raw::c_float,
    beta_fast: std::os::raw::c_float,
    beta_slow: std::os::raw::c_float
);

ffi_wrapper!(
    dsv4_fp8_kv_quantize_tensor,
    ffi::ds4_metal_dsv4_fp8_kv_quantize_tensor,
    x: *mut ffi::ds4_metal_tensor,
    n_tok: u32,
    head_dim: u32,
    n_rot: u32
);

// ============================================================================
// BASIC ARITHMETIC KERNELS
// ============================================================================

ffi_wrapper!(
    add_tensor,
    ffi::ds4_metal_add_tensor,
    out: *mut ffi::ds4_metal_tensor,
    a: *const ffi::ds4_metal_tensor,
    b: *const ffi::ds4_metal_tensor,
    n: u32
);

ffi_wrapper!(
    swiglu_tensor,
    ffi::ds4_metal_swiglu_tensor,
    out: *mut ffi::ds4_metal_tensor,
    gate: *const ffi::ds4_metal_tensor,
    up: *const ffi::ds4_metal_tensor,
    n: u32,
    clamp: std::os::raw::c_float,
    weight: std::os::raw::c_float
);

ffi_wrapper!(
    repeat_hc_tensor,
    ffi::ds4_metal_repeat_hc_tensor,
    out: *mut ffi::ds4_metal_tensor,
    row: *const ffi::ds4_metal_tensor,
    n_embd: u32,
    n_hc: u32
);

// ============================================================================
// KV CACHE STORAGE KERNELS
// ============================================================================

ffi_wrapper!(
    kv_fp8_store_raw_tensor,
    ffi::ds4_metal_kv_fp8_store_raw_tensor,
    kv: *mut ffi::ds4_metal_tensor,
    raw_cache: *mut ffi::ds4_metal_tensor,
    raw_cap: u32,
    row: u32,
    head_dim: u32,
    n_rot: u32
);

ffi_wrapper!(
    store_raw_kv_tensor,
    ffi::ds4_metal_store_raw_kv_tensor,
    raw_cache: *mut ffi::ds4_metal_tensor,
    kv: *const ffi::ds4_metal_tensor,
    raw_cap: u32,
    row: u32,
    head_dim: u32
);

ffi_wrapper!(
    store_raw_kv_batch_tensor,
    ffi::ds4_metal_store_raw_kv_batch_tensor,
    raw_cache: *mut ffi::ds4_metal_tensor,
    kv: *const ffi::ds4_metal_tensor,
    raw_cap: u32,
    pos0: u32,
    n_tokens: u32,
    head_dim: u32
);

// ============================================================================
// ATTENTION KERNELS
// ============================================================================

ffi_wrapper!(
    attention_decode_heads_tensor,
    ffi::ds4_metal_attention_decode_heads_tensor,
    heads: *mut ffi::ds4_metal_tensor,
    model_map: *const c_void,
    model_size: u64,
    sinks_offset: u64,
    q: *const ffi::ds4_metal_tensor,
    raw_kv: *const ffi::ds4_metal_tensor,
    n_raw: u32,
    raw_cap: u32,
    raw_start: u32,
    comp_kv: *const ffi::ds4_metal_tensor,
    n_comp: u32,
    comp_mask: *const ffi::ds4_metal_tensor,
    use_mask: u32,
    n_head: u32,
    head_dim: u32
);

ffi_wrapper!(
    attention_prefill_raw_heads_tensor,
    ffi::ds4_metal_attention_prefill_raw_heads_tensor,
    heads: *mut ffi::ds4_metal_tensor,
    model_map: *const c_void,
    model_size: u64,
    sinks_offset: u64,
    q: *const ffi::ds4_metal_tensor,
    raw_kv: *const ffi::ds4_metal_tensor,
    n_tokens: u32,
    window: u32,
    n_head: u32,
    head_dim: u32
);

ffi_wrapper!(
    attention_decode_raw_batch_heads_tensor,
    ffi::ds4_metal_attention_decode_raw_batch_heads_tensor,
    heads: *mut ffi::ds4_metal_tensor,
    model_map: *const c_void,
    model_size: u64,
    sinks_offset: u64,
    q: *const ffi::ds4_metal_tensor,
    raw_kv: *const ffi::ds4_metal_tensor,
    n_tokens: u32,
    pos0: u32,
    n_raw: u32,
    raw_cap: u32,
    raw_start: u32,
    window: u32,
    n_head: u32,
    head_dim: u32
);

ffi_wrapper!(
    attention_decode_mixed_batch_heads_tensor,
    ffi::ds4_metal_attention_decode_mixed_batch_heads_tensor,
    heads: *mut ffi::ds4_metal_tensor,
    model_map: *const c_void,
    model_size: u64,
    sinks_offset: u64,
    q: *const ffi::ds4_metal_tensor,
    raw_kv: *const ffi::ds4_metal_tensor,
    comp_kv: *const ffi::ds4_metal_tensor,
    comp_mask: *const ffi::ds4_metal_tensor,
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
    head_dim: u32
);

ffi_wrapper!(
    attention_indexed_mixed_batch_heads_tensor,
    ffi::ds4_metal_attention_indexed_mixed_batch_heads_tensor,
    heads: *mut ffi::ds4_metal_tensor,
    model_map: *const c_void,
    model_size: u64,
    sinks_offset: u64,
    q: *const ffi::ds4_metal_tensor,
    raw_kv: *const ffi::ds4_metal_tensor,
    comp_kv: *const ffi::ds4_metal_tensor,
    topk: *const ffi::ds4_metal_tensor,
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
    head_dim: u32
);

ffi_wrapper!(
    attention_prefill_static_mixed_heads_tensor,
    ffi::ds4_metal_attention_prefill_static_mixed_heads_tensor,
    heads: *mut ffi::ds4_metal_tensor,
    model_map: *const c_void,
    model_size: u64,
    sinks_offset: u64,
    q: *const ffi::ds4_metal_tensor,
    raw_kv: *const ffi::ds4_metal_tensor,
    comp_kv: *const ffi::ds4_metal_tensor,
    n_tokens: u32,
    n_comp: u32,
    window: u32,
    ratio: u32,
    n_head: u32,
    head_dim: u32
);

ffi_wrapper!(
    attention_prefill_masked_mixed_heads_tensor,
    ffi::ds4_metal_attention_prefill_masked_mixed_heads_tensor,
    heads: *mut ffi::ds4_metal_tensor,
    model_map: *const c_void,
    model_size: u64,
    sinks_offset: u64,
    q: *const ffi::ds4_metal_tensor,
    raw_kv: *const ffi::ds4_metal_tensor,
    comp_kv: *const ffi::ds4_metal_tensor,
    comp_mask: *const ffi::ds4_metal_tensor,
    n_tokens: u32,
    n_comp: u32,
    window: u32,
    ratio: u32,
    n_head: u32,
    head_dim: u32
);

ffi_wrapper!(
    attention_output_q8_batch_tensor,
    ffi::ds4_metal_attention_output_q8_batch_tensor,
    out: *mut ffi::ds4_metal_tensor,
    low: *mut ffi::ds4_metal_tensor,
    group_tmp: *mut ffi::ds4_metal_tensor,
    low_tmp: *mut ffi::ds4_metal_tensor,
    model_map: *const c_void,
    model_size: u64,
    out_a_offset: u64,
    out_b_offset: u64,
    group_dim: u64,
    rank: u64,
    n_groups: u32,
    out_dim: u64,
    heads: *const ffi::ds4_metal_tensor,
    n_tokens: u32
);

ffi_wrapper!(
    attention_output_low_q8_tensor,
    ffi::ds4_metal_attention_output_low_q8_tensor,
    low: *mut ffi::ds4_metal_tensor,
    model_map: *const c_void,
    model_size: u64,
    out_a_offset: u64,
    group_dim: u64,
    rank: u64,
    n_groups: u32,
    heads: *const ffi::ds4_metal_tensor
);

// ============================================================================
// MoE & FFN KERNELS
// ============================================================================

ffi_wrapper!(
    shared_gate_up_swiglu_q8_0_tensor,
    ffi::ds4_metal_shared_gate_up_swiglu_q8_0_tensor,
    gate: *mut ffi::ds4_metal_tensor,
    up: *mut ffi::ds4_metal_tensor,
    mid: *mut ffi::ds4_metal_tensor,
    model_map: *const c_void,
    model_size: u64,
    gate_offset: u64,
    up_offset: u64,
    in_dim: u64,
    out_dim: u64,
    x: *const ffi::ds4_metal_tensor
);

ffi_wrapper!(
    shared_down_hc_expand_q8_0_tensor,
    ffi::ds4_metal_shared_down_hc_expand_q8_0_tensor,
    out_hc: *mut ffi::ds4_metal_tensor,
    shared_out: *mut ffi::ds4_metal_tensor,
    model_map: *const c_void,
    model_size: u64,
    weight_offset: u64,
    in_dim: u64,
    out_dim: u64,
    shared_mid: *const ffi::ds4_metal_tensor,
    routed_out: *const ffi::ds4_metal_tensor,
    residual_hc: *const ffi::ds4_metal_tensor,
    split: *const ffi::ds4_metal_tensor,
    n_embd: u32,
    n_hc: u32
);

ffi_wrapper!(
    router_select_tensor,
    ffi::ds4_metal_router_select_tensor,
    selected: *mut ffi::ds4_metal_tensor,
    weights: *mut ffi::ds4_metal_tensor,
    probs: *mut ffi::ds4_metal_tensor,
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
    logits: *const ffi::ds4_metal_tensor
);

ffi_wrapper!(
    router_select_batch_tensor,
    ffi::ds4_metal_router_select_batch_tensor,
    selected: *mut ffi::ds4_metal_tensor,
    weights: *mut ffi::ds4_metal_tensor,
    probs: *mut ffi::ds4_metal_tensor,
    model_map: *const c_void,
    model_size: u64,
    bias_offset: u64,
    hash_offset: u64,
    hash_rows: u32,
    n_expert_groups: u32,
    n_group_used: u32,
    has_bias: std::os::raw::c_int,
    hash_mode: std::os::raw::c_int,
    logits: *const ffi::ds4_metal_tensor,
    tokens: *const ffi::ds4_metal_tensor,
    n_tokens: u32
);

ffi_wrapper!(
    routed_moe_one_tensor,
    ffi::ds4_metal_routed_moe_one_tensor,
    out: *mut ffi::ds4_metal_tensor,
    gate: *mut ffi::ds4_metal_tensor,
    up: *mut ffi::ds4_metal_tensor,
    mid: *mut ffi::ds4_metal_tensor,
    experts: *mut ffi::ds4_metal_tensor,
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
    selected: *const ffi::ds4_metal_tensor,
    weights: *const ffi::ds4_metal_tensor,
    n_expert: u32,
    clamp: std::os::raw::c_float,
    x: *const ffi::ds4_metal_tensor
);

ffi_wrapper!(
    routed_moe_batch_tensor,
    ffi::ds4_metal_routed_moe_batch_tensor,
    out: *mut ffi::ds4_metal_tensor,
    gate: *mut ffi::ds4_metal_tensor,
    up: *mut ffi::ds4_metal_tensor,
    mid: *mut ffi::ds4_metal_tensor,
    experts: *mut ffi::ds4_metal_tensor,
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
    selected: *const ffi::ds4_metal_tensor,
    weights: *const ffi::ds4_metal_tensor,
    n_expert: u32,
    clamp: std::os::raw::c_float,
    x: *const ffi::ds4_metal_tensor,
    n_tokens: u32
);

// ============================================================================
// HEAD-CONTROL (HC) FUSION KERNELS
// ============================================================================

ffi_wrapper!(
    hc_split_sinkhorn_tensor,
    ffi::ds4_metal_hc_split_sinkhorn_tensor,
    out: *mut ffi::ds4_metal_tensor,
    mix: *const ffi::ds4_metal_tensor,
    model_map: *const c_void,
    model_size: u64,
    scale_offset: u64,
    base_offset: u64,
    n_hc: u32,
    sinkhorn_iters: u32,
    eps: std::os::raw::c_float
);

ffi_wrapper!(
    hc_weighted_sum_tensor,
    ffi::ds4_metal_hc_weighted_sum_tensor,
    out: *mut ffi::ds4_metal_tensor,
    residual_hc: *const ffi::ds4_metal_tensor,
    weights: *const ffi::ds4_metal_tensor,
    n_embd: u32,
    n_hc: u32
);

ffi_wrapper!(
    hc_weighted_sum_split_tensor,
    ffi::ds4_metal_hc_weighted_sum_split_tensor,
    out: *mut ffi::ds4_metal_tensor,
    residual_hc: *const ffi::ds4_metal_tensor,
    split: *const ffi::ds4_metal_tensor,
    n_embd: u32,
    n_hc: u32
);

ffi_wrapper!(
    hc_split_weighted_sum_tensor,
    ffi::ds4_metal_hc_split_weighted_sum_tensor,
    out: *mut ffi::ds4_metal_tensor,
    split: *mut ffi::ds4_metal_tensor,
    mix: *const ffi::ds4_metal_tensor,
    residual_hc: *const ffi::ds4_metal_tensor,
    model_map: *const c_void,
    model_size: u64,
    scale_offset: u64,
    base_offset: u64,
    n_embd: u32,
    n_hc: u32,
    sinkhorn_iters: u32,
    eps: std::os::raw::c_float
);

ffi_wrapper!(
    hc_split_weighted_sum_norm_tensor,
    ffi::ds4_metal_hc_split_weighted_sum_norm_tensor,
    out: *mut ffi::ds4_metal_tensor,
    norm_out: *mut ffi::ds4_metal_tensor,
    split: *mut ffi::ds4_metal_tensor,
    mix: *const ffi::ds4_metal_tensor,
    residual_hc: *const ffi::ds4_metal_tensor,
    model_map: *const c_void,
    model_size: u64,
    scale_offset: u64,
    base_offset: u64,
    norm_weight_offset: u64,
    n_embd: u32,
    n_hc: u32,
    sinkhorn_iters: u32,
    eps: std::os::raw::c_float,
    norm_eps: std::os::raw::c_float
);

ffi_wrapper!(
    output_hc_weights_tensor,
    ffi::ds4_metal_output_hc_weights_tensor,
    out: *mut ffi::ds4_metal_tensor,
    pre: *const ffi::ds4_metal_tensor,
    model_map: *const c_void,
    model_size: u64,
    scale_offset: u64,
    base_offset: u64,
    n_hc: u32,
    eps: std::os::raw::c_float
);

ffi_wrapper!(
    hc_expand_tensor,
    ffi::ds4_metal_hc_expand_tensor,
    out_hc: *mut ffi::ds4_metal_tensor,
    block_out: *const ffi::ds4_metal_tensor,
    residual_hc: *const ffi::ds4_metal_tensor,
    post: *const ffi::ds4_metal_tensor,
    comb: *const ffi::ds4_metal_tensor,
    n_embd: u32,
    n_hc: u32
);

ffi_wrapper!(
    hc_expand_split_tensor,
    ffi::ds4_metal_hc_expand_split_tensor,
    out_hc: *mut ffi::ds4_metal_tensor,
    block_out: *const ffi::ds4_metal_tensor,
    residual_hc: *const ffi::ds4_metal_tensor,
    split: *const ffi::ds4_metal_tensor,
    n_embd: u32,
    n_hc: u32
);

ffi_wrapper!(
    hc_expand_add_split_tensor,
    ffi::ds4_metal_hc_expand_add_split_tensor,
    out_hc: *mut ffi::ds4_metal_tensor,
    block_out: *const ffi::ds4_metal_tensor,
    block_add: *const ffi::ds4_metal_tensor,
    residual_hc: *const ffi::ds4_metal_tensor,
    split: *const ffi::ds4_metal_tensor,
    n_embd: u32,
    n_hc: u32
);

// ============================================================================
// COMPRESSOR & INDEXER KERNELS (ATTENTION OPTIMIZATION)
// ============================================================================

ffi_wrapper!(
    compressor_update_tensor,
    ffi::ds4_metal_compressor_update_tensor,
    kv_cur: *const ffi::ds4_metal_tensor,
    sc_cur: *const ffi::ds4_metal_tensor,
    state_kv: *mut ffi::ds4_metal_tensor,
    state_score: *mut ffi::ds4_metal_tensor,
    comp_cache: *mut ffi::ds4_metal_tensor,
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
    freq_base: std::os::raw::c_float,
    freq_scale: std::os::raw::c_float,
    ext_factor: std::os::raw::c_float,
    attn_factor: std::os::raw::c_float,
    beta_fast: std::os::raw::c_float,
    beta_slow: std::os::raw::c_float,
    rms_eps: std::os::raw::c_float
);

ffi_wrapper!(
    compressor_store_batch_tensor,
    ffi::ds4_metal_compressor_store_batch_tensor,
    kv: *const ffi::ds4_metal_tensor,
    sc: *const ffi::ds4_metal_tensor,
    state_kv: *mut ffi::ds4_metal_tensor,
    state_score: *mut ffi::ds4_metal_tensor,
    model_map: *const c_void,
    model_size: u64,
    ape_offset: u64,
    ape_type: u32,
    head_dim: u32,
    ratio: u32,
    pos0: u32,
    n_tokens: u32
);

ffi_wrapper!(
    compressor_prefill_tensor,
    ffi::ds4_metal_compressor_prefill_tensor,
    comp_cache: *mut ffi::ds4_metal_tensor,
    state_kv: *mut ffi::ds4_metal_tensor,
    state_score: *mut ffi::ds4_metal_tensor,
    kv: *const ffi::ds4_metal_tensor,
    sc: *const ffi::ds4_metal_tensor,
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
    quantize_fp8: std::os::raw::c_int,
    freq_base: std::os::raw::c_float,
    freq_scale: std::os::raw::c_float,
    ext_factor: std::os::raw::c_float,
    attn_factor: std::os::raw::c_float,
    beta_fast: std::os::raw::c_float,
    beta_slow: std::os::raw::c_float,
    rms_eps: std::os::raw::c_float
);

ffi_wrapper!(
    compressor_prefill_ratio4_replay_tensor,
    ffi::ds4_metal_compressor_prefill_ratio4_replay_tensor,
    comp_cache: *mut ffi::ds4_metal_tensor,
    state_kv: *mut ffi::ds4_metal_tensor,
    state_score: *mut ffi::ds4_metal_tensor,
    kv: *const ffi::ds4_metal_tensor,
    sc: *const ffi::ds4_metal_tensor,
    model_map: *const c_void,
    model_size: u64,
    ape_offset: u64,
    ape_type: u32,
    norm_offset: u64,
    norm_type: u32,
    head_dim: u32,
    pos0: u32,
    n_tokens: u32,
    n_rot: u32,
    n_ctx_orig: u32,
    quantize_fp8: std::os::raw::c_int,
    freq_base: std::os::raw::c_float,
    freq_scale: std::os::raw::c_float,
    ext_factor: std::os::raw::c_float,
    attn_factor: std::os::raw::c_float,
    beta_fast: std::os::raw::c_float,
    beta_slow: std::os::raw::c_float,
    rms_eps: std::os::raw::c_float
);

ffi_wrapper!(
    compressor_prefill_state_ratio4_tensor,
    ffi::ds4_metal_compressor_prefill_state_ratio4_tensor,
    state_kv: *mut ffi::ds4_metal_tensor,
    state_score: *mut ffi::ds4_metal_tensor,
    kv_tail: *const ffi::ds4_metal_tensor,
    sc_tail: *const ffi::ds4_metal_tensor,
    model_map: *const c_void,
    model_size: u64,
    ape_offset: u64,
    ape_type: u32,
    head_dim: u32,
    pos0: u32
);

ffi_wrapper!(
    indexer_score_one_tensor,
    ffi::ds4_metal_indexer_score_one_tensor,
    scores: *mut ffi::ds4_metal_tensor,
    q: *const ffi::ds4_metal_tensor,
    weights: *const ffi::ds4_metal_tensor,
    index_comp: *const ffi::ds4_metal_tensor,
    n_comp: u32,
    n_head: u32,
    head_dim: u32,
    scale: std::os::raw::c_float
);

ffi_wrapper!(
    indexer_scores_prefill_tensor,
    ffi::ds4_metal_indexer_scores_prefill_tensor,
    scores: *mut ffi::ds4_metal_tensor,
    q: *const ffi::ds4_metal_tensor,
    weights: *const ffi::ds4_metal_tensor,
    index_comp: *const ffi::ds4_metal_tensor,
    n_comp: u32,
    n_tokens: u32,
    n_head: u32,
    head_dim: u32,
    ratio: u32,
    scale: std::os::raw::c_float
);

ffi_wrapper!(
    indexer_scores_decode_batch_tensor,
    ffi::ds4_metal_indexer_scores_decode_batch_tensor,
    scores: *mut ffi::ds4_metal_tensor,
    q: *const ffi::ds4_metal_tensor,
    weights: *const ffi::ds4_metal_tensor,
    index_comp: *const ffi::ds4_metal_tensor,
    n_comp: u32,
    n_tokens: u32,
    pos0: u32,
    n_head: u32,
    head_dim: u32,
    ratio: u32,
    scale: std::os::raw::c_float
);

ffi_wrapper!(
    indexer_topk_tensor,
    ffi::ds4_metal_indexer_topk_tensor,
    selected: *mut ffi::ds4_metal_tensor,
    scores: *const ffi::ds4_metal_tensor,
    n_comp: u32,
    n_tokens: u32,
    top_k: u32
);

ffi_wrapper!(
    dsv4_topk_mask_tensor,
    ffi::ds4_metal_dsv4_topk_mask_tensor,
    mask: *mut ffi::ds4_metal_tensor,
    topk: *const ffi::ds4_metal_tensor,
    n_comp: u32,
    n_tokens: u32,
    top_k: u32
);
