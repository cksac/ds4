#![allow(dead_code, non_camel_case_types)]

use libc::{c_char, c_float, c_int, c_void};

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ds4_context_memory {
    pub total_bytes: u64,
    pub raw_bytes: u64,
    pub compressed_bytes: u64,
    pub scratch_bytes: u64,
    pub prefill_cap: u32,
    pub raw_cap: u32,
    pub comp_cap: u32,
}

// ---------------------------------------------------------------------------
// Metal tensor and execution primitives.
// ---------------------------------------------------------------------------

#[repr(C)]
pub struct ds4_metal_tensor {
    _private: [u8; 0],
}

extern "C" {
    pub fn ds4_metal_init() -> c_int;
    pub fn ds4_metal_cleanup();
    pub fn ds4_metal_get_device() -> *mut c_void;  // Returns id<MTLDevice>
    pub fn ds4_metal_get_queue() -> *mut c_void;   // Returns id<MTLCommandQueue>
    pub fn ds4_metal_buffer_alloc(bytes: u64) -> *mut c_void;
    pub fn ds4_metal_tensor_bind_owned_buffer(buffer: *mut c_void, bytes: u64) -> *mut ds4_metal_tensor;
    pub fn ds4_metal_tensor_wrap_buffer(buffer: *mut c_void, offset: u64, bytes: u64) -> *mut ds4_metal_tensor;
    pub fn ds4_metal_tensor_alloc(bytes: u64) -> *mut ds4_metal_tensor;
    pub fn ds4_metal_tensor_view(base: *const ds4_metal_tensor, offset: u64, bytes: u64) -> *mut ds4_metal_tensor;
    pub fn ds4_metal_tensor_free(tensor: *mut ds4_metal_tensor);
    pub fn ds4_metal_tensor_bytes(tensor: *const ds4_metal_tensor) -> u64;
    pub fn ds4_metal_tensor_write(tensor: *mut ds4_metal_tensor, offset: u64, data: *const c_void, bytes: u64) -> c_int;
    pub fn ds4_metal_tensor_read(tensor: *const ds4_metal_tensor, offset: u64, data: *mut c_void, bytes: u64) -> c_int;
    pub fn ds4_metal_begin_commands() -> c_int;
    pub fn ds4_metal_end_commands() -> c_int;
    pub fn ds4_metal_synchronize() -> c_int;
    pub fn ds4_metal_set_model_map_range(model_map: *const c_void, model_size: u64, map_offset: u64, map_size: u64) -> c_int;
    pub fn ds4_metal_set_quality(quality: c_int);
    pub fn ds4_metal_print_memory_report(label: *const c_char);
    pub fn ds4_metal_embed_token_hc_tensor(out_hc: *mut ds4_metal_tensor, model_map: *const c_void, model_size: u64, weight_offset: u64, n_vocab: u32, token: u32, n_embd: u32, n_hc: u32) -> c_int;
    pub fn ds4_metal_embed_tokens_hc_tensor(out_hc: *mut ds4_metal_tensor, tokens: *const ds4_metal_tensor, model_map: *const c_void, model_size: u64, weight_offset: u64, n_vocab: u32, n_tokens: u32, n_embd: u32, n_hc: u32) -> c_int;
    pub fn ds4_metal_indexer_score_one_tensor(scores: *mut ds4_metal_tensor, q: *const ds4_metal_tensor, weights: *const ds4_metal_tensor, index_comp: *const ds4_metal_tensor, n_comp: u32, n_head: u32, head_dim: u32, scale: c_float) -> c_int;
    pub fn ds4_metal_indexer_scores_prefill_tensor(scores: *mut ds4_metal_tensor, q: *const ds4_metal_tensor, weights: *const ds4_metal_tensor, index_comp: *const ds4_metal_tensor, n_comp: u32, n_tokens: u32, n_head: u32, head_dim: u32, ratio: u32, scale: c_float) -> c_int;
    pub fn ds4_metal_indexer_scores_decode_batch_tensor(scores: *mut ds4_metal_tensor, q: *const ds4_metal_tensor, weights: *const ds4_metal_tensor, index_comp: *const ds4_metal_tensor, n_comp: u32, n_tokens: u32, pos0: u32, n_head: u32, head_dim: u32, ratio: u32, scale: c_float) -> c_int;
    pub fn ds4_metal_indexer_topk_tensor(selected: *mut ds4_metal_tensor, scores: *const ds4_metal_tensor, n_comp: u32, n_tokens: u32, top_k: u32) -> c_int;
    pub fn ds4_metal_dsv4_topk_mask_tensor(mask: *mut ds4_metal_tensor, topk: *const ds4_metal_tensor, n_comp: u32, n_tokens: u32, top_k: u32) -> c_int;
    pub fn ds4_metal_matmul_q8_0_tensor(out: *mut ds4_metal_tensor, model_map: *const c_void, model_size: u64, weight_offset: u64, in_dim: u64, out_dim: u64, x: *const ds4_metal_tensor, n_tok: u64) -> c_int;
    pub fn ds4_metal_shared_gate_up_swiglu_q8_0_tensor(gate: *mut ds4_metal_tensor, up: *mut ds4_metal_tensor, mid: *mut ds4_metal_tensor, model_map: *const c_void, model_size: u64, gate_offset: u64, up_offset: u64, in_dim: u64, out_dim: u64, x: *const ds4_metal_tensor) -> c_int;
    pub fn ds4_metal_matmul_f16_tensor(out: *mut ds4_metal_tensor, model_map: *const c_void, model_size: u64, weight_offset: u64, in_dim: u64, out_dim: u64, x: *const ds4_metal_tensor, n_tok: u64) -> c_int;
    pub fn ds4_metal_matmul_f16_pair_tensor(out_a: *mut ds4_metal_tensor, out_b: *mut ds4_metal_tensor, model_map: *const c_void, model_size: u64, weight_a_offset: u64, weight_b_offset: u64, in_dim: u64, out_dim: u64, x: *const ds4_metal_tensor, n_tok: u64) -> c_int;
    pub fn ds4_metal_matmul_f32_tensor(out: *mut ds4_metal_tensor, model_map: *const c_void, model_size: u64, weight_offset: u64, in_dim: u64, out_dim: u64, x: *const ds4_metal_tensor, n_tok: u64) -> c_int;
    pub fn ds4_metal_repeat_hc_tensor(out: *mut ds4_metal_tensor, row: *const ds4_metal_tensor, n_embd: u32, n_hc: u32) -> c_int;
    pub fn ds4_metal_rms_norm_plain_rows_tensor(out: *mut ds4_metal_tensor, x: *const ds4_metal_tensor, n: u32, rows: u32, eps: c_float) -> c_int;
    pub fn ds4_metal_rms_norm_weight_tensor(out: *mut ds4_metal_tensor, x: *const ds4_metal_tensor, model_map: *const c_void, model_size: u64, weight_offset: u64, n: u32, eps: c_float) -> c_int;
    pub fn ds4_metal_rms_norm_weight_rows_tensor(out: *mut ds4_metal_tensor, x: *const ds4_metal_tensor, model_map: *const c_void, model_size: u64, weight_offset: u64, n: u32, rows: u32, eps: c_float) -> c_int;
    pub fn ds4_metal_dsv4_qkv_rms_norm_rows_tensor(q_out: *mut ds4_metal_tensor, q: *const ds4_metal_tensor, model_map: *const c_void, model_size: u64, q_weight_offset: u64, q_n: u32, kv_out: *mut ds4_metal_tensor, kv: *const ds4_metal_tensor, kv_weight_offset: u64, kv_n: u32, rows: u32, eps: c_float) -> c_int;
    pub fn ds4_metal_head_rms_norm_tensor(x: *mut ds4_metal_tensor, n_tok: u32, n_head: u32, head_dim: u32, eps: c_float) -> c_int;
    pub fn ds4_metal_dsv4_fp8_kv_quantize_tensor(x: *mut ds4_metal_tensor, n_tok: u32, head_dim: u32, n_rot: u32) -> c_int;
    pub fn ds4_metal_rope_tail_tensor(x: *mut ds4_metal_tensor, n_tok: u32, n_head: u32, head_dim: u32, n_rot: u32, pos0: u32, n_ctx_orig: u32, inverse: bool, freq_base: c_float, freq_scale: c_float, ext_factor: c_float, attn_factor: c_float, beta_fast: c_float, beta_slow: c_float) -> c_int;
    pub fn ds4_metal_kv_fp8_store_raw_tensor(kv: *mut ds4_metal_tensor, raw_cache: *mut ds4_metal_tensor, raw_cap: u32, row: u32, head_dim: u32, n_rot: u32) -> c_int;
    pub fn ds4_metal_store_raw_kv_tensor(raw_cache: *mut ds4_metal_tensor, kv: *const ds4_metal_tensor, raw_cap: u32, row: u32, head_dim: u32) -> c_int;
    pub fn ds4_metal_store_raw_kv_batch_tensor(raw_cache: *mut ds4_metal_tensor, kv: *const ds4_metal_tensor, raw_cap: u32, pos0: u32, n_tokens: u32, head_dim: u32) -> c_int;
    pub fn ds4_metal_compressor_update_tensor(kv_cur: *const ds4_metal_tensor, sc_cur: *const ds4_metal_tensor, state_kv: *mut ds4_metal_tensor, state_score: *mut ds4_metal_tensor, comp_cache: *mut ds4_metal_tensor, model_map: *const c_void, model_size: u64, ape_offset: u64, ape_type: u32, norm_offset: u64, norm_type: u32, head_dim: u32, ratio: u32, pos: u32, comp_row: u32, n_rot: u32, n_ctx_orig: u32, freq_base: c_float, freq_scale: c_float, ext_factor: c_float, attn_factor: c_float, beta_fast: c_float, beta_slow: c_float, rms_eps: c_float) -> c_int;
    pub fn ds4_metal_compressor_store_batch_tensor(kv: *const ds4_metal_tensor, sc: *const ds4_metal_tensor, state_kv: *mut ds4_metal_tensor, state_score: *mut ds4_metal_tensor, model_map: *const c_void, model_size: u64, ape_offset: u64, ape_type: u32, head_dim: u32, ratio: u32, pos0: u32, n_tokens: u32) -> c_int;
    pub fn ds4_metal_compressor_prefill_tensor(comp_cache: *mut ds4_metal_tensor, state_kv: *mut ds4_metal_tensor, state_score: *mut ds4_metal_tensor, kv: *const ds4_metal_tensor, sc: *const ds4_metal_tensor, model_map: *const c_void, model_size: u64, ape_offset: u64, ape_type: u32, norm_offset: u64, norm_type: u32, head_dim: u32, ratio: u32, pos0: u32, n_tokens: u32, n_rot: u32, n_ctx_orig: u32, quantize_fp8: c_int, freq_base: c_float, freq_scale: c_float, ext_factor: c_float, attn_factor: c_float, beta_fast: c_float, beta_slow: c_float, rms_eps: c_float) -> c_int;
    pub fn ds4_metal_compressor_prefill_ratio4_replay_tensor(comp_cache: *mut ds4_metal_tensor, state_kv: *mut ds4_metal_tensor, state_score: *mut ds4_metal_tensor, kv: *const ds4_metal_tensor, sc: *const ds4_metal_tensor, model_map: *const c_void, model_size: u64, ape_offset: u64, ape_type: u32, norm_offset: u64, norm_type: u32, head_dim: u32, pos0: u32, n_tokens: u32, n_rot: u32, n_ctx_orig: u32, quantize_fp8: c_int, freq_base: c_float, freq_scale: c_float, ext_factor: c_float, attn_factor: c_float, beta_fast: c_float, beta_slow: c_float, rms_eps: c_float) -> c_int;
    pub fn ds4_metal_compressor_prefill_state_ratio4_tensor(state_kv: *mut ds4_metal_tensor, state_score: *mut ds4_metal_tensor, kv_tail: *const ds4_metal_tensor, sc_tail: *const ds4_metal_tensor, model_map: *const c_void, model_size: u64, ape_offset: u64, ape_type: u32, head_dim: u32, pos0: u32) -> c_int;
    pub fn ds4_metal_attention_decode_heads_tensor(heads: *mut ds4_metal_tensor, model_map: *const c_void, model_size: u64, sinks_offset: u64, q: *const ds4_metal_tensor, raw_kv: *const ds4_metal_tensor, n_raw: u32, raw_cap: u32, raw_start: u32, comp_kv: *const ds4_metal_tensor, n_comp: u32, comp_mask: *const ds4_metal_tensor, use_mask: u32, n_head: u32, head_dim: u32) -> c_int;
    pub fn ds4_metal_attention_prefill_raw_heads_tensor(heads: *mut ds4_metal_tensor, model_map: *const c_void, model_size: u64, sinks_offset: u64, q: *const ds4_metal_tensor, raw_kv: *const ds4_metal_tensor, n_tokens: u32, window: u32, n_head: u32, head_dim: u32) -> c_int;
    pub fn ds4_metal_attention_decode_raw_batch_heads_tensor(heads: *mut ds4_metal_tensor, model_map: *const c_void, model_size: u64, sinks_offset: u64, q: *const ds4_metal_tensor, raw_kv: *const ds4_metal_tensor, n_tokens: u32, pos0: u32, n_raw: u32, raw_cap: u32, raw_start: u32, window: u32, n_head: u32, head_dim: u32) -> c_int;
    pub fn ds4_metal_attention_decode_mixed_batch_heads_tensor(heads: *mut ds4_metal_tensor, model_map: *const c_void, model_size: u64, sinks_offset: u64, q: *const ds4_metal_tensor, raw_kv: *const ds4_metal_tensor, comp_kv: *const ds4_metal_tensor, comp_mask: *const ds4_metal_tensor, use_comp_mask: u32, n_tokens: u32, pos0: u32, n_raw: u32, raw_cap: u32, raw_start: u32, n_comp: u32, window: u32, ratio: u32, n_head: u32, head_dim: u32) -> c_int;
    pub fn ds4_metal_attention_indexed_mixed_batch_heads_tensor(heads: *mut ds4_metal_tensor, model_map: *const c_void, model_size: u64, sinks_offset: u64, q: *const ds4_metal_tensor, raw_kv: *const ds4_metal_tensor, comp_kv: *const ds4_metal_tensor, topk: *const ds4_metal_tensor, n_tokens: u32, pos0: u32, n_raw: u32, raw_cap: u32, raw_start: u32, n_comp: u32, top_k: u32, window: u32, ratio: u32, n_head: u32, head_dim: u32) -> c_int;
    pub fn ds4_metal_attention_prefill_static_mixed_heads_tensor(heads: *mut ds4_metal_tensor, model_map: *const c_void, model_size: u64, sinks_offset: u64, q: *const ds4_metal_tensor, raw_kv: *const ds4_metal_tensor, comp_kv: *const ds4_metal_tensor, n_tokens: u32, n_comp: u32, window: u32, ratio: u32, n_head: u32, head_dim: u32) -> c_int;
    pub fn ds4_metal_attention_prefill_masked_mixed_heads_tensor(heads: *mut ds4_metal_tensor, model_map: *const c_void, model_size: u64, sinks_offset: u64, q: *const ds4_metal_tensor, raw_kv: *const ds4_metal_tensor, comp_kv: *const ds4_metal_tensor, comp_mask: *const ds4_metal_tensor, n_tokens: u32, n_comp: u32, window: u32, ratio: u32, n_head: u32, head_dim: u32) -> c_int;
    pub fn ds4_metal_attention_output_q8_batch_tensor(out: *mut ds4_metal_tensor, low: *mut ds4_metal_tensor, group_tmp: *mut ds4_metal_tensor, low_tmp: *mut ds4_metal_tensor, model_map: *const c_void, model_size: u64, out_a_offset: u64, out_b_offset: u64, group_dim: u64, rank: u64, n_groups: u32, out_dim: u64, heads: *const ds4_metal_tensor, n_tokens: u32) -> c_int;
    pub fn ds4_metal_attention_output_low_q8_tensor(low: *mut ds4_metal_tensor, model_map: *const c_void, model_size: u64, out_a_offset: u64, group_dim: u64, rank: u64, n_groups: u32, heads: *const ds4_metal_tensor) -> c_int;
    pub fn ds4_metal_swiglu_tensor(out: *mut ds4_metal_tensor, gate: *const ds4_metal_tensor, up: *const ds4_metal_tensor, n: u32, clamp: c_float, weight: c_float) -> c_int;
    pub fn ds4_metal_add_tensor(out: *mut ds4_metal_tensor, a: *const ds4_metal_tensor, b: *const ds4_metal_tensor, n: u32) -> c_int;
    pub fn ds4_metal_router_select_tensor(selected: *mut ds4_metal_tensor, weights: *mut ds4_metal_tensor, probs: *mut ds4_metal_tensor, model_map: *const c_void, model_size: u64, bias_offset: u64, hash_offset: u64, hash_rows: u32, token: u32, n_expert_groups: u32, n_group_used: u32, has_bias: bool, hash_mode: bool, logits: *const ds4_metal_tensor) -> c_int;
    pub fn ds4_metal_router_select_batch_tensor(selected: *mut ds4_metal_tensor, weights: *mut ds4_metal_tensor, probs: *mut ds4_metal_tensor, model_map: *const c_void, model_size: u64, bias_offset: u64, hash_offset: u64, hash_rows: u32, n_expert_groups: u32, n_group_used: u32, has_bias: c_int, hash_mode: c_int, logits: *const ds4_metal_tensor, tokens: *const ds4_metal_tensor, n_tokens: u32) -> c_int;
    pub fn ds4_metal_routed_moe_one_tensor(out: *mut ds4_metal_tensor, gate: *mut ds4_metal_tensor, up: *mut ds4_metal_tensor, mid: *mut ds4_metal_tensor, experts: *mut ds4_metal_tensor, model_map: *const c_void, model_size: u64, gate_offset: u64, up_offset: u64, down_offset: u64, gate_type: u32, down_type: u32, gate_expert_bytes: u64, gate_row_bytes: u64, down_expert_bytes: u64, down_row_bytes: u64, expert_in_dim: u32, expert_mid_dim: u32, out_dim: u32, selected: *const ds4_metal_tensor, weights: *const ds4_metal_tensor, n_expert: u32, clamp: c_float, x: *const ds4_metal_tensor) -> c_int;
    pub fn ds4_metal_routed_moe_batch_tensor(out: *mut ds4_metal_tensor, gate: *mut ds4_metal_tensor, up: *mut ds4_metal_tensor, mid: *mut ds4_metal_tensor, experts: *mut ds4_metal_tensor, model_map: *const c_void, model_size: u64, gate_offset: u64, up_offset: u64, down_offset: u64, gate_type: u32, down_type: u32, gate_expert_bytes: u64, gate_row_bytes: u64, down_expert_bytes: u64, down_row_bytes: u64, expert_in_dim: u32, expert_mid_dim: u32, out_dim: u32, selected: *const ds4_metal_tensor, weights: *const ds4_metal_tensor, n_expert: u32, clamp: c_float, x: *const ds4_metal_tensor, n_tokens: u32) -> c_int;
    pub fn ds4_metal_hc_split_sinkhorn_tensor(out: *mut ds4_metal_tensor, mix: *const ds4_metal_tensor, model_map: *const c_void, model_size: u64, scale_offset: u64, base_offset: u64, n_hc: u32, sinkhorn_iters: u32, eps: c_float) -> c_int;
    pub fn ds4_metal_hc_weighted_sum_tensor(out: *mut ds4_metal_tensor, residual_hc: *const ds4_metal_tensor, weights: *const ds4_metal_tensor, n_embd: u32, n_hc: u32) -> c_int;
    pub fn ds4_metal_hc_weighted_sum_split_tensor(out: *mut ds4_metal_tensor, residual_hc: *const ds4_metal_tensor, split: *const ds4_metal_tensor, n_embd: u32, n_hc: u32) -> c_int;
    pub fn ds4_metal_hc_split_weighted_sum_tensor(out: *mut ds4_metal_tensor, split: *mut ds4_metal_tensor, mix: *const ds4_metal_tensor, residual_hc: *const ds4_metal_tensor, model_map: *const c_void, model_size: u64, scale_offset: u64, base_offset: u64, n_embd: u32, n_hc: u32, sinkhorn_iters: u32, eps: c_float) -> c_int;
    pub fn ds4_metal_hc_split_weighted_sum_norm_tensor(out: *mut ds4_metal_tensor, norm_out: *mut ds4_metal_tensor, split: *mut ds4_metal_tensor, mix: *const ds4_metal_tensor, residual_hc: *const ds4_metal_tensor, model_map: *const c_void, model_size: u64, scale_offset: u64, base_offset: u64, norm_weight_offset: u64, n_embd: u32, n_hc: u32, sinkhorn_iters: u32, eps: c_float, norm_eps: c_float) -> c_int;
    pub fn ds4_metal_output_hc_weights_tensor(out: *mut ds4_metal_tensor, pre: *const ds4_metal_tensor, model_map: *const c_void, model_size: u64, scale_offset: u64, base_offset: u64, n_hc: u32, eps: c_float) -> c_int;
    pub fn ds4_metal_hc_expand_tensor(out_hc: *mut ds4_metal_tensor, block_out: *const ds4_metal_tensor, residual_hc: *const ds4_metal_tensor, post: *const ds4_metal_tensor, comb: *const ds4_metal_tensor, n_embd: u32, n_hc: u32) -> c_int;
    pub fn ds4_metal_hc_expand_split_tensor(out_hc: *mut ds4_metal_tensor, block_out: *const ds4_metal_tensor, residual_hc: *const ds4_metal_tensor, split: *const ds4_metal_tensor, n_embd: u32, n_hc: u32) -> c_int;
    pub fn ds4_metal_hc_expand_add_split_tensor(out_hc: *mut ds4_metal_tensor, block_out: *const ds4_metal_tensor, block_add: *const ds4_metal_tensor, residual_hc: *const ds4_metal_tensor, split: *const ds4_metal_tensor, n_embd: u32, n_hc: u32) -> c_int;
    pub fn ds4_metal_shared_down_hc_expand_q8_0_tensor(out_hc: *mut ds4_metal_tensor, shared_out: *mut ds4_metal_tensor, model_map: *const c_void, model_size: u64, weight_offset: u64, in_dim: u64, out_dim: u64, shared_mid: *const ds4_metal_tensor, routed_out: *const ds4_metal_tensor, residual_hc: *const ds4_metal_tensor, split: *const ds4_metal_tensor, n_embd: u32, n_hc: u32) -> c_int;
    pub fn ds4_metal_matmul_q8_0_hc_expand_tensor(out_hc: *mut ds4_metal_tensor, block_out: *mut ds4_metal_tensor, model_map: *const c_void, model_size: u64, weight_offset: u64, in_dim: u64, out_dim: u64, x: *const ds4_metal_tensor, residual_hc: *const ds4_metal_tensor, split: *const ds4_metal_tensor, n_embd: u32, n_hc: u32) -> c_int;
}