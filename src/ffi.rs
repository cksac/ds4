#![allow(dead_code, non_camel_case_types)]

use libc::{c_char, c_float, c_int, c_void};

pub const DS4_MAX_DIMS: usize = 8;
pub const DS4_N_LAYER: usize = 43;

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ds4_backend {
    DS4_BACKEND_METAL = 0,
    DS4_BACKEND_CPU = 1,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ds4_think_mode {
    DS4_THINK_NONE = 0,
    DS4_THINK_HIGH = 1,
    DS4_THINK_MAX = 2,
}

#[repr(C)]
#[derive(Debug)]
pub struct ds4_tokens {
    pub v: *mut c_int,
    pub len: c_int,
    pub cap: c_int,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ds4_token_score {
    pub id: c_int,
    pub logit: c_float,
    pub logprob: c_float,
}

#[repr(C)]
pub struct ds4_engine {
    _private: [u8; 0],
}

#[repr(C)]
#[derive(Debug)]
pub struct ds4_engine_options {
    pub model_path: *const c_char,
    pub mtp_path: *const c_char,
    pub backend: ds4_backend,
    pub n_threads: c_int,
    pub mtp_draft_tokens: c_int,
    pub mtp_margin: c_float,
    pub warm_weights: bool,
    pub quality: bool,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ds4_gguf_map {
    pub map: *const c_void,
    pub size: u64,
}

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

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct ds4_bound_tensor_ref {
    pub present: bool,
    pub ndim: u32,
    pub dim: [u64; DS4_MAX_DIMS],
    pub tensor_type: u32,
    pub abs_offset: u64,
    pub bytes: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct ds4_layer_binding_ref {
    pub hc_attn_fn: ds4_bound_tensor_ref,
    pub hc_attn_scale: ds4_bound_tensor_ref,
    pub hc_attn_base: ds4_bound_tensor_ref,
    pub attn_norm: ds4_bound_tensor_ref,
    pub attn_q_a: ds4_bound_tensor_ref,
    pub attn_q_a_norm: ds4_bound_tensor_ref,
    pub attn_q_b: ds4_bound_tensor_ref,
    pub attn_kv: ds4_bound_tensor_ref,
    pub attn_kv_a_norm: ds4_bound_tensor_ref,
    pub attn_sinks: ds4_bound_tensor_ref,
    pub attn_output_a: ds4_bound_tensor_ref,
    pub attn_output_b: ds4_bound_tensor_ref,
    pub attn_compressor_ape: ds4_bound_tensor_ref,
    pub attn_compressor_kv: ds4_bound_tensor_ref,
    pub attn_compressor_gate: ds4_bound_tensor_ref,
    pub attn_compressor_norm: ds4_bound_tensor_ref,
    pub indexer_attn_q_b: ds4_bound_tensor_ref,
    pub indexer_proj: ds4_bound_tensor_ref,
    pub indexer_compressor_ape: ds4_bound_tensor_ref,
    pub indexer_compressor_kv: ds4_bound_tensor_ref,
    pub indexer_compressor_gate: ds4_bound_tensor_ref,
    pub indexer_compressor_norm: ds4_bound_tensor_ref,
    pub hc_ffn_fn: ds4_bound_tensor_ref,
    pub hc_ffn_scale: ds4_bound_tensor_ref,
    pub hc_ffn_base: ds4_bound_tensor_ref,
    pub ffn_norm: ds4_bound_tensor_ref,
    pub ffn_gate_tid2eid: ds4_bound_tensor_ref,
    pub ffn_gate_inp: ds4_bound_tensor_ref,
    pub ffn_exp_probs_b: ds4_bound_tensor_ref,
    pub ffn_gate_exps: ds4_bound_tensor_ref,
    pub ffn_up_exps: ds4_bound_tensor_ref,
    pub ffn_down_exps: ds4_bound_tensor_ref,
    pub ffn_gate_shexp: ds4_bound_tensor_ref,
    pub ffn_up_shexp: ds4_bound_tensor_ref,
    pub ffn_down_shexp: ds4_bound_tensor_ref,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ds4_weights_binding_ref {
    pub token_embd: ds4_bound_tensor_ref,
    pub output_hc_base: ds4_bound_tensor_ref,
    pub output_hc_fn: ds4_bound_tensor_ref,
    pub output_hc_scale: ds4_bound_tensor_ref,
    pub output_norm: ds4_bound_tensor_ref,
    pub output: ds4_bound_tensor_ref,
    pub layer: [ds4_layer_binding_ref; DS4_N_LAYER],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct ds4_mtp_weights_binding_ref {
    pub e_proj: ds4_bound_tensor_ref,
    pub h_proj: ds4_bound_tensor_ref,
    pub enorm: ds4_bound_tensor_ref,
    pub hnorm: ds4_bound_tensor_ref,
    pub norm: ds4_bound_tensor_ref,
    pub hc_head_base: ds4_bound_tensor_ref,
    pub hc_head_fn: ds4_bound_tensor_ref,
    pub hc_head_scale: ds4_bound_tensor_ref,
    pub block: ds4_layer_binding_ref,
}

extern "C" {
    pub fn ds4_engine_open(out: *mut *mut ds4_engine, opt: *const ds4_engine_options) -> c_int;
    pub fn ds4_engine_open_mapped(
        out: *mut *mut ds4_engine,
        opt: *const ds4_engine_options,
        model_map: *const ds4_gguf_map,
        mtp_map: *const ds4_gguf_map,
    ) -> c_int;
    pub fn ds4_engine_open_prebound_mapped(
        out: *mut *mut ds4_engine,
        opt: *const ds4_engine_options,
        model_map: *const ds4_gguf_map,
        model_bindings: *const ds4_weights_binding_ref,
        mtp_map: *const ds4_gguf_map,
        mtp_bindings: *const ds4_mtp_weights_binding_ref,
    ) -> c_int;
    pub fn ds4_engine_close(e: *mut ds4_engine);
    pub fn ds4_engine_dump_tokens(e: *mut ds4_engine, tokens: *const ds4_tokens);
    pub fn ds4_token_eos(e: *mut ds4_engine) -> c_int;

    pub fn ds4_engine_mtp_draft_tokens(e: *mut ds4_engine) -> c_int;
    pub fn ds4_tokens_push(tv: *mut ds4_tokens, token: c_int);
    pub fn ds4_tokens_free(tv: *mut ds4_tokens);
}

// ---------------------------------------------------------------------------
// Metal tensor and execution primitives.
// ---------------------------------------------------------------------------

#[repr(C)]
pub struct ds4_metal_tensor {
    _private: [u8; 0],
}

extern "C" {
    pub fn ds4_metal_tensor_alloc(bytes: u64) -> *mut ds4_metal_tensor;
    pub fn ds4_metal_tensor_view(
        base: *const ds4_metal_tensor,
        offset: u64,
        bytes: u64,
    ) -> *mut ds4_metal_tensor;
    pub fn ds4_metal_tensor_free(tensor: *mut ds4_metal_tensor);
    pub fn ds4_metal_tensor_bytes(tensor: *const ds4_metal_tensor) -> u64;
    pub fn ds4_metal_tensor_write(
        tensor: *mut ds4_metal_tensor,
        offset: u64,
        data: *const c_void,
        bytes: u64,
    ) -> c_int;
    pub fn ds4_metal_tensor_read(
        tensor: *const ds4_metal_tensor,
        offset: u64,
        data: *mut c_void,
        bytes: u64,
    ) -> c_int;
    pub fn ds4_metal_tensor_copy(
        dst: *mut ds4_metal_tensor,
        dst_offset: u64,
        src: *const ds4_metal_tensor,
        src_offset: u64,
        bytes: u64,
    ) -> c_int;

    pub fn ds4_metal_begin_commands() -> c_int;
    pub fn ds4_metal_flush_commands() -> c_int;
    pub fn ds4_metal_end_commands() -> c_int;
    pub fn ds4_metal_synchronize() -> c_int;

    // Embedding
    pub fn ds4_metal_embed_token_hc_tensor(
        out_hc: *mut ds4_metal_tensor,
        model_map: *const c_void,
        model_size: u64,
        weight_offset: u64,
        n_vocab: u32,
        token: u32,
        n_embd: u32,
        n_hc: u32,
    ) -> c_int;

    // HC split/expand kernels
    pub fn ds4_metal_hc_split_weighted_sum_norm_tensor(
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
        eps: c_float,
        norm_eps: c_float,
    ) -> c_int;

    pub fn ds4_metal_matmul_q8_0_hc_expand_tensor(
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
    ) -> c_int;

    pub fn ds4_metal_shared_down_hc_expand_q8_0_tensor(
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
    ) -> c_int;

    pub fn ds4_metal_hc_expand_add_split_tensor(
        out_hc: *mut ds4_metal_tensor,
        block_out: *const ds4_metal_tensor,
        block_add: *const ds4_metal_tensor,
        residual_hc: *const ds4_metal_tensor,
        split: *const ds4_metal_tensor,
        n_embd: u32,
        n_hc: u32,
    ) -> c_int;

    pub fn ds4_metal_output_hc_weights_tensor(
        out: *mut ds4_metal_tensor,
        pre: *const ds4_metal_tensor,
        model_map: *const c_void,
        model_size: u64,
        scale_offset: u64,
        base_offset: u64,
        n_hc: u32,
        eps: c_float,
    ) -> c_int;

    pub fn ds4_metal_hc_weighted_sum_tensor(
        out: *mut ds4_metal_tensor,
        residual_hc: *const ds4_metal_tensor,
        weights: *const ds4_metal_tensor,
        n_embd: u32,
        n_hc: u32,
    ) -> c_int;

    // Dense projections
    pub fn ds4_metal_matmul_q8_0_tensor(
        out: *mut ds4_metal_tensor,
        model_map: *const c_void,
        model_size: u64,
        weight_offset: u64,
        in_dim: u64,
        out_dim: u64,
        x: *const ds4_metal_tensor,
        n_tok: u64,
    ) -> c_int;

    pub fn ds4_metal_matmul_f16_tensor(
        out: *mut ds4_metal_tensor,
        model_map: *const c_void,
        model_size: u64,
        weight_offset: u64,
        in_dim: u64,
        out_dim: u64,
        x: *const ds4_metal_tensor,
        n_tok: u64,
    ) -> c_int;

    pub fn ds4_metal_matmul_f16_pair_tensor(
        out_a: *mut ds4_metal_tensor,
        out_b: *mut ds4_metal_tensor,
        model_map: *const c_void,
        model_size: u64,
        weight_a_offset: u64,
        weight_b_offset: u64,
        in_dim: u64,
        out_dim: u64,
        x: *const ds4_metal_tensor,
        n_tok: u64,
    ) -> c_int;

    // Norms
    pub fn ds4_metal_dsv4_qkv_rms_norm_rows_tensor(
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
        eps: c_float,
    ) -> c_int;

    pub fn ds4_metal_head_rms_norm_tensor(
        x: *mut ds4_metal_tensor,
        n_tok: u32,
        n_head: u32,
        head_dim: u32,
        eps: c_float,
    ) -> c_int;

    pub fn ds4_metal_rms_norm_weight_tensor(
        out: *mut ds4_metal_tensor,
        x: *const ds4_metal_tensor,
        model_map: *const c_void,
        model_size: u64,
        weight_offset: u64,
        n: u32,
        eps: c_float,
    ) -> c_int;

    pub fn ds4_metal_rms_norm_plain_tensor(
        out: *mut ds4_metal_tensor,
        x: *const ds4_metal_tensor,
        n: u32,
        eps: c_float,
    ) -> c_int;

    // RoPE
    pub fn ds4_metal_rope_tail_tensor(
        x: *mut ds4_metal_tensor,
        n_tok: u32,
        n_head: u32,
        head_dim: u32,
        n_rot: u32,
        pos0: u32,
        n_ctx_orig: u32,
        inverse: bool,
        freq_base: c_float,
        freq_scale: c_float,
        ext_factor: c_float,
        attn_factor: c_float,
        beta_fast: c_float,
        beta_slow: c_float,
    ) -> c_int;

    // KV cache store
    pub fn ds4_metal_kv_fp8_store_raw_tensor(
        kv: *mut ds4_metal_tensor,
        raw_cache: *mut ds4_metal_tensor,
        raw_cap: u32,
        row: u32,
        head_dim: u32,
        n_rot: u32,
    ) -> c_int;

    // Compressor
    pub fn ds4_metal_compressor_update_tensor(
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
        freq_base: c_float,
        freq_scale: c_float,
        ext_factor: c_float,
        attn_factor: c_float,
        beta_fast: c_float,
        beta_slow: c_float,
        rms_eps: c_float,
    ) -> c_int;

    pub fn ds4_metal_dsv4_fp8_kv_quantize_tensor(
        x: *mut ds4_metal_tensor,
        n_tok: u32,
        head_dim: u32,
        n_rot: u32,
    ) -> c_int;

    // Indexer
    pub fn ds4_metal_indexer_score_one_tensor(
        scores: *mut ds4_metal_tensor,
        q: *const ds4_metal_tensor,
        weights: *const ds4_metal_tensor,
        index_comp: *const ds4_metal_tensor,
        n_comp: u32,
        n_head: u32,
        head_dim: u32,
        scale: c_float,
    ) -> c_int;

    pub fn ds4_metal_indexer_topk_tensor(
        selected: *mut ds4_metal_tensor,
        scores: *const ds4_metal_tensor,
        n_comp: u32,
        n_tokens: u32,
        top_k: u32,
    ) -> c_int;

    // Attention
    pub fn ds4_metal_attention_indexed_mixed_batch_heads_tensor(
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
    ) -> c_int;

    pub fn ds4_metal_attention_decode_heads_tensor(
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
    ) -> c_int;

    // Attention output
    pub fn ds4_metal_attention_output_low_q8_tensor(
        low: *mut ds4_metal_tensor,
        model_map: *const c_void,
        model_size: u64,
        out_a_offset: u64,
        group_dim: u64,
        rank: u64,
        n_groups: u32,
        heads: *const ds4_metal_tensor,
    ) -> c_int;

    // Router
    pub fn ds4_metal_router_select_tensor(
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
    ) -> c_int;

    // MoE
    pub fn ds4_metal_routed_moe_one_tensor(
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
        clamp: c_float,
        x: *const ds4_metal_tensor,
    ) -> c_int;

    // Shared expert
    pub fn ds4_metal_shared_gate_up_swiglu_q8_0_tensor(
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
    ) -> c_int;
}