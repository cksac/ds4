//! GPU graph state — persistent tensors for decode and batched prefill.
//!
//! Mirrors `ds4_gpu_graph` from ds4.c. One graph is owned per session.

use super::tensor::Tensor;
use super::buffers::ScratchBuffer;

pub const N_LAYER: usize = 43;
pub const N_EMBD: u32 = 1536;  // per-HC embedding dim (1536 * 4 = 6144 total)
pub const N_HC: u32 = 4;       // head channels
pub const N_HEAD: u32 = 64;
pub const HEAD_DIM: u32 = 512;
pub const INDEXER_HEAD_DIM: u32 = 128;
pub const N_ROT: u32 = 128;    // RoPE tail dim
pub const INDEXER_TOP_K: u32 = 512;

// Expert counts
pub const N_EXPERTS: u32 = 256;
pub const N_EXPERTS_USED: u32 = 6;

/// Fixed tensors for single-token decode.
pub struct DecodeTensors {
    pub cur_hc: Tensor,
    pub flat_hc: Tensor,
    pub hc_mix: Tensor,
    pub hc_split: Tensor,
    pub hc_pre: Tensor,
    pub hc_post: Tensor,
    pub hc_comb: Tensor,
    pub attn_cur: Tensor,
    pub attn_norm: Tensor,
    pub qr: Tensor,
    pub qr_norm: Tensor,
    pub q: Tensor,
    pub kv_raw: Tensor,
    pub kv: Tensor,
}

/// Per-layer persistent KV state.
pub struct LayerKVState {
    pub raw_cache: Tensor,           // ring buffer, raw_cap rows
    pub attn_comp_cache: Option<Tensor>, // compressed cache (ratio-4 or ratio-128)
    pub attn_state_kv: Tensor,       // compressor frontier KV
    pub attn_state_score: Tensor,    // compressor frontier score
    pub index_comp_cache: Option<Tensor>, // indexer compressed cache (ratio-4 only)
    pub index_state_kv: Option<Tensor>,
    pub index_state_score: Option<Tensor>,
    pub n_comp: u32,
    pub n_index_comp: u32,
}

/// Per-layer work tensors (reused in-place per layer).
pub struct WorkTensors {
    pub comp_kv_cur: Tensor,
    pub comp_sc_cur: Tensor,
    pub indexer_q: Tensor,
    pub indexer_weights: Tensor,
    pub indexer_scores: Tensor,
    pub comp_mask: Tensor,
    pub comp_selected: Tensor,
    pub heads: Tensor,
    pub attn_low: Tensor,
    pub attn_out: Tensor,
    pub after_attn_hc: Tensor,
    pub ffn_cur: Tensor,
    pub ffn_norm: Tensor,
    pub shared_gate: Tensor,
    pub shared_up: Tensor,
    pub shared_mid: Tensor,
    pub shared_out: Tensor,
    pub router_logits: Tensor,
    pub router_probs: Tensor,
    pub router_selected: Tensor,
    pub router_weights: Tensor,
    pub routed_gate: Tensor,
    pub routed_up: Tensor,
    pub routed_mid: Tensor,
    pub routed_down: Tensor,
    pub routed_out: Tensor,
    pub ffn_out: Tensor,
    pub after_ffn_hc: Tensor,
    pub output_pre: Tensor,
    pub output_weights: Tensor,
    pub output_embd: Tensor,
    pub output_norm: Tensor,
    pub logits: Tensor,
}

/// Batched prefill tensors (layer-major, chunked).
pub struct PrefillTensors {
    pub tokens: Tensor,
    pub cur_hc: Tensor,
    pub next_hc: Tensor,
    pub flat_hc: Tensor,
    pub hc_mix: Tensor,
    pub hc_split: Tensor,
    pub attn_cur: Tensor,
    pub attn_norm: Tensor,
    pub qr: Tensor,
    pub qr_norm: Tensor,
    pub q: Tensor,
    pub kv_raw: Tensor,
    pub kv: Tensor,
    pub comp_kv: Tensor,
    pub comp_sc: Tensor,
    pub indexer_q: Tensor,
    pub indexer_weights: Tensor,
    pub heads: Tensor,
    pub attn_low: Tensor,
    pub attn_out: Tensor,
    pub group_tmp: Tensor,
    pub low_tmp: Tensor,
    pub after_attn_hc: Tensor,
    pub ffn_cur: Tensor,
    pub ffn_norm: Tensor,
    pub shared_gate: Tensor,
    pub shared_up: Tensor,
    pub shared_mid: Tensor,
    pub shared_out: Tensor,
    pub router_logits: Tensor,
    pub router_probs: Tensor,
    pub router_selected: Tensor,
    pub router_weights: Tensor,
    pub routed_gate: Tensor,
    pub routed_up: Tensor,
    pub routed_mid: Tensor,
    pub routed_down: Tensor,
    pub routed_out: Tensor,
    pub ffn_out: Tensor,
}

/// The full GPU graph state for one session.
pub struct GpuGraph {
    pub decode: DecodeTensors,
    pub layers: Vec<LayerKVState>,
    pub work: WorkTensors,
    pub prefill: PrefillTensors,
    pub raw_cap: u32,
    pub comp_cap: u32,
    pub layer_comp_cap: [u32; N_LAYER],
    pub prefill_cap: u32,
    pub raw_window: u32,
    pub quality: bool,
    pub materialize_ffn_out: bool,
    // Scratch buffers
    pub flash_attn_mask: ScratchBuffer,
    pub flash_attn_pad: ScratchBuffer,
    pub flash_attn_tmp: ScratchBuffer,
    pub flash_attn_blk: ScratchBuffer,
    pub flash_attn_ring: ScratchBuffer,
    pub flash_attn_kv: ScratchBuffer,
    pub compressor_pool_kv: ScratchBuffer,
    pub compressor_pool_score: ScratchBuffer,
    pub compressor_pool_score_cont: ScratchBuffer,
    pub compressor_pool_softmax: ScratchBuffer,
    pub compressor_pool_product: ScratchBuffer,
    pub compressor_store_ape: ScratchBuffer,
    pub compressor_store_score: ScratchBuffer,
    pub embed_rows: ScratchBuffer,
    pub router_selection: ScratchBuffer,
    pub router_weight_sum: ScratchBuffer,
    pub indexer_head_scores: ScratchBuffer,
    pub indexer_topk: ScratchBuffer,
    pub indexed_topk: ScratchBuffer,
    pub f16_round_scratch: ScratchBuffer,
    pub raw_store_round: ScratchBuffer,
    pub moe_gate_scratch: ScratchBuffer,
    pub moe_down_scratch: ScratchBuffer,
    pub moe_id_map: ScratchBuffer,
    pub attn_out_group_ids: ScratchBuffer,
}
