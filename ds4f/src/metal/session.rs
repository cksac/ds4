//! GPU session: owns the graph state and orchestrates kernel dispatch
//! for decode and prefill.
//!
//! Mirrors `ds4_session` + `metal_graph_encode_decode_layer` from ds4.c.

use anyhow::Result;
use objc2::rc::Retained;
use objc2::runtime::AnyObject;
use crate::metal::{buffers::*, commands::*, graph::*, pipeline::*, tensor::*};

// ── Model weight reference ────────────────────────────────────────────

/// Opaque reference to model weights (backed by C engine FFI initially).
pub struct ModelWeights {
    pub map: *const std::ffi::c_void,
    pub size: u64,
}

/// Per-layer weight offsets (mirrors ds4_layer_weights from ds4.c).
/// All offsets are absolute byte offsets into the mmap'd GGUF file.
pub struct LayerWeights {
    pub layer: u32,
    // Attention
    pub attn_norm: u64,
    pub attn_q_a: u64,
    pub attn_q_b: u64,
    pub attn_kv: u64,
    pub attn_out_a: u64,
    pub attn_out_b: u64,
    pub attn_out_group_dim: u64,
    // HC
    pub hc_attn_fn: u64,
    pub hc_attn_scale: u64,
    pub hc_attn_base: u64,
    pub hc_ffn_fn: u64,
    pub hc_ffn_scale: u64,
    pub hc_ffn_base: u64,
    // Shared FFN
    pub ffn_norm: u64,
    pub ffn_gate_shexp: u64,
    pub ffn_up_shexp: u64,
    pub ffn_down_shexp: u64,
    // Routed MoE
    pub ffn_gate_exps: u64,
    pub ffn_up_exps: u64,
    pub ffn_down_exps: u64,
    pub ffn_gate_type: u32,
    pub ffn_down_type: u32,
    pub ffn_gate_expert_bytes: u64,
    pub ffn_gate_row_bytes: u64,
    pub ffn_down_expert_bytes: u64,
    pub ffn_down_row_bytes: u64,
    pub ffn_expert_in_dim: u32,
    pub ffn_expert_mid_dim: u32,
    pub ffn_expert_out_dim: u32,
    // Compressor
    pub compress_ratio: u32,
    pub compress_ape: u64,
    pub compress_ape_type: u32,
    pub compress_norm: u64,
    pub compress_norm_type: u32,
    // RoPE
    pub rope_freq_base: f32,
    pub rope_freq_scale: f32,
    // Router
    pub router_bias: u64,
    pub router_hash: u64,
    pub router_hash_rows: u32,
    pub has_bias: bool,
    pub hash_mode: bool,
}

// ── Session ────────────────────────────────────────────────────────────

/// One inference session owning the live Metal graph state.
pub struct Session {
    pub graph: GpuGraph,
    pub device: Retained<AnyObject>,
    pub queue: Retained<AnyObject>,
    pub pipeline_cache: PipelineCache,
    pub model_views: ModelViews,
    pub batch: CommandBatch,
    pub weights: ModelWeights,
    pub layers: Vec<LayerWeights>,
    pub checkpoint: Vec<i32>,
    pub logits: Vec<f32>,
    pub ctx_size: i32,
    pub pos: i32, // current position in the sequence
    // Graph constants
    pub n_layer: u32,
    pub n_embd: u32,
    pub n_hc: u32,
    pub n_head: u32,
    pub head_dim: u32,
    pub n_rot: u32,
    pub n_vocab: u32,
    pub sink_offset: u64,
    pub output_norm_offset: u64,
    pub output_weight_offset: u64,
    pub indexer_n_head: u32,
    pub indexer_head_dim: u32,
    pub indexer_top_k: u32,
}

impl Session {
    /// Create a new session with allocated graph tensors.
    pub fn new(
        device: Retained<AnyObject>,
        queue: Retained<AnyObject>,
        library: Retained<AnyObject>,
        model_map: *const std::ffi::c_void,
        model_size: u64,
        ctx_size: i32,
        n_layer: u32, n_embd: u32, n_hc: u32,
        n_head: u32, head_dim: u32, n_rot: u32,
        n_vocab: u32,
    ) -> Result<Self> {
        let model_views = ModelViews::new();
        let pipeline_cache = PipelineCache::new(device.clone(), library.clone());
        let batch = CommandBatch::new(queue.clone());

        // Compute cache sizes
        let raw_cap = 256u32; // Small for decode — grows with ctx via reallocation
        let comp_cap = 256u32;
        let prefill_cap = 1u32; // Single-token decode; prefill allocates batch on demand
        let raw_window = 128u32;

        // Allocate decode tensors
        let hc_dim = n_hc as u64 * n_embd as u64; // 4 * 1536 = 6144
        let hc_dim_bytes = hc_dim * 4;
        let embd_bytes = n_embd as u64 * 4;
        let qr_dim = 256u64; // Q LoRA rank; TODO: read from model
        let q_dim = n_head as u64 * head_dim as u64; // 64 * 512 = 32768
        let kv_dim = n_head as u64 * head_dim as u64; // same as Q for DS4

        let decode = DecodeTensors {
            cur_hc: Tensor::alloc(&device, hc_dim_bytes).unwrap(),
            flat_hc: Tensor::alloc(&device, hc_dim_bytes).unwrap(),
            hc_mix: Tensor::alloc(&device, (2 * n_hc as u64 + n_hc as u64 * n_hc as u64) * 4).unwrap(),
            hc_split: Tensor::alloc(&device, (n_hc as u64 * n_hc as u64 * 3) * 4).unwrap(),
            hc_pre: Tensor::alloc(&device, n_hc as u64 * 4).unwrap(),
            hc_post: Tensor::alloc(&device, n_hc as u64 * 4).unwrap(),
            hc_comb: Tensor::alloc(&device, n_hc as u64 * n_hc as u64 * 4).unwrap(),
            attn_cur: Tensor::alloc(&device, embd_bytes).unwrap(),
            attn_norm: Tensor::alloc(&device, embd_bytes).unwrap(),
            qr: Tensor::alloc(&device, qr_dim * 4).unwrap(),
            qr_norm: Tensor::alloc(&device, qr_dim * 4).unwrap(),
            q: Tensor::alloc(&device, q_dim * 4).unwrap(),
            kv_raw: Tensor::alloc(&device, kv_dim * 4).unwrap(),
            kv: Tensor::alloc(&device, kv_dim * 4).unwrap(),
        };

        // Allocate per-layer KV state (minimal sizes — grows during session lifetime)
        let mut layers = Vec::with_capacity(n_layer as usize);
        let small_raw = 256u64 * head_dim as u64 * 4; // 256 rows
        let small_comp = 256u64 * head_dim as u64 * 4;
        let state_bytes = 4u64 * head_dim as u64 * 4;
        let small_index = 256u64 * 128 * 4;
        for _il in 0..n_layer as usize {
            layers.push(LayerKVState {
                raw_cache: Tensor::alloc(&device, small_raw).unwrap(),
                attn_comp_cache: Some(Tensor::alloc(&device, small_comp).unwrap()),
                attn_state_kv: Tensor::alloc(&device, state_bytes).unwrap(),
                attn_state_score: Tensor::alloc(&device, state_bytes).unwrap(),
                index_comp_cache: Some(Tensor::alloc(&device, small_index).unwrap()),
                index_state_kv: Some(Tensor::alloc(&device, state_bytes).unwrap()),
                index_state_score: Some(Tensor::alloc(&device, state_bytes).unwrap()),
                n_comp: 0,
                n_index_comp: 0,
            });
        }

        // Allocate per-layer work tensors
        let work = WorkTensors {
            comp_kv_cur: Tensor::alloc(&device, head_dim as u64 * 4).unwrap(),
            comp_sc_cur: Tensor::alloc(&device, 4).unwrap(),
            indexer_q: Tensor::alloc(&device, 64 * 128 * 4).unwrap(),
            indexer_weights: Tensor::alloc(&device, 64 * 4).unwrap(),
            indexer_scores: Tensor::alloc(&device, comp_cap as u64 * 4).unwrap(),
            comp_mask: Tensor::alloc(&device, comp_cap as u64 * 4).unwrap(),
            comp_selected: Tensor::alloc(&device, 512 * 4).unwrap(), // top_k indices
            heads: Tensor::alloc(&device, q_dim * 4).unwrap(),
            attn_low: Tensor::alloc(&device, qr_dim * 4).unwrap(),
            attn_out: Tensor::alloc(&device, embd_bytes).unwrap(),
            after_attn_hc: Tensor::alloc(&device, hc_dim_bytes).unwrap(),
            ffn_cur: Tensor::alloc(&device, embd_bytes).unwrap(),
            ffn_norm: Tensor::alloc(&device, embd_bytes).unwrap(),
            shared_gate: Tensor::alloc(&device, 2048 * 4).unwrap(), // shared_dim
            shared_up: Tensor::alloc(&device, 2048 * 4).unwrap(),
            shared_mid: Tensor::alloc(&device, 2048 * 4).unwrap(),
            shared_out: Tensor::alloc(&device, embd_bytes).unwrap(),
            router_logits: Tensor::alloc(&device, 256 * 4).unwrap(),
            router_probs: Tensor::alloc(&device, 256 * 4).unwrap(),
            router_selected: Tensor::alloc(&device, 6 * 4).unwrap(),
            router_weights: Tensor::alloc(&device, 6 * 4).unwrap(),
            routed_gate: Tensor::alloc(&device, 2048 * 4).unwrap(), // expert_mid_dim
            routed_up: Tensor::alloc(&device, 2048 * 4).unwrap(),
            routed_mid: Tensor::alloc(&device, 2048 * 4).unwrap(),
            routed_down: Tensor::alloc(&device, 2048 * 4).unwrap(),
            routed_out: Tensor::alloc(&device, embd_bytes).unwrap(),
            ffn_out: Tensor::alloc(&device, embd_bytes).unwrap(),
            after_ffn_hc: Tensor::alloc(&device, hc_dim_bytes).unwrap(),
            output_pre: Tensor::alloc(&device, n_hc as u64 * 4).unwrap(),
            output_weights: Tensor::alloc(&device, n_hc as u64 * 4).unwrap(),
            output_embd: Tensor::alloc(&device, embd_bytes).unwrap(),
            output_norm: Tensor::alloc(&device, embd_bytes).unwrap(),
            logits: Tensor::alloc(&device, n_vocab as u64 * 4).unwrap(),
        };

        // Allocate batch prefill tensors at minimal size (single token for decode)
        // Full prefill batch tensors are reallocated when prefill is performed
        let ptok = 1u64; // single-token placeholder
        let prefill = PrefillTensors {
            tokens: Tensor::alloc(&device, ptok * 4).unwrap(),
            cur_hc: Tensor::alloc(&device, ptok * hc_dim_bytes).unwrap(),
            next_hc: Tensor::alloc(&device, ptok * hc_dim_bytes).unwrap(),
            flat_hc: Tensor::alloc(&device, ptok * hc_dim_bytes).unwrap(),
            hc_mix: Tensor::alloc(&device, ptok * (2 * n_hc + n_hc * n_hc) as u64 * 4).unwrap(),
            hc_split: Tensor::alloc(&device, ptok * n_hc as u64 * n_hc as u64 * 3 * 4).unwrap(),
            attn_cur: Tensor::alloc(&device, ptok * embd_bytes).unwrap(),
            attn_norm: Tensor::alloc(&device, ptok * embd_bytes).unwrap(),
            qr: Tensor::alloc(&device, ptok * qr_dim * 4).unwrap(),
            qr_norm: Tensor::alloc(&device, ptok * qr_dim * 4).unwrap(),
            q: Tensor::alloc(&device, ptok * q_dim * 4).unwrap(),
            kv_raw: Tensor::alloc(&device, ptok * kv_dim * 4).unwrap(),
            kv: Tensor::alloc(&device, ptok * kv_dim * 4).unwrap(),
            comp_kv: Tensor::alloc(&device, ptok * head_dim as u64 * 4).unwrap(),
            comp_sc: Tensor::alloc(&device, ptok * 4).unwrap(),
            indexer_q: Tensor::alloc(&device, ptok * 64 * 128 * 4).unwrap(),
            indexer_weights: Tensor::alloc(&device, 64 * 4).unwrap(),
            heads: Tensor::alloc(&device, ptok * q_dim * 4).unwrap(),
            attn_low: Tensor::alloc(&device, ptok * qr_dim * 4).unwrap(),
            attn_out: Tensor::alloc(&device, ptok * embd_bytes).unwrap(),
            group_tmp: Tensor::alloc(&device, 4096 * 4).unwrap(),
            low_tmp: Tensor::alloc(&device, 4096 * 4).unwrap(),
            after_attn_hc: Tensor::alloc(&device, ptok * hc_dim_bytes).unwrap(),
            ffn_cur: Tensor::alloc(&device, ptok * embd_bytes).unwrap(),
            ffn_norm: Tensor::alloc(&device, ptok * embd_bytes).unwrap(),
            shared_gate: Tensor::alloc(&device, ptok * 2048 * 4).unwrap(),
            shared_up: Tensor::alloc(&device, ptok * 2048 * 4).unwrap(),
            shared_mid: Tensor::alloc(&device, ptok * 2048 * 4).unwrap(),
            shared_out: Tensor::alloc(&device, ptok * embd_bytes).unwrap(),
            router_logits: Tensor::alloc(&device, ptok * 256 * 4).unwrap(),
            router_probs: Tensor::alloc(&device, ptok * 256 * 4).unwrap(),
            router_selected: Tensor::alloc(&device, ptok * 6 * 4).unwrap(),
            router_weights: Tensor::alloc(&device, ptok * 6 * 4).unwrap(),
            routed_gate: Tensor::alloc(&device, ptok * 2048 * 4).unwrap(),
            routed_up: Tensor::alloc(&device, ptok * 2048 * 4).unwrap(),
            routed_mid: Tensor::alloc(&device, ptok * 2048 * 4).unwrap(),
            routed_down: Tensor::alloc(&device, ptok * 2048 * 4).unwrap(),
            routed_out: Tensor::alloc(&device, ptok * embd_bytes).unwrap(),
            ffn_out: Tensor::alloc(&device, ptok * embd_bytes).unwrap(),
        };

        let graph = GpuGraph {
            decode, layers, work, prefill,
            raw_cap, comp_cap,
            layer_comp_cap: [comp_cap; N_LAYER],
            prefill_cap, raw_window,
            quality: false, materialize_ffn_out: false,
            flash_attn_mask: ScratchBuffer::new("fa_mask"),
            flash_attn_pad: ScratchBuffer::new("fa_pad"),
            flash_attn_tmp: ScratchBuffer::new("fa_tmp"),
            flash_attn_blk: ScratchBuffer::new("fa_blk"),
            flash_attn_ring: ScratchBuffer::new("fa_ring"),
            flash_attn_kv: ScratchBuffer::new("fa_kv"),
            compressor_pool_kv: ScratchBuffer::new("cp_kv"),
            compressor_pool_score: ScratchBuffer::new("cp_sc"),
            compressor_pool_score_cont: ScratchBuffer::new("cp_scc"),
            compressor_pool_softmax: ScratchBuffer::new("cp_sm"),
            compressor_pool_product: ScratchBuffer::new("cp_pr"),
            compressor_store_ape: ScratchBuffer::new("cs_ape"),
            compressor_store_score: ScratchBuffer::new("cs_sc"),
            embed_rows: ScratchBuffer::new("emb"),
            router_selection: ScratchBuffer::new("rs"),
            router_weight_sum: ScratchBuffer::new("rw"),
            indexer_head_scores: ScratchBuffer::new("ihs"),
            indexer_topk: ScratchBuffer::new("itk"),
            indexed_topk: ScratchBuffer::new("idt"),
            f16_round_scratch: ScratchBuffer::new("f16r"),
            raw_store_round: ScratchBuffer::new("rsr"),
            moe_gate_scratch: ScratchBuffer::new("mgs"),
            moe_down_scratch: ScratchBuffer::new("mds"),
            moe_id_map: ScratchBuffer::new("mid"),
            attn_out_group_ids: ScratchBuffer::new("aog"),
        };

        Ok(Self {
            graph,
            device,
            queue,
            pipeline_cache,
            model_views,
            batch,
            weights: ModelWeights { map: std::ptr::null(), size: 0 },
            layers: Vec::new(),
            checkpoint: Vec::new(),
            logits: vec![0.0f32; n_vocab as usize],
            ctx_size,
            pos: 0,
            n_layer, n_embd, n_hc, n_head, head_dim, n_rot, n_vocab,
            sink_offset: 0,
            output_norm_offset: 0,
            output_weight_offset: 0,
            indexer_n_head: 64,
            indexer_head_dim: 128,
            indexer_top_k: 512,
        })
    }

    /// Set the model weight mapping.
    pub fn set_model_map(&mut self, map: *const std::ffi::c_void, size: u64, tensor_data_offset: u64, tensor_data_size: u64) -> Result<()> {
        self.weights = ModelWeights { map, size };
        self.model_views.map_model_range(&self.device, map, size, tensor_data_offset, tensor_data_size)
    }
}
