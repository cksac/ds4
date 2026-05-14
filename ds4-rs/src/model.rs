//! Fixed model structure: weights, layers, engine, session.

use crate::gguf::*;
use crate::tensor::GpuTensor;
use std::sync::Arc;

// ─── Per-layer weight bindings ───

pub struct LayerWeights {
    // HC attention projections
    pub hc_attn_fn: Option<GgufTensor>,
    pub hc_attn_scale: Option<GgufTensor>,
    pub hc_attn_base: Option<GgufTensor>,
    // Attention
    pub attn_norm: Option<GgufTensor>,
    pub attn_q_a: Option<GgufTensor>,
    pub attn_q_b: Option<GgufTensor>,
    pub attn_q_a_norm: Option<GgufTensor>,
    pub attn_kv_a: Option<GgufTensor>,
    pub attn_kv_b: Option<GgufTensor>,
    pub attn_kv_a_norm: Option<GgufTensor>,
    pub attn_sinks: Option<GgufTensor>,
    pub attn_output_a: Option<GgufTensor>,
    pub attn_output_b: Option<GgufTensor>,
    // Compressor (ratio != 0)
    pub attn_compressor_ape: Option<GgufTensor>,
    pub attn_compressor_kv: Option<GgufTensor>,
    pub attn_compressor_gate: Option<GgufTensor>,
    pub attn_compressor_norm: Option<GgufTensor>,
    // Indexer (ratio == 4)
    pub indexer_attn_q_b: Option<GgufTensor>,
    pub indexer_proj: Option<GgufTensor>,
    pub indexer_comp_ape: Option<GgufTensor>,
    pub indexer_comp_kv: Option<GgufTensor>,
    pub indexer_comp_gate: Option<GgufTensor>,
    pub indexer_comp_norm: Option<GgufTensor>,
    // HC FFN
    pub hc_ffn_fn: Option<GgufTensor>,
    pub hc_ffn_scale: Option<GgufTensor>,
    pub hc_ffn_base: Option<GgufTensor>,
    // FFN
    pub ffn_norm: Option<GgufTensor>,
    pub ffn_gate_inp: Option<GgufTensor>,
    pub ffn_gate_exps: Option<GgufTensor>,
    pub ffn_up_exps: Option<GgufTensor>,
    pub ffn_down_exps: Option<GgufTensor>,
    pub shared_gate: Option<GgufTensor>,
    pub shared_up: Option<GgufTensor>,
    pub shared_down: Option<GgufTensor>,
    pub ffn_exp_probs_b: Option<GgufTensor>, // bias suffix
    // Hash layers (layer < 3)
    pub ffn_gate_tid2eid: Option<GgufTensor>,
}

impl LayerWeights {
    pub fn new() -> Self {
        Self {
            hc_attn_fn: None, hc_attn_scale: None, hc_attn_base: None,
            attn_norm: None, attn_q_a: None, attn_q_b: None,
            attn_q_a_norm: None, attn_kv_a: None, attn_kv_b: None,
            attn_kv_a_norm: None, attn_sinks: None,
            attn_output_a: None, attn_output_b: None,
            attn_compressor_ape: None, attn_compressor_kv: None,
            attn_compressor_gate: None, attn_compressor_norm: None,
            indexer_attn_q_b: None, indexer_proj: None,
            indexer_comp_ape: None, indexer_comp_kv: None,
            indexer_comp_gate: None, indexer_comp_norm: None,
            hc_ffn_fn: None, hc_ffn_scale: None, hc_ffn_base: None,
            ffn_norm: None, ffn_gate_inp: None,
            ffn_gate_exps: None, ffn_up_exps: None, ffn_down_exps: None,
            shared_gate: None, shared_up: None, shared_down: None,
            ffn_exp_probs_b: None, ffn_gate_tid2eid: None,
        }
    }
}

// ─── MTP (Multi-Token Prediction) Weights ───

pub struct MtpWeights {
    pub hc_head_base: Option<GgufTensor>,
    pub hc_head_fn: Option<GgufTensor>,
    pub hc_head_scale: Option<GgufTensor>,
    pub e_proj: Option<GgufTensor>,
    pub h_proj: Option<GgufTensor>,
    pub enorm: Option<GgufTensor>,
    pub hnorm: Option<GgufTensor>,
    pub norm: Option<GgufTensor>,
    pub layer: Option<Box<LayerWeights>>,
}

impl MtpWeights {
    pub fn new() -> Self {
        MtpWeights {
            hc_head_base: None, hc_head_fn: None, hc_head_scale: None,
            e_proj: None, h_proj: None, enorm: None, hnorm: None, norm: None,
            layer: None,
        }
    }
}

// ─── Engine ───

pub struct EngineWeights {
    pub token_embd: Option<GgufTensor>,
    pub output_hc_base: Option<GgufTensor>,
    pub output_hc_fn: Option<GgufTensor>,
    pub output_hc_scale: Option<GgufTensor>,
    pub output_norm: Option<GgufTensor>,
    pub output: Option<GgufTensor>,
    pub layer: [LayerWeights; N_LAYER as usize],
}

impl EngineWeights {
    pub fn new() -> Self {
        EngineWeights {
            token_embd: None, output_hc_base: None, output_hc_fn: None,
            output_hc_scale: None, output_norm: None, output: None,
            layer: std::array::from_fn(|_| LayerWeights::new()),
        }
    }
}

pub struct Engine {
    pub model: GgufModel,
    pub weights: EngineWeights,
    pub backend: Backend,
    pub vocab: Vocab,
}

pub enum Backend { Metal, Cpu }

// ─── Session ───

pub struct Session {
    pub engine: Arc<Engine>,
    pub graph: Option<GpuGraph>,
    pub logits: Vec<f32>,
    pub checkpoint: Vec<i32>,
}

// ─── GPU Graph ───

pub struct GpuGraph {
    // Decode tensors
    pub cur_hc: Vec<GpuTensor>,
    pub flat_hc: Vec<GpuTensor>,
    pub hc_mix: Vec<GpuTensor>,
    pub hc_split: Vec<GpuTensor>,
    pub q: Vec<GpuTensor>,
    pub kv_raw: Vec<GpuTensor>,

    // KV cache (per layer)
    pub layer_raw_cache: Vec<GpuTensor>,
    pub layer_attn_comp_cache: Vec<GpuTensor>,
    pub layer_index_comp_cache: Vec<GpuTensor>,

    // Per-layer state frontiers
    pub attn_state_kv: Vec<GpuTensor>,
    pub attn_state_score: Vec<GpuTensor>,

    // Logits
    pub logits: GpuTensor,
}

// ─── Vocab ───

pub struct Vocab {
    pub tokens: Vec<String>,
    pub merges: Vec<(i32, i32)>,
    pub scores: Vec<f32>,
    pub bos_id: i32,
    pub eos_id: i32,
    pub unk_id: i32,
    pub sep_id: i32,
    pub pad_id: i32,
}
