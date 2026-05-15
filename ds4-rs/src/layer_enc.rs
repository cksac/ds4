//! Per-layer GPU encoding for DeepSeek V4 Flash inference.
//! Structural stubs — actual decode loop lives in graph.rs via `eval_token_decode()`.
#![allow(dead_code)]

use crate::tensor::GpuTensor;
use crate::model::*;
use crate::model_view::*;

// ─── HC pre-attention: split mixer + weighted sum ───
pub fn encode_hc_pre_attention(
    _hc_split: &GpuTensor, _hc_mix_in: &GpuTensor,
    _weights: &LayerWeights, _views: &ModelViews,
) -> Result<(), &'static str> {
    Ok(())
}

// ─── Q projection: matmul_q_a -> rms_norm -> matmul_q_b ───
pub fn encode_q_projection(
    _q_out: &GpuTensor, _x_in: &GpuTensor,
    _weights: &LayerWeights, _views: &ModelViews,
) -> Result<(), &'static str> {
    Ok(())
}

// ─── KV projection: matmul_kv_a -> rms_norm -> matmul_kv_b ───
pub fn encode_kv_projection(
    _kv_out: &GpuTensor, _x_in: &GpuTensor,
    _weights: &LayerWeights, _views: &ModelViews,
) -> Result<(), &'static str> {
    Ok(())
}

// ─── RoPE tail on Q and K ───
pub fn encode_rope_tail(
    _q: &GpuTensor, _kv: &GpuTensor, _pos: u32,
) -> Result<(), &'static str> {
    Ok(())
}

// ─── KV store into raw ring cache ───
pub fn encode_kv_store(
    _kv: &GpuTensor, _raw_cache: &GpuTensor, _pos: u32, _raw_cap: u32,
) -> Result<(), &'static str> {
    Ok(())
}

// ─── Compressor update (ratio != 0 layers) ───
pub fn encode_compressor_update(
    _kv_cur: &GpuTensor, _sc_cur: &GpuTensor,
    _state_kv: &GpuTensor, _state_score: &GpuTensor,
    _comp_cache: &GpuTensor, _weights: &LayerWeights, _views: &ModelViews,
) -> Result<(), &'static str> {
    Ok(())
}

// ─── Attention decode heads ───
pub fn encode_attention(
    _heads: &GpuTensor, _q: &GpuTensor, _raw_kv: &GpuTensor,
    _comp_kv: &GpuTensor, _comp_mask: &GpuTensor,
    _n_raw: u32, _raw_cap: u32, _raw_start: u32,
    _n_comp: u32, _use_mask: u32,
) -> Result<(), &'static str> {
    Ok(())
}
