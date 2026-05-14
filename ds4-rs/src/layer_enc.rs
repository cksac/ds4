//! Per-layer GPU encoding for DeepSeek V4 Flash inference.
//! Encodes one decode layer: HC mix, Q/KV proj, RoPE, attention, FFN.
//! Uses Metal kernel dispatch to the pipeline cache.
//!
//! Note: These functions are kept as a structural reference. The actual
//! decode loop now lives in graph.rs via `eval_token_decode()`.
#![allow(dead_code)]

use metal::*;
use crate::bridge;
use crate::tensor::GpuTensor;
use crate::model::*;
use crate::gguf::*;
use crate::model_view::*;
use crate::pipeline;

fn get_pipeline(name: &str) -> Option<ComputePipelineState> {
    if let Some(p) = pipeline::get_pipeline(name) { return Some(p); }
    let lib = bridge::with_library(|l| l.clone())?;
    let lib_ref = &*lib;
    if let Ok(fn_) = lib_ref.get_function(name, None) {
        let device = bridge::with_device(|d| d.clone())?;
        if let Ok(p) = device.new_compute_pipeline_state_with_function(&fn_) {
            pipeline::cache_pipeline(name, &p);
            return Some(p);
        }
    }
    None
}

// ─── HC pre-attention: split mixer + weighted sum ───
pub fn encode_hc_pre_attention(
    hc_split: &GpuTensor, hc_mix_in: &GpuTensor,
    weights: &LayerWeights, _views: &ModelViews,
) -> Result<(), &'static str> {
    // kernel_dsv4_hc_split_sinkhorn: split mixer into pre/post/comb weights
    // kernel_dsv4_hc_weighted_sum: reduce 4 residual HC streams to 1 embedding row
    // First: HC split
    if let Some(ref _w) = weights.hc_attn_fn {
        let pipeline = get_pipeline("kernel_dsv4_hc_split_sinkhorn").ok_or("hc split pipeline")?;
        bridge::with_queue(|queue| {
            let cb = queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pipeline);
            if let Some(ref buf) = hc_mix_in.buffer() {
                enc.set_buffer(0, Some(&**buf), hc_mix_in.offset_raw() as u64);
            }
            if let Some(ref buf) = hc_split.buffer() {
                enc.set_buffer(3, Some(&**buf), hc_split.offset_raw() as u64);
            }
            enc.dispatch_thread_groups(MTLSize { width: 1, height: 1, depth: 1 },
                MTLSize { width: 256, height: 1, depth: 1 });
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        });
    }
    Ok(())
}

// ─── Q projection: matmul_q_a -> rms_norm -> matmul_q_b ───
pub fn encode_q_projection(
    _q_out: &GpuTensor, _x_in: &GpuTensor,
    _weights: &LayerWeights, _views: &ModelViews,
) -> Result<(), &'static str> {
    // Step 1: matmul_f16 (Q_a projection)  in=4096, out=1024
    // Step 2: rms_norm on 1024-dim
    // Step 3: matmul_f16 (Q_b projection)  in=1024, out=64*512=32768
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
    q: &GpuTensor, kv: &GpuTensor, _pos: u32,
) -> Result<(), &'static str> {
    let pipeline = get_pipeline("kernel_dsv4_rope_tail_f32").ok_or("rope pipeline")?;
    let head_dim = N_HEAD_DIM as u64;
    let n_head = N_HEAD;
    let nth = std::cmp::min(256u64, head_dim).max(1);
    bridge::with_queue(|queue| {
        let cb = queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        if let Some(ref buf) = q.buffer() {
            enc.set_buffer(1, Some(&**buf), q.offset_raw() as u64);
        }
        enc.dispatch_thread_groups(MTLSize { width: n_head as u64, height: 1, depth: 1 },
            MTLSize { width: nth, height: 1, depth: 1 });
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
    });
    // Repeat for KV
    if let Some(ref buf) = kv.buffer() {
        bridge::with_queue(|queue| {
            let cb = queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pipeline);
            enc.set_buffer(1, Some(&**buf), kv.offset_raw() as u64);
            enc.dispatch_thread_groups(MTLSize { width: 1, height: 1, depth: 1 },
                MTLSize { width: nth, height: 1, depth: 1 });
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        });
    }
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

// ─── Attention output projection + HC expand ───
fn encode_attention_output(
    _attn_out: &GpuTensor, _heads: &GpuTensor,
    _weights: &LayerWeights, _views: &ModelViews,
) -> Result<(), &'static str> {
    // matmul_q8_0 (output_a) + matmul_f16 (output_b) + hc_expand
    Ok(())
}

// ─── Shared expert gate/up + SwiGLU ───
fn encode_shared_expert(
    _shared_mid: &GpuTensor, _x_in: &GpuTensor,
    _weights: &LayerWeights, _views: &ModelViews,
) -> Result<(), &'static str> {
    // kernel_dsv4_shared_gate_up_swiglu_q8_0
    Ok(())
}

// ─── Routed expert matvec ───
fn encode_routed_expert(
    _routed_out: &GpuTensor, _x_in: &GpuTensor,
    _weights: &LayerWeights, _views: &ModelViews,
    _selected: &[i32], _weights_val: &[f32],
) -> Result<(), &'static str> {
    Ok(())
}

// ─── HC expand after FFN ───
fn encode_hc_expand_ffn(
    _hc_out: &GpuTensor, _block_out: &GpuTensor,
    _residual_hc: &GpuTensor, _split: &GpuTensor,
    _weights: &LayerWeights,
) -> Result<(), &'static str> {
    // kernel_dsv4_hc_expand4
    Ok(())
}
