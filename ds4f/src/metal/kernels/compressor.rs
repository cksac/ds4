//! KV compression kernels: store, update, and prefill.

use anyhow::Result;
use crate::metal::{args::*, buffers::ModelViews, commands::CommandBatch, objc_ext, pipeline::PipelineCache, tensor::Tensor};
use super::dispatch;

/// Store one token's KV+score into the rolling compressor state with APE projection.
pub fn compressor_store_one(
    cache: &mut PipelineCache, batch: &mut CommandBatch, model_views: &ModelViews,
    model_map: *const std::ffi::c_void, model_size: u64,
    ape_offset: u64, ape_type: u32,
    kv_cur: &Tensor, sc_cur: &Tensor,
    state_kv: &Tensor, state_score: &Tensor,
    head_dim: u32, ratio: u32, pos: u32,
) -> Result<()> {
    let ape_bytes = head_dim as u64 * 4 * 2; // APE: two rows of head_dim floats
    let (abuf, aoff) = model_views.wrap_model_range(model_map, model_size, ape_offset, ape_bytes)?;
    let args = CompressorStoreOneArgs {
        head_dim: head_dim as i32, ratio: ratio as i32, pos: pos as i32, ape_type: ape_type as i32,
    };
    let threads = dispatch::threads_1d(head_dim as usize);
    dispatch::dispatch(
        cache, batch, "kernel_dsv4_compressor_store_one", &args,
        &[(&kv_cur.buffer, kv_cur.offset as usize), (&sc_cur.buffer, sc_cur.offset as usize),
          (&abuf, aoff as usize), (&state_kv.buffer, state_kv.offset as usize),
          (&state_score.buffer, state_score.offset as usize)],
        None,
        objc_ext::mtl_size(head_dim as usize, 1, 1),
        objc_ext::mtl_size(threads, 1, 1),
    )
}

/// Fused softmax-weighted pooling of compressed KV rows.
pub fn softmax_pool(
    cache: &mut PipelineCache, batch: &mut CommandBatch,
    dst: &Tensor, kv: &Tensor, score: &Tensor,
    n_rows: u32, n_comp: u32, head_dim: u32,
) -> Result<()> {
    let kv_row_stride = head_dim as u64 * 4;
    let kv_plane_stride = n_rows as u64 * kv_row_stride;
    let args = SoftmaxPoolArgs {
        ne00: n_rows as i64,
        ne01: head_dim as i64,
        ne02: n_comp as i64,
        nb00: kv_row_stride,
        nb01: 4,
        nb02: kv_plane_stride,
        nb10: 4,
        nb11: 0,
        nb12: 0,
        ne0: head_dim as i64,
        ne1: n_comp as i64,
        nb0: 4,
        nb1: head_dim as u64 * 4,
    };
    let total = head_dim as usize * n_comp as usize;
    let threads = dispatch::threads_1d(total);
    let groups = (total + threads - 1) / threads;
    dispatch::dispatch(
        cache, batch, "kernel_dsv4_softmax_pool", &args,
        &[(&kv.buffer, kv.offset as usize), (&score.buffer, score.offset as usize),
          (&dst.buffer, dst.offset as usize)],
        None,
        objc_ext::mtl_size(groups, 1, 1),
        objc_ext::mtl_size(threads, 1, 1),
    )
}

/// Shift ratio-4 compressor state (copies second half over first half).
pub fn ratio4_shift(
    cache: &mut PipelineCache, batch: &mut CommandBatch,
    state_kv: &Tensor, state_score: &Tensor,
    head_dim: u32, ratio: u32,
) -> Result<()> {
    let args = Ratio4ShiftArgs { width: head_dim };
    let total = head_dim as usize * ratio as usize;
    let threads = 256usize;
    let groups = (total + threads - 1) / threads;
    dispatch::dispatch(
        cache, batch, "kernel_dsv4_ratio4_shift_f32", &args,
        &[(&state_kv.buffer, state_kv.offset as usize), (&state_score.buffer, state_score.offset as usize)],
        None,
        objc_ext::mtl_size(groups, 1, 1),
        objc_ext::mtl_size(threads, 1, 1),
    )
}

/// Decode-time compressor update: store token, pool if buffer full, apply norm+RoPE.
/// This orchestrates several sub-kernels in sequence.
pub fn compressor_update(
    cache: &mut PipelineCache, batch: &mut CommandBatch, model_views: &ModelViews,
    model_map: *const std::ffi::c_void, model_size: u64,
    ape_offset: u64, ape_type: u32, norm_offset: u64, norm_type: u32,
    kv_cur: &Tensor, sc_cur: &Tensor,
    state_kv: &Tensor, state_score: &Tensor,
    comp_cache: &Tensor,
    head_dim: u32, ratio: u32, pos: u32, comp_row: u32, n_rot: u32, n_ctx_orig: u32,
    freq_base: f32, freq_scale: f32, ext_factor: f32, attn_factor: f32,
    beta_fast: f32, beta_slow: f32, rms_eps: f32,
) -> Result<()> {
    // Step 1: Store new token into rolling state
    compressor_store_one(cache, batch, model_views, model_map, model_size,
        ape_offset, ape_type, kv_cur, sc_cur, state_kv, state_score,
        head_dim, ratio, pos)?;

    // Step 2: If the rolling buffer is full (pos % ratio == ratio-1), emit a compressed row
    if (pos + 1) % ratio == 0 {
        // Softmax-pool the rolling state
        let coff = if ratio == 4 { 2u32 } else { 1u32 };
        softmax_pool(cache, batch, comp_cache, state_kv, state_score, coff * ratio, 1, head_dim)?;

        // Apply RMS norm (weighted if norm_type != 0, plain otherwise)
        if norm_type != 0 {
            super::norm::rms_norm_weight(cache, batch, model_views, model_map, model_size,
                norm_offset, comp_cache, comp_cache, head_dim, rms_eps)?;
        } else {
            super::norm::rms_norm_plain(cache, batch, comp_cache, comp_cache, head_dim, rms_eps)?;
        }

        // Apply RoPE to the tail of the compressed row
        super::rope::rope_tail(cache, batch, comp_cache,
            1, 1, head_dim, n_rot, comp_row, n_ctx_orig, false,
            freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow)?;

        // Shift ratio-4 state for next round
        if ratio == 4 {
            ratio4_shift(cache, batch, state_kv, state_score, head_dim, ratio)?;
        }
    }

    Ok(())
}
