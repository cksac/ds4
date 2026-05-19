//! RMS normalization kernels: plain, weighted, QKV fused, head norm.

use anyhow::Result;
use crate::metal::{args::*, buffers::ModelViews, commands::CommandBatch, objc_ext, pipeline::PipelineCache, tensor::Tensor};
use super::dispatch;

pub fn rms_norm_plain(
    cache: &mut PipelineCache, batch: &mut CommandBatch,
    out: &Tensor, x: &Tensor, n: u32, eps: f32,
) -> Result<()> { rms_norm_plain_rows(cache, batch, out, x, n, 1, eps) }

pub fn rms_norm_plain_rows(
    cache: &mut PipelineCache, batch: &mut CommandBatch,
    out: &Tensor, x: &Tensor, n: u32, rows: u32, eps: f32,
) -> Result<()> {
    if n == 0 || rows == 0 || (n & 3) != 0 { anyhow::bail!("invalid RMS norm dims"); }
    let args = RmsNormArgs::new(n, rows, eps);
    dispatch::dispatch(
        cache, batch, "kernel_rms_norm_f32_4", &args,
        &[(&x.buffer, x.offset as usize), (&x.buffer, x.offset as usize),
          (&x.buffer, x.offset as usize), (&out.buffer, out.offset as usize)],
        Some((32 * std::mem::size_of::<f32>(), 0)),
        objc_ext::mtl_size(rows as usize, 1, 1),
        objc_ext::mtl_size(dispatch::threads_pow2((n / 4) as usize), 1, 1),
    )
}

pub fn rms_norm_weight(
    cache: &mut PipelineCache, batch: &mut CommandBatch, model_views: &ModelViews,
    model_map: *const std::ffi::c_void, model_size: u64,
    weight_offset: u64, out: &Tensor, x: &Tensor, n: u32, eps: f32,
) -> Result<()> { rms_norm_weight_rows(cache, batch, model_views, model_map, model_size, weight_offset, out, x, n, 1, eps) }

pub fn rms_norm_weight_rows(
    cache: &mut PipelineCache, batch: &mut CommandBatch, model_views: &ModelViews,
    model_map: *const std::ffi::c_void, model_size: u64,
    weight_offset: u64, out: &Tensor, x: &Tensor, n: u32, rows: u32, eps: f32,
) -> Result<()> {
    if n == 0 || rows == 0 || (n & 3) != 0 { anyhow::bail!("invalid RMS norm dims"); }
    let row_bytes = n as u64 * std::mem::size_of::<f32>() as u64;
    let (wbuf, woff) = model_views.wrap_model_range(model_map, model_size, weight_offset, row_bytes)?;
    let args = RmsNormArgs::new(n, rows, eps);
    dispatch::dispatch(
        cache, batch, "kernel_rms_norm_mul_f32_4", &args,
        &[(&x.buffer, x.offset as usize), (&wbuf, woff as usize),
          (&x.buffer, x.offset as usize), (&out.buffer, out.offset as usize)],
        Some((32 * std::mem::size_of::<f32>(), 0)),
        objc_ext::mtl_size(rows as usize, 1, 1),
        objc_ext::mtl_size(dispatch::threads_pow2((n / 4) as usize), 1, 1),
    )
}

pub fn qkv_rms_norm_rows(
    cache: &mut PipelineCache, batch: &mut CommandBatch, model_views: &ModelViews,
    model_map: *const std::ffi::c_void, model_size: u64,
    q_weight_offset: u64, kv_weight_offset: u64,
    q_out: &Tensor, kv_out: &Tensor, q: &Tensor, kv: &Tensor,
    q_n: u32, kv_n: u32, rows: u32, eps: f32,
) -> Result<()> {
    let q_row_bytes = q_n as u64 * 4;
    let kv_row_bytes = kv_n as u64 * 4;
    let (q_wbuf, q_woff) = model_views.wrap_model_range(model_map, model_size, q_weight_offset, q_row_bytes)?;
    let (kv_wbuf, kv_woff) = model_views.wrap_model_range(model_map, model_size, kv_weight_offset, kv_row_bytes)?;
    let args = QkvRmsNormArgs { q_n: q_n as i32, q_n4: (q_n / 4) as i32, kv_n: kv_n as i32, kv_n4: (kv_n / 4) as i32, q_row_stride: q_row_bytes, kv_row_stride: kv_row_bytes, eps };
    dispatch::dispatch(
        cache, batch, "kernel_dsv4_qkv_rms_norm_f32_4", &args,
        &[(&q.buffer, q.offset as usize), (&q_wbuf, q_woff as usize),
          (&q_out.buffer, q_out.offset as usize), (&kv.buffer, kv.offset as usize),
          (&kv_wbuf, kv_woff as usize), (&kv_out.buffer, kv_out.offset as usize)],
        Some((32 * std::mem::size_of::<f32>(), 0)),
        // Grid: x=rows, y=2 (0=Q, 1=KV)
        objc_ext::mtl_size(rows as usize, 2, 1),
        objc_ext::mtl_size(dispatch::threads_pow2((q_n.max(kv_n) / 4) as usize), 1, 1),
    )
}

pub fn head_rms_norm(
    cache: &mut PipelineCache, batch: &mut CommandBatch,
    x: &Tensor, n_tok: u32, n_head: u32, head_dim: u32, eps: f32,
) -> Result<()> {
    let args = RmsNormArgs::new_3d(head_dim, n_head, n_tok, eps);
    let ne00_t = (head_dim / 4) as usize;
    let nth = dispatch::threads_pow2(ne00_t);
    dispatch::dispatch(
        cache, batch, "kernel_rms_norm_f32_4", &args,
        &[(&x.buffer, x.offset as usize), (&x.buffer, x.offset as usize),
          (&x.buffer, x.offset as usize), (&x.buffer, x.offset as usize)],
        Some((32 * std::mem::size_of::<f32>(), 0)),
        // 3D grid: dim0=nef1(=n_head), dim1=nef2(=n_tok)
        objc_ext::mtl_size(n_head as usize, n_tok as usize, 1),
        objc_ext::mtl_size(nth, 1, 1),
    )
}
