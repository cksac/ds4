//! RoPE tail and FP8 KV quantization kernels.

use anyhow::Result;
use crate::metal::{args::*, commands::CommandBatch, objc_ext, pipeline::PipelineCache, tensor::Tensor};
use super::dispatch;

pub fn rope_tail(
    cache: &mut PipelineCache, batch: &mut CommandBatch,
    x: &Tensor, n_tokens: u32, n_head: u32, head_dim: u32, n_rot: u32,
    pos0: u32, n_ctx_orig: u32, inverse: bool,
    freq_base: f32, freq_scale: f32, ext_factor: f32, attn_factor: f32,
    beta_fast: f32, beta_slow: f32,
) -> Result<()> {
    let row_bytes = head_dim as u64 * 4;
    let tok_bytes = n_head as u64 * row_bytes;
    let total_bytes = n_tokens as u64 * tok_bytes;
    let args = RopeTailArgs {
        ne00: head_dim as i64,
        ne01: n_head as i64,
        ne02: n_tokens as i64,
        ne03: 1,
        nb00: 4,
        nb01: row_bytes,
        nb02: tok_bytes,
        nb03: total_bytes,
        nb0: 4,
        nb1: row_bytes,
        nb2: tok_bytes,
        nb3: total_bytes,
        n_dims: n_rot as i32,
        mode: 0,
        n_ctx_orig: n_ctx_orig as i32,
        inverse: inverse as i32,
        freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow,
        src2: 0,
    };
    let total = n_tokens as usize * n_head as usize;
    dispatch::dispatch(
        cache, batch, "kernel_dsv4_rope_tail_f32", &args,
        &[(&x.buffer, x.offset as usize), (&x.buffer, x.offset as usize),
          (&x.buffer, x.offset as usize), (&x.buffer, x.offset as usize)],
        None,
        objc_ext::mtl_size(total, 1, 1),
        objc_ext::mtl_size(32, 1, 1),
    )
}

pub fn fp8_kv_quantize(
    cache: &mut PipelineCache, batch: &mut CommandBatch,
    x: &Tensor, n_tokens: u32, head_dim: u32, n_rot: u32,
) -> Result<()> {
    let row_bytes = head_dim as u64 * 4;
    let total_bytes = n_tokens as u64 * row_bytes;
    let args = Fp8KvQuantizeArgs {
        ne00: head_dim as i64,
        ne01: n_tokens as i64,
        ne02: 1,
        ne03: 1,
        nb00: 4,
        nb01: row_bytes,
        nb02: total_bytes,
        nb03: total_bytes,
        nb0: 4,
        nb1: row_bytes,
        nb2: total_bytes,
        nb3: total_bytes,
        n_rot: n_rot as i32,
    };
    let threads = 64usize;
    dispatch::dispatch(
        cache, batch, "kernel_dsv4_fp8_kv_quantize_f32", &args,
        &[(&x.buffer, x.offset as usize), (&x.buffer, x.offset as usize)],
        Some((threads * std::mem::size_of::<f32>(), 0)),
        objc_ext::mtl_size(n_tokens as usize, 1, 1),
        objc_ext::mtl_size(threads, 1, 1),
    )
}

pub fn kv_fp8_store_raw(
    cache: &mut PipelineCache, batch: &mut CommandBatch,
    kv: &Tensor, raw_cache: &Tensor, raw_cap: u32, row: u32,
    head_dim: u32, n_rot: u32,
) -> Result<()> {
    let _ = raw_cap;
    let args = KvFp8StoreArgs {
        head_dim: head_dim as i32, n_rot: n_rot as i32,
        raw_row: row as i32,
    };
    let threads = 64;
    dispatch::dispatch(
        cache, batch, "kernel_dsv4_kv_fp8_store_f32", &args,
        &[(&kv.buffer, kv.offset as usize), (&raw_cache.buffer, raw_cache.offset as usize)],
        Some((threads * std::mem::size_of::<f32>(), 0)),
        objc_ext::mtl_size(1, 1, 1),
        objc_ext::mtl_size(threads, 1, 1),
    )
}
