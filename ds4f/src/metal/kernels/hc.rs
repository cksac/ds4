//! Hyper-Connection kernels: split, weighted sum, expand, fused tails.

use anyhow::Result;
use crate::metal::{args::*, buffers::ModelViews, commands::CommandBatch, objc_ext, pipeline::PipelineCache, tensor::Tensor};
use super::dispatch;

pub fn hc_split_sinkhorn(
    cache: &mut PipelineCache, batch: &mut CommandBatch, model_views: &ModelViews,
    model_map: *const std::ffi::c_void, model_size: u64,
    scale_offset: u64, base_offset: u64,
    out: &Tensor, mix: &Tensor, n_hc: u32, sinkhorn_iters: u32, eps: f32,
) -> Result<()> {
    let scale_bytes = n_hc as u64 * std::mem::size_of::<f32>() as u64;
    let n_rows = mix.bytes / (n_hc as u64 * n_hc as u64 * 4); // mix_hc * n_hc floats per row
    let (sbuf, soff) = model_views.wrap_model_range(model_map, model_size, scale_offset, scale_bytes)?;
    let (bbuf, boff) = model_views.wrap_model_range(model_map, model_size, base_offset, scale_bytes)?;
    let mix_hc = (mix.bytes / (n_rows * 4)) as i64; // total entries per row in mix
    let args = HcSplitSinkhornArgs {
        n_hc: n_hc as i32, sinkhorn_iters: sinkhorn_iters as i32,
        n_rows: n_rows as i64, mix_hc,
        nb01: n_hc as u64 * n_hc as u64 * 4, nb1: n_hc as u64 * 4,
        eps,
    };
    dispatch::dispatch(
        cache, batch, "kernel_dsv4_hc_split_sinkhorn", &args,
        &[(&mix.buffer, mix.offset as usize), (&sbuf, soff as usize),
          (&bbuf, boff as usize), (&out.buffer, out.offset as usize)],
        None,
        objc_ext::mtl_size((n_hc * n_hc * 3) as usize, n_rows as usize, 1),
        objc_ext::mtl_size(1, 1, 1),
    )
}

pub fn hc_weighted_sum(
    cache: &mut PipelineCache, batch: &mut CommandBatch,
    out: &Tensor, residual_hc: &Tensor, weights: &Tensor,
    n_embd: u32, n_hc: u32,
) -> Result<()> {
    let n_tokens: i64 = 1;
    let args = HcWeightedSumArgs {
        n_embd: n_embd as i64, n_hc: n_hc as i64, n_tokens,
        nb_x0: 4, nb_x1: n_embd as u64 * 4,
        nb_x2: n_hc as u64 * n_embd as u64 * 4,
        nb_w0: 4, nb_w1: n_hc as u64 * 4,
        nb0: 4, nb1: n_embd as u64 * 4,
    };
    let n_elem = n_embd as usize * n_tokens as usize;
    let threads = 256usize.min(n_elem.max(1));
    let groups = (n_elem + threads - 1) / threads;
    dispatch::dispatch(
        cache, batch, "kernel_dsv4_hc_weighted_sum", &args,
        &[(&residual_hc.buffer, residual_hc.offset as usize),
          (&weights.buffer, weights.offset as usize), (&out.buffer, out.offset as usize)],
        None,
        objc_ext::mtl_size(groups, 1, 1),
        objc_ext::mtl_size(threads, 1, 1),
    )
}

pub fn hc_split_weighted_sum(
    cache: &mut PipelineCache, batch: &mut CommandBatch, model_views: &ModelViews,
    model_map: *const std::ffi::c_void, model_size: u64,
    scale_offset: u64, base_offset: u64,
    out: &Tensor, split: &Tensor, mix: &Tensor, residual_hc: &Tensor,
    n_embd: u32, n_hc: u32, sinkhorn_iters: u32, eps: f32,
) -> Result<()> {
    let scale_bytes = 3 * 4; // 3 floats (pre_scale, post_scale, comb_scale)
    let (sbuf, soff) = model_views.wrap_model_range(model_map, model_size, scale_offset, scale_bytes)?;
    let mix_hc = 2 * n_hc as u64 + n_hc as u64 * n_hc as u64;
    let (bbuf, boff) = model_views.wrap_model_range(model_map, model_size, base_offset, mix_hc * 4)?;
    let n_rows: i64 = 1;
    let nb_x1 = n_embd as u64 * 4;
    let nb_x2 = n_hc as u64 * n_embd as u64 * 4;
    let nb1 = n_embd as u64 * 4;
    let args = HcSplitWeightedSumArgs {
        n_embd: n_embd as i64, n_hc: n_hc as i32, sinkhorn_iters: sinkhorn_iters as i32,
        n_rows, mix_hc: mix_hc as i64,
        nb_mix1: mix_hc * 4, nb_split1: mix_hc * 4,
        nb_x0: 4, nb_x1, nb_x2, nb0: 4, nb1, eps,
    };
    dispatch::dispatch(
        cache, batch, "kernel_dsv4_hc_split_weighted_sum", &args,
        &[(&mix.buffer, mix.offset as usize), (&sbuf, soff as usize),
          (&bbuf, boff as usize), (&residual_hc.buffer, residual_hc.offset as usize),
          (&split.buffer, split.offset as usize), (&out.buffer, out.offset as usize)],
        Some((n_hc as usize * 4, 0)),
        objc_ext::mtl_size(n_rows as usize, 1, 1),
        objc_ext::mtl_size(256, 1, 1),
    )
}

pub fn hc_split_weighted_sum_norm(
    cache: &mut PipelineCache, batch: &mut CommandBatch, model_views: &ModelViews,
    model_map: *const std::ffi::c_void, model_size: u64,
    scale_offset: u64, base_offset: u64, norm_weight_offset: u64,
    out: &Tensor, norm_out: &Tensor, split: &Tensor, mix: &Tensor, residual_hc: &Tensor,
    n_embd: u32, n_hc: u32, sinkhorn_iters: u32, eps: f32, norm_eps: f32,
) -> Result<()> {
    let scale_bytes = 3 * 4; // 3 floats
    let norm_bytes = n_embd as u64 * 4;
    let mix_hc = 2 * n_hc as u64 + n_hc as u64 * n_hc as u64;
    let (sbuf, soff) = model_views.wrap_model_range(model_map, model_size, scale_offset, scale_bytes)?;
    let (bbuf, boff) = model_views.wrap_model_range(model_map, model_size, base_offset, mix_hc * 4)?;
    let (nbuf, noff) = model_views.wrap_model_range(model_map, model_size, norm_weight_offset, norm_bytes)?;
    let n_rows: i64 = 1;
    let nb_x1 = n_embd as u64 * 4;
    let nb_x2 = n_hc as u64 * n_embd as u64 * 4;
    let nb1 = n_embd as u64 * 4;
    let nb_norm1 = n_embd as u64 * 4;
    let args = HcSplitWeightedSumNormArgs {
        n_embd: n_embd as i64, n_hc: n_hc as i32, sinkhorn_iters: sinkhorn_iters as i32,
        n_rows, mix_hc: mix_hc as i64,
        nb_mix1: mix_hc * 4, nb_split1: mix_hc * 4,
        nb_x0: 4, nb_x1, nb_x2, nb0: 4, nb1, nb_norm1, eps, norm_eps,
    };
    dispatch::dispatch(
        cache, batch, "kernel_dsv4_hc_split_weighted_sum_norm4", &args,
        &[(&mix.buffer, mix.offset as usize), (&sbuf, soff as usize),
          (&bbuf, boff as usize), (&residual_hc.buffer, residual_hc.offset as usize),
          (&split.buffer, split.offset as usize), (&out.buffer, out.offset as usize),
          (&nbuf, noff as usize), (&norm_out.buffer, norm_out.offset as usize)],
        Some((n_hc as usize * 4 + 32 * std::mem::size_of::<f32>(), 0)),
        objc_ext::mtl_size(n_rows as usize, 1, 1),
        objc_ext::mtl_size(1024, 1, 1),
    )
}

pub fn hc_expand(
    cache: &mut PipelineCache, batch: &mut CommandBatch,
    out_hc: &Tensor, block_out: &Tensor, residual_hc: &Tensor,
    post: &Tensor, comb: &Tensor, n_embd: u32, n_hc: u32,
) -> Result<()> {
    let n_tokens: i64 = 1;
    let args = HcExpandArgs {
        n_embd: n_embd as i64, n_hc: n_hc as i64, n_tokens,
        nb_block0: 4, nb_block1: n_embd as u64 * 4,
        nb_add0: 4, nb_add1: n_embd as u64 * 4,
        nb_res0: 4, nb_res1: n_embd as u64 * 4,
        nb_res2: n_hc as u64 * n_embd as u64 * 4,
        nb_post0: 4, nb_post1: n_hc as u64 * 4,
        nb_comb0: 4, nb_comb1: n_hc as u64 * 4,
        nb_comb2: n_hc as u64 * n_hc as u64 * 4,
        nb0: 4, nb1: n_embd as u64 * 4,
        nb2: n_hc as u64 * n_embd as u64 * 4,
        has_add: 0,
    };
    let (kernel, n_elem) = if n_hc == 4 {
        ("kernel_dsv4_hc_expand4", n_embd as usize * n_tokens as usize)
    } else {
        ("kernel_dsv4_hc_expand", n_embd as usize * n_hc as usize * n_tokens as usize)
    };
    let threads = 256usize.min(n_elem.max(1));
    let groups = (n_elem + threads - 1) / threads;
    // C sets block_out at index 5 as a dummy (unused when has_add=0)
    dispatch::dispatch(
        cache, batch, kernel, &args,
        &[(&block_out.buffer, block_out.offset as usize),
          (&residual_hc.buffer, residual_hc.offset as usize),
          (&post.buffer, post.offset as usize),
          (&comb.buffer, comb.offset as usize),
          (&block_out.buffer, block_out.offset as usize),
          (&out_hc.buffer, out_hc.offset as usize)],
        None,
        objc_ext::mtl_size(groups, 1, 1),
        objc_ext::mtl_size(threads, 1, 1),
    )
}

pub fn hc_expand_add_split(
    cache: &mut PipelineCache, batch: &mut CommandBatch,
    out_hc: &Tensor, block_out: &Tensor, block_add: &Tensor,
    residual_hc: &Tensor, split: &Tensor, n_embd: u32, n_hc: u32,
) -> Result<()> {
    let n_tokens: i64 = 1;
    let mix_hc = 2 * n_hc as u64 + n_hc as u64 * n_hc as u64;
    // Split layout (per token): pre[0..n_hc), post[n_hc..2*n_hc), comb[2*n_hc..mix_hc)
    let post_offset = n_hc as u64 * 4;
    let comb_offset = 2 * n_hc as u64 * 4;
    let args = HcExpandArgs {
        n_embd: n_embd as i64, n_hc: n_hc as i64, n_tokens,
        nb_block0: 4, nb_block1: n_embd as u64 * 4,
        nb_add0: 4, nb_add1: n_embd as u64 * 4,
        nb_res0: 4, nb_res1: n_embd as u64 * 4,
        nb_res2: n_hc as u64 * n_embd as u64 * 4,
        nb_post0: 4, nb_post1: mix_hc * 4,
        nb_comb0: 4, nb_comb1: n_hc as u64 * 4,
        nb_comb2: mix_hc * 4,
        nb0: 4, nb1: n_embd as u64 * 4,
        nb2: n_hc as u64 * n_embd as u64 * 4,
        has_add: 1,
    };
    let n_elem = n_embd as usize * n_tokens as usize;
    let threads = 256usize.min(n_elem.max(1));
    let groups = (n_elem + threads - 1) / threads;
    dispatch::dispatch(
        cache, batch, "kernel_dsv4_hc_expand4", &args,
        &[(&block_out.buffer, block_out.offset as usize),
          (&residual_hc.buffer, residual_hc.offset as usize),
          (&split.buffer, split.offset as usize + post_offset as usize),
          (&split.buffer, split.offset as usize + comb_offset as usize),
          (&block_add.buffer, block_add.offset as usize),
          (&out_hc.buffer, out_hc.offset as usize)],
        None,
        objc_ext::mtl_size(groups, 1, 1),
        objc_ext::mtl_size(threads, 1, 1),
    )
}
