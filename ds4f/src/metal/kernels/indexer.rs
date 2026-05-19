//! Ratio-4 indexer kernels: score computation, top-k selection, mask.

use anyhow::Result;
use crate::metal::{args::*, commands::CommandBatch, objc_ext, pipeline::PipelineCache, tensor::Tensor};
use super::dispatch;

/// Compute relevance scores between query heads and compressed KV entries (single token).
pub fn indexer_score_one(
    cache: &mut PipelineCache, batch: &mut CommandBatch,
    scores: &Tensor, q: &Tensor, weights: &Tensor, index_comp: &Tensor,
    n_comp: u32, n_head: u32, head_dim: u32, scale: f32,
) -> Result<()> {
    let (n_head, head_dim) = (n_head, head_dim);
    if n_head == 64 && head_dim == 128 {
        // Use the fused direct kernel for DS4's default indexer config
        indexer_score_one_direct(cache, batch, scores, q, weights, index_comp, n_comp, n_head, head_dim, scale)
    } else {
        // Generic path: matvec + weighted sum
        indexer_score_one_generic(cache, batch, scores, q, weights, index_comp, n_comp, n_head, head_dim, scale)
    }
}

fn indexer_score_one_direct(
    cache: &mut PipelineCache, batch: &mut CommandBatch,
    scores: &Tensor, q: &Tensor, weights: &Tensor, index_comp: &Tensor,
    n_comp: u32, n_head: u32, head_dim: u32, scale: f32,
) -> Result<()> {
    let args = IndexerScoresFusedArgs {
        n_comp,
        n_tokens: 1,
        n_head,
        head_dim,
        pos0: 0,
        ratio: 4,
        q_token_stride: n_head as u64 * head_dim as u64 * 4,
        q_head_stride: head_dim as u64 * 4,
        weights_token_stride: n_head as u64 * 4,
        index_row_stride: head_dim as u64 * 4,
        score_token_stride: n_comp as u64 * 4,
        scale,
    };
    dispatch::dispatch(
        cache, batch, "kernel_dsv4_indexer_score_one_direct", &args,
        &[(&q.buffer, q.offset as usize), (&weights.buffer, weights.offset as usize),
          (&index_comp.buffer, index_comp.offset as usize), (&scores.buffer, scores.offset as usize)],
        Some((4 * 32 * std::mem::size_of::<f32>(), 0)),
        objc_ext::mtl_size(n_comp as usize, 1, 1),
        objc_ext::mtl_size(32, 4, 1),
    )
}

fn indexer_score_one_generic(
    cache: &mut PipelineCache, batch: &mut CommandBatch,
    scores: &Tensor, q: &Tensor, weights: &Tensor, index_comp: &Tensor,
    n_comp: u32, n_head: u32, head_dim: u32, scale: f32,
) -> Result<()> {
    anyhow::bail!("indexer_score_one_generic not yet implemented")
}

/// Collapse per-head indexer scores using learned head weights.
pub fn indexer_weighted_sum(
    cache: &mut PipelineCache, batch: &mut CommandBatch,
    dst: &Tensor, scores: &Tensor, weights: &Tensor,
    n_comp: u32, n_head: u32, head_dim: u32, scale: f32,
) -> Result<()> {
    let args = IndexerWeightedSumArgs {
        ne00: n_comp as i64,
        ne01: 1,
        ne02: n_head as i64,
        nb00: 4,
        nb01: n_comp as u64 * 4,
        nb02: n_comp as u64 * 4,
        ne10: n_head as i64,
        ne11: 1,
        nb10: 4,
        nb11: n_head as u64 * 4,
        ne0: n_comp as i64,
        ne1: 1,
        nb0: 4,
        nb1: n_comp as u64 * 4,
        scale,
    };
    let total = n_comp as usize;
    let threads = 128;
    let groups = (total + threads - 1) / threads;
    dispatch::dispatch(
        cache, batch, "kernel_dsv4_indexer_weighted_sum", &args,
        &[(&scores.buffer, scores.offset as usize), (&weights.buffer, weights.offset as usize),
          (&dst.buffer, dst.offset as usize)],
        None, objc_ext::mtl_size(groups, 1, 1), objc_ext::mtl_size(threads, 1, 1),
    )
}

/// Top-k selection from indexer scores: argsort + merge for each token row.
pub fn indexer_topk(
    cache: &mut PipelineCache, batch: &mut CommandBatch,
    selected: &Tensor, scores: &Tensor,
    n_comp: u32, n_tokens: u32, top_k: u32,
) -> Result<()> {
    // Argsort each row of scores in descending order
    super::misc::argsort_f32_i32_desc(cache, batch, selected, scores, n_tokens, n_comp)?;

    // No merge step needed for single-pass argsort when n_comp <= 1024
    Ok(())
}

/// Build attention mask from top-k compressed indices.
pub fn topk_mask(
    cache: &mut PipelineCache, batch: &mut CommandBatch,
    mask: &Tensor, topk: &Tensor,
    n_tokens: u32, n_comp: u32, top_k: u32,
) -> Result<()> {
    // Fill mask with -inf
    let n_elems = n_tokens as u64 * n_comp as u64;
    super::misc::unary_fill(cache, batch, mask, f32::NEG_INFINITY, n_elems)?;

    // Scatter enabled positions (write 0.0)
    let args = TopkMaskArgs {
        ne00: top_k as i64,
        ne01: n_tokens as i64,
        nb00: 4,
        nb01: top_k as u64 * 4,
        ne0: n_comp as i64,
        ne1: n_tokens as i64,
        nb0: 4,
        nb1: n_comp as u64 * 4,
    };
    dispatch::dispatch(
        cache, batch, "kernel_dsv4_topk_mask_scatter", &args,
        &[(&topk.buffer, topk.offset as usize), (&mask.buffer, mask.offset as usize)],
        None, objc_ext::mtl_size(top_k as usize * n_tokens as usize, 1, 1),
        objc_ext::mtl_size(64, 1, 1),
    )
}
