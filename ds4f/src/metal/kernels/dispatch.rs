//! Shared dispatch helpers for all kernel modules.
//!
//! Every kernel dispatch follows the same pattern: make args, get pipeline,
//! get encoder, set bytes+buffers, dispatch, drop encoder.

use anyhow::Result;
use crate::metal::{commands::CommandBatch, objc_ext, pipeline::PipelineCache, tensor::Tensor};
use objc2::rc::Retained;
use objc2::runtime::AnyObject;
use objc2_metal::MTLSize;

/// Compute the number of threads per threadgroup for 1D elementwise ops.
pub fn threads_1d(n: usize) -> usize { 256usize.min(n.max(1)) }

/// Compute threadgroup size for reduction-style kernels (RMS norm etc.)
/// that need powers of two starting at 32.
pub fn threads_pow2(n: usize) -> usize {
    let mut nth = 32usize;
    while nth < n && nth < 1024 { nth *= 2; }
    if nth > n { n } else { nth.max(1) }
}

/// Dispatch using an already-compiled pipeline (no cache lookup).
pub fn dispatch_with_pipeline<A: Sized>(
    batch: &mut CommandBatch,
    pipeline: &Retained<AnyObject>,
    args: &A,
    bufs: &[(&Retained<AnyObject>, usize)],
    threadgroup_memory: Option<(usize, usize)>,
    grid: MTLSize,
    threads_per_group: MTLSize,
) -> Result<()> {
    let (cb, is_batch) = batch.command_buffer()?;
    let enc = batch.compute_encoder(&cb, !is_batch)?;
    unsafe {
        objc_ext::enc_set_compute_pipeline_state(&enc, pipeline);
        objc_ext::enc_set_bytes(&enc, args as *const A as *const _, std::mem::size_of::<A>(), 0);
        for (i, (buf, offset)) in bufs.iter().enumerate() {
            objc_ext::enc_set_buffer(&enc, buf, *offset, i + 1);
        }
        if let Some((len, idx)) = threadgroup_memory {
            objc_ext::enc_set_threadgroup_memory_length(&enc, len, idx);
        }
        objc_ext::enc_dispatch_threadgroups(&enc, grid, threads_per_group);
    }
    drop(enc);
    Ok(())
}

/// Dispatch a pipeline with args at index 0 and up to 6 buffers at indices 1..7.
#[allow(clippy::too_many_arguments)]
pub fn dispatch<A: Sized>(
    cache: &mut PipelineCache,
    batch: &mut CommandBatch,
    kernel_name: &str,
    args: &A,
    bufs: &[(&Retained<AnyObject>, usize)], // (buffer, offset)
    threadgroup_memory: Option<(usize, usize)>, // (length, index)
    grid: MTLSize,
    threads_per_group: MTLSize,
) -> Result<()> {
    let pipeline = cache.get(kernel_name)?;
    let (cb, is_batch) = batch.command_buffer()?;
    let enc = batch.compute_encoder(&cb, !is_batch)?;

    unsafe {
        objc_ext::enc_set_compute_pipeline_state(&enc, &pipeline);
        objc_ext::enc_set_bytes(&enc, args as *const A as *const _, std::mem::size_of::<A>(), 0);
        for (i, (buf, offset)) in bufs.iter().enumerate() {
            objc_ext::enc_set_buffer(&enc, buf, *offset, i + 1);
        }
        if let Some((len, idx)) = threadgroup_memory {
            objc_ext::enc_set_threadgroup_memory_length(&enc, len, idx);
        }
        objc_ext::enc_dispatch_threadgroups(&enc, grid, threads_per_group);
    }
    drop(enc);
    Ok(())
}
