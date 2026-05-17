//! RMS normalization kernels.

use anyhow::Result;
use crate::metal::{args::*, commands::CommandBatch, objc_ext, pipeline::PipelineCache, tensor::Tensor};

pub fn rms_norm_plain(
    cache: &mut PipelineCache, batch: &mut CommandBatch,
    out: &Tensor, x: &Tensor, n: u32, eps: f32,
) -> Result<()> {
    rms_norm_plain_rows(cache, batch, out, x, n, 1, eps)
}

pub fn rms_norm_plain_rows(
    cache: &mut PipelineCache, batch: &mut CommandBatch,
    out: &Tensor, x: &Tensor, n: u32, rows: u32, eps: f32,
) -> Result<()> {
    if n == 0 || rows == 0 || (n & 3) != 0 {
        anyhow::bail!("invalid RMS norm dims: n={}, rows={}", n, rows);
    }

    let args = RmsNormArgs::new(n, rows, eps);
    let pipeline = cache.get("kernel_rms_norm_f32_4")?;
    let (cb, is_batch) = batch.command_buffer()?;
    let enc = batch.compute_encoder(&cb, !is_batch)?;

    unsafe {
        objc_ext::enc_set_compute_pipeline_state(&enc, &pipeline);
        objc_ext::enc_set_bytes(&enc, &args as *const _ as _, std::mem::size_of_val(&args), 0);
        objc_ext::enc_set_buffer(&enc, &x.buffer, x.offset as usize, 1);
        objc_ext::enc_set_buffer(&enc, &x.buffer, x.offset as usize, 2);
        objc_ext::enc_set_buffer(&enc, &x.buffer, x.offset as usize, 3);
        objc_ext::enc_set_buffer(&enc, &out.buffer, out.offset as usize, 4);
        objc_ext::enc_set_threadgroup_memory_length(&enc, 32 * std::mem::size_of::<f32>(), 0);
        objc_ext::enc_dispatch_threadgroups(
            &enc,
            objc_ext::mtl_size(rows as usize, 1, 1),
            objc_ext::mtl_size(threads_for_norm(n), 1, 1),
        );
    }
    drop(enc);
    Ok(())
}

fn threads_for_norm(n: u32) -> usize {
    let ne00_t = (n / 4) as usize;
    let mut nth: usize = 32;
    while nth < ne00_t && nth < 1024 { nth *= 2; }
    if nth > ne00_t { nth = ne00_t; }
    if nth == 0 { 1 } else { nth }
}
