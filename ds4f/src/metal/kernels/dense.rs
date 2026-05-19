//! Dense matrix-vector and matrix-matrix multiplication kernels.
//! Q8_0, F16, F32 matvec + batched matmul + fused HC variants.

use anyhow::Result;
use crate::metal::{args::*, buffers::ModelViews, commands::CommandBatch, objc_ext, pipeline::PipelineCache, tensor::Tensor};
use super::dispatch;

/// Q8_0 matmul dispatching to matvec (1 tok), small-batch ext (2-8 tok), or matmul (>8 tok).
pub fn matmul_q8_0(
    cache: &mut PipelineCache, batch: &mut CommandBatch, model_views: &ModelViews,
    model_map: *const std::ffi::c_void, model_size: u64, weight_offset: u64,
    in_dim: u64, out_dim: u64, x: &Tensor, n_tok: u64,
    out: &Tensor,
) -> Result<()> {
    if n_tok == 0 { anyhow::bail!("matmul_q8_0: n_tok must be > 0"); }
    if n_tok == 1 {
        // Matvec path - single token decode
        matmul_q8_0_mv(cache, batch, model_views, model_map, model_size, weight_offset, in_dim, out_dim, x, out)
    } else if n_tok <= 8 {
        anyhow::bail!("matmul_q8_0: small-batch ext path not yet implemented");
    } else {
        anyhow::bail!("matmul_q8_0: matmul batched path not yet implemented");
    }
}

fn matmul_q8_0_mv(
    cache: &mut PipelineCache, batch: &mut CommandBatch, model_views: &ModelViews,
    model_map: *const std::ffi::c_void, model_size: u64, weight_offset: u64,
    in_dim: u64, out_dim: u64, x: &Tensor, out: &Tensor,
) -> Result<()> {
    let wbytes = in_dim * out_dim + out_dim * 2 * std::mem::size_of::<f32>() as u64; // Q8_0: qs + block scales
    let (wbuf, woff) = model_views.wrap_model_range(model_map, model_size, weight_offset, wbytes)?;
    let args = MulMvArgs {
        ne00: in_dim as i32, ne01: out_dim as i32, ne02: 1,
        nb00: in_dim as u64, nb01: in_dim * out_dim, nb02: in_dim * out_dim, nb03: in_dim * out_dim,
        ne10: in_dim as i32, ne11: 1, ne12: 1,
        nb10: 4, nb11: in_dim as u64 * 4, nb12: in_dim as u64 * 4, nb13: in_dim as u64 * 4,
        ne0: out_dim as i32, ne1: 1,
        nr0: 2, r2: 1i16, r3: 1i16,
    };
    let nsg = mv_nsg(out_dim as usize) as i32;
    let nr0 = 2u32; // N_R0_Q8_0
    let rows_per_tg = (out_dim as usize + nr0 as usize - 1) / nr0 as usize;
    let pipeline = cache.get_with_constants("kernel_mul_mv_q8_0_f32", &[(600, 37, nsg)])?;
    let (cb, is_batch) = batch.command_buffer()?;
    let enc = batch.compute_encoder(&cb, !is_batch)?;
    unsafe {
        objc_ext::enc_set_compute_pipeline_state(&enc, &pipeline);
        objc_ext::enc_set_bytes(&enc, &args as *const _ as _, std::mem::size_of_val(&args), 0);
        objc_ext::enc_set_buffer(&enc, &wbuf, woff as usize, 1);
        objc_ext::enc_set_buffer(&enc, &x.buffer, x.offset as usize, 2);
        objc_ext::enc_set_buffer(&enc, &out.buffer, out.offset as usize, 3);
        objc_ext::enc_set_threadgroup_memory_length(&enc, nsg as usize * 32 * nr0 as usize * std::mem::size_of::<f32>(), 0);
        objc_ext::enc_dispatch_threadgroups(&enc, objc_ext::mtl_size(rows_per_tg, 1, 1), objc_ext::mtl_size(32, nsg as usize, 1));
    }
    drop(enc);
    Ok(())
}

/// F16 matmul — uses matvec for single token decode.
pub fn matmul_f16(
    cache: &mut PipelineCache, batch: &mut CommandBatch, model_views: &ModelViews,
    model_map: *const std::ffi::c_void, model_size: u64, weight_offset: u64,
    in_dim: u64, out_dim: u64, x: &Tensor, n_tok: u64,
    out: &Tensor,
) -> Result<()> {
    if n_tok != 1 { anyhow::bail!("matmul_f16: only matvec (n_tok=1) implemented"); }
    let wbytes = in_dim * out_dim * 2; // f16 weights
    let (wbuf, woff) = model_views.wrap_model_range(model_map, model_size, weight_offset, wbytes)?;
    let args = MulMvArgs {
        ne00: in_dim as i32, ne01: out_dim as i32, ne02: 1,
        nb00: 2, nb01: in_dim as u64 * 2, nb02: in_dim as u64 * 2 * out_dim, nb03: in_dim as u64 * 2 * out_dim,
        ne10: in_dim as i32, ne11: 1, ne12: 1,
        nb10: 4, nb11: in_dim as u64 * 4, nb12: in_dim as u64 * 4, nb13: in_dim as u64 * 4,
        ne0: out_dim as i32, ne1: 1,
        nr0: 4, r2: 1i16, r3: 1i16,
    };
    let nsg = mv_nsg(out_dim as usize);
    let nr0 = 4u32; // N_R0 for F16
    let rows_per_tg = (out_dim as usize + nr0 as usize - 1) / nr0 as usize;
    dispatch::dispatch(
        cache, batch, "kernel_mul_mv_f16_f32", &args,
        &[(&wbuf, woff as usize), (&x.buffer, x.offset as usize), (&out.buffer, out.offset as usize)],
        Some((nsg * 32 * nr0 as usize * std::mem::size_of::<f32>(), 0)),
        objc_ext::mtl_size(rows_per_tg, 1, 1),
        objc_ext::mtl_size(32, nsg, 1),
    )
}

/// Fused shared expert gate+up Q8_0 projection + SwiGLU.
pub fn shared_gate_up_swiglu_q8_0(
    cache: &mut PipelineCache, batch: &mut CommandBatch, model_views: &ModelViews,
    model_map: *const std::ffi::c_void, model_size: u64,
    gate_offset: u64, up_offset: u64, in_dim: u64, out_dim: u64,
    x: &Tensor, gate: &Tensor, up: &Tensor, mid: &Tensor,
) -> Result<()> {
    let wbytes = in_dim * out_dim + out_dim * 2 * std::mem::size_of::<f32>() as u64;
    let (gbuf, goff) = model_views.wrap_model_range(model_map, model_size, gate_offset, wbytes)?;
    let (ubuf, uoff) = model_views.wrap_model_range(model_map, model_size, up_offset, wbytes)?;
    let args = MulMvArgs {
        ne00: in_dim as i32, ne01: out_dim as i32, ne02: 1,
        nb00: in_dim as u64, nb01: in_dim * out_dim, nb02: in_dim * out_dim, nb03: in_dim * out_dim,
        ne10: in_dim as i32, ne11: 1, ne12: 1,
        nb10: 4, nb11: in_dim * 4, nb12: in_dim * 4, nb13: in_dim * 4,
        ne0: out_dim as i32, ne1: 1,
        nr0: 2, r2: 1i16, r3: 1i16,
    };
    let nsg = mv_nsg(out_dim as usize);
    let nr0 = 2u32;
    let rows_per_tg = (out_dim as usize + nr0 as usize - 1) / nr0 as usize;
    dispatch::dispatch(
        cache, batch, "kernel_dsv4_shared_gate_up_swiglu_q8_0", &args,
        &[(&gbuf, goff as usize), (&ubuf, uoff as usize), (&x.buffer, x.offset as usize),
          (&gate.buffer, gate.offset as usize), (&up.buffer, up.offset as usize), (&mid.buffer, mid.offset as usize)],
        Some((nsg * 32 * nr0 as usize * std::mem::size_of::<f32>(), 0)),
        objc_ext::mtl_size(rows_per_tg, 1, 1),
        objc_ext::mtl_size(32, nsg, 1),
    )
}

pub fn mv_nsg(out_dim: usize) -> usize {
    match out_dim {
        0..=4096 => 4,
        4097..=8192 => 8,
        _ => 8,
    }
}

pub fn mul_mv_args(in_dim: u64, out_dim: u64, n_rows: u64) -> MulMvArgs {
    MulMvArgs {
        ne00: in_dim as i32, ne01: out_dim as i32, ne02: 1,
        nb00: in_dim as u64, nb01: in_dim * out_dim, nb02: in_dim * out_dim, nb03: in_dim * out_dim,
        ne10: in_dim as i32, ne11: n_rows as i32, ne12: 1,
        nb10: 4, nb11: in_dim * 4, nb12: in_dim * n_rows * 4, nb13: in_dim * n_rows * 4,
        ne0: out_dim as i32, ne1: n_rows as i32,
        nr0: 2, r2: n_rows as i16, r3: 1i16,
    }
}
