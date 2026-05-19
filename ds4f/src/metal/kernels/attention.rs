//! Flash attention and attention output projection kernels.

use anyhow::{Context, Result};
use crate::metal::{args::*, buffers::*, commands::CommandBatch, objc_ext, pipeline::PipelineCache, tensor::Tensor};
use super::dispatch;

pub fn attention_decode_heads(
    cache: &mut PipelineCache, batch: &mut CommandBatch, model_views: &ModelViews,
    model_map: *const std::ffi::c_void, model_size: u64, sinks_offset: u64,
    heads: &Tensor, q: &Tensor, raw_kv: &Tensor,
    n_raw: u32, raw_cap: u32, raw_start: u32,
    comp_kv: Option<&Tensor>, n_comp: u32, comp_mask: Option<&Tensor>,
    use_mask: bool, n_head: u32, head_dim: u32,
    fa_mask: &ScratchBuffer, fa_kv: &ScratchBuffer,
    fa_pad: &ScratchBuffer, fa_tmp: &ScratchBuffer,
    device: &objc2::runtime::AnyObject,
) -> Result<()> {
    if head_dim != 512 { anyhow::bail!("head_dim must be 512"); }
    if n_head == 0 || n_raw == 0 || raw_cap < n_raw { anyhow::bail!("invalid attention params"); }
    if n_comp == 0 {
        attention_decode_raw(cache, batch, model_views, model_map, model_size, sinks_offset,
            heads, q, raw_kv, n_raw as usize, raw_cap as usize, raw_start as usize,
            n_head as usize, head_dim as usize,
            fa_mask, fa_kv, fa_pad, fa_tmp, device)
    } else {
        anyhow::bail!("attention with compressed KV not yet implemented");
    }
}

fn attention_decode_raw(
    cache: &mut PipelineCache, batch: &mut CommandBatch, model_views: &ModelViews,
    model_map: *const std::ffi::c_void, model_size: u64, sinks_offset: u64,
    heads: &Tensor, q: &Tensor, raw_kv: &Tensor,
    n_raw: usize, raw_cap: usize, raw_start: usize,
    n_head: usize, head_dim: usize,
    fa_mask: &ScratchBuffer, fa_kv: &ScratchBuffer,
    fa_pad: &ScratchBuffer, fa_tmp: &ScratchBuffer,
    device: &objc2::runtime::AnyObject,
) -> Result<()> {
    let ncpsg: usize = 32; let nwg: usize = 32;
    let nsg = attn_vec_nsg(n_raw, nwg, ncpsg);
    let row_bytes_f16 = head_dim * 2;
    let need_pad = n_raw % ncpsg != 0;

    // Ensure scratch buffers (via shared ref — RefCell interior mutability)
    fa_mask.ensure(device, n_raw * 2)?;
    fa_kv.ensure(device, n_raw * row_bytes_f16)?;
    let pad_bytes = 2 * ncpsg * row_bytes_f16 + ncpsg * 2;
    fa_pad.ensure(device, pad_bytes)?;
    let tmp_bytes = n_head * head_dim * nwg * std::mem::size_of::<f32>()
        + n_head * 2 * nwg * std::mem::size_of::<f32>();
    fa_tmp.ensure(device, tmp_bytes)?;

    // Hold Ref borrows for the scope of this function
    let mask_borrow = fa_mask.buffer.borrow();
    let mask_buf = mask_borrow.as_ref().unwrap();
    let kv_borrow = fa_kv.buffer.borrow();
    let kv_buf = kv_borrow.as_ref().unwrap();
    let pad_borrow = fa_pad.buffer.borrow();
    let pad_buf = pad_borrow.as_ref().unwrap();
    let tmp_borrow = fa_tmp.buffer.borrow();
    let tmp_buf = tmp_borrow.as_ref().unwrap();

    unsafe { std::ptr::write_bytes(objc_ext::buffer_contents(mask_buf), 0, n_raw * 2); }

    let sink_bytes = n_head * std::mem::size_of::<f32>();
    let (sinks_buf, sinks_off) = model_views.wrap_model_range(model_map, model_size, sinks_offset, sink_bytes as u64)?;

    // ── Step 0: Copy raw F32 KV → F16 scratch buffer ─────────────────
    let cpy_n = (n_raw * head_dim) as u64;
    let cpy_threads: usize = 256;
    let cpy_groups = ((cpy_n as usize) + cpy_threads - 1) / cpy_threads;
    let cpy_pipeline = cache.get("kernel_cpy_f32_f16")?;
    let cpy_args = CpyArgs {
        nk0: cpy_n as i64,
        ne00: cpy_n as i64, ne01: 1, ne02: 1, ne03: 1,
        nb00: 4, nb01: cpy_n * 4, nb02: cpy_n * 4, nb03: cpy_n * 4,
        ne0: cpy_n as i64, ne1: 1, ne2: 1, ne3: 1,
        nb0: 2, nb1: cpy_n * 2, nb2: cpy_n * 2, nb3: cpy_n * 2,
    };
    dispatch::dispatch_with_pipeline(
        batch, &cpy_pipeline, &cpy_args,
        &[(&raw_kv.buffer, raw_kv.offset as usize), (kv_buf, 0)],
        None,
        objc_ext::mtl_size(cpy_groups, 1, 1),
        objc_ext::mtl_size(cpy_threads, 1, 1),
    )?;

    // ── Step 1: Pad K/V/mask (if n_raw not multiple of 32) ──────────
    if need_pad {
        let pad_constants: &[(usize, usize, i32)] = &[(100, 53, 1), (125, 29, ncpsg as i32)];
        let pad_pipeline = cache.get_with_constants("kernel_flash_attn_ext_pad", pad_constants)?;
        let pad_args = FlashAttnExtPadArgs {
            ne11: n_raw as i32, ne_12_2: 1, ne_12_3: 1,
            nb11: row_bytes_f16 as u64, nb12: (n_raw * row_bytes_f16) as u64, nb13: (n_raw * row_bytes_f16) as u64,
            nb21: row_bytes_f16 as u64, nb22: (n_raw * row_bytes_f16) as u64, nb23: (n_raw * row_bytes_f16) as u64,
            ne31: n_raw as i32, ne32: 1, ne33: 1, nb31: (n_raw * 2) as u64, nb32: (n_raw * 2) as u64, nb33: (n_raw * 2) as u64,
        };
        let (cb, is_batch) = batch.command_buffer()?;
        let enc = batch.compute_encoder(&cb, !is_batch)?;
        unsafe {
            objc_ext::enc_set_compute_pipeline_state(&enc, &pad_pipeline);
            objc_ext::enc_set_bytes(&enc, &pad_args as *const _ as _, std::mem::size_of_val(&pad_args), 0);
            objc_ext::enc_set_buffer(&enc, kv_buf, 0, 1);
            objc_ext::enc_set_buffer(&enc, kv_buf, 0, 2);
            objc_ext::enc_set_buffer(&enc, mask_buf, 0, 3);
            objc_ext::enc_set_buffer(&enc, pad_buf, 0, 4);
            objc_ext::enc_dispatch_threadgroups(&enc, objc_ext::mtl_size(ncpsg, 1, 1), objc_ext::mtl_size(32, 1, 1));
        }
        drop(enc);
    }

    // ── Step 2: Vectorized flash attention ──────────────────────────
    let has_kvpad: i32 = if need_pad { 1 } else { 0 };
    let vec_constants: &[(usize, usize, i32)] = &[
        (400, 53, 1), (401, 53, 1), (402, 53, 0), (403, 53, 0), (404, 53, has_kvpad),
        (420, 29, head_dim as i32), (421, 29, head_dim as i32),
        (422, 29, nsg as i32), (423, 29, nwg as i32),
    ];
    let vec_pipeline = cache.get_with_constants("kernel_flash_attn_ext_vec_f16_dk512_dv512", vec_constants)?;
    let vec_args = FlashAttnExtVecArgs {
        ne01: head_dim as i32, ne02: n_head as i32, ne03: 1,
        nb01: head_dim as u64 * 4, nb02: n_head as u64 * head_dim as u64 * 4, nb03: n_head as u64 * head_dim as u64 * 4,
        ne11: n_raw as i32, ne_12_2: 1, ne_12_3: 1,
        ns10: head_dim as i32,
        nb11: row_bytes_f16 as u64, nb12: (n_raw * row_bytes_f16) as u64, nb13: (n_raw * row_bytes_f16) as u64,
        ns20: head_dim as i32,
        nb21: row_bytes_f16 as u64, nb22: (n_raw * row_bytes_f16) as u64, nb23: (n_raw * row_bytes_f16) as u64,
        ne31: n_raw as i32, ne32: 1, ne33: 1,
        nb31: (n_raw * 2) as u64, nb32: (n_raw * 2) as u64, nb33: (n_raw * 2) as u64,
        ne1: n_head as i32, ne2: 1, ne3: 1,
        scale: 1.0 / (head_dim as f32).sqrt(),
        max_bias: 0.0, m0: 0.0, m1: 0.0, n_head_log2: 6, logit_softcap: 0.0,
    };

    let (cb, is_batch) = batch.command_buffer()?;
    let enc = batch.compute_encoder(&cb, !is_batch)?;
    unsafe {
        objc_ext::enc_set_compute_pipeline_state(&enc, &vec_pipeline);
        objc_ext::enc_set_bytes(&enc, &vec_args as *const _ as _, std::mem::size_of_val(&vec_args), 0);
        objc_ext::enc_set_buffer(&enc, &q.buffer, q.offset as usize, 1);
        objc_ext::enc_set_buffer(&enc, kv_buf, 0, 2);
        objc_ext::enc_set_buffer(&enc, kv_buf, 0, 3);
        objc_ext::enc_set_buffer(&enc, mask_buf, 0, 4);
        objc_ext::enc_set_buffer(&enc, &sinks_buf, sinks_off as usize, 5);
        objc_ext::enc_set_buffer(&enc, pad_buf, 0, 6);
        objc_ext::enc_set_buffer(&enc, tmp_buf, 0, 7);
        objc_ext::enc_set_threadgroup_memory_length(&enc, 32 * ncpsg * 2, 0);
        objc_ext::enc_dispatch_threadgroups(&enc, objc_ext::mtl_size(n_head, nwg, 1), objc_ext::mtl_size(32, nsg, 1));
    }
    drop(enc);

    // ── Step 3: Reduce split-K partial results ──────────────────────
    let reduce_constants: &[(usize, usize, i32)] = &[(500, 29, head_dim as i32), (501, 29, nwg as i32)];
    let reduce_pipeline = cache.get_with_constants("kernel_flash_attn_ext_vec_reduce", reduce_constants)?;
    let reduce_args = FlashAttnExtVecReduceArgs {
        nrows: head_dim as i32,
    };
    let (cb, is_batch) = batch.command_buffer()?;
    let enc = batch.compute_encoder(&cb, !is_batch)?;
    unsafe {
        objc_ext::enc_set_compute_pipeline_state(&enc, &reduce_pipeline);
        objc_ext::enc_set_bytes(&enc, &reduce_args as *const _ as _, std::mem::size_of_val(&reduce_args), 0);
        objc_ext::enc_set_buffer(&enc, tmp_buf, 0, 1);
        objc_ext::enc_set_buffer(&enc, &heads.buffer, heads.offset as usize, 2);
        objc_ext::enc_dispatch_threadgroups(&enc, objc_ext::mtl_size(n_head, 1, 1), objc_ext::mtl_size(head_dim, 1, 1));
    }
    drop(enc);
    Ok(())
}

pub fn attention_output_q8_batch(
    cache: &mut PipelineCache, batch: &mut CommandBatch, model_views: &ModelViews,
    model_map: *const std::ffi::c_void, model_size: u64,
    out_a_offset: u64, out_b_offset: u64,
    group_dim: u64, rank: u64, n_groups: u32, out_dim: u64,
    out: &Tensor, low: &Tensor, _group_tmp: &Tensor, _low_tmp: &Tensor,
    heads: &Tensor, n_tokens: u32,
) -> Result<()> {
    attention_output_low_q8(cache, batch, model_views, model_map, model_size,
        out_a_offset, group_dim, rank, n_groups, low, heads, n_tokens)?;
    let wbytes = rank * (n_groups as u64) * (out_dim / n_groups as u64);
    let (wbuf, woff) = model_views.wrap_model_range(model_map, model_size, out_b_offset, wbytes)?;
    let args = super::dense::mul_mv_args(rank, out_dim / n_groups as u64, n_groups as u64);
    let nsg = super::dense::mv_nsg(out_dim as usize);
    let pipeline = cache.get("kernel_mul_mv_q8_0_f32")?;
    let (cb, is_batch) = batch.command_buffer()?;
    let enc = batch.compute_encoder(&cb, !is_batch)?;
    unsafe {
        objc_ext::enc_set_compute_pipeline_state(&enc, &pipeline);
        objc_ext::enc_set_bytes(&enc, &args as *const _ as _, std::mem::size_of_val(&args), 0);
        objc_ext::enc_set_buffer(&enc, &wbuf, woff as usize, 1);
        objc_ext::enc_set_buffer(&enc, &low.buffer, low.offset as usize, 2);
        objc_ext::enc_set_buffer(&enc, &out.buffer, out.offset as usize, 3);
        objc_ext::enc_set_threadgroup_memory_length(&enc, nsg * 32 * 2 * std::mem::size_of::<f32>(), 0);
        objc_ext::enc_dispatch_threadgroups(&enc,
            objc_ext::mtl_size(out_dim as usize / (2 * nsg * 32).max(1) as usize, 1, 1),
            objc_ext::mtl_size(32, nsg, 1));
    }
    drop(enc);
    Ok(())
}

pub fn attention_output_low_q8(
    cache: &mut PipelineCache, batch: &mut CommandBatch, model_views: &ModelViews,
    model_map: *const std::ffi::c_void, model_size: u64,
    out_a_offset: u64, group_dim: u64, rank: u64, n_groups: u32,
    low: &Tensor, heads: &Tensor, _n_tokens: u32,
) -> Result<()> {
    let wbytes = group_dim * rank;
    let (wbuf, woff) = model_views.wrap_model_range(model_map, model_size, out_a_offset, wbytes)?;
    let pipeline = cache.get("kernel_mul_mv_id_q8_0_f32")?;
    let args = MulMvIdArgs {
        nei0: n_groups as i32, nei1: 1, nbi1: 4,
        ne00: group_dim as i32, ne01: rank as i32, ne02: n_groups as i32,
        nb00: group_dim as u64, nb01: wbytes, nb02: wbytes,
        ne10: group_dim as i32, ne11: 1, ne12: 1, ne13: 1,
        nb10: 4, nb11: group_dim as u64 * 4, nb12: group_dim as u64 * 4,
        ne0: rank as i32, ne1: 1,
        nb1: rank as u64 * 4,
        nr0: 2,
    };
    let nsg = super::dense::mv_nsg(rank as usize);
    let (cb, is_batch) = batch.command_buffer()?;
    let enc = batch.compute_encoder(&cb, !is_batch)?;
    unsafe {
        objc_ext::enc_set_compute_pipeline_state(&enc, &pipeline);
        objc_ext::enc_set_bytes(&enc, &args as *const _ as _, std::mem::size_of_val(&args), 0);
        objc_ext::enc_set_buffer(&enc, &wbuf, woff as usize, 1);
        objc_ext::enc_set_buffer(&enc, &heads.buffer, heads.offset as usize, 2);
        objc_ext::enc_set_buffer(&enc, &low.buffer, low.offset as usize, 3);
        objc_ext::enc_set_threadgroup_memory_length(&enc, nsg * 32 * 2 * std::mem::size_of::<f32>(), 0);
        objc_ext::enc_dispatch_threadgroups(&enc, objc_ext::mtl_size(rank as usize / 2, 1, 1), objc_ext::mtl_size(32, nsg, 1));
    }
    drop(enc);
    Ok(())
}

fn attn_vec_nsg(n_keys: usize, nwg: usize, ncpsg: usize) -> usize {
    let mut nsg = 1;
    while 2 * nwg * nsg * ncpsg < n_keys && nsg < 4 { nsg *= 2; }
    nsg
}
