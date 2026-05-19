//! Router selection and Mixture-of-Experts kernels.

use anyhow::Result;
use crate::metal::{args::*, buffers::ModelViews, commands::CommandBatch, objc_ext, pipeline::PipelineCache, tensor::Tensor};
use super::dispatch;

pub fn router_select(
    cache: &mut PipelineCache, batch: &mut CommandBatch, model_views: &ModelViews,
    model_map: *const std::ffi::c_void, model_size: u64,
    bias_offset: u64, hash_offset: u64, hash_rows: u32,
    n_expert_groups: u32, n_group_used: u32, has_bias: bool, hash_mode: bool,
    token: u32, logits: &Tensor, selected: &Tensor, weights: &Tensor, probs: &Tensor,
) -> Result<()> {
    let (bbuf, boff) = if has_bias {
        let bbytes = n_expert_groups as u64 * 4;
        let (b, o) = model_views.wrap_model_range(model_map, model_size, bias_offset, bbytes)?;
        (Some(b), Some(o))
    } else { (None, None) };
    let (hbuf, hoff) = if hash_mode {
        let hbytes = hash_rows as u64 * n_expert_groups as u64 * n_group_used as u64 * 4;
        let (h, o) = model_views.wrap_model_range(model_map, model_size, hash_offset, hbytes)?;
        (Some(h), Some(o))
    } else { (None, None) };
    let args = RouterSelectOneArgs {
        has_bias: if has_bias { 1u32 } else { 0u32 },
        hash_mode: if hash_mode { 1u32 } else { 0u32 },
        use_token_buffer: 0u32,
        token,
        hash_rows,
    };
    let bufs: Vec<_> = {
        let mut v = vec![(&logits.buffer, logits.offset as usize), (&selected.buffer, selected.offset as usize),
                         (&weights.buffer, weights.offset as usize), (&probs.buffer, probs.offset as usize)];
        if let (Some(b), Some(o)) = (&bbuf, boff) { v.push((b, o as usize)); }
        else { v.push((&logits.buffer, 0)); }
        if let (Some(h), Some(o)) = (&hbuf, hoff) { v.push((h, o as usize)); }
        else { v.push((&logits.buffer, 0)); }
        v
    };
    dispatch::dispatch(
        cache, batch, "kernel_dsv4_router_finalize_one", &args,
        &bufs,
        Some((n_expert_groups as usize * std::mem::size_of::<f32>(), 0)),
        objc_ext::mtl_size(1, 1, 1),
        objc_ext::mtl_size(256, 1, 1),
    )
}

pub fn router_weights_one(
    cache: &mut PipelineCache, batch: &mut CommandBatch,
    probs: &Tensor, selected: &Tensor, weights: &Tensor,
) -> Result<()> {
    dispatch::dispatch(
        cache, batch, "kernel_dsv4_router_weights_one", &(),
        &[(&probs.buffer, probs.offset as usize), (&selected.buffer, selected.offset as usize),
          (&weights.buffer, weights.offset as usize)],
        None,
        objc_ext::mtl_size(1, 1, 1),
        objc_ext::mtl_size(6, 1, 1),
    )
}

pub fn routed_moe_one(
    cache: &mut PipelineCache, batch: &mut CommandBatch, model_views: &ModelViews,
    model_map: *const std::ffi::c_void, model_size: u64,
    gate_offset: u64, up_offset: u64, down_offset: u64,
    gate_type: u32, down_type: u32,
    gate_expert_bytes: u64, gate_row_bytes: u64,
    down_expert_bytes: u64, down_row_bytes: u64,
    expert_in_dim: u32, expert_mid_dim: u32, out_dim: u32,
    x: &Tensor, selected: &Tensor, weights: &Tensor,
    n_expert: u32, clamp: f32,
    out: &Tensor, gate: &Tensor, up: &Tensor, mid: &Tensor,
) -> Result<()> {
    // Select kernel based on quantization type
    let (gate_kernel, up_kernel, down_kernel, pair_kernel, swiglu_kernel) = match gate_type {
        16 => { // IQ2_XXS
            ("kernel_mul_mv_id_iq2_xxs_f32", "kernel_mul_mv_id_iq2_xxs_pair_f32",
             "kernel_mul_mv_id_q8_0_f32", "kernel_mul_mv_id_iq2_xxs_pair_f32",
             "kernel_mul_mv_id_iq2_xxs_pair_swiglu_f32")
        }
        10 => { // Q2_K
            ("kernel_mul_mv_id_q2_k_f32", "kernel_mul_mv_id_q2_k_f32",
             "kernel_mul_mv_id_q8_0_f32", "kernel_mul_mv_id_q2_k_f32",
             "kernel_mul_mv_id_q2_k_f32")
        }
        12 => { // Q4_K
            ("kernel_mul_mv_id_q4_k_f32", "kernel_mul_mv_id_q4_k_pair_f32",
             "kernel_mul_mv_id_q8_0_f32", "kernel_mul_mv_id_q4_k_pair_f32",
             "kernel_mul_mv_id_q4_k_pair_swiglu_f32")
        }
        _ => anyhow::bail!("unsupported gate quant type: {}", gate_type),
    };

    let (gbuf, goff) = model_views.wrap_model_range(model_map, model_size, gate_offset, gate_expert_bytes)?;
    let (ubuf, uoff) = model_views.wrap_model_range(model_map, model_size, up_offset, gate_expert_bytes)?;
    let (dbuf, doff) = model_views.wrap_model_range(model_map, model_size, down_offset, down_expert_bytes)?;

    let n_total_experts: i32 = 256;
    let gate_nb00 = gate_row_bytes / std::cmp::max(1u64, expert_in_dim as u64 / 256);
    let gate_nr0: i32 = 2;
    let nsg = if expert_mid_dim <= 4096 { 4usize } else { 8 };
    let args = MulMvIdArgs {
        nei0: n_expert as i32,
        nei1: 1,
        nbi1: n_expert as u64 * 4,
        ne00: expert_in_dim as i32,
        ne01: expert_mid_dim as i32,
        ne02: n_total_experts,
        nb00: gate_nb00,
        nb01: gate_row_bytes,
        nb02: gate_expert_bytes,
        ne10: expert_in_dim as i32,
        ne11: 1,
        ne12: 1,
        ne13: 1,
        nb10: 4,
        nb11: expert_in_dim as u64 * 4,
        nb12: expert_in_dim as u64 * 4,
        ne0: expert_mid_dim as i32,
        ne1: n_expert as i32,
        nb1: expert_mid_dim as u64 * 4,
        nr0: gate_nr0,
    };

    // Gate projection (matvec per expert) — compile with correct nsg
    let gate_pipeline = cache.get_with_constants(gate_kernel, &[(600, 37, nsg as i32)])?;
    let gate_row_groups = (expert_mid_dim as usize + gate_nr0 as usize - 1) / gate_nr0 as usize;
    dispatch::dispatch_with_pipeline(
        batch, &gate_pipeline, &args,
        &[(&gbuf, goff as usize), (&x.buffer, x.offset as usize),
          (&gate.buffer, gate.offset as usize), (&selected.buffer, selected.offset as usize)],
        Some((nsg * 32 * gate_nr0 as usize * std::mem::size_of::<f32>(), 0)),
        objc_ext::mtl_size(gate_row_groups, 1, n_expert as usize),
        objc_ext::mtl_size(32, nsg, 1),
    )?;

    // SwiGLU activation
    super::misc::swiglu(cache, batch, mid, gate, up, expert_mid_dim, clamp, 1.5)?;

    // Down projection
    let down_nb00 = down_row_bytes / std::cmp::max(1u64, expert_mid_dim as u64 / 256);
    let down_nr0: i32 = 2;
    let d_nsg = if out_dim <= 4096 { 4usize } else { 8 };
    let down_args = MulMvIdArgs {
        nei0: n_expert as i32,
        nei1: 1,
        nbi1: n_expert as u64 * 4,
        ne00: expert_mid_dim as i32,
        ne01: out_dim as i32,
        ne02: n_total_experts,
        nb00: down_nb00,
        nb01: down_row_bytes,
        nb02: down_expert_bytes,
        ne10: expert_mid_dim as i32,
        ne11: n_expert as i32,
        ne12: 1,
        ne13: 1,
        nb10: 4,
        nb11: expert_mid_dim as u64 * 4,
        nb12: n_expert as u64 * expert_mid_dim as u64 * 4,
        ne0: out_dim as i32,
        ne1: n_expert as i32,
        nb1: out_dim as u64 * 4,
        nr0: down_nr0,
    };
    let down_pipeline = cache.get_with_constants(down_kernel, &[(600, 37, d_nsg as i32)])?;
    let down_row_groups = (out_dim as usize + down_nr0 as usize - 1) / down_nr0 as usize;
    dispatch::dispatch_with_pipeline(
        batch, &down_pipeline, &down_args,
        &[(&dbuf, doff as usize), (&mid.buffer, mid.offset as usize),
          (&out.buffer, out.offset as usize), (&selected.buffer, selected.offset as usize)],
        Some((d_nsg * 32 * down_nr0 as usize * std::mem::size_of::<f32>(), 0)),
        objc_ext::mtl_size(down_row_groups, 1, n_expert as usize),
        objc_ext::mtl_size(32, d_nsg, 1),
    )?;

    Ok(())
}
