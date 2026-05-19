//! Miscellaneous kernel dispatchers: swiglu, add, embed, copy, fill,
//! unary, softmax, argsort, get/set rows, sum_rows, concat, binary ops,
//! repeat, directional steering.

use anyhow::Result;
use crate::metal::{args::*, commands::CommandBatch, objc_ext, pipeline::PipelineCache, tensor::Tensor};
use super::dispatch;

// ── SwiGLU ───────────────────────────────────────────────────────────

pub fn swiglu(
    cache: &mut PipelineCache, batch: &mut CommandBatch,
    out: &Tensor, gate: &Tensor, up: &Tensor,
    n: u32, clamp: f32, weight: f32,
) -> Result<()> {
    let row_bytes = n as u64 * 4;
    let args = SwigluArgs {
        ne00: n as i32,
        nb01: row_bytes,
        ne10: n as i32,
        nb11: row_bytes,
        ne0: n as i32,
        nb1: row_bytes,
        i00: 0,
        i10: 0,
        alpha: clamp,
        limit: weight,
    };
    let pipeline = cache.get("kernel_swiglu_f32")?;
    let (cb, is_batch) = batch.command_buffer()?;
    let enc = batch.compute_encoder(&cb, !is_batch)?;
    let threads = dispatch::threads_1d(n as usize);
    unsafe {
        objc_ext::enc_set_compute_pipeline_state(&enc, &pipeline);
        objc_ext::enc_set_bytes(&enc, &args as *const _ as _, std::mem::size_of_val(&args), 0);
        objc_ext::enc_set_buffer(&enc, &gate.buffer, gate.offset as usize, 1);
        objc_ext::enc_set_buffer(&enc, &up.buffer, up.offset as usize, 2);
        objc_ext::enc_set_buffer(&enc, &out.buffer, out.offset as usize, 3);
        objc_ext::enc_dispatch_threadgroups(&enc, objc_ext::mtl_size(1, 1, 1), objc_ext::mtl_size(threads, 1, 1));
    }
    drop(enc);
    Ok(())
}

// ── Elementwise add ───────────────────────────────────────────────────

pub fn add(
    cache: &mut PipelineCache, batch: &mut CommandBatch,
    out: &Tensor, a: &Tensor, b: &Tensor, n: u32,
) -> Result<()> {
    bin_op(cache, batch, out, a, b, n, 0, 0) // OP_BIN_ADD
}

// ── Elementwise multiply ─────────────────────────────────────────────

pub fn mul(
    cache: &mut PipelineCache, batch: &mut CommandBatch,
    out: &Tensor, a: &Tensor, b: &Tensor, n: u32,
) -> Result<()> {
    bin_op(cache, batch, out, a, b, n, 2, 0) // OP_BIN_MUL
}

// ── Scalar multiply ───────────────────────────────────────────────────

pub fn mul_scalar(
    cache: &mut PipelineCache, batch: &mut CommandBatch,
    out: &Tensor, a: &Tensor, n: u32, scalar: f32,
) -> Result<()> {
    bin_op(cache, batch, out, a, a, n, 2, 0)
}

// ── Row-wise division ─────────────────────────────────────────────────

pub fn div_row(
    cache: &mut PipelineCache, batch: &mut CommandBatch,
    out: &Tensor, a: &Tensor, b: &Tensor, n: u32,
) -> Result<()> {
    bin_op(cache, batch, out, a, b, n, 5, 1)
}

fn bin_op(
    cache: &mut PipelineCache, batch: &mut CommandBatch,
    out: &Tensor, a: &Tensor, b: &Tensor, n: u32, _op: i32, _bcast: i32,
) -> Result<()> {
    let nb = n as u64 * 4;
    let args = BinArgs {
        ne00: n as i32, ne01: 1, ne02: 1, ne03: 1,
        nb00: nb, nb01: nb, nb02: nb, nb03: nb,
        ne10: n as i32, ne11: 1, ne12: 1, ne13: 1,
        nb10: nb, nb11: nb, nb12: nb, nb13: nb,
        ne0: n as i32, ne1: 1, ne2: 1, ne3: 1,
        nb0: nb, nb1: nb, nb2: nb, nb3: nb,
        offs: 0,
        o1: [0u64; 8],
    };
    let threads = dispatch::threads_1d(n as usize);
    let groups = (n as usize + threads - 1) / threads;
    dispatch::dispatch(
        cache, batch, "kernel_bin_fuse_f32_f32_f32", &args,
        &[(&a.buffer, a.offset as usize), (&b.buffer, b.offset as usize), (&out.buffer, out.offset as usize)],
        None, objc_ext::mtl_size(groups, 1, 1), objc_ext::mtl_size(threads, 1, 1),
    )
}

// ── Copy / conversion ─────────────────────────────────────────────────

pub fn copy_f32_f32(
    cache: &mut PipelineCache, batch: &mut CommandBatch,
    dst: &Tensor, src: &Tensor, n: u64,
) -> Result<()> {
    let args = CpyArgs {
        nk0: n as i64, ne00: n as i64, ne01: 1, ne02: 1, ne03: 1,
        nb00: 4, nb01: n * 4, nb02: n * 4, nb03: n * 4,
        ne0: n as i64, ne1: 1, ne2: 1, ne3: 1,
        nb0: 4, nb1: n * 4, nb2: n * 4, nb3: n * 4,
    };
    let threads = 256usize;
    let groups = (n as usize + threads - 1) / threads;
    dispatch::dispatch(
        cache, batch, "kernel_cpy_f32_f32", &args,
        &[(&src.buffer, src.offset as usize), (&dst.buffer, dst.offset as usize)],
        None, objc_ext::mtl_size(groups, 1, 1), objc_ext::mtl_size(threads, 1, 1),
    )
}

pub fn copy_f32_f16(
    cache: &mut PipelineCache, batch: &mut CommandBatch,
    dst: &Tensor, src: &Tensor, n: u64,
) -> Result<()> {
    let args = CpyArgs {
        nk0: n as i64, ne00: n as i64, ne01: 1, ne02: 1, ne03: 1,
        nb00: 4, nb01: n * 4, nb02: n * 4, nb03: n * 4,
        ne0: n as i64, ne1: 1, ne2: 1, ne3: 1,
        nb0: 2, nb1: n * 2, nb2: n * 2, nb3: n * 2,
    };
    let threads = 256usize;
    let groups = (n as usize + threads - 1) / threads;
    dispatch::dispatch(
        cache, batch, "kernel_cpy_f32_f16", &args,
        &[(&src.buffer, src.offset as usize), (&dst.buffer, dst.offset as usize)],
        None, objc_ext::mtl_size(groups, 1, 1), objc_ext::mtl_size(threads, 1, 1),
    )
}

pub fn copy_f16_f32(
    cache: &mut PipelineCache, batch: &mut CommandBatch,
    dst: &Tensor, src: &Tensor, n: u64,
) -> Result<()> {
    let args = CpyArgs {
        nk0: n as i64, ne00: n as i64, ne01: 1, ne02: 1, ne03: 1,
        nb00: 2, nb01: n * 2, nb02: n * 2, nb03: n * 2,
        ne0: n as i64, ne1: 1, ne2: 1, ne3: 1,
        nb0: 4, nb1: n * 4, nb2: n * 4, nb3: n * 4,
    };
    let threads = 256usize;
    let groups = (n as usize + threads - 1) / threads;
    dispatch::dispatch(
        cache, batch, "kernel_cpy_f16_f32", &args,
        &[(&src.buffer, src.offset as usize), (&dst.buffer, dst.offset as usize)],
        None, objc_ext::mtl_size(groups, 1, 1), objc_ext::mtl_size(threads, 1, 1),
    )
}

// ── Unary ops (with function constant specialization) ──────────────────
// The Metal shader uses function constant at index 1200 (FC_UNARY) to select
// the operation. Each variant needs its own pipeline compiled with the op number.
// FC_UNARY_CNT at index 1201 selects count mode (used by fill).

fn unary_dispatch(
    cache: &mut PipelineCache, batch: &mut CommandBatch,
    dst: &Tensor, src: &Tensor, args: &UnaryArgs, n: u64,
    op: i32, cnt: bool,
) -> Result<()> {
    let key = format!("kernel_unary_f32_f32_op{}_cnt{}", op, cnt as i32);
    let constants: &[(usize, usize, i32)] = &[(1200, 37, op), (1201, 53, cnt as i32)];
    // Use the real host function name; key is only for caching
    let pipeline = cache.get_with_constants("kernel_unary_f32_f32", constants)?;
    cache.put(&key, pipeline.clone());
    let threads = dispatch::threads_1d(n as usize);
    let groups = if cnt { 1usize } else { (n as usize + threads - 1) / threads };
    dispatch::dispatch_with_pipeline(
        batch, &pipeline, args,
        &[(&src.buffer, src.offset as usize), (&dst.buffer, dst.offset as usize)],
        None,
        objc_ext::mtl_size(groups, 1, 1),
        objc_ext::mtl_size(threads, 1, 1),
    )
}

pub fn unary_fill(
    cache: &mut PipelineCache, batch: &mut CommandBatch,
    dst: &Tensor, value: f32, n: u64,
) -> Result<()> {
    let args = UnaryArgs {
        ne00: n as i32, ne01: 1, ne02: 1, ne03: 1,
        nb00: 4, nb01: n * 4, nb02: n * 4, nb03: n * 4,
        ne0: n as i32, ne1: 1, ne2: 1, ne3: 1,
        nb0: 4, nb1: n * 4, nb2: n * 4, nb3: n * 4,
        slope: 0.0, scale: 0.0, bias: 0.0, val: value, min: 0.0, max: 0.0,
    };
    unary_dispatch(cache, batch, dst, dst, &args, n, 11, true) // OP_UNARY_NUM_FILL=11, cnt=true
}

pub fn unary_sigmoid(cache: &mut PipelineCache, batch: &mut CommandBatch, dst: &Tensor, src: &Tensor, n: u64) -> Result<()> {
    let args = UnaryArgs { ne00: n as i32, ne01: 1, ne02: 1, ne03: 1, nb00: 4, nb01: n * 4, nb02: n * 4, nb03: n * 4, ne0: n as i32, ne1: 1, ne2: 1, ne3: 1, nb0: 4, nb1: n * 4, nb2: n * 4, nb3: n * 4, slope: 0.0, scale: 0.0, bias: 0.0, val: 0.0, min: 0.0, max: 0.0 };
    unary_dispatch(cache, batch, dst, src, &args, n, 102, false)
}

pub fn unary_silu(cache: &mut PipelineCache, batch: &mut CommandBatch, dst: &Tensor, src: &Tensor, n: u64) -> Result<()> {
    let args = UnaryArgs { ne00: n as i32, ne01: 1, ne02: 1, ne03: 1, nb00: 4, nb01: n * 4, nb02: n * 4, nb03: n * 4, ne0: n as i32, ne1: 1, ne2: 1, ne3: 1, nb0: 4, nb1: n * 4, nb2: n * 4, nb3: n * 4, slope: 0.0, scale: 0.0, bias: 0.0, val: 0.0, min: 0.0, max: 0.0 };
    unary_dispatch(cache, batch, dst, src, &args, n, 106, false)
}

pub fn unary_softplus(cache: &mut PipelineCache, batch: &mut CommandBatch, dst: &Tensor, src: &Tensor, n: u64) -> Result<()> {
    let args = UnaryArgs { ne00: n as i32, ne01: 1, ne02: 1, ne03: 1, nb00: 4, nb01: n * 4, nb02: n * 4, nb03: n * 4, ne0: n as i32, ne1: 1, ne2: 1, ne3: 1, nb0: 4, nb1: n * 4, nb2: n * 4, nb3: n * 4, slope: 0.0, scale: 0.0, bias: 0.0, val: 0.0, min: 0.0, max: 0.0 };
    unary_dispatch(cache, batch, dst, src, &args, n, 115, false)
}

pub fn unary_sqrt(cache: &mut PipelineCache, batch: &mut CommandBatch, dst: &Tensor, src: &Tensor, n: u64) -> Result<()> {
    let args = UnaryArgs { ne00: n as i32, ne01: 1, ne02: 1, ne03: 1, nb00: 4, nb01: n * 4, nb02: n * 4, nb03: n * 4, ne0: n as i32, ne1: 1, ne2: 1, ne3: 1, nb0: 4, nb1: n * 4, nb2: n * 4, nb3: n * 4, slope: 0.0, scale: 0.0, bias: 0.0, val: 0.0, min: 0.0, max: 0.0 };
    unary_dispatch(cache, batch, dst, src, &args, n, 14, false)
}

pub fn unary_clamp(cache: &mut PipelineCache, batch: &mut CommandBatch, dst: &Tensor, src: &Tensor, n: u64, min: f32, max: f32) -> Result<()> {
    let args = UnaryArgs { ne00: n as i32, ne01: 1, ne02: 1, ne03: 1, nb00: 4, nb01: n * 4, nb02: n * 4, nb03: n * 4, ne0: n as i32, ne1: 1, ne2: 1, ne3: 1, nb0: 4, nb1: n * 4, nb2: n * 4, nb3: n * 4, slope: 0.0, scale: 0.0, bias: 0.0, val: 0.0, min, max };
    unary_dispatch(cache, batch, dst, src, &args, n, 12, false)
}

pub fn unary_scale(cache: &mut PipelineCache, batch: &mut CommandBatch, dst: &Tensor, src: &Tensor, n: u64, scale: f32, bias: f32) -> Result<()> {
    let args = UnaryArgs { ne00: n as i32, ne01: 1, ne02: 1, ne03: 1, nb00: 4, nb01: n * 4, nb02: n * 4, nb03: n * 4, ne0: n as i32, ne1: 1, ne2: 1, ne3: 1, nb0: 4, nb1: n * 4, nb2: n * 4, nb3: n * 4, slope: 0.0, scale, bias, val: 0.0, min: 0.0, max: 0.0 };
    unary_dispatch(cache, batch, dst, src, &args, n, 10, false)
}

// ── Get rows / Set rows ──────────────────────────────────────────────

pub fn get_rows_f32(
    cache: &mut PipelineCache, batch: &mut CommandBatch,
    dst: &Tensor, src: &Tensor, ids: &Tensor,
    n_rows: u32, row_bytes: u64,
) -> Result<()> {
    let args = GetRowsArgs {
        ne00: (row_bytes / 4) as i32, nb00: 4, nb01: row_bytes,
        nb02: row_bytes * n_rows as u64, nb03: row_bytes * n_rows as u64,
        ne10: n_rows as i32, nb10: 4, nb11: n_rows as u64 * 4,
        ne0: (row_bytes / 4) as i32, nb0: 4, nb1: row_bytes,
        nb2: row_bytes, nb3: row_bytes,
    };
    let threads = dispatch::threads_1d(row_bytes as usize / 4);
    dispatch::dispatch(
        cache, batch, "kernel_get_rows_f32", &args,
        &[(&src.buffer, src.offset as usize), (&ids.buffer, ids.offset as usize),
          (&dst.buffer, dst.offset as usize)],
        None,
        objc_ext::mtl_size(n_rows as usize, 1, 1),
        objc_ext::mtl_size(threads, 1, 1),
    )
}

pub fn set_rows_f32_i32(
    cache: &mut PipelineCache, batch: &mut CommandBatch,
    dst: &Tensor, src: &Tensor, ids: &Tensor,
    n_rows: u32, row_bytes: u64,
) -> Result<()> {
    let args = SetRowsArgs {
        ne00: (row_bytes / 4) as i32, nb00: 4, nb01: row_bytes,
        nb02: row_bytes * n_rows as u64, nb03: row_bytes * n_rows as u64,
        ne10: n_rows as i32, nb10: 4, nb11: n_rows as u64 * 4,
        ne0: (row_bytes / 4) as i32, nb0: 4, nb1: row_bytes,
        nb2: row_bytes * n_rows as u64, nb3: row_bytes * n_rows as u64,
    };
    let threads = dispatch::threads_1d(row_bytes as usize / 4);
    dispatch::dispatch(
        cache, batch, "kernel_set_rows_f32_i32", &args,
        &[(&src.buffer, src.offset as usize), (&ids.buffer, ids.offset as usize),
          (&dst.buffer, dst.offset as usize)],
        None,
        objc_ext::mtl_size(n_rows as usize, 1, 1),
        objc_ext::mtl_size(threads, 1, 1),
    )
}

// ── Repeat (broadcast row to HC width) ────────────────────────────────

pub fn repeat_f32(
    cache: &mut PipelineCache, batch: &mut CommandBatch,
    dst: &Tensor, src: &Tensor, n_embd: u32, n_hc: u32,
) -> Result<()> {
    let args = RepeatArgs {
        ne00: n_embd as i32,
        nb00: 4, nb01: n_embd as u64 * 4, nb02: n_embd as u64 * 4, nb03: n_embd as u64 * 4,
        ne0: n_embd as i32, ne1: n_hc as i32, ne2: 1, ne3: 1,
        nb0: 4, nb1: n_embd as u64 * 4, nb2: n_embd as u64 * 4 * n_hc as u64, nb3: n_embd as u64 * 4 * n_hc as u64,
    };
    let threads = dispatch::threads_1d(n_embd as usize);
    dispatch::dispatch(
        cache, batch, "kernel_repeat_f32", &args,
        &[(&src.buffer, src.offset as usize), (&dst.buffer, dst.offset as usize)],
        None,
        objc_ext::mtl_size(n_hc as usize, 1, 1),
        objc_ext::mtl_size(threads, 1, 1),
    )
}

// ── Sum rows ──────────────────────────────────────────────────────────

pub fn sum_rows_f32(
    cache: &mut PipelineCache, batch: &mut CommandBatch,
    dst: &Tensor, src: &Tensor, n_elems: u32, n_rows: u32, mean: bool,
) -> Result<()> {
    let args = SumRowsArgs {
        ne00: n_elems as i32, ne01: n_rows as i32,
        nb00: 4, nb01: n_elems as u64 * 4, nb02: n_elems as u64 * 4 * n_rows as u64, nb03: n_elems as u64 * 4 * n_rows as u64,
        ne0: 1, ne1: 1,
        nb0: 4, nb1: 4, nb2: 4, nb3: 4,
        mean: mean as i32,
    };
    let threads = dispatch::threads_pow2(n_elems as usize);
    dispatch::dispatch(
        cache, batch, "kernel_sum_rows_f32_f32", &args,
        &[(&src.buffer, src.offset as usize), (&dst.buffer, dst.offset as usize)],
        Some((threads * std::mem::size_of::<f32>(), 0)),
        objc_ext::mtl_size(n_rows as usize, 1, 1),
        objc_ext::mtl_size(threads, 1, 1),
    )
}

// ── Concat ────────────────────────────────────────────────────────────

pub fn concat(
    cache: &mut PipelineCache, batch: &mut CommandBatch,
    dst: &Tensor, a: &Tensor, b: &Tensor, n_a: u32, n_b: u32,
) -> Result<()> {
    let args = ConcatArgs {
        ne00: n_a as i32, ne01: 1, nb00: 4, nb01: n_a as u64 * 4, nb02: n_a as u64 * 4, nb03: n_a as u64 * 4,
        ne10: n_b as i32, ne11: 1, nb10: 4, nb11: n_b as u64 * 4, nb12: n_b as u64 * 4, nb13: n_b as u64 * 4,
        ne0: (n_a + n_b) as i32, ne1: 1,
        nb0: 4, nb1: (n_a + n_b) as u64 * 4, nb2: (n_a + n_b) as u64 * 4, nb3: (n_a + n_b) as u64 * 4,
    };
    let threads = 256usize;
    let groups = ((n_a + n_b) as usize + threads - 1) / threads;
    dispatch::dispatch(
        cache, batch, "kernel_concat", &args,
        &[(&a.buffer, a.offset as usize), (&b.buffer, b.offset as usize), (&dst.buffer, dst.offset as usize)],
        None, objc_ext::mtl_size(groups, 1, 1), objc_ext::mtl_size(threads, 1, 1),
    )
}

// ── Softmax ───────────────────────────────────────────────────────────

pub fn softmax_f32(
    cache: &mut PipelineCache, batch: &mut CommandBatch,
    dst: &Tensor, src: &Tensor, mask: Option<&Tensor>, sink: Option<&Tensor>,
    n_rows: u32, n_cols: u32, scale: f32,
) -> Result<()> {
    let args = SoftMaxArgs {
        ne00: n_cols as i32, ne01: n_rows as i32, ne02: 1,
        nb00: 4, nb01: n_cols as u64 * 4, nb02: n_cols as u64 * 4 * n_rows as u64, nb03: n_cols as u64 * 4 * n_rows as u64,
        ne0: n_cols as i32, ne1: 1, nb0: 4, nb1: n_cols as u64 * 4, nb2: n_cols as u64 * 4, nb3: n_cols as u64 * 4,
        scale, max_bias: 0.0, mask_nrows: n_rows as i32, n_rows_log2: 0,
    };
    let threads = dispatch::threads_pow2(n_cols as usize);
    let bufs: Vec<_> = {
        let mut v = vec![(&src.buffer, src.offset as usize), (&dst.buffer, dst.offset as usize)];
        if let Some(m) = mask { v.push((&m.buffer, m.offset as usize)); } else { v.push((&dst.buffer, 0)); }
        if let Some(s) = sink { v.push((&s.buffer, s.offset as usize)); } else { v.push((&dst.buffer, 0)); }
        v
    };
    dispatch::dispatch(
        cache, batch, "kernel_soft_max_f32", &args, &bufs,
        Some((threads * std::mem::size_of::<f32>(), 0)),
        objc_ext::mtl_size(n_rows as usize, 1, 1), objc_ext::mtl_size(threads, 1, 1),
    )
}

// ── Argsort ───────────────────────────────────────────────────────────

pub fn argsort_f32_i32_desc(
    cache: &mut PipelineCache, batch: &mut CommandBatch,
    dst: &Tensor, src: &Tensor, n_rows: u32, n_cols: u32,
) -> Result<()> {
    let args = ArgsortArgs {
        ne00: n_cols as i32, nb00: 4, nb01: n_cols as u64 * 4,
        ne0: n_cols as i32, nb0: 4, nb1: n_cols as u64 * 4,
        order: 1, n_passes: 0, // descending
    };
    let threads = dispatch::threads_pow2(n_cols as usize);
    dispatch::dispatch(
        cache, batch, "kernel_argsort_f32_i32_desc", &args,
        &[(&src.buffer, src.offset as usize), (&dst.buffer, dst.offset as usize)],
        Some((threads * std::mem::size_of::<i32>(), 0)),
        objc_ext::mtl_size(n_rows as usize, 1, 1), objc_ext::mtl_size(threads, 1, 1),
    )
}

pub fn argsort_merge_f32_i32_desc(
    cache: &mut PipelineCache, batch: &mut CommandBatch,
    dst: &Tensor, src: &Tensor, tmp: &Tensor, n_rows: u32, n_cols: u32, run_len: u32,
) -> Result<()> {
    let args = ArgsortMergeArgs {
        ne00: n_cols as i32, nb00: 4, nb01: n_cols as u64 * 4,
        ne0: n_cols as i32, nb0: 4, nb1: n_cols as u64 * 4,
        order: 1, run_len: run_len as i32,
    };
    let threads = dispatch::threads_pow2(n_cols as usize);
    dispatch::dispatch(
        cache, batch, "kernel_argsort_merge_f32_i32_desc", &args,
        &[(&src.buffer, src.offset as usize), (&tmp.buffer, tmp.offset as usize), (&dst.buffer, dst.offset as usize)],
        Some((threads * std::mem::size_of::<i32>(), 0)),
        objc_ext::mtl_size(n_rows as usize, 1, 1), objc_ext::mtl_size(threads, 1, 1),
    )
}

// ── Directional steering ──────────────────────────────────────────────

pub fn directional_steering_project(
    cache: &mut PipelineCache, batch: &mut CommandBatch,
    x: &Tensor, directions: &Tensor, layer: u32, width: u32, rows: u32, scale: f32,
) -> Result<()> {
    let nth = std::cmp::min(256u32, width).next_power_of_two();
    let args = DirectionalSteeringArgs { width, rows, layer, n_threads: nth, scale };
    dispatch::dispatch(
        cache, batch, "kernel_dsv4_directional_steering_project_f32", &args,
        &[(&x.buffer, x.offset as usize), (&directions.buffer, directions.offset as usize)],
        Some((nth as usize * std::mem::size_of::<f32>(), 0)),
        objc_ext::mtl_size(rows as usize, 1, 1),
        objc_ext::mtl_size(nth as usize, 1, 1),
    )
}
