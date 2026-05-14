use metal::*;
use crate::pipeline;
use crate::bridge;
use crate::tensor::GpuTensor;
use crate::gguf::N_HEAD_DIM;
use crate::metal_args;
use std::ffi::c_void;
use std::time::{Duration, Instant};

fn wait_gpu(cb: &CommandBufferRef) -> Result<(), &'static str> {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        match cb.status() {
            MTLCommandBufferStatus::NotEnqueued | MTLCommandBufferStatus::Scheduled => {}
            _ => return Ok(()),
        }
        if Instant::now() >= deadline { return Err("gpu timeout"); }
        std::thread::sleep(Duration::from_millis(10));
    }
}

const FC_MUL_MV: u64 = 600;
#[allow(dead_code)]
const FC_UNARY: u64 = 1200;
#[allow(dead_code)]
const FC_BIN: u64 = 1300;
const FC_FLASH_ATTN_EXT: u64 = 300;
#[allow(dead_code)]
const FC_FLASH_ATTN_EXT_VEC: u64 = 400;

fn get_or_create_pipeline(name: &str) -> Option<ComputePipelineState> {
    if let Some(p) = pipeline::get_pipeline(name) { return Some(p); }
    let lib = bridge::with_library(|l| l.clone())?;
    let fn_ = lib.get_function(name, None).ok()?;
    let device = bridge::with_device(|d| d.clone())?;
    let p = device.new_compute_pipeline_state_with_function(&fn_).ok()?;
    pipeline::cache_pipeline(name, &p);
    Some(p)
}

fn make_matmul_pipeline(name: &str, nsg: i16, nxpsg: i16) -> Option<ComputePipelineState> {
    let key = format!("{}_{}_{}", name, nsg, nxpsg);
    if let Some(p) = pipeline::get_pipeline(&key) { return Some(p); }
    let lib = bridge::with_library(|l| l.clone())?;
    let fcv = FunctionConstantValues::new();
    unsafe {
        fcv.set_constant_value_at_index(
            FC_MUL_MV, MTLDataType::Short, &nsg as *const i16 as *const c_void);
        fcv.set_constant_value_at_index(
            FC_MUL_MV + 1, MTLDataType::Short, &nxpsg as *const i16 as *const c_void);
    }
    let fn_ = lib.get_function(name, Some(fcv)).ok()?;
    let device = bridge::with_device(|d| d.clone())?;
    let p = device.new_compute_pipeline_state_with_function(&fn_).ok()?;
    pipeline::cache_pipeline(&key, &p);
    Some(p)
}

fn dispatch_with_args<T: Sized>(
    pipeline: &ComputePipelineState, args: &T,
    buffers: &[(Option<&Buffer>, u64)],
    threads: (u64, u64, u64), tg_size: (u64, u64, u64),
) -> Result<(), &'static str> {
    bridge::with_queue(|queue| {
        let cb = queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pipeline);
        let args_bytes = unsafe {
            std::slice::from_raw_parts(
                (args as *const T) as *const u8, std::mem::size_of::<T>())
        };
        let device = bridge::with_device(|d| d.clone()).unwrap();
        let args_buf = device.new_buffer_with_data(
            args_bytes.as_ptr() as *const c_void,
            args_bytes.len() as u64, MTLResourceOptions::StorageModeShared);
        enc.set_buffer(0, Some(&args_buf), 0);
        for (i, &(ref buf, off)) in buffers.iter().enumerate() {
            if let Some(b) = buf { enc.set_buffer(i as u64 + 1, Some(&**b), off); }
        }
        enc.dispatch_thread_groups(
            MTLSize { width: threads.0, height: threads.1, depth: threads.2 },
            MTLSize { width: tg_size.0, height: tg_size.1, depth: tg_size.2 });
        enc.end_encoding();
        cb.commit();
        wait_gpu(&cb)?;
        if cb.status() == MTLCommandBufferStatus::Error { Err("kernel failed") } else { Ok(()) }
    }).unwrap()
}

// ─── Struct args matching Metal shaders ───

#[repr(C)]
pub struct MulMvArgs {
    ne00: i32, ne01: i32, ne02: i32,
    nb00: u64, nb01: u64, nb02: u64, nb03: u64,
    ne10: i32, ne11: i32, ne12: i32,
    nb10: u64, nb11: u64, nb12: u64, nb13: u64,
    ne0: i32, ne1: i32, nr0: i32, r2: i16, r3: i16,
}

#[repr(C)]
pub struct RmsNormArgs {
    ne00: i32, ne00_t: i32,
    nb1: u64, nb2: u64, nb3: u64, eps: f32,
    nef1: [i32; 3], nef2: [i32; 3], nef3: [i32; 3],
    nbf1: [u64; 3], nbf2: [u64; 3], nbf3: [u64; 3],
}

#[repr(C)]
pub struct RopeTailArgs {
    ne00: i64, ne01: i64, ne02: i64, ne03: i64,
    nb00: u64, nb01: u64, nb02: u64, nb03: u64,
    nb0: u64, nb1: u64, nb2: u64, nb3: u64,
    n_dims: i32, mode: i32, n_ctx_orig: i32, inverse: i32,
    freq_base: f32, freq_scale: f32, ext_factor: f32,
    attn_factor: f32, beta_fast: f32, beta_slow: f32,
    src2: u8, _padding: [u8; 7],
}

#[repr(C)]
pub struct HcSplitSinkhornArgs {
    n_hc: i32, sinkhorn_iters: i32, n_rows: i64,
    mix_hc: i64, nb01: u64, nb1: u64, eps: f32,
}

#[repr(C)]
pub struct HcWeightedSumArgs {
    n_embd: i64, n_hc: i64, n_tokens: i64,
    nb_x0: u64, nb_x1: u64, nb_x2: u64,
    nb_w0: u64, nb_w1: u64, nb0: u64, nb1: u64,
}

#[repr(C)]
pub struct HcExpandArgs {
    n_embd: i64, n_hc: i64, n_tokens: i64,
    nb_block0: u64, nb_block1: u64,
    nb_add0: u64, nb_add1: u64,
    nb_res0: u64, nb_res1: u64, nb_res2: u64,
    nb_post0: u64, nb_post1: u64,
    nb_comb0: u64, nb_comb1: u64, nb_comb2: u64,
    nb0: u64, nb1: u64, nb2: u64,
    has_add: i32,
}

#[repr(C)]
pub struct SoftMaxArgs {
    ne00: i32, ne01: i32, ne02: i32,
    nb01: u64, nb02: u64, nb03: u64,
    ne11: i32, ne12: i32, ne13: i32,
    nb11: u64, nb12: u64, nb13: u64,
    nb1: u64, nb2: u64, nb3: u64,
    scale: f32, max_bias: f32, m0: f32, m1: f32, n_head_log2: i32,
}

#[repr(C)]
pub struct GluArgs {
    ne00: i32, nb01: u64, ne10: i32, nb11: u64,
    ne0: i32, nb1: u64, i00: i32, i10: i32, alpha: f32, limit: f32,
}

#[repr(C)]
pub struct IndexedAttentionArgs {
    n_tokens: u32, n_head: u32, n_raw: u32, raw_cap: u32,
    raw_start: u32, n_comp: u32, top_k: u32, pos0: u32,
    window: u32, ratio: u32,
    q_token_stride: u64, q_head_stride: u64,
    raw_row_stride: u64, comp_row_stride: u64,
    topk_token_stride: u64, dst_token_stride: u64, dst_head_stride: u64,
    scale: f32,
}

#[repr(C)]
pub struct DirectionalSteeringProjectArgs {
    width: u32, rows: u32, layer: u32, n_threads: u32, scale: f32,
}

#[repr(C)]
pub struct MoeSwigluWeightArgs {
    width: u32, rows: u32, gate_row_stride: u64, up_row_stride: u64,
    mid_row_stride: u64, weight_stride: u64,
    write_clamped: u32, clamp_value: f32,
}

#[repr(C)]
pub struct KvFp8StoreArgs {
    head_dim: i32, n_rot: i32, raw_row: i32,
}

#[repr(C)]
pub struct CompressorStoreOneArgs {
    width: u32, ratio: u32, pos: u32, ape_type: u32,
}

// Convenience wrapper using cached pipeline name
fn dispatch_pipeline<T: Sized>(
    pipeline_name: &str, args: &T,
    buffers: &[(Option<&Buffer>, u64)],
    threads: (u64, u64, u64), tg_size: (u64, u64, u64),
) -> Result<(), &'static str> {
    let p = get_or_create_pipeline(pipeline_name).ok_or("pipeline not found")?;
    dispatch_with_args(&p, args, buffers, threads, tg_size)
}

// ─── RMS Norm: args(0), src0(1), src1_0(2), src1_1(3), dst(4) ───
pub fn rms_norm_plain(
    out: &GpuTensor, x: &GpuTensor, n: u32, eps: f32,
) -> Result<(), &'static str> {
    let n4 = (n / 4) as i32;
    let row_bytes = (n * 4) as u64;
    let args = RmsNormArgs {
        ne00: n as i32, ne00_t: n4,
        nb1: row_bytes, nb2: row_bytes, nb3: row_bytes, eps,
        nef1: [n4, 0, 0], nef2: [0; 3], nef3: [0; 3],
        nbf1: [row_bytes, 0, 0], nbf2: [0; 3], nbf3: [0; 3],
    };
    dispatch_pipeline("kernel_rms_norm_f32_4", &args,
        &[(x.buffer(), x.offset_raw()), (None, 0), (None, 0),
          (out.buffer(), out.offset_raw())],
        (1, 1, 1), (256, 1, 1))
}

// ─── RoPE: args(0), src0(1-x), src1(2-pos), src2(3), dst(4) ───
pub fn rope_tail(
    x: &GpuTensor, _n_tok: u32, n_head: u32, head_dim: u32,
    n_rot: u32, pos: u32, n_ctx_orig: u32, _inverse: bool,
    freq_base: f32, freq_scale: f32, ext_factor: f32,
    attn_factor: f32, beta_fast: f32, beta_slow: f32,
) -> Result<(), &'static str> {
    let stride = (head_dim * 4) as u64;
    let args = RopeTailArgs {
        ne00: head_dim as i64, ne01: n_head as i64, ne02: 1, ne03: 1,
        nb00: 4, nb01: stride, nb02: stride, nb03: stride,
        nb0: 4, nb1: stride, nb2: stride, nb3: stride,
        n_dims: n_rot as i32, mode: 2, n_ctx_orig: n_ctx_orig as i32,
        inverse: 0, freq_base, freq_scale, ext_factor, attn_factor,
        beta_fast, beta_slow, src2: 0, _padding: [0; 7],
    };
    let device = bridge::with_device(|d| d.clone()).unwrap();
    let pos_data: [i32; 1] = [pos as i32];
    let pos_buf = device.new_buffer_with_data(
        pos_data.as_ptr() as *const c_void, 4,
        MTLResourceOptions::StorageModeShared);
    dispatch_pipeline("kernel_dsv4_rope_tail_f32", &args,
        &[(x.buffer(), x.offset_raw()), (Some(&pos_buf), 0), (None, 0),
          (x.buffer(), x.offset_raw())],
        (n_head as u64, 1, 1),
        (std::cmp::min(256u64, head_dim as u64).max(1), 1, 1))
}

// ─── HC Split Sinkhorn: args(0), mixes(1), scale(2), base(3), dst(4) ───
pub fn hc_split_sinkhorn(
    out: &GpuTensor, mixes: &GpuTensor,
    _model_map: &[u8], _model_size: u64, _scale_offset: u64, _base_offset: u64,
    n_hc: u32, sinkhorn_iters: u32, eps: f32,
) -> Result<(), &'static str> {
    let args = HcSplitSinkhornArgs {
        n_hc: n_hc as i32, sinkhorn_iters: sinkhorn_iters as i32,
        n_rows: 1, mix_hc: (n_hc + n_hc * n_hc + n_hc) as i64,
        nb01: 0, nb1: 0, eps,
    };
    let device = bridge::with_device(|d| d.clone()).unwrap();
    let zero = device.new_buffer(64, MTLResourceOptions::StorageModeShared);
    dispatch_pipeline("kernel_dsv4_hc_split_sinkhorn", &args,
        &[(mixes.buffer(), mixes.offset_raw()), (Some(&zero), 0),
          (Some(&zero), 0), (out.buffer(), out.offset_raw())],
        (1, 1, 1), (256, 1, 1))
}

// ─── HC Weighted Sum: args(0), x(1), weights(2), dst(3) ───
pub fn hc_weighted_sum(
    out: &GpuTensor, residual_hc: &GpuTensor, weights: &GpuTensor,
    n_embd: u32, n_hc: u32,
) -> Result<(), &'static str> {
    let args = HcWeightedSumArgs {
        n_embd: n_embd as i64, n_hc: n_hc as i64, n_tokens: 1,
        nb_x0: 4, nb_x1: (n_embd * 4) as u64, nb_x2: 0,
        nb_w0: 4, nb_w1: 0, nb0: 4, nb1: 0,
    };
    dispatch_pipeline("kernel_dsv4_hc_weighted_sum", &args,
        &[(residual_hc.buffer(), residual_hc.offset_raw()),
          (weights.buffer(), weights.offset_raw()),
          (out.buffer(), out.offset_raw())],
        (n_embd as u64, 1, 1), (256, 1, 1))
}

// ─── HC Expand: args(0), block_out(1), block_add(2), residual_hc(3), post(4), comb(5), out(6) ───
pub fn hc_expand(
    out_hc: &GpuTensor, block_out: &GpuTensor, residual_hc: &GpuTensor,
    split: &GpuTensor, n_embd: u32, n_hc: u32,
) -> Result<(), &'static str> {
    let stride = (n_embd * 4) as u64;
    let args = HcExpandArgs {
        n_embd: n_embd as i64, n_hc: n_hc as i64, n_tokens: 1,
        nb_block0: 4, nb_block1: stride,
        nb_add0: 0, nb_add1: 0,
        nb_res0: 4, nb_res1: stride, nb_res2: stride * n_hc as u64,
        nb_post0: 4, nb_post1: 0,
        nb_comb0: 4, nb_comb1: (n_hc * 4) as u64, nb_comb2: (n_hc * n_hc * 4) as u64,
        nb0: 4, nb1: stride, nb2: stride * n_hc as u64,
        has_add: 0,
    };
    dispatch_pipeline("kernel_dsv4_hc_expand4", &args,
        &[(block_out.buffer(), block_out.offset_raw()),
          (None, 0),
          (residual_hc.buffer(), residual_hc.offset_raw()),
          (split.buffer(), split.offset_raw()),
          (split.buffer(), split.offset_raw()),
          (out_hc.buffer(), out_hc.offset_raw())],
        (n_embd as u64, 1, 1), (256, 1, 1))
}

// ─── Softmax: args(0), src0(1), src1(2), src2(3), dst(4) ───
pub fn softmax(
    out: &GpuTensor, src: &GpuTensor, _mask: Option<&GpuTensor>,
    n: u32, rows: u32, scale: f32,
) -> Result<(), &'static str> {
    let args = SoftMaxArgs {
        ne00: n as i32, ne01: rows as i32, ne02: 1,
        nb01: (n * 4) as u64, nb02: 0, nb03: 0,
        ne11: 0, ne12: 0, ne13: 0, nb11: 0, nb12: 0, nb13: 0,
        nb1: (n * 4) as u64, nb2: 0, nb3: 0,
        scale, max_bias: 0.0, m0: 0.0, m1: 0.0, n_head_log2: 0,
    };
    dispatch_pipeline("kernel_soft_max", &args,
        &[(src.buffer(), src.offset_raw()), (None, 0), (None, 0),
          (out.buffer(), out.offset_raw())],
        (rows as u64, 1, 1), (256, 1, 1))
}

// ─── SwiGLU: args(0), src0(1-gate), src1(2-up), dst(3) ───
pub fn swiglu(
    out: &GpuTensor, gate: &GpuTensor, up: &GpuTensor,
    n: u32, _clamp: f32, _weight: f32,
) -> Result<(), &'static str> {
    let row_stride = (n * 4) as u64;
    let args = GluArgs {
        ne00: 0, nb01: row_stride, ne10: 0, nb11: row_stride,
        ne0: n as i32, nb1: row_stride, i00: 0, i10: 0,
        alpha: 1.0, limit: _clamp,
    };
    dispatch_pipeline("kernel_swiglu_f32", &args,
        &[(gate.buffer(), gate.offset_raw()),
          (up.buffer(), up.offset_raw()),
          (out.buffer(), out.offset_raw())],
        (1, 1, 1), (256, 1, 1))
}

// ─── Matmul: F16 weight × f32 activation ───
pub fn matmul_f16(
    out: &GpuTensor, model_map: &[u8], _model_size: u64,
    weight_offset: u64, in_dim: u64, out_dim: u64,
    x: &GpuTensor, n_tok: u64,
) -> Result<(), &'static str> {
    let nr0 = if out_dim % 4 == 0 { 4 } else { 2 };
    let args = MulMvArgs {
        ne00: in_dim as i32, ne01: out_dim as i32, ne02: 1,
        nb00: 2, nb01: (in_dim * 2) as u64, nb02: 0, nb03: 0,
        ne10: in_dim as i32, ne11: n_tok as i32, ne12: 1,
        nb10: 4, nb11: (in_dim * 4) as u64, nb12: 0, nb13: 0,
        ne0: out_dim as i32, ne1: n_tok as i32,
        nr0, r2: 1, r3: 1,
    };
    let device = bridge::with_device(|d| d.clone()).unwrap();
    let wdata = &model_map[weight_offset as usize..];
    let wbuf = device.new_buffer_with_data(
        wdata.as_ptr() as *const c_void,
        (in_dim * out_dim * 2) as u64,
        MTLResourceOptions::StorageModeShared);
    let p = make_matmul_pipeline("kernel_mul_mv_f16_f32", if nr0 == 4 {4} else {2}, 1)
        .ok_or("f16 matmul pipeline")?;
    dispatch_with_args(&p, &args,
        &[(Some(&wbuf), 0), (x.buffer(), x.offset_raw()),
          (out.buffer(), out.offset_raw())],
        ((out_dim / nr0 as u64 + 1) * n_tok, 1, 1), (256, 1, 1))
}

// ─── Matmul: Q8_0 weight × f32 activation ───
pub fn matmul_q8_0(
    out: &GpuTensor, model_map: &[u8], _model_size: u64,
    weight_offset: u64, in_dim: u64, out_dim: u64,
    x: &GpuTensor, n_tok: u64,
) -> Result<(), &'static str> {
    let nr0 = if out_dim % 4 == 0 { 4 } else { 2 };
    let args = MulMvArgs {
        ne00: in_dim as i32, ne01: out_dim as i32, ne02: 1,
        nb00: 34, nb01: 34 * (in_dim as u64 / 32), nb02: 0, nb03: 0,
        ne10: in_dim as i32, ne11: n_tok as i32, ne12: 1,
        nb10: 4, nb11: (in_dim * 4) as u64, nb12: 0, nb13: 0,
        ne0: out_dim as i32, ne1: n_tok as i32,
        nr0, r2: 1, r3: 1,
    };
    let device = bridge::with_device(|d| d.clone()).unwrap();
    let wdata = &model_map[weight_offset as usize..];
    let wbuf = device.new_buffer_with_data(
        wdata.as_ptr() as *const c_void,
        args.nb01 as u64 * out_dim / nr0 as u64,
        MTLResourceOptions::StorageModeShared);
    let p = make_matmul_pipeline("kernel_mul_mv_q8_0_f32", if nr0 == 4 {4} else {2}, 1)
        .ok_or("q8_0 matmul pipeline")?;
    dispatch_with_args(&p, &args,
        &[(Some(&wbuf), 0), (x.buffer(), x.offset_raw()),
          (out.buffer(), out.offset_raw())],
        ((out_dim / nr0 as u64 + 1) * n_tok, 1, 1), (256, 1, 1))
}

// ─── KV FP8 Store + raw cache write ───
pub fn kv_fp8_store(
    kv: &GpuTensor, raw_cache: &GpuTensor,
    head_dim: u32, n_rot: u32, raw_row: u32,
) -> Result<(), &'static str> {
    let args = KvFp8StoreArgs {
        head_dim: head_dim as i32, n_rot: n_rot as i32, raw_row: raw_row as i32,
    };
    dispatch_pipeline("kernel_dsv4_kv_fp8_store_f32", &args,
        &[(kv.buffer(), kv.offset_raw()),
          (raw_cache.buffer(), raw_cache.offset_raw())],
        (1, 1, 1), (64, 1, 1))
}

// ─── Compressor store one ───
pub fn compressor_store_one(
    kv: &GpuTensor, score: &GpuTensor,
    model_map: &[u8], ape_offset: u64,
    state_kv: &GpuTensor, state_score: &GpuTensor,
    width: u32, ratio: u32, pos: u32,
) -> Result<(), &'static str> {
    let args = CompressorStoreOneArgs {
        width, ratio, pos, ape_type: 1, // half
    };
    let device = bridge::with_device(|d| d.clone()).unwrap();
    let ape_data = &model_map[ape_offset as usize..];
    let ape_buf = device.new_buffer_with_data(
        ape_data.as_ptr() as *const c_void,
        ratio as u64 * width as u64 * 2, // half
        MTLResourceOptions::StorageModeShared);
    dispatch_pipeline("kernel_dsv4_compressor_store_one", &args,
        &[(kv.buffer(), kv.offset_raw()),
          (score.buffer(), score.offset_raw()),
          (Some(&ape_buf), 0),
          (state_kv.buffer(), state_kv.offset_raw()),
          (state_score.buffer(), state_score.offset_raw())],
        (width as u64, 1, 1), (256, 1, 1))
}

// ─── Embedding: get_rows ───
pub fn embed_tokens(
    out: &GpuTensor, model_map: &[u8], _model_size: u64,
    weight_offset: u64, n_vocab: u32, n_embd: u32,
    tokens: &[i32],
) -> Result<(), &'static str> {
    let device = bridge::with_device(|d| d.clone()).unwrap();
    let wdata = &model_map[weight_offset as usize..];
    let wbuf = device.new_buffer_with_data(
        wdata.as_ptr() as *const c_void,
        (n_vocab as u64 * n_embd as u64 * 2) as u64,
        MTLResourceOptions::StorageModeShared);
    let tbuf = device.new_buffer_with_data(
        tokens.as_ptr() as *const c_void,
        (tokens.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared);
    use crate::metal_args::GetRowsArgs;
    let args = GetRowsArgs {
        ne00t: n_embd as i32,
        ne00: n_embd as i32,
        nb01: n_embd as u64 * 2,
        nb02: 0, nb03: 0,
        ne10: n_embd as i32,
        nb10: 4,
        nb11: 0, nb12: 0,
        nb1: n_embd as u64 * 4,
        nb2: 0, nb3: 0,
    };
    let abuf = device.new_buffer_with_data(
        &args as *const GetRowsArgs as *const c_void,
        std::mem::size_of::<GetRowsArgs>() as u64,
        MTLResourceOptions::StorageModeShared);
    let p = get_or_create_pipeline("kernel_get_rows_f16").ok_or("get_rows pipeline")?;
    bridge::with_queue(|queue| {
        let cb = queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&p);
        enc.set_buffer(0, Some(&abuf), 0);
        enc.set_buffer(1, Some(&wbuf), 0);
        enc.set_buffer(2, Some(&tbuf), 0);
        enc.set_buffer(3, Some(&**out.buffer().unwrap()), out.offset_raw());
        let n = tokens.len() as u64 * n_embd as u64;
        enc.dispatch_thread_groups(MTLSize { width: (n + 31) / 32, height: 1, depth: 1 },
            MTLSize { width: 32, height: 1, depth: 1 });
        enc.end_encoding();
        cb.commit();
        wait_gpu(&cb)?;
        if cb.status() == MTLCommandBufferStatus::Error { Err("embed failed") } else { Ok(()) }
    }).unwrap()
}

// ─── Router finalize one ───
pub fn router_select_one(
    probs: &GpuTensor, bias: &GpuTensor,
    selected: &GpuTensor, _n_expert: u32, has_bias: bool,
) -> Result<(), &'static str> {
    let args = metal_args::RouterSelectOneArgs {
        has_bias: has_bias as u32, hash_mode: 0, use_token_buffer: 0,
        token: 0, hash_rows: 0,
    };
    let device = bridge::with_device(|d| d.clone()).unwrap();
    let zero = device.new_buffer(4, MTLResourceOptions::StorageModeShared);
    dispatch_pipeline("kernel_dsv4_router_finalize_one", &args,
        &[(probs.buffer(), probs.offset_raw()),
          (bias.buffer(), bias.offset_raw()),
          (Some(&zero), 0), // hash placeholder
          (Some(&zero), 0), // tokens placeholder
          (selected.buffer(), selected.offset_raw())],
        (1, 1, 1), (256, 1, 1))
}

// ─── Router weights one ───
pub fn router_weights_one(
    probs: &GpuTensor, selected: &GpuTensor, weights: &GpuTensor,
) -> Result<(), &'static str> {
    let p = get_or_create_pipeline("kernel_dsv4_router_weights_one")
        .ok_or("router weights pipeline")?;
    bridge::with_queue(|queue| {
        let cb = queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&p);
        enc.set_buffer(0, Some(&**probs.buffer().unwrap()), probs.offset_raw());
        enc.set_buffer(1, Some(&**selected.buffer().unwrap()), selected.offset_raw());
        enc.set_buffer(2, Some(&**weights.buffer().unwrap()), weights.offset_raw());
        enc.dispatch_thread_groups(MTLSize { width: 1, height: 1, depth: 1 },
            MTLSize { width: 6, height: 1, depth: 1 });
        enc.end_encoding();
        cb.commit();
        wait_gpu(&cb)?;
        if cb.status() == MTLCommandBufferStatus::Error { Err("router weights failed") } else { Ok(()) }
    }).unwrap()
}

// ─── MoE SwiGLU with route weight ───
pub fn moe_swiglu_weight(
    gate: &GpuTensor, up: &GpuTensor, mid: &GpuTensor,
    weights: &GpuTensor, width: u32,
) -> Result<(), &'static str> {
    let args = MoeSwigluWeightArgs {
        width, rows: 1,
        gate_row_stride: width as u64 * 4,
        up_row_stride: width as u64 * 4,
        mid_row_stride: width as u64 * 4,
        weight_stride: 4,
        write_clamped: 0, clamp_value: 0.0,
    };
    dispatch_pipeline("kernel_dsv4_moe_swiglu_weight", &args,
        &[(gate.buffer(), gate.offset_raw()),
          (up.buffer(), up.offset_raw()),
          (mid.buffer(), mid.offset_raw()),
          (weights.buffer(), weights.offset_raw())],
        (1, 1, 1), (256, 1, 1))
}

// ─── Indexed mixed attention heads8 ───
pub fn indexed_attention(
    heads: &GpuTensor, q: &GpuTensor,
    raw_kv: &GpuTensor, comp_kv: &GpuTensor,
    topk: &GpuTensor, sinks: &GpuTensor,
    n_tokens: u32, n_head: u32, n_raw: u32, raw_cap: u32,
    raw_start: u32, n_comp: u32, topk_k: u32, pos0: u32,
    window: u32, ratio: u32, scale: f32,
    q_head_stride: u64, dst_head_stride: u64,
) -> Result<(), &'static str> {
    let args = IndexedAttentionArgs {
        n_tokens, n_head, n_raw, raw_cap, raw_start, n_comp,
        top_k: topk_k, pos0, window, ratio,
        q_token_stride: (n_head as u64 * q_head_stride),
        q_head_stride,
        raw_row_stride: (N_HEAD_DIM as u64 * 4),
        comp_row_stride: (N_HEAD_DIM as u64 * 4),
        topk_token_stride: (topk_k as u64 * 4),
        dst_token_stride: (n_head as u64 * dst_head_stride),
        dst_head_stride,
        scale,
    };
    let device = bridge::with_device(|d| d.clone()).unwrap();
    let sink_buf = sinks.buffer().cloned().unwrap_or_else(||
        device.new_buffer(4, MTLResourceOptions::StorageModeShared));
    dispatch_pipeline("kernel_dsv4_indexed_mixed_attention_heads8", &args,
        &[(q.buffer(), q.offset_raw()),
          (raw_kv.buffer(), raw_kv.offset_raw()),
          (comp_kv.buffer(), comp_kv.offset_raw()),
          (topk.buffer(), topk.offset_raw()),
          (Some(&sink_buf), 0),
          (heads.buffer(), heads.offset_raw())],
        (n_tokens as u64, (n_head / 8 + 1) as u64, 1), (128, 1, 1))
}

// ─── Directional steering ───
pub fn directional_steering_project(
    x: &GpuTensor, directions: &GpuTensor,
    layer: u32, width: u32, rows: u32, scale: f32,
) -> Result<(), &'static str> {
    let args = DirectionalSteeringProjectArgs {
        width, rows, layer, n_threads: 256, scale,
    };
    dispatch_pipeline("kernel_dsv4_directional_steering_project_f32", &args,
        &[(x.buffer(), x.offset_raw()),
          (directions.buffer(), directions.offset_raw())],
        (rows as u64, 1, 1), (256, 1, 1))
}

// ─── New struct args for remaining ops ───

#[repr(C)]
pub struct MulMvIdArgs {
    nei0: i32, nei1: i32, nbi1: u64,
    ne00: i32, ne01: i32, ne02: i32,
    nb00: u64, nb01: u64, nb02: u64,
    ne10: i32, ne11: i32, ne12: i32, ne13: i32,
    nb10: u64, nb11: u64, nb12: u64,
    ne0: i32, ne1: i32, nb1: u64, nr0: i32,
}

#[repr(C)]
pub struct IndexerScoresFusedArgs {
    n_comp: u32, n_tokens: u32, n_head: u32, head_dim: u32,
    pos0: u32, ratio: u32,
    q_token_stride: u64, q_head_stride: u64,
    weights_token_stride: u64, index_row_stride: u64,
    score_token_stride: u64, scale: f32,
}

#[repr(C)]
pub struct SoftmaxPoolArgs {
    ne00: i64, ne01: i64, ne02: i64,
    nb00: u64, nb01: u64, nb02: u64,
    nb10: u64, nb11: u64, nb12: u64,
    ne0: i64, ne1: i64, nb0: u64, nb1: u64,
}

// ─── Expert matmul: IQ2_XXS pair (gate+up) ───
pub fn matmul_id_iq2_xxs_pair(
    dst_gate: &GpuTensor, dst_up: &GpuTensor,
    model_map: &[u8], gate_offset: u64, _up_offset: u64,
    in_dim: u32, out_dim: u32, n_expert: u32,
    x: &GpuTensor, selected: &[i32],
) -> Result<(), &'static str> {
    let nr0: u32 = if out_dim % 4 == 0 { 4 } else { 2 };
    let args = MulMvIdArgs {
        nei0: 1, nei1: selected.len() as i32, nbi1: 4,
        ne00: in_dim as i32, ne01: out_dim as i32,
        ne02: (n_expert * in_dim / 32 * 66) as i32,
        nb00: 66, nb01: (in_dim as u64 / 32 * 66),
        nb02: (in_dim as u64 / 32 * 66 * out_dim as u64),
        ne10: in_dim as i32, ne11: 1, ne12: 1, ne13: 1,
        nb10: 4, nb11: (in_dim * 4) as u64,
        nb12: (in_dim * 4) as u64,
        ne0: out_dim as i32, ne1: 1, nb1: (out_dim * 4) as u64,
        nr0: nr0 as i32,
    };
    let device = bridge::with_device(|d| d.clone()).unwrap();
    // Create weight buffer with gate+up interleaved for pair kernel
    let wdata = &model_map[gate_offset as usize..];
    let wbuf = device.new_buffer_with_data(
        wdata.as_ptr() as *const c_void,
        (in_dim as u64 / 32 * 66 * out_dim as u64 * n_expert as u64) * 2,
        MTLResourceOptions::StorageModeShared);
    let sel_buf = device.new_buffer_with_data(
        selected.as_ptr() as *const c_void,
        (selected.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared);
    let nsg: i16 = if nr0 == 4 { 4 } else { 2 };
    let p = make_matmul_pipeline("kernel_mul_mv_id_iq2_xxs_pair_f32", nsg, 1)
        .ok_or("iq2 pair pipeline")?;
    dispatch_with_args(&p, &args,
        &[(Some(&wbuf), 0), (Some(&wbuf), (in_dim as u64 / 32 * 66 * out_dim as u64 * n_expert as u64)),
          (x.buffer(), x.offset_raw()),
          (dst_gate.buffer(), dst_gate.offset_raw()),
          (dst_up.buffer(), dst_up.offset_raw()),
          (Some(&sel_buf), 0)],
        ((out_dim / nr0 + 1) as u64, 1, selected.len() as u64),
        (256, 1, 1))
}

// ─── Expert matmul: Q2_K sum6 (gate+up for Q2 experts) ───
#[allow(non_snake_case)]
pub fn matmul_id_q2_K_sum6(
    dst_gate: &GpuTensor, dst_up: &GpuTensor,
    model_map: &[u8], gate_offset: u64, _up_offset: u64,
    in_dim: u32, out_dim: u32, n_expert: u32,
    x: &GpuTensor, selected: &[i32],
) -> Result<(), &'static str> {
    let nr0: u32 = 4;
    let args = MulMvIdArgs {
        nei0: 1, nei1: selected.len() as i32, nbi1: 4,
        ne00: in_dim as i32, ne01: out_dim as i32,
        ne02: (n_expert * in_dim / 256 * 84) as i32,
        nb00: 84, nb01: (in_dim as u64 / 256 * 84),
        nb02: (in_dim as u64 / 256 * 84 * out_dim as u64),
        ne10: in_dim as i32, ne11: 1, ne12: 1, ne13: 1,
        nb10: 4, nb11: (in_dim * 4) as u64,
        nb12: (in_dim * 4) as u64,
        ne0: out_dim as i32, ne1: 1, nb1: (out_dim * 4) as u64,
        nr0: nr0 as i32,
    };
    let device = bridge::with_device(|d| d.clone()).unwrap();
    let step = in_dim as u64 / 256 * 84 * out_dim as u64;
    let wdata = &model_map[gate_offset as usize..];
    let wbuf = device.new_buffer_with_data(
        wdata.as_ptr() as *const c_void,
        step * n_expert as u64 * 2,
        MTLResourceOptions::StorageModeShared);
    let sel_buf = device.new_buffer_with_data(
        selected.as_ptr() as *const c_void,
        (selected.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared);
    let p = make_matmul_pipeline("kernel_mul_mv_id_q2_K_sum6_f32", 4, 1)
        .ok_or("q2 sum6 pipeline")?;
    dispatch_with_args(&p, &args,
        &[(Some(&wbuf), 0), (Some(&wbuf), step * n_expert as u64),
          (x.buffer(), x.offset_raw()),
          (dst_gate.buffer(), dst_gate.offset_raw()),
          (dst_up.buffer(), dst_up.offset_raw()),
          (Some(&sel_buf), 0)],
        ((out_dim / nr0 + 1) as u64, 1, selected.len() as u64),
        (256, 1, 1))
}

// ─── Expert matmul: Q4_K sum6 (gate+up for Q4 experts) ───
#[allow(non_snake_case)]
pub fn matmul_id_q4_K_sum6(
    dst_gate: &GpuTensor, dst_up: &GpuTensor,
    model_map: &[u8], gate_offset: u64, _up_offset: u64,
    in_dim: u32, out_dim: u32, n_expert: u32,
    x: &GpuTensor, selected: &[i32],
) -> Result<(), &'static str> {
    let nr0: u32 = 2;
    let args = MulMvIdArgs {
        nei0: 1, nei1: selected.len() as i32, nbi1: 4,
        ne00: in_dim as i32, ne01: out_dim as i32,
        ne02: (n_expert * in_dim / 256 * 144) as i32,
        nb00: 144, nb01: (in_dim as u64 / 256 * 144),
        nb02: (in_dim as u64 / 256 * 144 * out_dim as u64),
        ne10: in_dim as i32, ne11: 1, ne12: 1, ne13: 1,
        nb10: 4, nb11: (in_dim * 4) as u64,
        nb12: (in_dim * 4) as u64,
        ne0: out_dim as i32, ne1: 1, nb1: (out_dim * 4) as u64,
        nr0: nr0 as i32,
    };
    let device = bridge::with_device(|d| d.clone()).unwrap();
    let step = in_dim as u64 / 256 * 144 * out_dim as u64;
    let wdata = &model_map[gate_offset as usize..];
    let wbuf = device.new_buffer_with_data(
        wdata.as_ptr() as *const c_void,
        step * n_expert as u64 * 2,
        MTLResourceOptions::StorageModeShared);
    let sel_buf = device.new_buffer_with_data(
        selected.as_ptr() as *const c_void,
        (selected.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared);
    let p = make_matmul_pipeline("kernel_mul_mv_id_q4_K_sum6_f32", 2, 1)
        .ok_or("q4 sum6 pipeline")?;
    dispatch_with_args(&p, &args,
        &[(Some(&wbuf), 0), (Some(&wbuf), step * n_expert as u64),
          (x.buffer(), x.offset_raw()),
          (dst_gate.buffer(), dst_gate.offset_raw()),
          (dst_up.buffer(), dst_up.offset_raw()),
          (Some(&sel_buf), 0)],
        ((out_dim / nr0 + 1) as u64, 1, selected.len() as u64),
        (256, 1, 1))
}

// ─── Indexer score one for decode ───
pub fn indexer_score_one(
    scores: &GpuTensor, q: &GpuTensor,
    model_map: &[u8], weight_offset: u64,
    index_comp: &GpuTensor,
    n_comp: u32, n_head: u32, head_dim: u32,
    pos0: u32, ratio: u32, scale: f32,
) -> Result<(), &'static str> {
    let args = IndexerScoresFusedArgs {
        n_comp, n_tokens: 1, n_head, head_dim, pos0, ratio,
        q_token_stride: (n_head as u64 * head_dim as u64 * 4),
        q_head_stride: (head_dim as u64 * 4),
        weights_token_stride: 0,
        index_row_stride: (head_dim as u64 * 4),
        score_token_stride: 0, scale,
    };
    let device = bridge::with_device(|d| d.clone()).unwrap();
    let wdata = &model_map[weight_offset as usize..];
    let wbuf = device.new_buffer_with_data(
        wdata.as_ptr() as *const c_void, n_head as u64 * 4,
        MTLResourceOptions::StorageModeShared);
    dispatch_pipeline("kernel_dsv4_indexer_score_one_direct", &args,
        &[(q.buffer(), q.offset_raw()),
          (Some(&wbuf), 0),
          (index_comp.buffer(), index_comp.offset_raw()),
          (scores.buffer(), scores.offset_raw())],
        (n_comp as u64, 1, 1), (128, 1, 1))
}

// ─── Softmax pool for compressor rows ───
pub fn softmax_pool(
    dst: &GpuTensor, kv: &GpuTensor, score: &GpuTensor,
    n_rows: u32, head_dim: u32,
) -> Result<(), &'static str> {
    let args = SoftmaxPoolArgs {
        ne00: n_rows as i64, ne01: 1, ne02: 1,
        nb00: (head_dim * 4) as u64, nb01: 0, nb02: 0,
        nb10: (head_dim * 4) as u64, nb11: 0, nb12: 0,
        ne0: head_dim as i64, ne1: 1,
        nb0: 4, nb1: (head_dim * 4) as u64,
    };
    dispatch_pipeline("kernel_dsv4_softmax_pool", &args,
        &[(kv.buffer(), kv.offset_raw()),
          (score.buffer(), score.offset_raw()),
          (dst.buffer(), dst.offset_raw())],
        (head_dim as u64, 1, 1), (256, 1, 1))
}

// ─── Unary: softplus + sqrt for router probabilities ───
#[repr(C)]
struct UnaryArgs {
    ne00: i32, ne01: i32, ne02: i32, ne03: i32,
    nb00: u64, nb01: u64, nb02: u64, nb03: u64,
    ne0: i32, ne1: i32, ne2: i32, ne3: i32,
    nb0: u64, nb1: u64, nb2: u64, nb3: u64,
    slope: f32, scale: f32, bias: f32, val: f32, min_: f32, max_: f32,
}

pub fn softplus_sqrt(
    dst: &GpuTensor, src: &GpuTensor, n: u32,
) -> Result<(), &'static str> {
    let stride = (n * 4) as u64;
    let args = UnaryArgs {
        ne00: n as i32, ne01: 1, ne02: 1, ne03: 1,
        nb00: 4, nb01: stride, nb02: stride, nb03: stride,
        ne0: (n / 4) as i32, ne1: 1, ne2: 1, ne3: 1,
        nb0: 4, nb1: stride, nb2: stride, nb3: stride,
        slope: 0.0, scale: 0.0, bias: 0.0, val: 0.0, min_: 0.0, max_: 0.0,
    };
    dispatch_pipeline("kernel_dsv4_softplus_sqrt_f32_4", &args,
        &[(src.buffer(), src.offset_raw()),
          (dst.buffer(), dst.offset_raw())],
        (1, 1, 1), (256, 1, 1))
}

// ─── Use flash_attn_ext (prefill, many query tokens) ───
fn make_attn_prefill_pipeline(name: &str, nsg: i32) -> Option<ComputePipelineState> {
    let key = format!("{}_{}", name, nsg);
    if let Some(p) = pipeline::get_pipeline(&key) { return Some(p); }
    let lib = bridge::with_library(|l| l.clone())?;
    let fcv = FunctionConstantValues::new();
    unsafe {
        // has_mask=false, has_sinks=false, has_bias=false, has_scap=false, has_kvpad=false
        let f = false as u8;
        for i in 0..5 {
            fcv.set_constant_value_at_index(
                FC_FLASH_ATTN_EXT + i, MTLDataType::Bool, &f as *const u8 as *const c_void);
        }
        // nsg = 4 or 8
        fcv.set_constant_value_at_index(
            FC_FLASH_ATTN_EXT + 22, MTLDataType::Int, &nsg as *const i32 as *const c_void);
        // ns10 = 0, ns20 = 0
        let zero: i32 = 0;
        fcv.set_constant_value_at_index(
            FC_FLASH_ATTN_EXT + 20, MTLDataType::Int, &zero as *const i32 as *const c_void);
        fcv.set_constant_value_at_index(
            FC_FLASH_ATTN_EXT + 21, MTLDataType::Int, &zero as *const i32 as *const c_void);
    }
    let fn_ = lib.get_function(name, Some(fcv)).ok()?;
    let device = bridge::with_device(|d| d.clone())?;
    let p = device.new_compute_pipeline_state_with_function(&fn_).ok()?;
    pipeline::cache_pipeline(&key, &p);
    Some(p)
}

#[repr(C)]
pub struct FlashAttnExtArgs {
    ne01: i32, ne02: i32, ne03: i32,
    nb01: u64, nb02: u64, nb03: u64,
    ne11: i32, ne_12_2: i32, ne_12_3: i32, ns10: i32,
    nb11: u64, nb12: u64, nb13: u64,
    ns20: i32, nb21: u64, nb22: u64, nb23: u64,
    ne31: i32, ne32: i32, ne33: i32,
    nb31: u64, nb32: u64, nb33: u64,
    ne1: i32, ne2: i32, ne3: i32,
    scale: f32, max_bias: f32, m0: f32, m1: f32,
    n_head_log2: i32, logit_softcap: f32,
}

pub fn flash_attn_prefill(
    dst: &GpuTensor, q: &GpuTensor, k: &GpuTensor, v: &GpuTensor,
    n_tokens: u32, n_kv: u32, n_head: u32, head_dim: u32,
    scale: f32,
) -> Result<(), &'static str> {
    let q_stride = (n_head * head_dim * 4) as u64;
    let kv_stride = (head_dim * 2) as u64; // F16 K/V
    let args = FlashAttnExtArgs {
        ne01: n_tokens as i32, ne02: 1, ne03: 1,
        nb01: q_stride, nb02: q_stride, nb03: q_stride,
        ne11: n_kv as i32, ne_12_2: 1, ne_12_3: 1, ns10: n_kv as i32,
        nb11: kv_stride, nb12: kv_stride, nb13: kv_stride,
        ns20: n_kv as i32, nb21: kv_stride, nb22: kv_stride, nb23: kv_stride,
        ne31: n_kv as i32, ne32: 1, ne33: 1,
        nb31: 2, nb32: 0, nb33: 0,
        ne1: (n_head * head_dim) as i32, ne2: 1, ne3: 1,
        scale, max_bias: 0.0, m0: 0.0, m1: 0.0,
        n_head_log2: 0, logit_softcap: 0.0,
    };
    let p = make_attn_prefill_pipeline("kernel_flash_attn_ext_f16_dk512_dv512", 4)
        .ok_or("attn prefill pipeline")?;
    let device = bridge::with_device(|d| d.clone()).unwrap();
    // Create a dummy 1-byte pad/blk buffer (kernel expects them)
    let pad = device.new_buffer(1, MTLResourceOptions::StorageModePrivate);
    let blk = device.new_buffer(1, MTLResourceOptions::StorageModePrivate);
    dispatch_with_args(&p, &args,
        &[(q.buffer(), q.offset_raw()),
          (k.buffer(), k.offset_raw()),
          (v.buffer(), v.offset_raw()),
          (None, 0), (None, 0), // mask, sinks
          (Some(&pad), 0), (Some(&blk), 0),
          (dst.buffer(), dst.offset_raw())],
        (n_tokens as u64, n_head as u64, 1), (256, 1, 1))
}

// ─── Repeat: repeat source row across dimensions ───
#[repr(C)]
struct RepeatArgs {
    ne00: i32, ne01: i32, ne02: i32, ne03: i32,
    nb00: u64, nb01: u64, nb02: u64, nb03: u64,
    ne0: i32, ne1: i32, ne2: i32, ne3: i32,
    nb0: u64, nb1: u64, nb2: u64, nb3: u64,
}

pub fn repeat_f32(
    dst: &GpuTensor, src: &GpuTensor,
    ne0: i32, ne1: i32,
    ne00: i32, ne01: i32,
) -> Result<(), &'static str> {
    let args = RepeatArgs {
        ne00, ne01, ne02: 1, ne03: 1,
        nb00: 4, nb01: (ne00 * 4) as u64, nb02: 0, nb03: 0,
        ne0, ne1, ne2: 1, ne3: 1,
        nb0: 4, nb1: (ne0 * 4) as u64, nb2: 0, nb3: 0,
    };
    dispatch_pipeline("kernel_repeat_f32", &args,
        &[(src.buffer(), src.offset_raw()),
          (dst.buffer(), dst.offset_raw())],
        (ne1 as u64, 1, 1), (256, 1, 1))
}

// ─── KV store raw (no FP8) for prefill compatibility ───
// The kernel_dsv4_kv_fp8_store_f32 already handles this.
// Re-export with clearer name:
#[allow(unused_imports)]
pub use kv_fp8_store as kv_fp8_store_raw;

// ─── Compressor prefill state: ratio4 shift ───
pub fn compressor_ratio4_shift(
    state_kv: &GpuTensor, state_score: &GpuTensor, width: u32,
) -> Result<(), &'static str> {
    let args = metal_args::Ratio4ShiftArgs { width };
    dispatch_pipeline("kernel_dsv4_ratio4_shift_f32", &args,
        &[(state_kv.buffer(), state_kv.offset_raw()),
          (state_score.buffer(), state_score.offset_raw())],
        (4 * width as u64, 1, 1), (256, 1, 1))
}
