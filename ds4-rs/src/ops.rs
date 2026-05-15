use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLDataType,
    MTLDevice, MTLFunctionConstantValues, MTLLibrary, MTLResourceOptions, MTLSize,
};
use objc2_foundation::NSString;
use std::ptr::NonNull;
use std::ffi::c_void;
use crate::bridge;
use crate::pipeline;
use crate::tensor::GpuTensor;
use crate::model_view::ModelViews;
use crate::gguf::N_HEAD_DIM;
use crate::metal_args;

// ─── GPU wait ───────────────────────────────────────────────────────────────

fn wait_gpu(cb: &ProtocolObject<dyn MTLCommandBuffer>) -> Result<(), &'static str> {
    cb.waitUntilCompleted();
    if cb.status() == objc2_metal::MTLCommandBufferStatus::Error {
        Err("gpu kernel failed")
    } else {
        Ok(())
    }
}

// ─── Function-constant indices ───────────────────────────────────────────────

const FC_MUL_MV: usize = 600;
const FC_FLASH_ATTN_EXT: usize = 300;

// ─── Pipeline helpers ────────────────────────────────────────────────────────

fn get_or_create_pipeline(name: &str)
    -> Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>
{
    if let Some(p) = pipeline::get_pipeline(name) { return Some(p); }
    let lib = bridge::library()?;
    let ns_name = NSString::from_str(name);
    let fn_ = lib.newFunctionWithName(&*ns_name)?;
    let device = bridge::device()?;
    let p = device.newComputePipelineStateWithFunction_error(&*fn_).ok()?;
    pipeline::cache_pipeline(name, p.clone());
    Some(p)
}

fn make_matmul_pipeline(name: &str, nsg: i16, nxpsg: i16)
    -> Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>
{
    let key = format!("{}_{}_{}", name, nsg, nxpsg);
    if let Some(p) = pipeline::get_pipeline(&key) { return Some(p); }
    let lib = bridge::library()?;
    let fcv = MTLFunctionConstantValues::new();
    let mut nsg_v = nsg;
    let mut nxpsg_v = nxpsg;
    unsafe {
        fcv.setConstantValue_type_atIndex(
            NonNull::new(&mut nsg_v as *mut i16 as *mut c_void).unwrap(),
            MTLDataType::Short, FC_MUL_MV);
        fcv.setConstantValue_type_atIndex(
            NonNull::new(&mut nxpsg_v as *mut i16 as *mut c_void).unwrap(),
            MTLDataType::Short, FC_MUL_MV + 1);
    }
    let ns_name = NSString::from_str(name);
    let fn_ = lib.newFunctionWithName_constantValues_error(&*ns_name, &*fcv).ok()?;
    let device = bridge::device()?;
    let p = device.newComputePipelineStateWithFunction_error(&*fn_).ok()?;
    pipeline::cache_pipeline(&key, p.clone());
    Some(p)
}

fn make_attn_prefill_pipeline(name: &str, nsg: i32)
    -> Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>
{
    let key = format!("{}_{}", name, nsg);
    if let Some(p) = pipeline::get_pipeline(&key) { return Some(p); }
    let lib = bridge::library()?;
    let fcv = MTLFunctionConstantValues::new();
    let mut f_false: u8 = 0;
    let mut nsg_v = nsg;
    let mut zero_v: i32 = 0;
    unsafe {
        for i in 0..5usize {
            fcv.setConstantValue_type_atIndex(
                NonNull::new(&mut f_false as *mut u8 as *mut c_void).unwrap(),
                MTLDataType::Bool, FC_FLASH_ATTN_EXT + i);
        }
        fcv.setConstantValue_type_atIndex(
            NonNull::new(&mut nsg_v as *mut i32 as *mut c_void).unwrap(),
            MTLDataType::Int, FC_FLASH_ATTN_EXT + 22);
        fcv.setConstantValue_type_atIndex(
            NonNull::new(&mut zero_v as *mut i32 as *mut c_void).unwrap(),
            MTLDataType::Int, FC_FLASH_ATTN_EXT + 20);
        fcv.setConstantValue_type_atIndex(
            NonNull::new(&mut zero_v as *mut i32 as *mut c_void).unwrap(),
            MTLDataType::Int, FC_FLASH_ATTN_EXT + 21);
    }
    let ns_name = NSString::from_str(name);
    let fn_ = lib.newFunctionWithName_constantValues_error(&*ns_name, &*fcv).ok()?;
    let device = bridge::device()?;
    let p = device.newComputePipelineStateWithFunction_error(&*fn_).ok()?;
    pipeline::cache_pipeline(&key, p.clone());
    Some(p)
}

// ─── Core dispatch ───────────────────────────────────────────────────────────

/// Dispatch: args struct at buffer-slot 0; data buffers at slots 1..N.
fn dispatch_with_args<T: Sized>(
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    args: &T,
    buffers: &[(Option<&ProtocolObject<dyn MTLBuffer>>, u64)],
    threads: (usize, usize, usize),
    tg_size: (usize, usize, usize),
) -> Result<(), &'static str> {
    let queue  = bridge::queue().ok_or("no queue")?;
    let device = bridge::device().ok_or("no device")?;
    let cb  = queue.commandBuffer().ok_or("no command buffer")?;
    let enc = cb.computeCommandEncoder().ok_or("no encoder")?;
    enc.setComputePipelineState(pipeline);

    let args_buf = unsafe {
        device.newBufferWithBytes_length_options(
            NonNull::new(args as *const T as *mut c_void).unwrap(),
            std::mem::size_of::<T>(),
            MTLResourceOptions::StorageModeShared,
        )
    }.ok_or("args buffer alloc")?;
    unsafe { enc.setBuffer_offset_atIndex(Some(&*args_buf), 0, 0); }

    for (i, (buf, off)) in buffers.iter().enumerate() {
        if let Some(b) = buf {
            unsafe { enc.setBuffer_offset_atIndex(Some(*b), *off as usize, i + 1); }
        }
    }
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize { width: threads.0, height: threads.1, depth: threads.2 },
        MTLSize { width: tg_size.0, height: tg_size.1, depth: tg_size.2 },
    );
    enc.endEncoding();
    cb.commit();
    wait_gpu(&*cb)
}

/// Dispatch without args struct; buffers at slots 0..N-1.
fn dispatch_raw(
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    buffers: &[(Option<&ProtocolObject<dyn MTLBuffer>>, u64)],
    threads: (usize, usize, usize),
    tg_size: (usize, usize, usize),
) -> Result<(), &'static str> {
    let queue = bridge::queue().ok_or("no queue")?;
    let cb  = queue.commandBuffer().ok_or("no command buffer")?;
    let enc = cb.computeCommandEncoder().ok_or("no encoder")?;
    enc.setComputePipelineState(pipeline);

    for (i, (buf, off)) in buffers.iter().enumerate() {
        if let Some(b) = buf {
            unsafe { enc.setBuffer_offset_atIndex(Some(*b), *off as usize, i); }
        }
    }
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize { width: threads.0, height: threads.1, depth: threads.2 },
        MTLSize { width: tg_size.0, height: tg_size.1, depth: tg_size.2 },
    );
    enc.endEncoding();
    cb.commit();
    wait_gpu(&*cb)
}

fn dispatch_pipeline<T: Sized>(
    pipeline_name: &str, args: &T,
    buffers: &[(Option<&ProtocolObject<dyn MTLBuffer>>, u64)],
    threads: (usize, usize, usize),
    tg_size: (usize, usize, usize),
) -> Result<(), &'static str> {
    let p = get_or_create_pipeline(pipeline_name).ok_or("pipeline not found")?;
    dispatch_with_args(&*p, args, buffers, threads, tg_size)
}

// ─── Arg structs ─────────────────────────────────────────────────────────────

#[repr(C)]
struct MulMvArgs {
    ne00: i32, ne01: i32, ne02: i32,
    nb00: u64, nb01: u64, nb02: u64, nb03: u64,
    ne10: i32, ne11: i32, ne12: i32,
    nb10: u64, nb11: u64, nb12: u64, nb13: u64,
    ne0: i32, ne1: i32, nr0: i32, r2: i16, r3: i16,
}

#[repr(C)]
struct RmsNormArgs {
    ne00: i32, ne00_t: i32,
    nb1: u64, nb2: u64, nb3: u64, eps: f32,
    nef1: [i32; 3], nef2: [i32; 3], nef3: [i32; 3],
    nbf1: [u64; 3], nbf2: [u64; 3], nbf3: [u64; 3],
}

#[repr(C)]
struct RopeTailArgs {
    ne00: i64, ne01: i64, ne02: i64, ne03: i64,
    nb00: u64, nb01: u64, nb02: u64, nb03: u64,
    nb0: u64, nb1: u64, nb2: u64, nb3: u64,
    n_dims: i32, mode: i32, n_ctx_orig: i32, inverse: i32,
    freq_base: f32, freq_scale: f32, ext_factor: f32,
    attn_factor: f32, beta_fast: f32, beta_slow: f32,
    src2: u8, _padding: [u8; 7],
}

#[repr(C)]
struct HcSplitSinkhornArgs {
    n_hc: i32, sinkhorn_iters: i32, n_rows: i64,
    mix_hc: i64, nb01: u64, nb1: u64, eps: f32,
}

#[repr(C)]
struct HcWeightedSumArgs {
    n_embd: i64, n_hc: i64, n_tokens: i64,
    nb_x0: u64, nb_x1: u64, nb_x2: u64,
    nb_w0: u64, nb_w1: u64, nb0: u64, nb1: u64,
}

#[repr(C)]
struct HcExpandArgs {
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
struct SoftMaxArgs {
    ne00: i32, ne01: i32, ne02: i32,
    nb01: u64, nb02: u64, nb03: u64,
    ne11: i32, ne12: i32, ne13: i32,
    nb11: u64, nb12: u64, nb13: u64,
    nb1: u64, nb2: u64, nb3: u64,
    scale: f32, max_bias: f32, m0: f32, m1: f32, n_head_log2: i32,
}

#[repr(C)]
struct GluArgs {
    ne00: i32, nb01: u64, ne10: i32, nb11: u64,
    ne0: i32, nb1: u64, i00: i32, i10: i32, alpha: f32, limit: f32,
}

#[repr(C)]
struct IndexedAttentionArgs {
    n_tokens: u32, n_head: u32, n_raw: u32, raw_cap: u32,
    raw_start: u32, n_comp: u32, top_k: u32, pos0: u32,
    window: u32, ratio: u32,
    q_token_stride: u64, q_head_stride: u64,
    raw_row_stride: u64, comp_row_stride: u64,
    topk_token_stride: u64, dst_token_stride: u64, dst_head_stride: u64,
    scale: f32,
}

#[repr(C)]
struct DirectionalSteeringProjectArgs {
    width: u32, rows: u32, layer: u32, n_threads: u32, scale: f32,
}

#[repr(C)]
struct MoeSwigluWeightArgs {
    width: u32, rows: u32, gate_row_stride: u64, up_row_stride: u64,
    mid_row_stride: u64, weight_stride: u64,
    write_clamped: u32, clamp_value: f32,
}

#[repr(C)]
struct KvFp8StoreArgs {
    head_dim: i32, n_rot: i32, raw_row: i32,
}

#[repr(C)]
struct CompressorStoreOneArgs {
    width: u32, ratio: u32, pos: u32, ape_type: u32,
}

#[repr(C)]
struct MulMvIdArgs {
    nei0: i32, nei1: i32, nbi1: u64,
    ne00: i32, ne01: i32, ne02: i32,
    nb00: u64, nb01: u64, nb02: u64,
    ne10: i32, ne11: i32, ne12: i32, ne13: i32,
    nb10: u64, nb11: u64, nb12: u64,
    ne0: i32, ne1: i32, nb1: u64, nr0: i32,
}

#[repr(C)]
struct IndexerScoresFusedArgs {
    n_comp: u32, n_tokens: u32, n_head: u32, head_dim: u32,
    pos0: u32, ratio: u32,
    q_token_stride: u64, q_head_stride: u64,
    weights_token_stride: u64, index_row_stride: u64,
    score_token_stride: u64, scale: f32,
}

#[repr(C)]
struct SoftmaxPoolArgs {
    ne00: i64, ne01: i64, ne02: i64,
    nb00: u64, nb01: u64, nb02: u64,
    nb10: u64, nb11: u64, nb12: u64,
    ne0: i64, ne1: i64, nb0: u64, nb1: u64,
}

#[repr(C)]
struct UnaryArgs {
    ne00: i32, ne01: i32, ne02: i32, ne03: i32,
    nb00: u64, nb01: u64, nb02: u64, nb03: u64,
    ne0: i32, ne1: i32, ne2: i32, ne3: i32,
    nb0: u64, nb1: u64, nb2: u64, nb3: u64,
    slope: f32, scale: f32, bias: f32, val: f32, min_: f32, max_: f32,
}

#[repr(C)]
struct FlashAttnExtArgs {
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

#[repr(C)]
struct RepeatArgs {
    ne00: i32, ne01: i32, ne02: i32, ne03: i32,
    nb00: u64, nb01: u64, nb02: u64, nb03: u64,
    ne0: i32, ne1: i32, ne2: i32, ne3: i32,
    nb0: u64, nb1: u64, nb2: u64, nb3: u64,
}

// ─── RMS Norm ────────────────────────────────────────────────────────────────

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
        &[(x.buf_ref(), x.offset_raw()), (None, 0), (None, 0),
          (out.buf_ref(), out.offset_raw())],
        (1, 1, 1), (256, 1, 1))
}

// ─── RoPE ────────────────────────────────────────────────────────────────────

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
    let device = bridge::device().ok_or("no device")?;
    let mut pos_val = pos as i32;
    let pos_buf = unsafe {
        device.newBufferWithBytes_length_options(
            NonNull::new(&mut pos_val as *mut i32 as *mut c_void).unwrap(),
            4, MTLResourceOptions::StorageModeShared,
        )
    }.ok_or("pos buffer")?;
    dispatch_pipeline("kernel_dsv4_rope_tail_f32", &args,
        &[(x.buf_ref(), x.offset_raw()), (Some(&*pos_buf), 0), (None, 0),
          (x.buf_ref(), x.offset_raw())],
        (n_head as usize, 1, 1),
        (std::cmp::min(256usize, head_dim as usize).max(1), 1, 1))
}

// ─── HC Split Sinkhorn ───────────────────────────────────────────────────────

pub fn hc_split_sinkhorn(
    out: &GpuTensor, mixes: &GpuTensor,
    n_hc: u32, sinkhorn_iters: u32, eps: f32,
) -> Result<(), &'static str> {
    let args = HcSplitSinkhornArgs {
        n_hc: n_hc as i32, sinkhorn_iters: sinkhorn_iters as i32,
        n_rows: 1, mix_hc: (n_hc + n_hc * n_hc + n_hc) as i64,
        nb01: 0, nb1: 0, eps,
    };
    let device = bridge::device().ok_or("no device")?;
    let zero = unsafe {
        device.newBufferWithLength_options(64, MTLResourceOptions::StorageModeShared)
    }.ok_or("zero buffer")?;
    dispatch_pipeline("kernel_dsv4_hc_split_sinkhorn", &args,
        &[(mixes.buf_ref(), mixes.offset_raw()), (Some(&*zero), 0),
          (Some(&*zero), 0), (out.buf_ref(), out.offset_raw())],
        (1, 1, 1), (256, 1, 1))
}

// ─── HC Weighted Sum ─────────────────────────────────────────────────────────

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
        &[(residual_hc.buf_ref(), residual_hc.offset_raw()),
          (weights.buf_ref(), weights.offset_raw()),
          (out.buf_ref(), out.offset_raw())],
        (n_embd as usize, 1, 1), (256, 1, 1))
}

// ─── HC Expand ───────────────────────────────────────────────────────────────

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
        &[(block_out.buf_ref(), block_out.offset_raw()),
          (None, 0),
          (residual_hc.buf_ref(), residual_hc.offset_raw()),
          (split.buf_ref(), split.offset_raw()),
          (split.buf_ref(), split.offset_raw()),
          (out_hc.buf_ref(), out_hc.offset_raw())],
        (n_embd as usize, 1, 1), (256, 1, 1))
}

// ─── Softmax ─────────────────────────────────────────────────────────────────

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
        &[(src.buf_ref(), src.offset_raw()), (None, 0), (None, 0),
          (out.buf_ref(), out.offset_raw())],
        (rows as usize, 1, 1), (256, 1, 1))
}

// ─── SwiGLU ──────────────────────────────────────────────────────────────────

pub fn swiglu(
    out: &GpuTensor, gate: &GpuTensor, up: &GpuTensor,
    n: u32, clamp: f32, _weight: f32,
) -> Result<(), &'static str> {
    let row_stride = (n * 4) as u64;
    let args = GluArgs {
        ne00: 0, nb01: row_stride, ne10: 0, nb11: row_stride,
        ne0: n as i32, nb1: row_stride, i00: 0, i10: 0,
        alpha: 1.0, limit: clamp,
    };
    dispatch_pipeline("kernel_swiglu_f32", &args,
        &[(gate.buf_ref(), gate.offset_raw()),
          (up.buf_ref(), up.offset_raw()),
          (out.buf_ref(), out.offset_raw())],
        (1, 1, 1), (256, 1, 1))
}

// ─── F16 matmul ──────────────────────────────────────────────────────────────

pub fn matmul_f16(
    out: &GpuTensor, views: &ModelViews,
    weight_offset: u64, in_dim: u64, out_dim: u64,
    x: &GpuTensor, n_tok: u64,
) -> Result<(), &'static str> {
    let nr0: i32 = if out_dim % 4 == 0 { 4 } else { 2 };
    let args = MulMvArgs {
        ne00: in_dim as i32, ne01: out_dim as i32, ne02: 1,
        nb00: 2, nb01: in_dim * 2, nb02: 0, nb03: 0,
        ne10: in_dim as i32, ne11: n_tok as i32, ne12: 1,
        nb10: 4, nb11: in_dim * 4, nb12: 0, nb13: 0,
        ne0: out_dim as i32, ne1: n_tok as i32,
        nr0, r2: 1, r3: 1,
    };
    let weight_bytes = in_dim * out_dim * 2;
    let (wbuf, woff) = views.find_view(weight_offset, weight_bytes)
        .ok_or("f16 weight view not found")?;
    let p = make_matmul_pipeline("kernel_mul_mv_f16_f32", nr0 as i16, 1)
        .ok_or("f16 matmul pipeline")?;
    dispatch_with_args(&*p, &args,
        &[(Some(wbuf), woff), (x.buf_ref(), x.offset_raw()),
          (out.buf_ref(), out.offset_raw())],
        ((out_dim / nr0 as u64 + 1) as usize * n_tok as usize, 1, 1),
        (256, 1, 1))
}

// ─── Q8_0 matmul ─────────────────────────────────────────────────────────────

pub fn matmul_q8_0(
    out: &GpuTensor, views: &ModelViews,
    weight_offset: u64, in_dim: u64, out_dim: u64,
    x: &GpuTensor, n_tok: u64,
) -> Result<(), &'static str> {
    let nr0: i32 = if out_dim % 4 == 0 { 4 } else { 2 };
    let nb01 = in_dim / 32 * 34;
    let args = MulMvArgs {
        ne00: in_dim as i32, ne01: out_dim as i32, ne02: 1,
        nb00: 34, nb01, nb02: 0, nb03: 0,
        ne10: in_dim as i32, ne11: n_tok as i32, ne12: 1,
        nb10: 4, nb11: in_dim * 4, nb12: 0, nb13: 0,
        ne0: out_dim as i32, ne1: n_tok as i32,
        nr0, r2: 1, r3: 1,
    };
    let weight_bytes = nb01 * out_dim;
    let (wbuf, woff) = views.find_view(weight_offset, weight_bytes)
        .ok_or("q8_0 weight view not found")?;
    let p = make_matmul_pipeline("kernel_mul_mv_q8_0_f32", nr0 as i16, 1)
        .ok_or("q8_0 matmul pipeline")?;
    dispatch_with_args(&*p, &args,
        &[(Some(wbuf), woff), (x.buf_ref(), x.offset_raw()),
          (out.buf_ref(), out.offset_raw())],
        ((out_dim / nr0 as u64 + 1) as usize * n_tok as usize, 1, 1),
        (256, 1, 1))
}

// ─── KV FP8 Store ────────────────────────────────────────────────────────────

pub fn kv_fp8_store(
    kv: &GpuTensor, raw_cache: &GpuTensor,
    head_dim: u32, n_rot: u32, raw_row: u32,
) -> Result<(), &'static str> {
    let args = KvFp8StoreArgs {
        head_dim: head_dim as i32, n_rot: n_rot as i32, raw_row: raw_row as i32,
    };
    dispatch_pipeline("kernel_dsv4_kv_fp8_store_f32", &args,
        &[(kv.buf_ref(), kv.offset_raw()),
          (raw_cache.buf_ref(), raw_cache.offset_raw())],
        (1, 1, 1), (64, 1, 1))
}

// ─── Compressor Store One ────────────────────────────────────────────────────

pub fn compressor_store_one(
    kv: &GpuTensor, score: &GpuTensor,
    views: &ModelViews, ape_offset: u64,
    state_kv: &GpuTensor, state_score: &GpuTensor,
    width: u32, ratio: u32, pos: u32,
) -> Result<(), &'static str> {
    let ape_bytes = ratio as u64 * width as u64 * 2; // f16
    let (ape_buf, ape_off) = views.find_view(ape_offset, ape_bytes)
        .ok_or("ape view not found")?;
    let args = CompressorStoreOneArgs { width, ratio, pos, ape_type: 1 };
    dispatch_pipeline("kernel_dsv4_compressor_store_one", &args,
        &[(kv.buf_ref(), kv.offset_raw()),
          (score.buf_ref(), score.offset_raw()),
          (Some(ape_buf), ape_off),
          (state_kv.buf_ref(), state_kv.offset_raw()),
          (state_score.buf_ref(), state_score.offset_raw())],
        (width as usize, 1, 1), (256, 1, 1))
}

// ─── Embedding: get_rows f16 ─────────────────────────────────────────────────

pub fn embed_tokens(
    out: &GpuTensor, views: &ModelViews,
    weight_offset: u64, n_vocab: u32, n_embd: u32,
    tokens: &[i32],
) -> Result<(), &'static str> {
    let weight_bytes = n_vocab as u64 * n_embd as u64 * 2;
    let (wbuf, woff) = views.find_view(weight_offset, weight_bytes)
        .ok_or("embed weight view not found")?;
    let device = bridge::device().ok_or("no device")?;
    let tbuf = unsafe {
        device.newBufferWithBytes_length_options(
            NonNull::new(tokens.as_ptr() as *mut c_void).unwrap(),
            tokens.len() * 4,
            MTLResourceOptions::StorageModeShared,
        )
    }.ok_or("token buffer")?;
    let args = metal_args::GetRowsArgs {
        ne00t: n_embd as i32, ne00: n_embd as i32,
        nb01: n_embd as u64 * 2,
        nb02: 0, nb03: 0,
        ne10: n_embd as i32, nb10: 4,
        nb11: 0, nb12: 0,
        nb1: n_embd as u64 * 4,
        nb2: 0, nb3: 0,
    };
    let n = tokens.len() * n_embd as usize;
    let p = get_or_create_pipeline("kernel_get_rows_f16").ok_or("get_rows pipeline")?;
    dispatch_with_args(&*p, &args,
        &[(Some(wbuf), woff), (Some(&*tbuf), 0), (out.buf_ref(), out.offset_raw())],
        ((n + 31) / 32, 1, 1), (32, 1, 1))
}

// ─── Router finalize one ─────────────────────────────────────────────────────

pub fn router_select_one(
    probs: &GpuTensor, bias: &GpuTensor,
    selected: &GpuTensor, _n_expert: u32, has_bias: bool,
) -> Result<(), &'static str> {
    let args = metal_args::RouterSelectOneArgs {
        has_bias: has_bias as u32, hash_mode: 0, use_token_buffer: 0,
        token: 0, hash_rows: 0,
    };
    let device = bridge::device().ok_or("no device")?;
    let zero = unsafe {
        device.newBufferWithLength_options(4, MTLResourceOptions::StorageModeShared)
    }.ok_or("zero buffer")?;
    dispatch_pipeline("kernel_dsv4_router_finalize_one", &args,
        &[(probs.buf_ref(), probs.offset_raw()),
          (bias.buf_ref(), bias.offset_raw()),
          (Some(&*zero), 0),
          (Some(&*zero), 0),
          (selected.buf_ref(), selected.offset_raw())],
        (1, 1, 1), (256, 1, 1))
}

// ─── Router weights one ──────────────────────────────────────────────────────

pub fn router_weights_one(
    probs: &GpuTensor, selected: &GpuTensor, weights: &GpuTensor,
) -> Result<(), &'static str> {
    let p = get_or_create_pipeline("kernel_dsv4_router_weights_one")
        .ok_or("router weights pipeline")?;
    dispatch_raw(&*p,
        &[(probs.buf_ref(), probs.offset_raw()),
          (selected.buf_ref(), selected.offset_raw()),
          (weights.buf_ref(), weights.offset_raw())],
        (1, 1, 1), (6, 1, 1))
}

// ─── MoE SwiGLU with route weight ────────────────────────────────────────────

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
        &[(gate.buf_ref(), gate.offset_raw()),
          (up.buf_ref(), up.offset_raw()),
          (mid.buf_ref(), mid.offset_raw()),
          (weights.buf_ref(), weights.offset_raw())],
        (1, 1, 1), (256, 1, 1))
}

// ─── Indexed mixed attention ─────────────────────────────────────────────────

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
        q_token_stride: n_head as u64 * q_head_stride,
        q_head_stride,
        raw_row_stride: N_HEAD_DIM as u64 * 4,
        comp_row_stride: N_HEAD_DIM as u64 * 4,
        topk_token_stride: topk_k as u64 * 4,
        dst_token_stride: n_head as u64 * dst_head_stride,
        dst_head_stride,
        scale,
    };
    let device = bridge::device().ok_or("no device")?;
    let _dummy: Option<Retained<ProtocolObject<dyn MTLBuffer>>> = if sinks.buf_ref().is_none() {
        Some(unsafe {
            device.newBufferWithLength_options(4, MTLResourceOptions::StorageModeShared)
        }.ok_or("dummy buffer")?)
    } else {
        None
    };
    let sink_ref = sinks.buf_ref().or(_dummy.as_deref());
    dispatch_pipeline("kernel_dsv4_indexed_mixed_attention_heads8", &args,
        &[(q.buf_ref(), q.offset_raw()),
          (raw_kv.buf_ref(), raw_kv.offset_raw()),
          (comp_kv.buf_ref(), comp_kv.offset_raw()),
          (topk.buf_ref(), topk.offset_raw()),
          (sink_ref, 0),
          (heads.buf_ref(), heads.offset_raw())],
        (n_tokens as usize, (n_head / 8 + 1) as usize, 1), (128, 1, 1))
}

// ─── Directional steering project ────────────────────────────────────────────

pub fn directional_steering_project(
    x: &GpuTensor, directions: &GpuTensor,
    layer: u32, width: u32, rows: u32, scale: f32,
) -> Result<(), &'static str> {
    let args = DirectionalSteeringProjectArgs {
        width, rows, layer, n_threads: 256, scale,
    };
    dispatch_pipeline("kernel_dsv4_directional_steering_project_f32", &args,
        &[(x.buf_ref(), x.offset_raw()),
          (directions.buf_ref(), directions.offset_raw())],
        (rows as usize, 1, 1), (256, 1, 1))
}

// ─── Expert matmul: IQ2_XXS pair ─────────────────────────────────────────────

pub fn matmul_id_iq2_xxs_pair(
    dst_gate: &GpuTensor, dst_up: &GpuTensor,
    views: &ModelViews, gate_offset: u64, _up_offset: u64,
    in_dim: u32, out_dim: u32, n_expert: u32,
    x: &GpuTensor, selected: &[i32],
) -> Result<(), &'static str> {
    let nr0: u32 = if out_dim % 4 == 0 { 4 } else { 2 };
    let step = in_dim as u64 / 32 * 66 * out_dim as u64;
    let total_bytes = step * n_expert as u64 * 2; // gate + up interleaved
    let (wbuf, woff) = views.find_view(gate_offset, total_bytes)
        .ok_or("iq2 expert view not found")?;
    let args = MulMvIdArgs {
        nei0: 1, nei1: selected.len() as i32, nbi1: 4,
        ne00: in_dim as i32, ne01: out_dim as i32,
        ne02: (n_expert * in_dim / 32 * 66) as i32,
        nb00: 66, nb01: in_dim as u64 / 32 * 66,
        nb02: in_dim as u64 / 32 * 66 * out_dim as u64,
        ne10: in_dim as i32, ne11: 1, ne12: 1, ne13: 1,
        nb10: 4, nb11: in_dim as u64 * 4,
        nb12: in_dim as u64 * 4,
        ne0: out_dim as i32, ne1: 1, nb1: out_dim as u64 * 4,
        nr0: nr0 as i32,
    };
    let device = bridge::device().ok_or("no device")?;
    let sel_buf = unsafe {
        device.newBufferWithBytes_length_options(
            NonNull::new(selected.as_ptr() as *mut c_void).unwrap(),
            selected.len() * 4,
            MTLResourceOptions::StorageModeShared,
        )
    }.ok_or("sel buffer")?;
    let nsg: i16 = if nr0 == 4 { 4 } else { 2 };
    let p = make_matmul_pipeline("kernel_mul_mv_id_iq2_xxs_pair_f32", nsg, 1)
        .ok_or("iq2 pair pipeline")?;
    let up_off = woff + step * n_expert as u64;
    dispatch_with_args(&*p, &args,
        &[(Some(wbuf), woff),
          (Some(wbuf), up_off),
          (x.buf_ref(), x.offset_raw()),
          (dst_gate.buf_ref(), dst_gate.offset_raw()),
          (dst_up.buf_ref(), dst_up.offset_raw()),
          (Some(&*sel_buf), 0)],
        ((out_dim / nr0 + 1) as usize, 1, selected.len()),
        (256, 1, 1))
}

// ─── Expert matmul: Q2_K sum6 ────────────────────────────────────────────────

#[allow(non_snake_case)]
pub fn matmul_id_q2_K_sum6(
    dst_gate: &GpuTensor, dst_up: &GpuTensor,
    views: &ModelViews, gate_offset: u64, _up_offset: u64,
    in_dim: u32, out_dim: u32, n_expert: u32,
    x: &GpuTensor, selected: &[i32],
) -> Result<(), &'static str> {
    let nr0: u32 = 4;
    let step = in_dim as u64 / 256 * 84 * out_dim as u64;
    let total_bytes = step * n_expert as u64 * 2;
    let (wbuf, woff) = views.find_view(gate_offset, total_bytes)
        .ok_or("q2_K expert view not found")?;
    let args = MulMvIdArgs {
        nei0: 1, nei1: selected.len() as i32, nbi1: 4,
        ne00: in_dim as i32, ne01: out_dim as i32,
        ne02: (n_expert * in_dim / 256 * 84) as i32,
        nb00: 84, nb01: in_dim as u64 / 256 * 84,
        nb02: in_dim as u64 / 256 * 84 * out_dim as u64,
        ne10: in_dim as i32, ne11: 1, ne12: 1, ne13: 1,
        nb10: 4, nb11: in_dim as u64 * 4,
        nb12: in_dim as u64 * 4,
        ne0: out_dim as i32, ne1: 1, nb1: out_dim as u64 * 4,
        nr0: nr0 as i32,
    };
    let device = bridge::device().ok_or("no device")?;
    let sel_buf = unsafe {
        device.newBufferWithBytes_length_options(
            NonNull::new(selected.as_ptr() as *mut c_void).unwrap(),
            selected.len() * 4,
            MTLResourceOptions::StorageModeShared,
        )
    }.ok_or("sel buffer")?;
    let p = make_matmul_pipeline("kernel_mul_mv_id_q2_K_sum6_f32", 4, 1)
        .ok_or("q2 sum6 pipeline")?;
    let up_off = woff + step * n_expert as u64;
    dispatch_with_args(&*p, &args,
        &[(Some(wbuf), woff),
          (Some(wbuf), up_off),
          (x.buf_ref(), x.offset_raw()),
          (dst_gate.buf_ref(), dst_gate.offset_raw()),
          (dst_up.buf_ref(), dst_up.offset_raw()),
          (Some(&*sel_buf), 0)],
        ((out_dim / nr0 + 1) as usize, 1, selected.len()),
        (256, 1, 1))
}

// ─── Expert matmul: Q4_K sum6 ────────────────────────────────────────────────

#[allow(non_snake_case)]
pub fn matmul_id_q4_K_sum6(
    dst_gate: &GpuTensor, dst_up: &GpuTensor,
    views: &ModelViews, gate_offset: u64, _up_offset: u64,
    in_dim: u32, out_dim: u32, n_expert: u32,
    x: &GpuTensor, selected: &[i32],
) -> Result<(), &'static str> {
    let nr0: u32 = 2;
    let step = in_dim as u64 / 256 * 144 * out_dim as u64;
    let total_bytes = step * n_expert as u64 * 2;
    let (wbuf, woff) = views.find_view(gate_offset, total_bytes)
        .ok_or("q4_K expert view not found")?;
    let args = MulMvIdArgs {
        nei0: 1, nei1: selected.len() as i32, nbi1: 4,
        ne00: in_dim as i32, ne01: out_dim as i32,
        ne02: (n_expert * in_dim / 256 * 144) as i32,
        nb00: 144, nb01: in_dim as u64 / 256 * 144,
        nb02: in_dim as u64 / 256 * 144 * out_dim as u64,
        ne10: in_dim as i32, ne11: 1, ne12: 1, ne13: 1,
        nb10: 4, nb11: in_dim as u64 * 4,
        nb12: in_dim as u64 * 4,
        ne0: out_dim as i32, ne1: 1, nb1: out_dim as u64 * 4,
        nr0: nr0 as i32,
    };
    let device = bridge::device().ok_or("no device")?;
    let sel_buf = unsafe {
        device.newBufferWithBytes_length_options(
            NonNull::new(selected.as_ptr() as *mut c_void).unwrap(),
            selected.len() * 4,
            MTLResourceOptions::StorageModeShared,
        )
    }.ok_or("sel buffer")?;
    let p = make_matmul_pipeline("kernel_mul_mv_id_q4_K_sum6_f32", 2, 1)
        .ok_or("q4 sum6 pipeline")?;
    let up_off = woff + step * n_expert as u64;
    dispatch_with_args(&*p, &args,
        &[(Some(wbuf), woff),
          (Some(wbuf), up_off),
          (x.buf_ref(), x.offset_raw()),
          (dst_gate.buf_ref(), dst_gate.offset_raw()),
          (dst_up.buf_ref(), dst_up.offset_raw()),
          (Some(&*sel_buf), 0)],
        ((out_dim / nr0 + 1) as usize, 1, selected.len()),
        (256, 1, 1))
}

// ─── Indexer score one ───────────────────────────────────────────────────────

pub fn indexer_score_one(
    scores: &GpuTensor, q: &GpuTensor,
    views: &ModelViews, weight_offset: u64,
    index_comp: &GpuTensor,
    n_comp: u32, n_head: u32, head_dim: u32,
    pos0: u32, ratio: u32, scale: f32,
) -> Result<(), &'static str> {
    let args = IndexerScoresFusedArgs {
        n_comp, n_tokens: 1, n_head, head_dim, pos0, ratio,
        q_token_stride: n_head as u64 * head_dim as u64 * 4,
        q_head_stride: head_dim as u64 * 4,
        weights_token_stride: 0,
        index_row_stride: head_dim as u64 * 4,
        score_token_stride: 0, scale,
    };
    let weight_bytes = n_head as u64 * 4;
    let (w_opt, w_off): (Option<&ProtocolObject<dyn MTLBuffer>>, u64) = if weight_offset != 0 {
        views.find_view(weight_offset, weight_bytes)
            .map(|(b, o)| (Some(b), o))
            .unwrap_or((None, 0))
    } else {
        (None, 0)
    };
    dispatch_pipeline("kernel_dsv4_indexer_score_one_direct", &args,
        &[(q.buf_ref(), q.offset_raw()),
          (w_opt, w_off),
          (index_comp.buf_ref(), index_comp.offset_raw()),
          (scores.buf_ref(), scores.offset_raw())],
        (n_comp as usize, 1, 1), (128, 1, 1))
}

// ─── Softmax pool ────────────────────────────────────────────────────────────

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
        &[(kv.buf_ref(), kv.offset_raw()),
          (score.buf_ref(), score.offset_raw()),
          (dst.buf_ref(), dst.offset_raw())],
        (head_dim as usize, 1, 1), (256, 1, 1))
}

// ─── Softplus + sqrt ─────────────────────────────────────────────────────────

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
        &[(src.buf_ref(), src.offset_raw()),
          (dst.buf_ref(), dst.offset_raw())],
        (1, 1, 1), (256, 1, 1))
}

// ─── Flash attention (prefill) ───────────────────────────────────────────────

pub fn flash_attn_prefill(
    dst: &GpuTensor, q: &GpuTensor, k: &GpuTensor, v: &GpuTensor,
    n_tokens: u32, n_kv: u32, n_head: u32, head_dim: u32,
    scale: f32,
) -> Result<(), &'static str> {
    let q_stride = (n_head * head_dim * 4) as u64;
    let kv_stride = (head_dim * 2) as u64;
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
    let device = bridge::device().ok_or("no device")?;
    let pad = unsafe {
        device.newBufferWithLength_options(1, MTLResourceOptions::StorageModePrivate)
    }.ok_or("pad buffer")?;
    let blk = unsafe {
        device.newBufferWithLength_options(1, MTLResourceOptions::StorageModePrivate)
    }.ok_or("blk buffer")?;
    dispatch_with_args(&*p, &args,
        &[(q.buf_ref(), q.offset_raw()),
          (k.buf_ref(), k.offset_raw()),
          (v.buf_ref(), v.offset_raw()),
          (None, 0), (None, 0),
          (Some(&*pad), 0), (Some(&*blk), 0),
          (dst.buf_ref(), dst.offset_raw())],
        (n_tokens as usize, n_head as usize, 1), (256, 1, 1))
}

// ─── Repeat f32 ──────────────────────────────────────────────────────────────

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
        &[(src.buf_ref(), src.offset_raw()),
          (dst.buf_ref(), dst.offset_raw())],
        (ne1 as usize, 1, 1), (256, 1, 1))
}

// ─── Compressor ratio4 shift ─────────────────────────────────────────────────

pub fn compressor_ratio4_shift(
    state_kv: &GpuTensor, state_score: &GpuTensor, width: u32,
) -> Result<(), &'static str> {
    let args = metal_args::Ratio4ShiftArgs { width };
    dispatch_pipeline("kernel_dsv4_ratio4_shift_f32", &args,
        &[(state_kv.buf_ref(), state_kv.offset_raw()),
          (state_score.buf_ref(), state_score.offset_raw())],
        (4 * width as usize, 1, 1), (256, 1, 1))
}

// Re-export alias
#[allow(unused_imports)]
pub use kv_fp8_store as kv_fp8_store_raw;
