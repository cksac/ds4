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

fn wait_gpu(cb: &ProtocolObject<dyn MTLCommandBuffer>) -> Result<(), &'static str> {
    cb.waitUntilCompleted();
    if cb.status() == objc2_metal::MTLCommandBufferStatus::Error {
        Err("gpu kernel failed")
    } else {
        Ok(())
    }
}

const FC_MUL_MV: usize = 600;
const FC_BIN: usize = 1300;
const FC_UNARY: usize = 1200;
const FC_SUM_ROWS: usize = 1400;

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

fn make_bin_pipeline(op: i16, f: i16, rb: bool, cb: bool)
    -> Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>
{
    let key = format!("kernel_bin_fuse_f32_f32_f32_{}_{}_{}_{}", op, f, rb, cb);
    if let Some(p) = pipeline::get_pipeline(&key) { return Some(p); }
    let lib = bridge::library()?;
    let fcv = MTLFunctionConstantValues::new();
    let mut op_v = op; let mut f_v = f; let mut rb_v = rb as u8; let mut cb_v = cb as u8;
    unsafe {
        fcv.setConstantValue_type_atIndex(
            NonNull::new(&mut op_v as *mut i16 as *mut c_void).unwrap(),
            MTLDataType::Short, FC_BIN + 0);
        fcv.setConstantValue_type_atIndex(
            NonNull::new(&mut f_v as *mut i16 as *mut c_void).unwrap(),
            MTLDataType::Short, FC_BIN + 1);
        fcv.setConstantValue_type_atIndex(
            NonNull::new(&mut rb_v as *mut u8 as *mut c_void).unwrap(),
            MTLDataType::Bool, FC_BIN + 2);
        fcv.setConstantValue_type_atIndex(
            NonNull::new(&mut cb_v as *mut u8 as *mut c_void).unwrap(),
            MTLDataType::Bool, FC_BIN + 3);
    }
    let ns_name = NSString::from_str("kernel_bin_fuse_f32_f32_f32");
    let fn_ = lib.newFunctionWithName_constantValues_error(&*ns_name, &*fcv).ok()?;
    let device = bridge::device()?;
    let p = device.newComputePipelineStateWithFunction_error(&*fn_).ok()?;
    pipeline::cache_pipeline(&key, p.clone());
    Some(p)
}

fn make_unary_pipeline(op: i16)
    -> Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>
{
    let key = format!("kernel_unary_f32_f32_4_{}", op);
    if let Some(p) = pipeline::get_pipeline(&key) { return Some(p); }
    let lib = bridge::library()?;
    let fcv = MTLFunctionConstantValues::new();
    let mut op_v = op; let mut cnt_v: u8 = 0;
    unsafe {
        fcv.setConstantValue_type_atIndex(
            NonNull::new(&mut op_v as *mut i16 as *mut c_void).unwrap(),
            MTLDataType::Short, FC_UNARY + 0);
        fcv.setConstantValue_type_atIndex(
            NonNull::new(&mut cnt_v as *mut u8 as *mut c_void).unwrap(),
            MTLDataType::Bool, FC_UNARY + 1);
    }
    let ns_name = NSString::from_str("kernel_unary_f32_f32_4");
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
                MTLDataType::Bool, 300 + i);
        }
        fcv.setConstantValue_type_atIndex(
            NonNull::new(&mut nsg_v as *mut i32 as *mut c_void).unwrap(),
            MTLDataType::Int, 300 + 22);
        fcv.setConstantValue_type_atIndex(
            NonNull::new(&mut zero_v as *mut i32 as *mut c_void).unwrap(),
            MTLDataType::Int, 300 + 20);
        fcv.setConstantValue_type_atIndex(
            NonNull::new(&mut zero_v as *mut i32 as *mut c_void).unwrap(),
            MTLDataType::Int, 300 + 21);
    }
    let ns_name = NSString::from_str(name);
    let fn_ = lib.newFunctionWithName_constantValues_error(&*ns_name, &*fcv).ok()?;
    let device = bridge::device()?;
    let p = device.newComputePipelineStateWithFunction_error(&*fn_).ok()?;
    pipeline::cache_pipeline(&key, p.clone());
    Some(p)
}

fn dispatch_with_args_tg<T: Sized>(
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    args: &T,
    buffers: &[(Option<&ProtocolObject<dyn MTLBuffer>>, u64)],
    threads: (usize, usize, usize),
    tg_size: (usize, usize, usize),
    tg_mem: u64,
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
    if tg_mem > 0 {
        unsafe { enc.setThreadgroupMemoryLength_atIndex(tg_mem as usize, 0); }
    }
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize { width: threads.0, height: threads.1, depth: threads.2 },
        MTLSize { width: tg_size.0, height: tg_size.1, depth: tg_size.2 },
    );
    enc.endEncoding();
    cb.commit();
    wait_gpu(&*cb)
}

fn dispatch_with_args<T: Sized>(
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    args: &T,
    buffers: &[(Option<&ProtocolObject<dyn MTLBuffer>>, u64)],
    threads: (usize, usize, usize),
    tg_size: (usize, usize, usize),
) -> Result<(), &'static str> {
    dispatch_with_args_tg(pipeline, args, buffers, threads, tg_size, 0)
}

fn dispatch_pipeline_tg<T: Sized>(
    pipeline_name: &str, args: &T,
    buffers: &[(Option<&ProtocolObject<dyn MTLBuffer>>, u64)],
    threads: (usize, usize, usize),
    tg_size: (usize, usize, usize),
    tg_mem: u64,
) -> Result<(), &'static str> {
    let p = get_or_create_pipeline(pipeline_name).ok_or("pipeline not found")?;
    dispatch_with_args_tg(&*p, args, buffers, threads, tg_size, tg_mem)
}

fn dispatch_pipeline<T: Sized>(
    pipeline_name: &str, args: &T,
    buffers: &[(Option<&ProtocolObject<dyn MTLBuffer>>, u64)],
    threads: (usize, usize, usize),
    tg_size: (usize, usize, usize),
) -> Result<(), &'static str> {
    dispatch_pipeline_tg(pipeline_name, args, buffers, threads, tg_size, 0)
}

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

fn rms_norm_threads(n: u32) -> u32 {
    let mut nth = 32u32;
    let ne00_t = n / 4;
    while nth < ne00_t && nth < 1024 { nth <<= 1; }
    if nth > ne00_t { nth = ne00_t; }
    nth.max(1)
}

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

#[repr(C)]
struct SumRowsArgs {
    ne00: i64, ne01: i64, ne02: i64, ne03: i64,
    nb00: u64, nb01: u64, nb02: u64, nb03: u64,
    ne0: i64, ne1: i64, ne2: i64, ne3: i64,
    nb0: u64, nb1: u64, nb2: u64, nb3: u64,
}

// ─── RMS Norm ───────────────────────────────────────────────────────────

pub fn rms_norm_plain(
    out: &GpuTensor, x: &GpuTensor, n: u32, eps: f32,
) -> Result<(), &'static str> {
    rms_norm_plain_rows(out, x, n, 1, eps)
}

pub fn rms_norm_plain_rows(
    out: &GpuTensor, x: &GpuTensor, n: u32, rows: u32, eps: f32,
) -> Result<(), &'static str> {
    let n4 = (n / 4) as i32;
    let row_bytes = (n * 4) as u64;
    let plane_bytes = row_bytes * rows as u64;
    let args = RmsNormArgs {
        ne00: n as i32, ne00_t: n4,
        nb1: row_bytes, nb2: plane_bytes, nb3: plane_bytes, eps,
        nef1: [rows as i32, 1, 1], nef2: [1, 1, 1], nef3: [1, 1, 1],
        nbf1: [row_bytes, 0, 0], nbf2: [0; 3], nbf3: [0; 3],
    };
    let nth = rms_norm_threads(n) as usize;
    dispatch_pipeline_tg("kernel_rms_norm_f32_4", &args,
        &[(x.buf_ref(), x.offset_raw()),
          (x.buf_ref(), x.offset_raw()),
          (x.buf_ref(), x.offset_raw()),
          (out.buf_ref(), out.offset_raw())],
        (rows as usize, 1, 1), (nth, 1, 1), 128)
}

pub fn rms_norm_weight(
    out: &GpuTensor, x: &GpuTensor,
    views: &ModelViews, weight_offset: u64,
    n: u32, eps: f32,
) -> Result<(), &'static str> {
    let row_bytes = (n * 4) as u64;
    let (wbuf, woff) = views.find_view(weight_offset, row_bytes)
        .ok_or("rms_norm weight view")?;
    let n4 = (n / 4) as i32;
    let args = RmsNormArgs {
        ne00: n as i32, ne00_t: n4,
        nb1: row_bytes, nb2: row_bytes, nb3: row_bytes, eps,
        nef1: [1, 1, 1], nef2: [1, 1, 1], nef3: [1, 1, 1],
        nbf1: [row_bytes, 0, 0], nbf2: [0; 3], nbf3: [0; 3],
    };
    let nth = rms_norm_threads(n) as usize;
    dispatch_pipeline_tg("kernel_rms_norm_mul_f32_4", &args,
        &[(x.buf_ref(), x.offset_raw()),
          (Some(wbuf), woff),
          (x.buf_ref(), x.offset_raw()),
          (out.buf_ref(), out.offset_raw())],
        (1, 1, 1), (nth, 1, 1), 128)
}

pub fn head_rms_norm(
    x: &GpuTensor, n_tok: u32, n_head: u32, head_dim: u32, eps: f32,
) -> Result<(), &'static str> {
    let n4 = (head_dim / 4) as i32;
    let row_bytes = (head_dim * 4) as u64;
    let head_bytes = row_bytes * n_head as u64;
    let plane_bytes = head_bytes * n_tok as u64;
    let args = RmsNormArgs {
        ne00: head_dim as i32, ne00_t: n4,
        nb1: row_bytes, nb2: head_bytes, nb3: plane_bytes, eps,
        nef1: [n_head as i32, 1, 1], nef2: [n_tok as i32, 1, 1], nef3: [1, 1, 1],
        nbf1: [row_bytes, 0, 0], nbf2: [head_bytes, 0, 0], nbf3: [plane_bytes, 0, 0],
    };
    let nth = rms_norm_threads(head_dim) as usize;
    dispatch_pipeline_tg("kernel_rms_norm_f32_4", &args,
        &[(x.buf_ref(), x.offset_raw()),
          (x.buf_ref(), x.offset_raw()),
          (x.buf_ref(), x.offset_raw()),
          (x.buf_ref(), x.offset_raw())],
        (n_head as usize, n_tok as usize, 1), (nth, 1, 1), 128)
}

// ─── RoPE ───────────────────────────────────────────────────────────────

pub fn rope_tail(
    x: &GpuTensor, _n_tok: u32, n_head: u32, head_dim: u32,
    n_rot: u32, pos: u32, n_ctx_orig: u32, inverse: bool,
    freq_base: f32, freq_scale: f32, ext_factor: f32,
    attn_factor: f32, beta_fast: f32, beta_slow: f32,
) -> Result<(), &'static str> {
    let stride = (head_dim * 4) as u64;
    let args = RopeTailArgs {
        ne00: head_dim as i64, ne01: n_head as i64, ne02: 1, ne03: 1,
        nb00: 4, nb01: stride, nb02: stride, nb03: stride,
        nb0: 4, nb1: stride, nb2: stride, nb3: stride,
        n_dims: n_rot as i32, mode: 2, n_ctx_orig: n_ctx_orig as i32,
        inverse: inverse as i32, freq_base, freq_scale, ext_factor, attn_factor,
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

// ─── HC Split Sinkhorn ──────────────────────────────────────────────────

pub fn hc_split_sinkhorn(
    out: &GpuTensor, mixes: &GpuTensor,
    views: &ModelViews,
    scale_offset: u64, base_offset: u64,
    n_hc: u32, sinkhorn_iters: u32, eps: f32,
) -> Result<(), &'static str> {
    let mix_hc = (2 * n_hc + n_hc * n_hc) as u64;
    let mix_bytes = mix_hc * 4;
    let scale_bytes = 3 * 4;
    let (scale_buf, scale_inner) = views.find_view(scale_offset, scale_bytes)
        .ok_or("hc_split_sinkhorn scale view")?;
    let (base_buf, base_inner) = views.find_view(base_offset, mix_bytes)
        .ok_or("hc_split_sinkhorn base view")?;
    let args = HcSplitSinkhornArgs {
        n_hc: n_hc as i32, sinkhorn_iters: sinkhorn_iters as i32,
        n_rows: 1, mix_hc: mix_hc as i64,
        nb01: mix_bytes, nb1: mix_bytes, eps,
    };
    let nth = 1usize.max(256.min(1));
    let n_tg = (1 + nth - 1) / nth;
    dispatch_pipeline("kernel_dsv4_hc_split_sinkhorn", &args,
        &[(mixes.buf_ref(), mixes.offset_raw()),
          (Some(scale_buf), scale_inner),
          (Some(base_buf), base_inner),
          (out.buf_ref(), out.offset_raw())],
        (n_tg, 1, 1), (nth, 1, 1))
}

// ─── HC Weighted Sum ────────────────────────────────────────────────────

pub fn hc_weighted_sum(
    out: &GpuTensor, residual_hc: &GpuTensor, weights: &GpuTensor,
    n_embd: u32, n_hc: u32,
) -> Result<(), &'static str> {
    let n_tokens = 1u64;
    let n_elem = n_embd as u64 * n_tokens;
    let nth = 256.min(n_elem as usize).max(1);
    let n_tg = (n_elem as usize + nth - 1) / nth;
    let args = HcWeightedSumArgs {
        n_embd: n_embd as i64, n_hc: n_hc as i64, n_tokens: n_tokens as i64,
        nb_x0: 4, nb_x1: (n_embd * 4) as u64,
        nb_x2: (n_hc as u64 * n_embd as u64 * 4),
        nb_w0: 4, nb_w1: (n_hc * 4) as u64,
        nb0: 4, nb1: (n_embd * 4) as u64,
    };
    dispatch_pipeline("kernel_dsv4_hc_weighted_sum", &args,
        &[(residual_hc.buf_ref(), residual_hc.offset_raw()),
          (weights.buf_ref(), weights.offset_raw()),
          (out.buf_ref(), out.offset_raw())],
        (n_tg, 1, 1), (nth, 1, 1))
}

// ─── HC Expand ──────────────────────────────────────────────────────────

pub fn hc_expand_tensor(
    out_hc: &GpuTensor, block_out: &GpuTensor,
    residual_hc: &GpuTensor,
    post: &GpuTensor, comb: &GpuTensor,
    n_embd: u32, n_hc: u32,
) -> Result<(), &'static str> {
    let n_tokens = 1u64;
    let n_elem = if n_hc == 4 { n_embd as u64 } else { n_embd as u64 * n_hc as u64 };
    let nth = 256u64.min(n_elem).max(1) as usize;
    let n_tg = (n_elem as usize + nth - 1) / nth;
    let mix_hc = (2 * n_hc + n_hc * n_hc) as u64;
    let args = HcExpandArgs {
        n_embd: n_embd as i64, n_hc: n_hc as i64, n_tokens: n_tokens as i64,
        nb_block0: 4, nb_block1: (n_embd * 4) as u64,
        nb_add0: 4, nb_add1: (n_embd * 4) as u64,
        nb_res0: 4, nb_res1: (n_embd * 4) as u64,
        nb_res2: (n_hc as u64 * n_embd as u64 * 4),
        nb_post0: 4, nb_post1: (n_hc * 4) as u64,
        nb_comb0: 4, nb_comb1: (n_hc * 4) as u64,
        nb_comb2: (n_hc as u64 * n_hc as u64 * 4),
        nb0: 4, nb1: (n_embd * 4) as u64,
        nb2: (n_hc as u64 * n_embd as u64 * 4),
        has_add: 0,
    };
    dispatch_pipeline("kernel_dsv4_hc_expand4", &args,
        &[(block_out.buf_ref(), block_out.offset_raw()),
          (residual_hc.buf_ref(), residual_hc.offset_raw()),
          (post.buf_ref(), post.offset_raw()),
          (comb.buf_ref(), comb.offset_raw()),
          (block_out.buf_ref(), block_out.offset_raw()),
          (out_hc.buf_ref(), out_hc.offset_raw())],
        (n_tg, 1, 1), (nth, 1, 1))
}

pub fn hc_expand_add_split_tensor(
    out_hc: &GpuTensor, block_out: &GpuTensor,
    block_add: &GpuTensor,
    residual_hc: &GpuTensor, split: &GpuTensor,
    n_embd: u32, n_hc: u32,
) -> Result<(), &'static str> {
    let n_tokens = 1u64;
    let n_elem = if n_hc == 4 { n_embd as u64 } else { n_embd as u64 * n_hc as u64 };
    let nth = 256u64.min(n_elem).max(1) as usize;
    let n_tg = (n_elem as usize + nth - 1) / nth;
    let mix_hc = (2 * n_hc + n_hc * n_hc) as u64;
    let post_off = (n_hc * 4) as u64;
    let comb_off = (2 * n_hc * 4) as u64;
    let args = HcExpandArgs {
        n_embd: n_embd as i64, n_hc: n_hc as i64, n_tokens: n_tokens as i64,
        nb_block0: 4, nb_block1: (n_embd * 4) as u64,
        nb_add0: 4, nb_add1: (n_embd * 4) as u64,
        nb_res0: 4, nb_res1: (n_embd * 4) as u64,
        nb_res2: (n_hc as u64 * n_embd as u64 * 4),
        nb_post0: 4, nb_post1: mix_hc * 4,
        nb_comb0: 4, nb_comb1: (n_hc * 4) as u64,
        nb_comb2: mix_hc * 4,
        nb0: 4, nb1: (n_embd * 4) as u64,
        nb2: (n_hc as u64 * n_embd as u64 * 4),
        has_add: 1,
    };
    dispatch_pipeline("kernel_dsv4_hc_expand4", &args,
        &[(block_out.buf_ref(), block_out.offset_raw()),
          (residual_hc.buf_ref(), residual_hc.offset_raw()),
          (split.buf_ref(), split.offset_raw() + post_off),
          (split.buf_ref(), split.offset_raw() + comb_off),
          (block_add.buf_ref(), block_add.offset_raw()),
          (out_hc.buf_ref(), out_hc.offset_raw())],
        (n_tg, 1, 1), (nth, 1, 1))
}

// ─── Output HC Weights ──────────────────────────────────────────────────

pub fn output_hc_weights(
    out: &GpuTensor, pre: &GpuTensor,
    views: &ModelViews,
    scale_offset: u64, base_offset: u64,
    n_hc: u32, eps: f32,
) -> Result<(), &'static str> {
    let row_bytes = (n_hc * 4) as u64;
    let (scale_buf, scale_inner) = views.find_view(scale_offset, 4)
        .ok_or("output_hc_weights scale view")?;
    let (base_buf, base_inner) = views.find_view(base_offset, row_bytes)
        .ok_or("output_hc_weights base view")?;

    let mul_p = make_bin_pipeline(2, 1, false, true).ok_or("mul_scalar pipeline")?;
    let add_p = make_bin_pipeline(0, 1, false, false).ok_or("add pipeline")?;
    let sig_p = make_unary_pipeline(102).ok_or("sigmoid pipeline")?;
    let sca_p = make_unary_pipeline(10).ok_or("scale pipeline")?;

    let n_tokens = 1u32;
    let mul_args = metal_args::BinArgs {
        ne00: n_hc as i32, ne01: n_tokens as i32, ne02: 1, ne03: 1,
        nb00: 4, nb01: row_bytes, nb02: row_bytes, nb03: row_bytes,
        ne10: 1, ne11: 1, ne12: 1, ne13: 1,
        nb10: 4, nb11: 4, nb12: 4, nb13: 4,
        ne0: n_hc as i32, ne1: n_tokens as i32, ne2: 1, ne3: 1,
        nb0: 4, nb1: row_bytes, nb2: row_bytes, nb3: row_bytes,
        offs: 0, o1: [0; 8],
    };
    let add_args = metal_args::BinArgs {
        ne00: n_hc as i32, ne01: n_tokens as i32, ne02: 1, ne03: 1,
        nb00: 4, nb01: row_bytes, nb02: row_bytes, nb03: row_bytes,
        ne10: n_hc as i32, ne11: 1, ne12: 1, ne13: 1,
        nb10: 4, nb11: row_bytes, nb12: row_bytes, nb13: row_bytes,
        ne0: n_hc as i32, ne1: n_tokens as i32, ne2: 1, ne3: 1,
        nb0: 4, nb1: row_bytes, nb2: row_bytes, nb3: row_bytes,
        offs: 0, o1: [0; 8],
    };
    let unary_c4 = n_hc / 4;
    let sig_args = UnaryArgs {
        ne00: unary_c4 as i32, ne01: n_tokens as i32, ne02: 1, ne03: 1,
        nb00: 4, nb01: row_bytes, nb02: row_bytes, nb03: row_bytes,
        ne0: unary_c4 as i32, ne1: n_tokens as i32, ne2: 1, ne3: 1,
        nb0: 4, nb1: row_bytes, nb2: row_bytes, nb3: row_bytes,
        slope: 0.0, scale: 0.0, bias: 0.0, val: 0.0, min_: 0.0, max_: 0.0,
    };
    let sca_args = UnaryArgs {
        ne00: unary_c4 as i32, ne01: n_tokens as i32, ne02: 1, ne03: 1,
        nb00: 4, nb01: row_bytes, nb02: row_bytes, nb03: row_bytes,
        ne0: unary_c4 as i32, ne1: n_tokens as i32, ne2: 1, ne3: 1,
        nb0: 4, nb1: row_bytes, nb2: row_bytes, nb3: row_bytes,
        slope: 0.0, scale: 1.0, bias: eps, val: 0.0, min_: 0.0, max_: 0.0,
    };

    let mut mul_nth = 1usize;
    while 2 * mul_nth < mul_args.ne0 as usize && mul_nth < 256 { mul_nth *= 2; }
    let mut add_nth = 1usize;
    while 2 * add_nth < add_args.ne0 as usize && add_nth < 256 { add_nth *= 2; }
    let unary_nth = (unary_c4 as usize).min(256).max(1);
    let unary_nk0 = (unary_c4 as usize + unary_nth - 1) / unary_nth;

    let queue = bridge::queue().ok_or("no queue")?;
    let cb = queue.commandBuffer().ok_or("no command buffer")?;
    let enc = cb.computeCommandEncoder().ok_or("no encoder")?;

    let pre_off = pre.offset_raw();
    let out_off = out.offset_raw();
    let out_ref = out.buf_ref();
    let pre_ref = pre.buf_ref();

    // Dispatch 1: bin_mul_scalar
    unsafe {
        let args_buf = bridge::device().ok_or("no device")?.newBufferWithBytes_length_options(
            NonNull::new(&mul_args as *const _ as *mut c_void).unwrap(),
            std::mem::size_of::<metal_args::BinArgs>(),
            MTLResourceOptions::StorageModeShared,
        ).ok_or("args buf")?;
        enc.setComputePipelineState(&*mul_p);
        enc.setBuffer_offset_atIndex(Some(&*args_buf), 0, 0);
        if let Some(b) = pre_ref { enc.setBuffer_offset_atIndex(Some(b), pre_off as usize, 1); }
        enc.setBuffer_offset_atIndex(Some(scale_buf), scale_inner as usize, 2);
        if let Some(b) = out_ref { enc.setBuffer_offset_atIndex(Some(b), out_off as usize, 3); }
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize { width: mul_args.ne01 as usize, height: 1, depth: 1 },
            MTLSize { width: mul_nth, height: 1, depth: 1 },
        );
    }

    // Dispatch 2: add
    unsafe {
        let args_buf = bridge::device().ok_or("no device")?.newBufferWithBytes_length_options(
            NonNull::new(&add_args as *const _ as *mut c_void).unwrap(),
            std::mem::size_of::<metal_args::BinArgs>(),
            MTLResourceOptions::StorageModeShared,
        ).ok_or("args buf")?;
        enc.setComputePipelineState(&*add_p);
        enc.setBuffer_offset_atIndex(Some(&*args_buf), 0, 0);
        if let Some(b) = out_ref { enc.setBuffer_offset_atIndex(Some(b), out_off as usize, 1); }
        enc.setBuffer_offset_atIndex(Some(base_buf), base_inner as usize, 2);
        if let Some(b) = out_ref { enc.setBuffer_offset_atIndex(Some(b), out_off as usize, 3); }
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize { width: add_args.ne01 as usize, height: 1, depth: 1 },
            MTLSize { width: add_nth, height: 1, depth: 1 },
        );
    }

    // Dispatch 3: sigmoid
    unsafe {
        let args_buf = bridge::device().ok_or("no device")?.newBufferWithBytes_length_options(
            NonNull::new(&sig_args as *const _ as *mut c_void).unwrap(),
            std::mem::size_of::<UnaryArgs>(),
            MTLResourceOptions::StorageModeShared,
        ).ok_or("args buf")?;
        enc.setComputePipelineState(&*sig_p);
        enc.setBuffer_offset_atIndex(Some(&*args_buf), 0, 0);
        if let Some(b) = out_ref { enc.setBuffer_offset_atIndex(Some(b), out_off as usize, 1); }
        if let Some(b) = out_ref { enc.setBuffer_offset_atIndex(Some(b), out_off as usize, 2); }
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize { width: unary_nk0 * sig_args.ne01 as usize, height: 1, depth: 1 },
            MTLSize { width: unary_nth, height: 1, depth: 1 },
        );
    }

    // Dispatch 4: scale
    unsafe {
        let args_buf = bridge::device().ok_or("no device")?.newBufferWithBytes_length_options(
            NonNull::new(&sca_args as *const _ as *mut c_void).unwrap(),
            std::mem::size_of::<UnaryArgs>(),
            MTLResourceOptions::StorageModeShared,
        ).ok_or("args buf")?;
        enc.setComputePipelineState(&*sca_p);
        enc.setBuffer_offset_atIndex(Some(&*args_buf), 0, 0);
        if let Some(b) = out_ref { enc.setBuffer_offset_atIndex(Some(b), out_off as usize, 1); }
        if let Some(b) = out_ref { enc.setBuffer_offset_atIndex(Some(b), out_off as usize, 2); }
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize { width: unary_nk0 * sca_args.ne01 as usize, height: 1, depth: 1 },
            MTLSize { width: unary_nth, height: 1, depth: 1 },
        );
    }

    enc.endEncoding();
    cb.commit();
    wait_gpu(&*cb)
}

// ─── Softmax ────────────────────────────────────────────────────────────

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

// ─── SwiGLU ─────────────────────────────────────────────────────────────

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

// ─── F16 matmul ─────────────────────────────────────────────────────────

pub fn matmul_f16(
    out: &GpuTensor, views: &ModelViews,
    weight_offset: u64, in_dim: u64, out_dim: u64,
    x: &GpuTensor, n_tok: u64,
) -> Result<(), &'static str> {
    let nsg = std::cmp::min(8i16, ((in_dim + 127) / 128) as i16).max(1);
    let nr0: i32 = 2;
    let use_4 = in_dim % 4 == 0;
    let pipeline_name = if use_4 { "kernel_mul_mv_f16_f32_4" } else { "kernel_mul_mv_f16_f32" };
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
    let p = make_matmul_pipeline(pipeline_name, nsg, 1)
        .ok_or("f16 matmul pipeline")?;
    let smem = (32 * nr0 as u64 * 4) as u64;
    dispatch_with_args_tg(&*p, &args,
        &[(Some(wbuf), woff), (x.buf_ref(), x.offset_raw()),
          (out.buf_ref(), out.offset_raw())],
        ((out_dim / nr0 as u64 + 1) as usize * n_tok as usize, 1, 1),
        (32, nsg as usize, 1), smem)
}

// ─── Q8_0 matmul ────────────────────────────────────────────────────────

pub fn matmul_q8_0(
    out: &GpuTensor, views: &ModelViews,
    weight_offset: u64, in_dim: u64, out_dim: u64,
    x: &GpuTensor, n_tok: u64,
) -> Result<(), &'static str> {
    let nsg: i16 = 4;
    let nr0: i32 = 2;
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
    let p = make_matmul_pipeline("kernel_mul_mv_q8_0_f32", nsg, 1)
        .ok_or("q8_0 matmul pipeline")?;
    let smem = (32 * nr0 as u64 * 4) as u64;
    dispatch_with_args_tg(&*p, &args,
        &[(Some(wbuf), woff), (x.buf_ref(), x.offset_raw()),
          (out.buf_ref(), out.offset_raw())],
        ((out_dim / nr0 as u64 + 1) as usize * n_tok as usize, 1, 1),
        (32, nsg as usize, 1), smem)
}

// ─── KV FP8 Store ───────────────────────────────────────────────────────

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

// ─── Compressor Store One ───────────────────────────────────────────────

pub fn compressor_store_one(
    kv: &GpuTensor, score: &GpuTensor,
    views: &ModelViews, ape_offset: u64,
    state_kv: &GpuTensor, state_score: &GpuTensor,
    width: u32, ratio: u32, pos: u32,
) -> Result<(), &'static str> {
    let ape_bytes = ratio as u64 * width as u64 * 2;
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

// ─── Embedding ──────────────────────────────────────────────────────────

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

// ─── Router ─────────────────────────────────────────────────────────────

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

// ─── MoE SwiGLU with route weight ───────────────────────────────────────

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

// ─── Indexed mixed attention ────────────────────────────────────────────

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
    // kernel uses threadgroup float4 *kv_shared[128] = 2048 bytes
    dispatch_pipeline_tg("kernel_dsv4_indexed_mixed_attention_heads8", &args,
        &[(q.buf_ref(), q.offset_raw()),
          (raw_kv.buf_ref(), raw_kv.offset_raw()),
          (comp_kv.buf_ref(), comp_kv.offset_raw()),
          (topk.buf_ref(), topk.offset_raw()),
          (sink_ref, 0),
          (heads.buf_ref(), heads.offset_raw())],
        (n_tokens as usize, (n_head / 8 + 1) as usize, 1), (128, 1, 1),
        2048) // 128 * sizeof(float4)
}

// ─── Directional steering project ───────────────────────────────────────

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

// ─── Expert matmul: IQ2_XXS pair ────────────────────────────────────────

pub fn matmul_id_iq2_xxs_pair(
    dst_gate: &GpuTensor, dst_up: &GpuTensor,
    views: &ModelViews, gate_offset: u64, _up_offset: u64,
    in_dim: u32, out_dim: u32, n_expert: u32,
    x: &GpuTensor, selected: &[i32],
) -> Result<(), &'static str> {
    let nr0: u32 = 4;
    let step = in_dim as u64 / 256 * 66 * out_dim as u64;
    let total_bytes = step * n_expert as u64 * 2;
    let (wbuf, woff) = views.find_view(gate_offset, total_bytes)
        .ok_or("iq2 expert view not found")?;
    let args = MulMvIdArgs {
        nei0: 1, nei1: selected.len() as i32, nbi1: 4,
        ne00: in_dim as i32, ne01: out_dim as i32,
        ne02: (n_expert * in_dim / 256 * 66) as i32,
        nb00: 66, nb01: in_dim as u64 / 256 * 66,
        nb02: in_dim as u64 / 256 * 66 * out_dim as u64,
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
    let nsg: i16 = 4;
    let p = make_matmul_pipeline("kernel_mul_mv_id_iq2_xxs_pair_f32", nsg, 1)
        .ok_or("iq2 pair pipeline")?;
    let up_off = woff + step * n_expert as u64;
    let smem = (32 * nr0 * 4) as u64;
    dispatch_with_args_tg(&*p, &args,
        &[(Some(wbuf), woff),
          (Some(wbuf), up_off),
          (x.buf_ref(), x.offset_raw()),
          (dst_gate.buf_ref(), dst_gate.offset_raw()),
          (dst_up.buf_ref(), dst_up.offset_raw()),
          (Some(&*sel_buf), 0)],
        ((out_dim / nr0 + 1) as usize, 1, selected.len()),
        (32, nsg as usize, 1), smem)
}

// ─── Expert matmul: Q2_K sum6 ───────────────────────────────────────────

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
    let nsg: i16 = 4;
    let p = make_matmul_pipeline("kernel_mul_mv_id_q2_K_sum6_f32", nsg, 1)
        .ok_or("q2 sum6 pipeline")?;
    let up_off = woff + step * n_expert as u64;
    let smem = (32 * nr0 * 4) as u64;
    dispatch_with_args_tg(&*p, &args,
        &[(Some(wbuf), woff),
          (Some(wbuf), up_off),
          (x.buf_ref(), x.offset_raw()),
          (dst_gate.buf_ref(), dst_gate.offset_raw()),
          (dst_up.buf_ref(), dst_up.offset_raw()),
          (Some(&*sel_buf), 0)],
        ((out_dim / nr0 + 1) as usize, 1, selected.len()),
        (32, nsg as usize, 1), smem)
}

// ─── Expert matmul: Q4_K sum6 ───────────────────────────────────────────

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
    let nsg: i16 = 4;
    let p = make_matmul_pipeline("kernel_mul_mv_id_q4_K_sum6_f32", nsg, 1)
        .ok_or("q4 sum6 pipeline")?;
    let up_off = woff + step * n_expert as u64;
    let smem = (32 * nr0 * 4) as u64;
    dispatch_with_args_tg(&*p, &args,
        &[(Some(wbuf), woff),
          (Some(wbuf), up_off),
          (x.buf_ref(), x.offset_raw()),
          (dst_gate.buf_ref(), dst_gate.offset_raw()),
          (dst_up.buf_ref(), dst_up.offset_raw()),
          (Some(&*sel_buf), 0)],
        ((out_dim / nr0 + 1) as usize, 1, selected.len()),
        (32, nsg as usize, 1), smem)
}

// ─── Expert matmul: Q8_0 id (single output) ────────────────────────────

pub fn matmul_id_q8_0_f32(
    dst: &GpuTensor,
    views: &ModelViews, weight_offset: u64,
    in_dim: u32, out_dim: u32, n_expert: u32,
    x: &GpuTensor, selected: &[i32],
) -> Result<(), &'static str> {
    let nr0: u32 = 2;
    let nb01 = in_dim as u64 / 32 * 34;
    let step = nb01 * out_dim as u64;
    let total_bytes = step * n_expert as u64;
    let (wbuf, woff) = views.find_view(weight_offset, total_bytes)
        .ok_or("q8_0 id expert view not found")?;
    let args = MulMvIdArgs {
        nei0: 1, nei1: selected.len() as i32, nbi1: 4,
        ne00: in_dim as i32, ne01: out_dim as i32,
        ne02: (n_expert * in_dim / 32 * 34) as i32,
        nb00: 34, nb01, nb02: nb01 * out_dim as u64,
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
    let nsg: i16 = 4;
    let p = make_matmul_pipeline("kernel_mul_mv_id_q8_0_f32", nsg, 1)
        .ok_or("q8_0 id pipeline")?;
    let smem = (32 * nr0 * 4) as u64;
    dispatch_with_args_tg(&*p, &args,
        &[(Some(wbuf), woff),
          (x.buf_ref(), x.offset_raw()),
          (dst.buf_ref(), dst.offset_raw()),
          (Some(&*sel_buf), 0)],
        ((out_dim / nr0 + 1) as usize, 1, selected.len()),
        (32, nsg as usize, 1), smem)
}

// ─── Expert matmul: Q2_K id (single output for down projection) ─────────

pub fn matmul_id_q2_K_f32(
    dst: &GpuTensor,
    views: &ModelViews, weight_offset: u64,
    in_dim: u32, out_dim: u32, n_expert: u32,
    x: &GpuTensor, selected: &[i32],
) -> Result<(), &'static str> {
    let nr0: u32 = 4;
    let nb01 = in_dim as u64 / 256 * 84;
    let step = nb01 * out_dim as u64;
    let total_bytes = step * n_expert as u64;
    let (wbuf, woff) = views.find_view(weight_offset, total_bytes)
        .ok_or("q2_K id expert view not found")?;
    let args = MulMvIdArgs {
        nei0: 1, nei1: selected.len() as i32, nbi1: 4,
        ne00: in_dim as i32, ne01: out_dim as i32,
        ne02: (n_expert * in_dim / 256 * 84) as i32,
        nb00: 84, nb01, nb02: nb01 * out_dim as u64,
        ne10: in_dim as i32, ne11: 1, ne12: 1, ne13: 1,
        nb10: 4, nb11: in_dim as u64 * 4, nb12: in_dim as u64 * 4,
        ne0: out_dim as i32, ne1: 1, nb1: out_dim as u64 * 4,
        nr0: nr0 as i32,
    };
    let device = bridge::device().ok_or("no device")?;
    let sel_buf = unsafe {
        device.newBufferWithBytes_length_options(
            NonNull::new(selected.as_ptr() as *mut c_void).unwrap(),
            selected.len() * 4, MTLResourceOptions::StorageModeShared,
        )
    }.ok_or("sel buffer")?;
    let nsg: i16 = 4;
    let p = make_matmul_pipeline("kernel_mul_mv_id_q2_K_f32", nsg, 1)
        .ok_or("q2_K id pipeline")?;
    let smem = (32 * nr0 * 4) as u64;
    dispatch_with_args_tg(&*p, &args,
        &[(Some(wbuf), woff), (x.buf_ref(), x.offset_raw()),
          (dst.buf_ref(), dst.offset_raw()), (Some(&*sel_buf), 0)],
        ((out_dim / nr0 + 1) as usize, 1, selected.len()),
        (32, nsg as usize, 1), smem)
}

// ─── Expert matmul: Q4_K id (single output for down projection) ─────────

pub fn matmul_id_q4_K_f32(
    dst: &GpuTensor,
    views: &ModelViews, weight_offset: u64,
    in_dim: u32, out_dim: u32, n_expert: u32,
    x: &GpuTensor, selected: &[i32],
) -> Result<(), &'static str> {
    let nr0: u32 = 2;
    let nb01 = in_dim as u64 / 256 * 144;
    let step = nb01 * out_dim as u64;
    let total_bytes = step * n_expert as u64;
    let (wbuf, woff) = views.find_view(weight_offset, total_bytes)
        .ok_or("q4_K id expert view not found")?;
    let args = MulMvIdArgs {
        nei0: 1, nei1: selected.len() as i32, nbi1: 4,
        ne00: in_dim as i32, ne01: out_dim as i32,
        ne02: (n_expert * in_dim / 256 * 144) as i32,
        nb00: 144, nb01, nb02: nb01 * out_dim as u64,
        ne10: in_dim as i32, ne11: 1, ne12: 1, ne13: 1,
        nb10: 4, nb11: in_dim as u64 * 4, nb12: in_dim as u64 * 4,
        ne0: out_dim as i32, ne1: 1, nb1: out_dim as u64 * 4,
        nr0: nr0 as i32,
    };
    let device = bridge::device().ok_or("no device")?;
    let sel_buf = unsafe {
        device.newBufferWithBytes_length_options(
            NonNull::new(selected.as_ptr() as *mut c_void).unwrap(),
            selected.len() * 4, MTLResourceOptions::StorageModeShared,
        )
    }.ok_or("sel buffer")?;
    let nsg: i16 = 4;
    let p = make_matmul_pipeline("kernel_mul_mv_id_q4_K_f32", nsg, 1)
        .ok_or("q4_K id pipeline")?;
    let smem = (32 * nr0 * 4) as u64;
    dispatch_with_args_tg(&*p, &args,
        &[(Some(wbuf), woff), (x.buf_ref(), x.offset_raw()),
          (dst.buf_ref(), dst.offset_raw()), (Some(&*sel_buf), 0)],
        ((out_dim / nr0 + 1) as usize, 1, selected.len()),
        (32, nsg as usize, 1), smem)
}

// ─── Expert matmul: IQ2_XXS id (single output for down projection) ──────

pub fn matmul_id_iq2_xxs_f32(
    dst: &GpuTensor,
    views: &ModelViews, weight_offset: u64,
    in_dim: u32, out_dim: u32, n_expert: u32,
    x: &GpuTensor, selected: &[i32],
) -> Result<(), &'static str> {
    let nr0: u32 = 4;
    let nb01 = in_dim as u64 / 256 * 66;
    let step = nb01 * out_dim as u64;
    let total_bytes = step * n_expert as u64;
    let (wbuf, woff) = views.find_view(weight_offset, total_bytes)
        .ok_or("iq2_xxs id expert view not found")?;
    let args = MulMvIdArgs {
        nei0: 1, nei1: selected.len() as i32, nbi1: 4,
        ne00: in_dim as i32, ne01: out_dim as i32,
        ne02: (n_expert * in_dim / 256 * 66) as i32,
        nb00: 66, nb01, nb02: nb01 * out_dim as u64,
        ne10: in_dim as i32, ne11: 1, ne12: 1, ne13: 1,
        nb10: 4, nb11: in_dim as u64 * 4, nb12: in_dim as u64 * 4,
        ne0: out_dim as i32, ne1: 1, nb1: out_dim as u64 * 4,
        nr0: nr0 as i32,
    };
    let device = bridge::device().ok_or("no device")?;
    let sel_buf = unsafe {
        device.newBufferWithBytes_length_options(
            NonNull::new(selected.as_ptr() as *mut c_void).unwrap(),
            selected.len() * 4, MTLResourceOptions::StorageModeShared,
        )
    }.ok_or("sel buffer")?;
    let nsg: i16 = 4;
    let p = make_matmul_pipeline("kernel_mul_mv_id_iq2_xxs_f32", nsg, 1)
        .ok_or("iq2_xxs id pipeline")?;
    let smem = (32 * nr0 * 4) as u64;
    dispatch_with_args_tg(&*p, &args,
        &[(Some(wbuf), woff), (x.buf_ref(), x.offset_raw()),
          (dst.buf_ref(), dst.offset_raw()), (Some(&*sel_buf), 0)],
        ((out_dim / nr0 + 1) as usize, 1, selected.len()),
        (32, nsg as usize, 1), smem)
}

// ─── Indexer score one ──────────────────────────────────────────────────

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
    let (w_opt, w_off) = if weight_offset != 0 {
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

// ─── Softmax pool ───────────────────────────────────────────────────────

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

// ─── Softplus + sqrt ────────────────────────────────────────────────────

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

// ─── Flash attention (prefill) ──────────────────────────────────────────

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

// ─── Repeat f32 ─────────────────────────────────────────────────────────

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

// ─── Compressor ratio4 shift ────────────────────────────────────────────

pub fn compressor_ratio4_shift(
    state_kv: &GpuTensor, state_score: &GpuTensor, width: u32,
) -> Result<(), &'static str> {
    let args = metal_args::Ratio4ShiftArgs { width };
    dispatch_pipeline("kernel_dsv4_ratio4_shift_f32", &args,
        &[(state_kv.buf_ref(), state_kv.offset_raw()),
          (state_score.buf_ref(), state_score.offset_raw())],
        (4 * width as usize, 1, 1), (256, 1, 1))
}

#[allow(unused_imports)]
pub use kv_fp8_store as kv_fp8_store_raw;
