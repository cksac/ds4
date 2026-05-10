use crate::ffi;
use anyhow::{bail, Result};
use libc::c_void;

#[derive(Clone, Copy)]
pub(crate) struct ModelMapRange {
    pub model_map: *const c_void,
    pub model_size: u64,
    pub map_offset: u64,
    pub map_size: u64,
}

pub(crate) fn tensor_alloc(bytes: u64) -> *mut ffi::ds4_metal_tensor {
    unsafe { ffi::ds4_metal_tensor_alloc(bytes) }
}

pub(crate) fn initialize(quality: bool, model_ranges: &[ModelMapRange]) -> Result<()> {
    let ok = unsafe { ffi::ds4_metal_init() };
    if ok == 0 {
        bail!("Metal backend unavailable; failed to initialize runtime")
    }

    unsafe { ffi::ds4_metal_set_quality(quality as i32) };

    for range in model_ranges {
        let ok = unsafe {
            ffi::ds4_metal_set_model_map_range(
                range.model_map,
                range.model_size,
                range.map_offset,
                range.map_size,
            )
        };
        if ok == 0 {
            bail!("Metal failed to map model views")
        }
    }

    Ok(())
}

pub(crate) fn begin_commands() -> Result<()> {
    let ok = unsafe { ffi::ds4_metal_begin_commands() };
    if ok == 0 {
        bail!("ds4_metal_begin_commands failed")
    }
    Ok(())
}

pub(crate) fn buffer_alloc(bytes: u64) -> *mut c_void {
    unsafe { ffi::ds4_metal_buffer_alloc(bytes) }
}

pub(crate) fn tensor_bind_owned_buffer(buffer: *mut c_void, bytes: u64) -> *mut ffi::ds4_metal_tensor {
    unsafe { ffi::ds4_metal_tensor_bind_owned_buffer(buffer, bytes) }
}

pub(crate) fn tensor_wrap_buffer(
    buffer: *mut c_void,
    offset: u64,
    bytes: u64,
) -> *mut ffi::ds4_metal_tensor {
    unsafe { ffi::ds4_metal_tensor_wrap_buffer(buffer, offset, bytes) }
}

pub(crate) fn tensor_view(
    base: *const ffi::ds4_metal_tensor,
    offset: u64,
    bytes: u64,
) -> *mut ffi::ds4_metal_tensor {
    unsafe { ffi::ds4_metal_tensor_view(base, offset, bytes) }
}

pub(crate) fn tensor_free(tensor: *mut ffi::ds4_metal_tensor) {
    unsafe { ffi::ds4_metal_tensor_free(tensor) }
}

pub(crate) fn tensor_write(
    tensor: *mut ffi::ds4_metal_tensor,
    offset: u64,
    data: *const c_void,
    bytes: u64,
) -> i32 {
    unsafe { ffi::ds4_metal_tensor_write(tensor, offset, data, bytes) }
}

pub(crate) fn tensor_read(
    tensor: *const ffi::ds4_metal_tensor,
    offset: u64,
    data: *mut c_void,
    bytes: u64,
) -> i32 {
    unsafe { ffi::ds4_metal_tensor_read(tensor, offset, data, bytes) }
}

pub(crate) fn end_commands() -> Result<()> {
    let ok = unsafe { ffi::ds4_metal_end_commands() };
    if ok == 0 {
        bail!("ds4_metal_end_commands failed")
    }
    Ok(())
}

pub(crate) fn synchronize() -> Result<()> {
    let ok = unsafe { ffi::ds4_metal_synchronize() };
    if ok == 0 {
        bail!("ds4_metal_synchronize failed")
    }
    Ok(())
}

pub(crate) fn cleanup() {
    unsafe { ffi::ds4_metal_cleanup() }
}

// Note: This wrapper is called with MetalTensor references on macOS
#[cfg(target_os = "macos")]
pub(crate) fn rms_norm_plain_tensor(
    out: &crate::MetalTensor,
    x: &crate::MetalTensor,
    n: u32,
    eps: f32,
) -> i32 {
    unsafe { 
        ffi::ds4_metal_rms_norm_plain_rows_tensor(
            out.ptr.as_ptr(), 
            x.ptr.as_ptr() as *const _,
            n,
            1,
            eps,
        )
    }
}

#[cfg(not(target_os = "macos"))]
pub(crate) fn rms_norm_plain_tensor(
    out: *mut ffi::ds4_metal_tensor,
    x: *const ffi::ds4_metal_tensor,
    n: u32,
    eps: f32,
) -> i32 {
    unsafe { ffi::ds4_metal_rms_norm_plain_rows_tensor(out, x, n, 1, eps) }
}

// ============================================================================
// NATIVE RUST TENSOR IMPLEMENTATIONS (Porting Cluster: Tensor Lifecycle)
// ============================================================================
// These functions gradually replace the C equivalents in ds4_metal.m
// Eventually, build.rs won't need to compile ds4_metal.m for these

#[cfg(target_os = "macos")]
pub(crate) mod native {
    use super::*;
    use crate::metal_native::NativeMetalTensor;
    
    /// Native Rust tensor allocation (replaces ds4_metal_tensor_alloc)
    /// Allocates GPU memory directly via Metal
    pub fn tensor_alloc(bytes: u64) -> Option<NativeMetalTensor> {
        NativeMetalTensor::new(bytes)
    }
    
    /// Native tensor view (replaces ds4_metal_tensor_view)
    pub fn tensor_view(base: &NativeMetalTensor, offset: u64, bytes: u64) -> Option<NativeMetalTensor> {
        base.view(offset, bytes)
    }
}
