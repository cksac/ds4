//! Metal FFI layer using `msg_send!` with `AnyObject`.
//!
//! objc2-metal 0.3.2 uses protocol traits that require `ProtocolObject<dyn Trait>`
//! wrappers, making direct use verbose.  This module wraps all Metal calls
//! with `AnyObject` and `msg_send!`, matching how the C code uses bare `id`.

use objc2::msg_send;
use objc2::rc::Retained;
use objc2::runtime::AnyObject;
use objc2_foundation::{NSError, NSString};
use objc2_metal::MTLSize;

// ── Convenience ────────────────────────────────────────────────────────

pub fn mtl_size(width: usize, height: usize, depth: usize) -> MTLSize {
    MTLSize { width, height, depth }
}

// ── MTLDevice ──────────────────────────────────────────────────────────

pub unsafe fn mtl_create_system_default_device() -> Option<Retained<AnyObject>> {
    extern "C" {
        fn MTLCreateSystemDefaultDevice() -> *mut AnyObject;
    }
    let ptr = MTLCreateSystemDefaultDevice();
    if ptr.is_null() { None } else { Some(Retained::from_raw(ptr).unwrap()) }
}

pub unsafe fn device_name(device: &AnyObject) -> Retained<NSString> {
    msg_send![device, name]
}

pub unsafe fn device_new_command_queue(device: &AnyObject) -> Option<Retained<AnyObject>> {
    msg_send![device, newCommandQueue]
}

pub unsafe fn device_new_library_with_source(
    device: &AnyObject, source: &NSString, options: &AnyObject,
) -> Result<Retained<AnyObject>, Retained<NSError>> {
    msg_send![device, newLibraryWithSource:source, options:options, error:_]
}

pub unsafe fn device_max_buffer_length(device: &AnyObject) -> usize {
    msg_send![device, maxBufferLength]
}

pub unsafe fn device_new_buffer_with_length(device: &AnyObject, length: usize) -> Option<Retained<AnyObject>> {
    msg_send![device, newBufferWithLength:length, options:0usize]
}

pub unsafe fn device_new_buffer_with_bytes_no_copy(
    device: &AnyObject, bytes: *mut std::ffi::c_void, length: usize,
) -> Option<Retained<AnyObject>> {
    let nil: Option<&objc2::runtime::AnyObject> = None;
    msg_send![device, newBufferWithBytesNoCopy:bytes, length:length, options:0usize, deallocator:nil]
}

// ── MTLCommandQueue ────────────────────────────────────────────────────

pub unsafe fn queue_command_buffer(queue: &AnyObject) -> Option<Retained<AnyObject>> {
    msg_send![queue, commandBuffer]
}

// ── MTLCommandBuffer ───────────────────────────────────────────────────

pub unsafe fn cb_commit(cb: &AnyObject) { let _: () = msg_send![cb, commit]; }
pub unsafe fn cb_wait_until_completed(cb: &AnyObject) { let _: () = msg_send![cb, waitUntilCompleted]; }
pub unsafe fn cb_compute_command_encoder(cb: &AnyObject) -> Retained<AnyObject> { msg_send![cb, computeCommandEncoder] }
pub unsafe fn cb_blit_command_encoder(cb: &AnyObject) -> Retained<AnyObject> { msg_send![cb, blitCommandEncoder] }
pub unsafe fn cb_status(cb: &AnyObject) -> usize { msg_send![cb, status] }
pub unsafe fn cb_error(cb: &AnyObject) -> Option<Retained<NSError>> { msg_send![cb, error] }

// ── MTLComputeCommandEncoder ───────────────────────────────────────────

pub unsafe fn enc_set_compute_pipeline_state(enc: &AnyObject, pipeline: &AnyObject) {
    let _: () = msg_send![enc, setComputePipelineState:pipeline];
}
pub unsafe fn enc_set_bytes(enc: &AnyObject, bytes: *const std::ffi::c_void, length: usize, index: usize) {
    let _: () = msg_send![enc, setBytes:bytes, length:length, atIndex:index];
}
pub unsafe fn enc_set_buffer(enc: &AnyObject, buffer: &AnyObject, offset: usize, index: usize) {
    let _: () = msg_send![enc, setBuffer:buffer, offset:offset, atIndex:index];
}
pub unsafe fn enc_set_threadgroup_memory_length(enc: &AnyObject, length: usize, index: usize) {
    let _: () = msg_send![enc, setThreadgroupMemoryLength:length, atIndex:index];
}
pub unsafe fn enc_dispatch_threadgroups(enc: &AnyObject, grid: MTLSize, threads: MTLSize) {
    let _: () = msg_send![enc, dispatchThreadgroups:grid, threadsPerThreadgroup:threads];
}
pub unsafe fn enc_end_encoding(enc: &AnyObject) { let _: () = msg_send![enc, endEncoding]; }

// ── MTLBlitCommandEncoder ──────────────────────────────────────────────

pub unsafe fn blit_copy(blit: &AnyObject, src: &AnyObject, src_off: usize, dst: &AnyObject, dst_off: usize, size: usize) {
    let _: () = msg_send![blit, copyFromBuffer:src, sourceOffset:src_off, toBuffer:dst, destinationOffset:dst_off, size:size];
}
pub unsafe fn blit_end_encoding(blit: &AnyObject) { let _: () = msg_send![blit, endEncoding]; }

// ── MTLLibrary ─────────────────────────────────────────────────────────

pub unsafe fn library_new_function(library: &AnyObject, name: &NSString) -> Option<Retained<AnyObject>> {
    msg_send![library, newFunctionWithName:name]
}
pub unsafe fn library_new_function_with_constants(
    library: &AnyObject, name: &NSString, constants: &AnyObject,
) -> Result<Retained<AnyObject>, Retained<NSError>> {
    msg_send![library, newFunctionWithName:name, constantValues:constants, error:_]
}

// ── MTLDevice (pipeline) ───────────────────────────────────────────────

pub unsafe fn device_new_compute_pipeline(
    device: &AnyObject, function: &AnyObject,
) -> Result<Retained<AnyObject>, Retained<NSError>> {
    msg_send![device, newComputePipelineStateWithFunction:function, error:_]
}

// ── MTLFunctionConstantValues ──────────────────────────────────────────

pub unsafe fn fn_constants_new() -> Retained<AnyObject> {
    msg_send![objc2::class!(MTLFunctionConstantValues), new]
}
pub unsafe fn fn_constants_set_value(constants: &AnyObject, value: *const std::ffi::c_void, ty: usize, index: usize) {
    // MTLFunctionConstantValues expects: setConstantValue:(const void *)value type:(MTLDataType)type atIndex:(NSUInteger)index
    // We need to be very precise about the ObjC types here.
    use objc2_foundation::NSUInteger;
    let _: () = msg_send![
        constants,
        setConstantValue: value,
        type: ty as NSUInteger,
        atIndex: index as NSUInteger
    ];
}

// ── MTLCompileOptions ──────────────────────────────────────────────────

pub unsafe fn compile_options_new() -> Retained<AnyObject> {
    msg_send![objc2::class!(MTLCompileOptions), new]
}

// ── MTLBuffer ──────────────────────────────────────────────────────────

pub unsafe fn buffer_contents(buffer: &AnyObject) -> *mut std::ffi::c_void {
    msg_send![buffer, contents]
}
pub unsafe fn buffer_set_label(buffer: &AnyObject, label: &NSString) {
    let _: () = msg_send![buffer, setLabel:label];
}
