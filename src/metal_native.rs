// Native Rust Metal tensor management - porting ds4_metal.m functionality
// This module gradually replaces C implementations with Rust equivalents
//
// Porting strategy:
// 1. Get Metal device/queue from C via FFI getters
// 2. Implement buffer allocation using objc2-metal
// 3. Build tensor wrappers around MTLBuffers
// 4. Replace C functions one-by-one
// 5. Remove from build.rs when all ported

use std::sync::OnceLock;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLDevice, MTLBuffer, MTLCommandQueue};
use objc2_foundation::NSUInteger;
use crate::ffi;

// Lazily-cached Metal context obtained from C initialization
pub(crate) struct MetalContext {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
}

impl MetalContext {
    fn get_or_init() -> Option<&'static MetalContext> {
        static CONTEXT: OnceLock<Option<MetalContext>> = OnceLock::new();
        
        CONTEXT.get_or_init(Self::initialize).as_ref()
    }

    fn initialize() -> Option<MetalContext> {
        // Get device and queue from C code's initialization
        let device_ptr = unsafe { ffi::ds4_metal_get_device() };
        let queue_ptr = unsafe { ffi::ds4_metal_get_queue() };
        
        if device_ptr.is_null() || queue_ptr.is_null() {
            return None;
        }
        
        // Convert raw Objective-C pointers to Rust Retained objects
        let device = unsafe {
            Retained::from_raw(device_ptr as *mut ProtocolObject<dyn MTLDevice>)?
        };
        
        let queue = unsafe {
            Retained::from_raw(queue_ptr as *mut ProtocolObject<dyn MTLCommandQueue>)?
        };
        
        Some(MetalContext { device, queue })
    }

    pub fn device(&self) -> &ProtocolObject<dyn MTLDevice> {
        &self.device
    }

    pub fn queue(&self) -> &ProtocolObject<dyn MTLCommandQueue> {
        &self.queue
    }
}

// Native Rust buffer allocation using Metal directly  
#[cfg(target_os = "macos")]
pub(crate) fn native_buffer_alloc_impl(bytes: u64) -> Option<Retained<ProtocolObject<dyn MTLBuffer>>> {
    use objc2::msg_send;
    use objc2::sel;
    use objc2_foundation::NSUInteger;
    
    let ctx = MetalContext::get_or_init()?;
    
    // Direct Objective-C message send equivalent to:
    // [device newBufferWithLength:bytes options:0]
    let device = ctx.device();
    unsafe {
        let buffer = msg_send![
            device,
            newBufferWithLength: bytes as NSUInteger,
            options: 0u64
        ];
        Retained::from_raw(buffer)
    }
}

#[cfg(not(target_os = "macos"))]
pub(crate) fn native_buffer_alloc_impl(_bytes: u64) -> Option<Retained<ProtocolObject<dyn MTLBuffer>>> {
    None
}

// Fallback: use C version if native allocation unavailable
pub(crate) fn native_buffer_alloc(bytes: u64) -> Option<Retained<ProtocolObject<dyn MTLBuffer>>> {
    // Try native implementation first
    #[cfg(target_os = "macos")]
    if let Some(buf) = native_buffer_alloc_impl(bytes) {
        return Some(buf);
    }
    
    // Fall back to C version
    let ptr = unsafe { ffi::ds4_metal_buffer_alloc(bytes) };
    if ptr.is_null() {
        return None;
    }
    
    // Convert raw pointer to Retained ProtocolObject
    unsafe {
        Retained::from_raw(ptr as *mut ProtocolObject<dyn MTLBuffer>)
    }
}

// Native Rust tensor wrapper
#[derive(Debug)]
pub(crate) struct NativeMetalTensor {
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    offset: u64,
    bytes: u64,
}

impl NativeMetalTensor {
    /// Allocate a new tensor with owned GPU memory
    pub fn new(bytes: u64) -> Option<Self> {
        native_buffer_alloc(bytes).map(|buffer| NativeMetalTensor {
            buffer,
            offset: 0,
            bytes,
        })
    }

    /// Create a view (sub-tensor) of this tensor
    pub fn view(&self, offset: u64, bytes: u64) -> Option<Self> {
        if offset > self.bytes || bytes > self.bytes - offset {
            return None;
        }
        
        Some(NativeMetalTensor {
            buffer: self.buffer.clone(),
            offset: self.offset + offset,
            bytes,
        })
    }

    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    pub fn offset(&self) -> u64 {
        self.offset
    }

    pub fn buffer(&self) -> &ProtocolObject<dyn MTLBuffer> {
        &self.buffer
    }

    /// Read tensor contents to CPU
    pub fn read(&self, offset: u64, data: &mut [u8]) -> bool {
        let total_available = self.bytes.saturating_sub(offset);
        if data.len() as u64 > total_available {
            return false;
        }
        
        unsafe {
            let src = (self.buffer.contents().as_ptr() as *const u8).add((self.offset + offset) as usize);
            std::ptr::copy_nonoverlapping(src, data.as_mut_ptr(), data.len());
        }
        true
    }

    /// Write tensor contents from CPU
    pub fn write(&self, offset: u64, data: &[u8]) -> bool {
        let total_available = self.bytes.saturating_sub(offset);
        if data.len() as u64 > total_available {
            return false;
        }
        
        unsafe {
            let dst = (self.buffer.contents().as_ptr() as *mut u8).add((self.offset + offset) as usize);
            std::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len());
        }
        true
    }
}

