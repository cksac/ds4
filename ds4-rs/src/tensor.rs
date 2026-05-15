use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};
use std::sync::atomic::{AtomicU64, Ordering};
use crate::bridge;

static LIVE_BYTES: AtomicU64 = AtomicU64::new(0);
static PEAK_BYTES: AtomicU64 = AtomicU64::new(0);

// MTLBuffer is not Send in objc2-metal. Wrap for use across threads.
pub struct SendBuffer(pub Option<Retained<ProtocolObject<dyn MTLBuffer>>>);
unsafe impl Send for SendBuffer {}
unsafe impl Sync for SendBuffer {}

pub struct GpuTensor {
    buffer: SendBuffer,
    offset: u64,
    bytes: u64,
    owned: bool,
}

impl GpuTensor {
    pub fn wrap(buffer: Retained<ProtocolObject<dyn MTLBuffer>>, offset: u64, bytes: u64) -> Self {
        GpuTensor { buffer: SendBuffer(Some(buffer)), offset, bytes, owned: false }
    }

    pub fn alloc(bytes: u64) -> Result<Self, &'static str> {
        let dev = bridge::device().ok_or("no device")?;
        let buf = unsafe {
            dev.newBufferWithLength_options(bytes as usize, MTLResourceOptions::StorageModeShared)
        }.ok_or("buffer alloc failed")?;
        LIVE_BYTES.fetch_add(bytes, Ordering::Relaxed);
        PEAK_BYTES.fetch_add(bytes, Ordering::Relaxed);
        Ok(GpuTensor { buffer: SendBuffer(Some(buf)), offset: 0, bytes, owned: true })
    }

    pub fn buf_ref(&self) -> Option<&ProtocolObject<dyn MTLBuffer>> {
        self.buffer.0.as_deref()
    }

    pub fn retain_buf(&self) -> Option<Retained<ProtocolObject<dyn MTLBuffer>>> {
        self.buffer.0.clone()
    }

    pub fn bytes(&self) -> u64 { self.bytes }
    pub fn offset_raw(&self) -> u64 { self.offset }

    pub fn fill_f32(&mut self, value: f32, count: u64) -> Result<(), &'static str> {
        if let Some(ref buf) = self.buffer.0 {
            let ptr = buf.contents().as_ptr() as *mut f32;
            for i in 0..(count as usize) {
                unsafe { *ptr.add(i) = value; }
            }
        }
        Ok(())
    }

    pub fn read_i32_slice(&self, offset: u64, dst: &mut [i32]) -> Result<(), &'static str> {
        if let Some(ref buf) = self.buffer.0 {
            let base = buf.contents().as_ptr() as usize
                + self.offset as usize + offset as usize;
            let ptr = base as *const i32;
            let contents = unsafe { std::slice::from_raw_parts(ptr, dst.len()) };
            dst.copy_from_slice(contents);
            Ok(())
        } else {
            Err("no buffer")
        }
    }

    pub fn read_bytes(&self) -> Result<Vec<u8>, &'static str> {
        if let Some(ref buf) = self.buffer.0 {
            let ptr = (buf.contents().as_ptr() as usize + self.offset as usize) as *const u8;
            let slice = unsafe { std::slice::from_raw_parts(ptr, self.bytes as usize) };
            Ok(slice.to_vec())
        } else {
            Err("no buffer")
        }
    }

    pub fn write_bytes(&mut self, data: &[u8]) -> Result<(), &'static str> {
        if data.len() as u64 != self.bytes {
            return Err("size mismatch");
        }
        if let Some(ref buf) = self.buffer.0 {
            let ptr = (buf.contents().as_ptr() as usize + self.offset as usize) as *mut u8;
            unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len()); }
            Ok(())
        } else {
            Err("no buffer")
        }
    }
}

impl Drop for GpuTensor {
    fn drop(&mut self) {
        if self.owned {
            LIVE_BYTES.fetch_sub(self.bytes, Ordering::Relaxed);
        }
    }
}
