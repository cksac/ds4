use metal::*;
use std::sync::atomic::{AtomicU64, Ordering};
use crate::bridge;

static LIVE_BYTES: AtomicU64 = AtomicU64::new(0);
static PEAK_BYTES: AtomicU64 = AtomicU64::new(0);

// Buffer from metal 0.20 is not Send. We wrap it in a Send-safe wrapper.
pub struct SendBuffer(Option<Buffer>);
unsafe impl Send for SendBuffer {}
unsafe impl Sync for SendBuffer {}

pub struct GpuTensor {
    buffer: SendBuffer,
    offset: u64,
    bytes: u64,
    owned: bool,
}

impl GpuTensor {
    pub fn wrap(buffer: Buffer, offset: u64, bytes: u64) -> Self {
        GpuTensor { buffer: SendBuffer(Some(buffer)), offset, bytes, owned: false }
    }

    pub fn alloc(bytes: u64) -> Result<Self, &'static str> {
        let dev = bridge::with_device(|d| d.clone()).ok_or("no device")?;
        let buf = dev.new_buffer(bytes, MTLResourceOptions::StorageModeShared);
        LIVE_BYTES.fetch_add(bytes, Ordering::Relaxed);
        PEAK_BYTES.fetch_add(bytes, Ordering::Relaxed);
        Ok(GpuTensor { buffer: SendBuffer(Some(buf)), offset: 0, bytes, owned: true })
    }

    pub fn buffer(&self) -> Option<&Buffer> { self.buffer.0.as_ref() }
    pub fn bytes(&self) -> u64 { self.bytes }
    pub fn offset_raw(&self) -> u64 { self.offset }

    pub fn fill_f32(&mut self, value: f32, count: u64) -> Result<(), &'static str> {
        if let Some(ref buf) = self.buffer.0 {
            let ptr = buf.contents() as *mut f32;
            for i in 0..(count as usize) {
                unsafe { *ptr.add(i) = value; }
            }
        }
        Ok(())
    }

    pub fn read_i32_slice(&self, offset: u64, dst: &mut [i32]) -> Result<(), &'static str> {
        if let Some(ref buf) = self.buffer.0 {
            let contents = unsafe {
                std::slice::from_raw_parts(
                    buf.contents().add(self.offset as usize).add(offset as usize) as *const i32,
                    dst.len(),
                )
            };
            dst.copy_from_slice(contents);
            Ok(())
        } else {
            Err("no buffer")
        }
    }

    pub fn read_bytes(&self) -> Result<Vec<u8>, &'static str> {
        if let Some(ref buf) = self.buffer.0 {
            let ptr = unsafe {
                buf.contents().add(self.offset as usize) as *mut u8
            };
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
            let ptr = unsafe {
                buf.contents().add(self.offset as usize) as *mut u8
            };
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
