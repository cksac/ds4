use objc2::rc::Retained;
use objc2::runtime::AnyObject;

use super::objc_ext;

#[derive(Debug)]
pub struct Tensor {
    pub buffer: Retained<AnyObject>,
    pub offset: u64,
    pub bytes: u64,
    owner: bool,
}

impl Tensor {
    pub fn alloc(device: &AnyObject, bytes: u64) -> Option<Self> {
        if bytes == 0 { return None; }
        let buffer = unsafe { objc_ext::device_new_buffer_with_length(device, bytes as usize) }?;
        Some(Self { buffer, offset: 0, bytes, owner: true })
    }

    pub fn alloc_managed(device: &AnyObject, bytes: u64) -> Option<Self> { Self::alloc(device, bytes) }

    pub fn view(base: &Tensor, offset: u64, bytes: u64) -> Option<Self> {
        if offset > base.bytes || bytes > base.bytes - offset { return None; }
        Some(Self { buffer: base.buffer.clone(), offset: base.offset + offset, bytes, owner: false })
    }

    pub fn contents(&self) -> *mut u8 {
        unsafe { (objc_ext::buffer_contents(&self.buffer) as *mut u8).add(self.offset as usize) }
    }

    pub fn fill_f32(&self, value: f32, count: u64) -> Result<(), &'static str> {
        if count > self.bytes / std::mem::size_of::<f32>() as u64 { return Err("count exceeds tensor capacity"); }
        let p = self.contents() as *mut f32;
        if p.is_null() && count != 0 { return Err("tensor contents not accessible"); }
        unsafe { for i in 0..count as usize { *p.add(i) = value; } }
        Ok(())
    }

    pub fn write(&self, offset: u64, data: &[u8]) -> Result<(), &'static str> {
        let len = data.len() as u64;
        if offset > self.bytes || len > self.bytes - offset { return Err("tensor write out of bounds"); }
        if !data.is_empty() {
            unsafe {
                let dst = objc_ext::buffer_contents(&self.buffer) as *mut u8;
                std::ptr::copy_nonoverlapping(data.as_ptr(), dst.add((self.offset + offset) as usize), data.len());
            }
        }
        Ok(())
    }

    pub fn read(&self, offset: u64, data: &mut [u8]) -> Result<(), &'static str> {
        let len = data.len() as u64;
        if offset > self.bytes || len > self.bytes - offset { return Err("tensor read out of bounds"); }
        if !data.is_empty() {
            unsafe {
                let src = objc_ext::buffer_contents(&self.buffer) as *const u8;
                std::ptr::copy_nonoverlapping(src.add((self.offset + offset) as usize), data.as_mut_ptr(), data.len());
            }
        }
        Ok(())
    }
}
