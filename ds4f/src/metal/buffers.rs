use anyhow::{Context, Result};
use objc2::rc::Retained;
use objc2::runtime::AnyObject;
use objc2_foundation::NSString;

use super::objc_ext;

pub const MAX_MODEL_VIEWS: usize = 16;
pub const MAX_TENSOR_BYTES: u64 = 704_643_072;

pub struct ModelView {
    pub buffer: Retained<AnyObject>,
    pub model_map: *const std::ffi::c_void,
    pub model_size: u64,
    pub model_offset: u64,
    pub bytes: u64,
}

pub struct ModelViews {
    views: Vec<ModelView>,
    wrap_count: u64,
    wrap_bytes: u64,
    wrap_max_bytes: u64,
}

impl ModelViews {
    pub fn new() -> Self {
        Self { views: Vec::with_capacity(MAX_MODEL_VIEWS), wrap_count: 0, wrap_bytes: 0, wrap_max_bytes: 0 }
    }

    pub fn map_model_range(
        &mut self, device: &AnyObject, model_map: *const std::ffi::c_void,
        model_size: u64, map_offset: u64, map_size: u64,
    ) -> Result<()> {
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as u64;
        let model_addr = model_map as u64;

        if model_addr & (page - 1) != 0 { anyhow::bail!("model mmap base is not page aligned"); }
        if map_offset > model_size || map_size > model_size - map_offset {
            anyhow::bail!("model mapped range is outside the GGUF mapping");
        }

        let page_model_offset = map_offset & !(page - 1);
        let leading = map_offset - page_model_offset;
        let mapped_model_size = round_up_u64(leading + map_size, page);

        let mut max_buffer = unsafe { objc_ext::device_max_buffer_length(device) } as u64;
        max_buffer &= !(page - 1);

        let overlap = round_up_u64(MAX_TENSOR_BYTES, page) + page;
        if max_buffer == 0 || max_buffer <= overlap {
            anyhow::bail!("Metal maxBufferLength too small for model views");
        }

        let step = max_buffer - overlap;
        let mut off: u64 = 0;
        while off < mapped_model_size {
            if self.views.len() >= MAX_MODEL_VIEWS { anyhow::bail!("too many model views"); }
            let mut view_bytes = mapped_model_size - off;
            if view_bytes > max_buffer { view_bytes = max_buffer; }

            let ptr = (model_addr + page_model_offset + off) as *mut std::ffi::c_void;
            let buffer = unsafe { objc_ext::device_new_buffer_with_bytes_no_copy(device, ptr, view_bytes as usize) }
                .context("failed to wrap mmap region as MTLBuffer")?;
            unsafe { objc_ext::buffer_set_label(&buffer, &NSString::from_str(&format!("ds4_model_view_{}", self.views.len()))); }

            self.views.push(ModelView { buffer, model_map, model_size, model_offset: page_model_offset + off, bytes: view_bytes });
            self.wrap_count += 1;
            self.wrap_bytes += view_bytes;
            if view_bytes > self.wrap_max_bytes { self.wrap_max_bytes = view_bytes; }
            if off + view_bytes >= mapped_model_size { break; }
            off += step;
        }
        Ok(())
    }

    pub fn wrap_model_range(
        &self, model_map: *const std::ffi::c_void, model_size: u64, offset: u64, len: u64,
    ) -> Result<(Retained<AnyObject>, u64)> {
        if model_size == 0 || offset > model_size || len > model_size - offset {
            anyhow::bail!("model range outside mapped model");
        }
        let end = offset + len;
        for view in &self.views {
            if view.model_map != model_map || view.model_size != model_size { continue; }
            let view_start = view.model_offset;
            let view_end = view_start + view.bytes;
            if offset >= view_start && end <= view_end {
                return Ok((view.buffer.clone(), offset - view_start));
            }
        }
        anyhow::bail!("model range not covered by mapped views");
    }
}

pub struct ScratchBuffer {
    pub buffer: Option<Retained<AnyObject>>,
    pub capacity: usize,
    label: &'static str,
}

impl ScratchBuffer {
    pub fn new(label: &'static str) -> Self { Self { buffer: None, capacity: 0, label } }
    pub fn ensure(&mut self, device: &AnyObject, bytes: usize) -> Result<()> {
        if self.capacity >= bytes && self.buffer.is_some() { return Ok(()); }
        let size = if bytes == 0 { 1 } else { bytes };
        let buffer = unsafe { objc_ext::device_new_buffer_with_length(device, size) }
            .with_context(|| format!("failed to allocate scratch buffer '{}'", self.label))?;
        unsafe { objc_ext::buffer_set_label(&buffer, &NSString::from_str(self.label)); }
        self.buffer = Some(buffer);
        self.capacity = size;
        Ok(())
    }
}

fn round_up_u64(v: u64, align: u64) -> u64 { (v + align - 1) & !(align - 1) }
