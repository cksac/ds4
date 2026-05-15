use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};
use std::ptr::NonNull;
use std::sync::Arc;
use memmap2::Mmap;
use crate::gguf::GgufModel;
use crate::bridge;

// Keep the mmap alive as long as ModelViews exists.
// Each ViewEntry wraps one no-copy MTLBuffer view of the mmap.

// MTLBuffer is not Send; wrap it.
struct SendBuf(Retained<ProtocolObject<dyn MTLBuffer>>);
unsafe impl Send for SendBuf {}
unsafe impl Sync for SendBuf {}

struct ViewEntry {
    buf: SendBuf,
    /// Absolute file offset at which this MTLBuffer begins (page-aligned).
    model_offset: u64,
    /// Byte length of this MTLBuffer.
    bytes: u64,
}

pub struct ModelViews {
    entries: Vec<ViewEntry>,
    _mmap: Arc<Mmap>,
}

const DS4_METAL_MODEL_MAX_TENSOR_BYTES: u64 = 704_643_072;

fn round_up(v: u64, align: u64) -> u64 {
    (v + align - 1) & !(align - 1)
}

impl ModelViews {
    pub fn new(model: &GgufModel) -> Result<Self, &'static str> {
        let device = bridge::device().ok_or("no Metal device")?;
        let mmap = model.map.clone();

        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as u64;
        let mmap_ptr = mmap.as_ptr() as usize;

        // Verify the mmap base is page-aligned (always true for mmap).
        if mmap_ptr & (page as usize - 1) != 0 {
            return Err("mmap base not page-aligned");
        }

        let tensor_data_pos = model.tensor_data_pos;
        let model_size = model.size;

        // Round down tensor_data_pos to a page boundary.
        let page_model_offset: u64 = tensor_data_pos & !(page - 1);
        let leading = tensor_data_pos - page_model_offset;
        let map_size = model_size - tensor_data_pos;
        let mapped_model_size = round_up(leading + map_size, page);

        let max_buffer_raw = unsafe { device.maxBufferLength() } as u64;
        let max_buffer = max_buffer_raw & !(page - 1);

        let overlap = round_up(DS4_METAL_MODEL_MAX_TENSOR_BYTES, page) + page;
        if max_buffer == 0 || max_buffer <= overlap {
            return Err("Metal maxBufferLength too small for model views");
        }

        let step = max_buffer - overlap;
        let mut entries = Vec::new();
        let mut off: u64 = 0;

        loop {
            let view_bytes = (mapped_model_size - off).min(max_buffer);
            let raw_ptr = (mmap_ptr + page_model_offset as usize + off as usize) as *mut std::ffi::c_void;
            let nn_ptr = NonNull::new(raw_ptr).ok_or("null model view pointer")?;

            let buf = unsafe {
                device.newBufferWithBytesNoCopy_length_options_deallocator(
                    nn_ptr,
                    view_bytes as usize,
                    MTLResourceOptions::StorageModeShared,
                    None,
                )
            }.ok_or("failed to create no-copy Metal buffer")?;

            entries.push(ViewEntry {
                buf: SendBuf(buf),
                model_offset: page_model_offset + off,
                bytes: view_bytes,
            });

            if off + view_bytes >= mapped_model_size { break; }
            off += step;
        }

        eprintln!(
            "ds4: Metal mapped mmaped model as {} overlapping shared buffer{}",
            entries.len(),
            if entries.len() == 1 { "" } else { "s" }
        );

        Ok(ModelViews { entries, _mmap: mmap })
    }

    /// Find a MTLBuffer that fully contains the range [abs_offset, abs_offset+size).
    /// Returns (&MTLBuffer, inner_offset) where inner_offset is the byte offset
    /// within the returned buffer at which abs_offset resides.
    pub fn find_view(
        &self, abs_offset: u64, size: u64,
    ) -> Option<(&ProtocolObject<dyn MTLBuffer>, u64)> {
        for entry in &self.entries {
            let view_start = entry.model_offset;
            let view_end = view_start + entry.bytes;
            if abs_offset >= view_start && abs_offset + size <= view_end {
                let inner = abs_offset - view_start;
                return Some((&*entry.buf.0, inner));
            }
        }
        None
    }

    /// Like find_view but returns an owned Retained buffer (cloned).
    /// Needed when callers must own the buffer (e.g. GpuTensor::wrap).
    pub fn find_view_retained(
        &self, abs_offset: u64, size: u64,
    ) -> Option<(Retained<ProtocolObject<dyn MTLBuffer>>, u64)> {
        for entry in &self.entries {
            let view_start = entry.model_offset;
            let view_end = view_start + entry.bytes;
            if abs_offset >= view_start && abs_offset + size <= view_end {
                let inner = abs_offset - view_start;
                return Some((entry.buf.0.clone(), inner));
            }
        }
        None
    }
}
