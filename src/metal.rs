//! Metal GPU runtime — device, queue, command buffer, pipeline, and buffer management.
//!
//! This is the pure-Rust Metal runtime using objc2-metal. It manages:
//! - Device and queue lifecycle
//! - Metal library (shader) loading and runtime compilation
//! - Compute pipeline caching (with function constants)
//! - Command buffer batching and synchronization
//! - Model weight mmap wrapping as no-copy MTLBuffers
//! - GPU buffer allocation

use anyhow::{anyhow, bail, Result};
use objc2::rc::{autoreleasepool, Retained};
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLDevice, MTLLibrary, MTLResourceOptions,
    MTLSize,
};
use std::cell::RefCell;
use std::collections::HashMap;

/// Maximum number of mmap model views (allows models > maxBufferLength).
const MAX_MODEL_VIEWS: usize = 16;

/// Largest single tensor expected in the model file.
const MODEL_MAX_TENSOR_BYTES: u64 = 704_643_072;

// ============================================================================
// Model mmap view — a page-aligned no-copy MTLBuffer slice.
// ============================================================================

struct ModelView {
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    model_offset: u64,
    bytes: u64,
}

// ============================================================================
// Thread-local Metal context.
// ============================================================================

struct MetalContext {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    library: Retained<ProtocolObject<dyn MTLLibrary>>,
    batch_cb: Option<Retained<ProtocolObject<dyn MTLCommandBuffer>>>,
    batch_enc: Option<Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>>,
    pipeline_cache: HashMap<String, Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    model_views: Vec<ModelView>,
    quality_mode: bool,
}

thread_local! {
    static METAL_CTX: RefCell<Option<MetalContext>> = const { RefCell::new(None) };
}

fn with_ctx<T>(f: impl FnOnce(&mut MetalContext) -> Result<T>) -> Result<T> {
    METAL_CTX.with(|cell| {
        let mut opt = cell.borrow_mut();
        let ctx = opt.as_mut().ok_or_else(|| anyhow!("Metal not initialized"))?;
        f(ctx)
    })
}

// ============================================================================
// Initialization / teardown.
// ============================================================================

/// Initialize the Metal runtime.
pub fn initialize(quality: bool) -> Result<()> {
    METAL_CTX.with(|cell| {
        let mut opt = cell.borrow_mut();
        if opt.is_some() {
            return Ok(());
        }
        autoreleasepool(|_| {
            let device = objc2_metal::MTLCreateSystemDefaultDevice()
                .ok_or_else(|| anyhow!("No Metal device available"))?;
            let queue = device
                .newCommandQueue()
                .ok_or_else(|| anyhow!("Failed to create Metal command queue"))?;
            let library = load_metal_library(&device)?;

            *opt = Some(MetalContext {
                device,
                queue,
                library,
                batch_cb: None,
                batch_enc: None,
                pipeline_cache: HashMap::new(),
                model_views: Vec::new(),
                quality_mode: quality,
            });
            Ok(())
        })
    })
}

/// Tear down the Metal runtime.
pub fn cleanup() {
    METAL_CTX.with(|cell| {
        let _ = cell.borrow_mut().take();
    });
}

/// Whether quality mode is enabled.
pub fn quality_mode() -> bool {
    with_ctx(|ctx| Ok(ctx.quality_mode)).unwrap_or(false)
}

// ============================================================================
// Metal library loading (runtime shader compilation).
// ============================================================================

fn load_metal_library(
    device: &ProtocolObject<dyn MTLDevice>,
) -> Result<Retained<ProtocolObject<dyn MTLLibrary>>> {
    autoreleasepool(|_| {
        // Try a pre-compiled metallib from env.
        if let Ok(path) = std::env::var("METAL_LIBRARY_PATH") {
            let ns_path = NSString::from_str(&path);
            if let Ok(lib) = unsafe {
                device.newLibraryWithURL_error(
                    &objc2_foundation::NSURL::fileURLWithPath(&ns_path),
                )
            } {
                return Ok(lib);
            }
        }

        // Compile shaders at runtime from metal/ directory.
        if let Some(lib) = compile_metal_library_runtime(device)? {
            return Ok(lib);
        }

        // Fallback: Xcode default library.
        device
            .newDefaultLibrary()
            .ok_or_else(|| anyhow!("Failed to load Metal library"))
    })
}

fn compile_metal_library_runtime(
    device: &ProtocolObject<dyn MTLDevice>,
) -> Result<Option<Retained<ProtocolObject<dyn MTLLibrary>>>> {
    use std::path::{Path, PathBuf};
    use std::process::Command;

    let metal_dir = find_metal_dir();
    let metal_dir = match metal_dir {
        Some(d) => d,
        None => return Ok(None),
    };

    let cache_dir = metal_dir.parent().unwrap_or(Path::new("."));
    let metallib_path = cache_dir.join("target").join("metal_kernels.metallib");

    // Use cache if up-to-date.
    if metallib_path.exists() && !sources_newer_than(&metal_dir, &metallib_path) {
        let ns_path = NSString::from_str(&metallib_path.to_string_lossy());
        if let Ok(lib) = unsafe {
            device.newLibraryWithURL_error(
                &objc2_foundation::NSURL::fileURLWithPath(&ns_path),
            )
        } {
            return Ok(Some(lib));
        }
    }

    eprintln!("ds4: Compiling Metal shaders from {:?}...", metal_dir);
    let tmp_dir = cache_dir.join("target").join("metal_tmp");
    std::fs::create_dir_all(&tmp_dir)?;
    std::fs::create_dir_all(metallib_path.parent().unwrap())?;

    let mut air_files = Vec::new();
    for entry in std::fs::read_dir(&metal_dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().map_or(true, |ext| ext != "metal") {
            continue;
        }
        let air_file = tmp_dir.join(format!(
            "{}.air",
            path.file_stem().unwrap().to_string_lossy()
        ));

        let status = Command::new("xcrun")
            .args(["-sdk", "macosx", "metal"])
            .args(["-I", &metal_dir.parent().unwrap_or(Path::new(".")).to_string_lossy()])
            .args(["-I", &metal_dir.to_string_lossy()])
            .args(["-ffast-math", "-Wall", "-Werror"])
            .arg("-c")
            .arg(&path)
            .arg("-o")
            .arg(&air_file)
            .status()?;
        if !status.success() {
            bail!("Failed to compile Metal shader: {:?}", path);
        }
        air_files.push(air_file);
    }
    if air_files.is_empty() {
        return Ok(None);
    }

    let mut cmd = Command::new("xcrun");
    cmd.args(["-sdk", "macosx", "metallib"]);
    for air in &air_files {
        cmd.arg(air);
    }
    cmd.arg("-o").arg(&metallib_path);
    let status = cmd.status()?;
    if !status.success() {
        bail!("Failed to link Metal library");
    }
    let _ = std::fs::remove_dir_all(&tmp_dir);

    eprintln!("ds4: Metal shaders compiled to {:?}", metallib_path);
    let ns_path = NSString::from_str(&metallib_path.to_string_lossy());
    let lib = unsafe {
        device.newLibraryWithURL_error(
            &objc2_foundation::NSURL::fileURLWithPath(&ns_path),
        )
    }
    .map_err(|e| anyhow!("Failed to load compiled metallib: {e}"))?;
    Ok(Some(lib))
}

fn find_metal_dir() -> Option<std::path::PathBuf> {
    if let Ok(exe) = std::env::current_exe() {
        if let Some(target_dir) = exe.parent() {
            for ancestor in target_dir.ancestors().skip(1) {
                let candidate = ancestor.join("metal");
                if candidate.is_dir() {
                    return Some(candidate);
                }
            }
        }
    }
    let cwd_candidate = std::path::PathBuf::from("metal");
    if cwd_candidate.is_dir() {
        return Some(cwd_candidate);
    }
    None
}

fn sources_newer_than(dir: &std::path::Path, reference: &std::path::Path) -> bool {
    let ref_time = match std::fs::metadata(reference).and_then(|m| m.modified()) {
        Ok(t) => t,
        Err(_) => return true,
    };
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().map_or(false, |ext| ext == "metal" || ext == "h") {
                if let Ok(src_time) = std::fs::metadata(&path).and_then(|m| m.modified()) {
                    if src_time > ref_time {
                        return true;
                    }
                }
            }
        }
    }
    false
}

// ============================================================================
// Command buffer batching.
// ============================================================================

/// Begin a new command batch.
pub fn begin_commands() -> Result<()> {
    with_ctx(|ctx| {
        if ctx.batch_cb.is_some() {
            bail!("Metal command batch already in progress");
        }
        let cb = ctx
            .queue
            .commandBuffer()
            .ok_or_else(|| anyhow!("Failed to create command buffer"))?;
        ctx.batch_cb = Some(cb);
        Ok(())
    })
}

/// End the current command batch, commit and wait.
pub fn end_commands() -> Result<()> {
    with_ctx(|ctx| {
        // End any active encoder first.
        if let Some(enc) = ctx.batch_enc.take() {
            enc.endEncoding();
        }
        let cb = ctx
            .batch_cb
            .take()
            .ok_or_else(|| anyhow!("No active command batch"))?;
        cb.commit();
        cb.waitUntilCompleted();
        Ok(())
    })
}

/// Synchronize — end current batch, start a new one.
pub fn synchronize() -> Result<()> {
    end_commands()?;
    begin_commands()
}

/// Get or create the current compute encoder for this batch.
pub fn current_encoder() -> Result<Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>> {
    with_ctx(|ctx| {
        if let Some(ref enc) = ctx.batch_enc {
            return Ok(enc.clone());
        }
        let cb = ctx
            .batch_cb
            .as_ref()
            .ok_or_else(|| anyhow!("No active command batch"))?;
        let enc = cb
            .computeCommandEncoder()
            .ok_or_else(|| anyhow!("Failed to create compute encoder"))?;
        ctx.batch_enc = Some(enc.clone());
        Ok(enc)
    })
}

// ============================================================================
// Pipeline caching.
// ============================================================================

/// Get or create a compute pipeline for a named kernel function.
pub fn get_pipeline(name: &str) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>> {
    with_ctx(|ctx| {
        if let Some(pso) = ctx.pipeline_cache.get(name) {
            return Ok(pso.clone());
        }
        let ns_name = NSString::from_str(name);
        let function = ctx
            .library
            .newFunctionWithName(&ns_name)
            .ok_or_else(|| anyhow!("Metal function '{name}' not found"))?;
        let pso = ctx.device.newComputePipelineStateWithFunction_error(&function)
            .map_err(|e| anyhow!("Failed to create pipeline for '{name}': {e}"))?;
        ctx.pipeline_cache.insert(name.to_string(), pso.clone());
        Ok(pso)
    })
}

// ============================================================================
// Buffer allocation.
// ============================================================================

/// Allocate a GPU-visible buffer.
pub fn buffer_alloc(bytes: u64) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>> {
    with_ctx(|ctx| {
        let opts = MTLResourceOptions::MTLResourceStorageModeShared;
        ctx.device
            .newBufferWithLength_options(bytes as usize, opts)
            .ok_or_else(|| anyhow!("Failed to allocate {bytes} byte Metal buffer"))
    })
}

// ============================================================================
// Model mmap views.
// ============================================================================

/// Wrap a region of a memory-mapped model file as no-copy MTLBuffers.
pub fn map_model_range(
    model_map: *const std::ffi::c_void,
    model_size: u64,
    map_offset: u64,
    map_size: u64,
) -> Result<()> {
    with_ctx(|ctx| {
        let page = page_size() as u64;
        let model_addr = model_map as u64;

        if (model_addr & (page - 1)) != 0 {
            bail!("Model mmap base is not page-aligned");
        }
        if map_offset > model_size || map_size > model_size - map_offset {
            bail!("Model mapped range is outside the GGUF mapping");
        }
        if ctx.model_views.len() >= MAX_MODEL_VIEWS {
            bail!("Too many model views (max {MAX_MODEL_VIEWS})");
        }

        let page_offset = map_offset & !(page - 1);
        let leading = map_offset - page_offset;
        let mapped_size = round_up(leading + map_size, page);
        let mut max_buffer = ctx.device.maxBufferLength() as u64;
        max_buffer &= !(page - 1);

        let overlap = round_up(MODEL_MAX_TENSOR_BYTES, page) + page;
        if max_buffer == 0 || max_buffer <= overlap {
            bail!("Metal maxBufferLength is too small for model views");
        }
        let step = max_buffer - overlap;

        let opts = MTLResourceOptions::MTLResourceStorageModeShared
            | MTLResourceOptions::MTLResourceCPUCacheModeWriteCombined;

        let mut off = 0u64;
        while off < mapped_size {
            let chunk = (mapped_size - off).min(max_buffer);
            let addr = (model_addr + page_offset + off) as *mut std::ffi::c_void;
            let buffer = unsafe {
                ctx.device
                    .newBufferWithBytesNoCopy_length_options_deallocator(
                        std::ptr::NonNull::new(addr).unwrap(),
                        chunk as usize,
                        opts,
                        None,
                    )
            }
            .ok_or_else(|| anyhow!("Failed to create no-copy buffer at offset {off}"))?;

            ctx.model_views.push(ModelView {
                buffer,
                model_offset: page_offset + off,
                bytes: chunk,
            });

            off += step;
        }
        Ok(())
    })
}

/// Find the MTLBuffer and local offset for a given model file offset.
pub fn model_buffer_for_offset(
    abs_offset: u64,
) -> Result<(Retained<ProtocolObject<dyn MTLBuffer>>, u64)> {
    with_ctx(|ctx| {
        for view in &ctx.model_views {
            if abs_offset >= view.model_offset
                && abs_offset < view.model_offset + view.bytes
            {
                let local = abs_offset - view.model_offset;
                return Ok((view.buffer.clone(), local));
            }
        }
        bail!("Model offset {abs_offset} not covered by any MTLBuffer view")
    })
}

// ============================================================================
// Helpers.
// ============================================================================

pub fn mtl_size(x: u64, y: u64, z: u64) -> MTLSize {
    MTLSize {
        width: x as usize,
        height: y as usize,
        depth: z as usize,
    }
}

fn page_size() -> usize {
    unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize }
}

fn round_up(x: u64, align: u64) -> u64 {
    (x + align - 1) & !(align - 1)
}
