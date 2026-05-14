use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLCreateSystemDefaultDevice, MTLCompileOptions, MTLDevice,
    MTLCommandQueue, MTLLibrary, MTLResourceOptions,
};
use objc2_foundation::{NSString, NSURL};
use std::sync::Mutex;

// MTLDevice, MTLCommandQueue, MTLLibrary are documented as thread-safe.
// objc2-metal does not impl Send for ProtocolObject<dyn MTL*>, so we wrap.
struct SendRetained<T: ?Sized>(Retained<ProtocolObject<T>>);
unsafe impl<T: ?Sized> Send for SendRetained<T> {}

static INITIALIZED: Mutex<bool> = Mutex::new(false);
static DEVICE:  Mutex<Option<SendRetained<dyn MTLDevice>>>      = Mutex::new(None);
static QUEUE:   Mutex<Option<SendRetained<dyn MTLCommandQueue>>> = Mutex::new(None);
static LIBRARY: Mutex<Option<SendRetained<dyn MTLLibrary>>>      = Mutex::new(None);

pub fn device() -> Option<Retained<ProtocolObject<dyn MTLDevice>>> {
    DEVICE.lock().unwrap().as_ref().map(|s| s.0.clone())
}
pub fn queue() -> Option<Retained<ProtocolObject<dyn MTLCommandQueue>>> {
    QUEUE.lock().unwrap().as_ref().map(|s| s.0.clone())
}
pub fn library() -> Option<Retained<ProtocolObject<dyn MTLLibrary>>> {
    LIBRARY.lock().unwrap().as_ref().map(|s| s.0.clone())
}

pub fn device_name() -> String {
    DEVICE.lock().unwrap().as_ref()
        .map(|s| s.0.name().to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

fn build_metal_source() -> String {
    let mut source = String::new();
    source.push_str("#include <metal_stdlib>\nusing namespace metal;\n\n");
    source.push_str("constant float DS4_M_PI_F = 3.14159265358979323846f;\n");
    source.push_str("#define MAX(x,y) ((x)>(y)?(x):(y))\n#define MIN(x,y) ((x)<(y)?(x):(y))\n");
    source.push_str("#define SWAP(x, y) { auto tmp = (x); (x) = (y); (y) = tmp; }\n");
    source.push_str("#define QK8_0 32\n#define N_SIMDWIDTH 32\n#define N_R0_Q8_0 2\n");
    source.push_str("#define N_SG_Q8_0 4\n#define FC_MUL_MV 600\n#define FC_MUL_MM 700\n");
    source.push_str("#define FC_BIN 1300\n#define FC_UNARY 1200\n");
    source.push_str("#define FOR_UNROLL(x) _Pragma(\"clang loop unroll(full)\") for (x)\n");
    source.push_str("struct block_q8_0 { half d; int8_t qs[QK8_0]; };\n");
    source.push_str("enum ds4_sort_order { DS4_SORT_ORDER_ASC, DS4_SORT_ORDER_DESC };\n\n");
    source.push_str("static void dequantize_f16_t4(device const half4 * src, short il, thread float4 & reg) {\n");
    source.push_str("    reg = (float4)(*src);\n}\n\n");

    let metal_dir = std::path::Path::new("src/metal");
    let mut files: Vec<std::path::PathBuf> = std::fs::read_dir(metal_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "metal"))
        .map(|e| e.path())
        .collect();
    files.sort();
    for f in &files {
        let content = std::fs::read_to_string(f).unwrap();
        source.push_str(&format!("\n// {}\n", f.file_name().unwrap().to_str().unwrap()));
        source.push_str(&content);
    }
    source
}

pub fn init() -> Result<(), Box<dyn std::error::Error>> {
    let mut init = INITIALIZED.lock().unwrap();
    if *init { return Ok(()); }

    let device = MTLCreateSystemDefaultDevice().ok_or("no Metal device")?;
    let queue = device.newCommandQueue().ok_or("no command queue")?;

    // Try precompiled metallib first, fall back to source compilation
    let metallib_env = option_env!("DS4_METALLIB_PATH");
    let precompiled = metallib_env
        .and_then(|p| if std::path::Path::new(p).exists() { Some(p.to_string()) } else { None })
        .or_else(|| {
            std::env::var("OUT_DIR").ok().map(|d| {
                std::path::Path::new(&d).join("ds4_kernels.metallib").to_string_lossy().to_string()
            }).filter(|p| std::path::Path::new(p).exists())
        });

    let library = if let Some(ref path) = precompiled {
        eprintln!("ds4: loading precompiled Metal library from {}", path);
        let url = NSURL::from_file_path(std::path::Path::new(path))
            .ok_or("bad metallib path")?;
        device.newLibraryWithURL_error(&url)
            .map_err(|e| format!("metallib load error: {}", e))?
    } else {
        eprintln!("ds4: compiling Metal library from source");
        let source_str = build_metal_source();
        let ns_source = NSString::from_str(&source_str);
        let opts = MTLCompileOptions::new();
        device.newLibraryWithSource_options_error(&ns_source, Some(&opts))
            .map_err(|e| format!("Metal compile error: {}", e))?
    };

    *DEVICE.lock().unwrap()  = Some(SendRetained(device));
    *QUEUE.lock().unwrap()   = Some(SendRetained(queue));
    *LIBRARY.lock().unwrap() = Some(SendRetained(library));

    *init = true;
    crate::pipeline::init_cache();
    Ok(())
}

pub fn cleanup() {
    let mut init = INITIALIZED.lock().unwrap();
    if !*init { return; }
    *DEVICE.lock().unwrap()  = None;
    *QUEUE.lock().unwrap()   = None;
    *LIBRARY.lock().unwrap() = None;
    *init = false;
}
