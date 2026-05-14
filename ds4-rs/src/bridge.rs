use metal::*;
use std::cell::RefCell;
use std::sync::Mutex;

static INITIALIZED: Mutex<bool> = Mutex::new(false);

thread_local! {
    pub static DEVICE: RefCell<Option<Device>> = RefCell::new(None);
    pub static QUEUE: RefCell<Option<CommandQueue>> = RefCell::new(None);
    pub static LIBRARY: RefCell<Option<Library>> = RefCell::new(None);
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
    // Missing dequantize_f16_t4 needed by dense.metal template instantiations
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

    let device = Device::system_default().ok_or("no Metal device")?;
    let queue = device.new_command_queue();

    let library: Library;
    if let Ok(out_dir) = std::env::var("OUT_DIR") {
        let metallib = std::path::Path::new(&out_dir).join("ds4_kernels.metallib");
        if metallib.exists() {
            library = device.new_library_with_file(metallib)?;
        } else {
            let source = build_metal_source();
            let opts = CompileOptions::new();
            library = device.new_library_with_source(&source, &opts)?;
        }
    } else {
        let source = build_metal_source();
        let opts = CompileOptions::new();
        library = device.new_library_with_source(&source, &opts)?;
    }

    DEVICE.with(|d| *d.borrow_mut() = Some(device));
    QUEUE.with(|q| *q.borrow_mut() = Some(queue));
    LIBRARY.with(|l| *l.borrow_mut() = Some(library));

    *init = true;
    crate::pipeline::init_cache();
    Ok(())
}

pub fn cleanup() {
    let mut init = INITIALIZED.lock().unwrap();
    if !*init { return; }
    DEVICE.with(|d| *d.borrow_mut() = None);
    QUEUE.with(|q| *q.borrow_mut() = None);
    LIBRARY.with(|l| *l.borrow_mut() = None);
    *init = false;
}

pub fn with_device<F: FnOnce(&Device) -> R, R>(f: F) -> Option<R> {
    DEVICE.with(|d| d.borrow().as_ref().map(|dev| f(dev)))
}

pub fn with_queue<F: FnOnce(&CommandQueue) -> R, R>(f: F) -> Option<R> {
    QUEUE.with(|q| q.borrow().as_ref().map(|q| f(q)))
}

pub fn with_library<F: FnOnce(&Library) -> R, R>(f: F) -> Option<R> {
    LIBRARY.with(|l| l.borrow().as_ref().map(|lib| f(lib)))
}
