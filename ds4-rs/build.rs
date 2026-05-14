use std::path::Path;
use std::process::Command;

fn main() {
    let root = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    // Re-run if any metal shader changes or this build.rs changes
    println!("cargo:rerun-if-changed={}/build.rs", root);
    let metal_dir = Path::new(&root).join("src/metal");
    if !metal_dir.exists() {
        return;
    }
    for entry in std::fs::read_dir(&metal_dir).unwrap() {
        let entry = entry.unwrap();
        if entry.path().extension().map_or(false, |ext| ext == "metal") {
            println!("cargo:rerun-if-changed={}", entry.path().display());
        }
    }

    let out_dir = std::env::var("OUT_DIR").unwrap();

    // Build the preamble (must match bridge.rs::build_metal_source)
    let preamble = "\
#include <metal_stdlib>
using namespace metal;
constant float DS4_M_PI_F = 3.14159265358979323846f;
#define MAX(x,y) ((x)>(y)?(x):(y))
#define MIN(x,y) ((x)<(y)?(x):(y))
#define SWAP(x, y) { auto tmp = (x); (x) = (y); (y) = tmp; }
#define QK8_0 32
#define N_SIMDWIDTH 32
#define N_R0_Q8_0 2
#define N_SG_Q8_0 4
#define FC_MUL_MV 600
#define FC_MUL_MM 700
#define FC_BIN 1300
#define FC_UNARY 1200
#define FOR_UNROLL(x) _Pragma(\"clang loop unroll(full)\") for (x)
struct block_q8_0 { half d; int8_t qs[QK8_0]; };
enum ds4_sort_order { DS4_SORT_ORDER_ASC, DS4_SORT_ORDER_DESC };
static void dequantize_f16_t4(device const half4 * src, short il, thread float4 & reg) {
    reg = (float4)(*src);
}
";

    // Collect and concatenate all .metal files
    let mut source = preamble.to_string();
    let mut files: Vec<_> = std::fs::read_dir(metal_dir)
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

    // Write concatenated source to a temp file
    let src_path = Path::new(&out_dir).join("ds4_kernels.metal");
    std::fs::write(&src_path, &source).unwrap();

    // Compile to AIR
    let air_path = Path::new(&out_dir).join("ds4_kernels.air");
    let status = Command::new("xcrun")
        .args(["-sdk", "macosx", "metal", "-std=metal3.0", "-O3"])
        .arg("-o").arg(&air_path)
        .arg("-c").arg(&src_path)
        .status();
    match status {
        Ok(s) if s.success() => {}
        Ok(s) => {
            eprintln!("Metal compilation failed with exit code: {}", s);
            eprintln!("Falling back to runtime compilation.");
            return;
        }
        Err(e) => {
            eprintln!("Could not invoke Metal compiler: {}", e);
            eprintln!("Falling back to runtime compilation.");
            return;
        }
    }

    // Link into metallib
    let lib_path = Path::new(&out_dir).join("ds4_kernels.metallib");
    let status = Command::new("xcrun")
        .args(["-sdk", "macosx", "metallib"])
        .arg("-o").arg(&lib_path)
        .arg(&air_path)
        .status();
    match status {
        Ok(s) if s.success() => {
            // Expose path so bridge.rs can load it at runtime
            println!("cargo:rustc-env=DS4_METALLIB_PATH={}", lib_path.display());
        }
        Ok(s) => {
            eprintln!("Metallib linking failed with exit code: {}", s);
        }
        Err(e) => {
            eprintln!("Could not invoke metallib linker: {}", e);
        }
    }
}
