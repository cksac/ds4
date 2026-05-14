use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();
    let dest = PathBuf::from(out_dir);

    let metal_dir = PathBuf::from("src/metal");
    let mut metal_files: Vec<PathBuf> = std::fs::read_dir(&metal_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "metal"))
        .map(|e| e.path())
        .collect();
    metal_files.sort();

    eprintln!("ds4-rs build: concatenating {} metal kernel files", metal_files.len());
    let mut combined: Vec<u8> = Vec::new();
    // Include the preamble that ds4_metal.m embeds (types, macros)
    let preamble = br#"
#include <metal_stdlib>
using namespace metal;

#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define SWAP(x, y) { auto tmp = (x); (x) = (y); (y) = tmp; }
#define QK8_0 32
#define N_SIMDWIDTH 32
#define N_R0_Q8_0 2
#define N_SG_Q8_0 4
#define FC_MUL_MV 600
#define FC_MUL_MM 700
#define FC_BIN 1300
#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)
#define M_PI_F 3.14159265358979323846f

struct block_q8_0 {
    half d;
    int8_t qs[QK8_0];
};

enum ds4_sort_order {
    DS4_SORT_ORDER_ASC,
    DS4_SORT_ORDER_DESC,
};
"#;
    combined.extend_from_slice(preamble);

    for f in &metal_files {
        let name = f.file_name().unwrap().to_str().unwrap();
        let header = format!("\n// >>> {} included by build.rs\n", name);
        combined.extend(header.as_bytes());
        let content = std::fs::read(f).unwrap();
        combined.extend(content);
        combined.push(b'\n');
    }

    let combined_path = dest.join("ds4_kernels.metal");
    std::fs::write(&combined_path, &combined).unwrap();
    println!("cargo:rerun-if-changed=src/metal/");
    println!("cargo:rerun-if-changed=build.rs");

    // Compile with xcrun metal - compile to .air, then metallib
    let air_path = dest.join("kernels.air");
    let metallib_path = dest.join("ds4_kernels.metallib");

    let compile = Command::new("xcrun")
        .args(["metal", "-O3", "-c"])
        .arg(combined_path.to_str().unwrap())
        .arg("-o")
        .arg(air_path.to_str().unwrap())
        .output();

    match compile {
        Ok(s) if s.status.success() => {
            eprintln!("ds4-rs build: Metal kernel compiled to .air");
            let link = Command::new("xcrun")
                .args(["metallib", "-o"])
                .arg(metallib_path.to_str().unwrap())
                .arg(air_path.to_str().unwrap())
                .output();
            match link {
                Ok(s) if s.status.success() => {
                    eprintln!("ds4-rs build: Metal kernel metallib created");
                }
                Ok(s) => {
                    eprintln!("ds4-rs build: metallib link failed: {}", 
                        String::from_utf8_lossy(&s.stderr));
                }
                Err(e) => {
                    eprintln!("ds4-rs build: metallib link error: {}", e);
                }
            }
        }
        Ok(s) => {
            eprintln!("ds4-rs build: Metal compilation failed (non-fatal, can use JIT):\n{}",
                String::from_utf8_lossy(&s.stderr));
        }
        Err(e) => {
            eprintln!("ds4-rs build: xcrun metal not available (non-fatal): {}", e);
        }
    }
}
