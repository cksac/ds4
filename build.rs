use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=ds4_metal.h");
    println!("cargo:rerun-if-changed=metal/ds4_metal_defs.h");

    if let Ok(entries) = std::fs::read_dir("metal") {
        for entry in entries.flatten() {
            println!("cargo:rerun-if-changed={}", entry.path().display());
        }
    }

    if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=Metal");

        // Always link against pre-compiled C library (contains Metal wrapper functions)
        println!("cargo:rustc-link-search=native=lib");
        println!("cargo:rustc-link-lib=static=ds4_metal");
        
        // Also compile Metal files to library for future Rust replacement
        let out_dir = std::env::var("OUT_DIR").unwrap();
        let metal_lib_path = compile_metal_library(&out_dir);
        
        // Pass the Metal library path to Rust code for future use
        if metal_lib_path.exists() {
            println!("cargo:rustc-env=METAL_LIBRARY_PATH={}", metal_lib_path.display());
        }
    }
}

fn compile_metal_library(out_dir: &str) -> PathBuf {
    let out_path = PathBuf::from(out_dir);
    let metal_lib = out_path.join("metal_kernels.metallib");
    
    // Skip if already compiled and up-to-date
    if metal_lib.exists() {
        return metal_lib;
    }

    // Read all .metal files in metal/ directory
    let metal_dir = PathBuf::from("metal");
    if !metal_dir.exists() {
        eprintln!("Warning: metal/ directory not found");
        return metal_lib;
    }

    // Compile individual .metal files to AIR
    let mut air_files = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&metal_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().map_or(false, |ext| ext == "metal") {
                let air_file = out_path.join(
                    format!("{}.air", path.file_stem().unwrap().to_string_lossy())
                );
                
                // Compile .metal to .air
                let status = Command::new("xcrun")
                    .args(&["-sdk", "macosx", "metal"])
                    .args(&[
                        "-I", ".",
                        "-I", "metal",
                        "-ffast-math",
                        "-Wall",
                        "-Werror",
                    ])
                    .arg("-c")
                    .arg(&path)
                    .arg("-o")
                    .arg(&air_file)
                    .status();

                match status {
                    Ok(s) if s.success() => {
                        air_files.push(air_file);
                    }
                    _ => {
                        eprintln!("Warning: Failed to compile {:?}", path);
                        // Continue with remaining files
                    }
                }
            }
        }
    }

    // Link .air files into .metallib
    if !air_files.is_empty() {
        let mut cmd = Command::new("xcrun");
        cmd.args(&["-sdk", "macosx", "metallib"]);
        for air in &air_files {
            cmd.arg(air);
        }
        cmd.arg("-o").arg(&metal_lib);

        match cmd.status() {
            Ok(s) if s.success() => {
                println!("cargo:warning=Compiled Metal library: {:?}", metal_lib);
            }
            _ => {
                eprintln!("Warning: Failed to link Metal library");
            }
        }
    }

    metal_lib
}