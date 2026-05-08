fn main() {
    println!("cargo:rerun-if-changed=ds4.c");
    println!("cargo:rerun-if-changed=ds4.h");
    println!("cargo:rerun-if-changed=ds4_metal.h");
    println!("cargo:rerun-if-changed=ds4_metal.m");

    if let Ok(entries) = std::fs::read_dir("metal") {
        for entry in entries.flatten() {
            println!("cargo:rerun-if-changed={}", entry.path().display());
        }
    }

    let mut build = cc::Build::new();
    build
        .include(".")
        .file("ds4.c")
        .flag_if_supported("-std=c99")
        .flag_if_supported("-O3")
        .flag_if_supported("-ffast-math")
        .flag_if_supported("-mcpu=native");

    if cfg!(target_os = "macos") {
        build.file("ds4_metal.m").flag_if_supported("-fobjc-arc");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=Metal");
    } else {
        build.define("DS4_NO_METAL", None);
    }

    build.compile("ds4ffi");
}