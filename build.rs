fn main() {
    println!("cargo:rerun-if-changed=ds4_metal.h");
    println!("cargo:rerun-if-changed=ds4_metal.m");

    if let Ok(entries) = std::fs::read_dir("metal") {
        for entry in entries.flatten() {
            println!("cargo:rerun-if-changed={}", entry.path().display());
        }
    }

    if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=Metal");

        let mut build = cc::Build::new();
        build
            .file("ds4_metal.m")
            .include(".")
            .flag_if_supported("-O3")
            .flag_if_supported("-ffast-math")
            .flag_if_supported("-mcpu=native");

        build.compile("ds4ffi");
    }
}