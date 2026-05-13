fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=Metal");
    }
}
