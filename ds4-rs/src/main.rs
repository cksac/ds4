//! DS4 Rust Inference Server -- Metal GPU Backend.
//!
//! Full inference engine: Metal init, GGUF load, weight bind, session,
//! per-layer GPU encoding, BPE tokenizer, HTTP server.

pub mod bridge;
pub mod tensor;
pub mod pipeline;
pub mod ops;
pub mod gguf;
pub mod model;
pub mod graph;
pub mod model_view;
pub mod weights;
pub mod layer_enc;
pub mod tokenizer;
pub mod session;
pub mod server;
pub mod metal_args;

use std::path::Path;
use std::sync::Mutex;

const PORT: u16 = 8080;

/// Global session state accessible from server handlers
pub static SESSION: std::sync::OnceLock<Mutex<session::SessionState>> = std::sync::OnceLock::new();

fn main() {
    println!("DS4 Rust Inference Server v0.1.0");
    println!("Backend: Metal (Apple GPU)");
    println!("Port: {}", PORT);

    // Step 1: Initialize Metal GPU
    println!("  Initializing Metal GPU...");
    if let Err(e) = bridge::init() {
        eprintln!("ERROR: Metal init failed: {}", e);
        std::process::exit(1);
    }
    println!("  Metal GPU ready.");

    // Step 2: Load model from GGUF file
    let model_path = std::env::var("DS4_MODEL").unwrap_or_else(|_| "ds4flash.gguf".to_string());
    println!("  Loading model: {}", model_path);

    let session = if Path::new(&model_path).exists() {
        match gguf::GgufModel::open(Path::new(&model_path)) {
            Ok(model) => {
                println!("  Model loaded: {} tensors, {} KV pairs", 
                    model.n_tensors, model.n_kv);
                let n_layer = gguf::N_LAYER as usize;
                let weights = weights::weights_bind(&model);
                println!("  Weights bound: {} layers", n_layer);
                match weights::validate_shapes(&model, &weights) {
                    Ok(_) => println!("  Shape validation OK"),
                    Err(e) => eprintln!("  Shape warnings: {}", e),
                }
                match session::SessionState::from_model(&model) {
                    Ok(s) => s,
                    Err(e) => {
                        eprintln!("  Session init: {} (stub mode)", e);
                        session::SessionState::new(65536)
                    }
                }
            }
            Err(e) => {
                eprintln!("WARN: Model loading failed: {}. Stub mode.", e);
                session::SessionState::new(65536)
            }
        }
    } else {
        eprintln!("WARN: Model file '{}' not found. Starting in stub mode.", model_path);
        eprintln!("  Set DS4_MODEL environment variable to load a real model.");
        session::SessionState::new(65536)
    };

    SESSION.set(Mutex::new(session)).ok();

    // Step 3: Start HTTP server
    println!("\nStarting HTTP server on port {}...", PORT);
    println!("  Health: http://localhost:{}/health", PORT);
    println!("  Models: http://localhost:{}/v1/models", PORT);
    println!("  Chat:   POST http://localhost:{}/v1/chat/completions", PORT);
    println!("  Text:   POST http://localhost:{}/v1/completions", PORT);
    println!();

    if let Err(e) = server::serve(PORT) {
        eprintln!("Server error: {}", e);
    }
}
