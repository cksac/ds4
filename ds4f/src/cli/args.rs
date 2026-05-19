//! CLI argument parsing for `ds4f run`.
//!
//! Mirrors the option parsing in ds4_cli.c `parse_options()`.

use clap::Args;

#[derive(Args, Debug, Clone)]
pub struct RunArgs {
    /// GGUF model path
    #[arg(short = 'm', long = "model", default_value = "ds4flash.gguf")]
    pub model_path: String,

    /// Optional MTP support GGUF for draft-token probes
    #[arg(long = "mtp")]
    pub mtp_path: Option<String>,

    /// Maximum autoregressive MTP draft tokens per speculative step
    #[arg(long = "mtp-draft", default_value = "1")]
    pub mtp_draft_tokens: i32,

    /// Minimum recursive-draft confidence for the fast N=2 verifier
    #[arg(long = "mtp-margin", default_value = "3.0")]
    pub mtp_margin: f32,

    /// Prompt to generate from (one-shot mode)
    #[arg(short = 'p', long = "prompt")]
    pub prompt: Option<String>,

    /// Read prompt text from file
    #[arg(long = "prompt-file")]
    pub prompt_file: Option<String>,

    /// System prompt (empty string disables default)
    #[arg(short = 's', long = "system", default_value = "You are a helpful assistant")]
    pub system: String,

    /// Maximum tokens to generate
    #[arg(short = 'n', long = "tokens", default_value = "50000")]
    pub n_predict: i32,

    /// Context size allocated for the session
    #[arg(short = 'c', long = "ctx", default_value = "32768")]
    pub ctx_size: i32,

    /// Sampling temperature (0 = greedy)
    #[arg(long = "temp", default_value = "1.0")]
    pub temperature: f32,

    /// Nucleus sampling probability
    #[arg(long = "top-p", default_value = "1.0")]
    pub top_p: f32,

    /// Keep tokens scoring at least F times the top token
    #[arg(long = "min-p", default_value = "0.05")]
    pub min_p: f32,

    /// Sampling seed for reproducible non-greedy runs
    #[arg(long = "seed")]
    pub seed: Option<u64>,

    /// CPU helper threads
    #[arg(short = 't', long = "threads", default_value = "4")]
    pub n_threads: i32,

    /// Backend: metal, cuda, or cpu
    #[arg(long = "backend", default_value = "metal")]
    pub backend: String,

    /// Prefer exact kernels over faster approximate paths
    #[arg(long = "quality")]
    pub quality: bool,

    /// Directional steering: load f32 direction vectors
    #[arg(long = "dir-steering-file")]
    pub directional_steering_file: Option<String>,

    /// Steering scale after FFN outputs
    #[arg(long = "dir-steering-ffn")]
    pub directional_steering_ffn: Option<f32>,

    /// Steering scale after attention outputs
    #[arg(long = "dir-steering-attn")]
    pub directional_steering_attn: Option<f32>,

    /// Touch mapped tensor pages before generation
    #[arg(long = "warm-weights")]
    pub warm_weights: bool,

    /// Thinking mode: think (default), think-max, nothink
    #[arg(long = "think-mode", default_value = "think")]
    pub think_mode: String,

    /// Load model and print summary only
    #[arg(long = "inspect")]
    pub inspect: bool,

    /// Tokenize prompt and print tokens without inference
    #[arg(long = "dump-tokens")]
    pub dump_tokens: bool,

    /// Write greedy continuation top-logprobs as JSON
    #[arg(long = "dump-logprobs")]
    pub dump_logprobs_path: Option<String>,

    /// Number of top alternatives for --dump-logprobs
    #[arg(long = "logprobs-top-k", default_value = "20")]
    pub dump_logprobs_top_k: i32,

    /// Run output HC/logits head test
    #[arg(long = "head-test")]
    pub head_test: bool,

    /// Run CPU whole-model pass for first prompt token
    #[arg(long = "first-token-test")]
    pub first_token_test: bool,

    /// Compare first GPU-resident graph stages with CPU
    #[arg(long = "metal-graph-test")]
    pub metal_graph_test: bool,

    /// Run GPU-resident self-token graph across all layers
    #[arg(long = "metal-graph-full-test")]
    pub metal_graph_full_test: bool,

    /// Compare CPU and GPU graph logits for full prompt
    #[arg(long = "metal-graph-prompt-test")]
    pub metal_graph_prompt_test: bool,

    /// Use native Rust Metal session for inference (instead of C engine)
    #[arg(long = "native")]
    pub native: bool,
}
