//! CLI argument parsing for `ds4f serve`.
//!
//! Mirrors the option parsing in ds4_server.c `parse_options()`.

use clap::Args;

#[derive(Args, Debug, Clone)]
pub struct ServeArgs {
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

    /// Context size allocated at startup
    #[arg(short = 'c', long = "ctx", default_value = "32768")]
    pub ctx_size: i32,

    /// Default max output tokens when client omits limit
    #[arg(short = 'n', long = "tokens", default_value = "393216")]
    pub default_tokens: i32,

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

    /// Touch mapped tensor pages before serving
    #[arg(long = "warm-weights")]
    pub warm_weights: bool,

    /// Bind address
    #[arg(long = "host", default_value = "127.0.0.1")]
    pub host: String,

    /// Bind port
    #[arg(long = "port", default_value = "8000")]
    pub port: u16,

    /// Add Access-Control-Allow-* headers
    #[arg(long = "cors")]
    pub cors: bool,

    /// Write session trace to file
    #[arg(long = "trace")]
    pub trace_path: Option<String>,

    /// Enable disk KV cache in directory
    #[arg(long = "kv-disk-dir")]
    pub kv_disk_dir: Option<String>,

    /// Disk budget for checkpoint files (MB)
    #[arg(long = "kv-disk-space-mb", default_value = "4096")]
    pub kv_disk_space_mb: u64,

    /// Do not save checkpoints shorter than N tokens
    #[arg(long = "kv-cache-min-tokens", default_value = "512")]
    pub kv_cache_min_tokens: usize,

    /// Cold first prompts up to this length are saved automatically
    #[arg(long = "kv-cache-cold-max-tokens", default_value = "30000")]
    pub kv_cache_cold_max_tokens: usize,

    /// Save at aligned frontiers spaced about N tokens apart
    #[arg(long = "kv-cache-continued-interval-tokens", default_value = "10000")]
    pub kv_cache_continued_interval_tokens: usize,

    /// Trim tail tokens before cold boundary saves
    #[arg(long = "kv-cache-boundary-trim-tokens", default_value = "32")]
    pub kv_cache_boundary_trim_tokens: usize,

    /// Align cold boundary saves to token multiple
    #[arg(long = "kv-cache-boundary-align-tokens", default_value = "2048")]
    pub kv_cache_boundary_align_tokens: usize,

    /// Reject checkpoints from same model with different quantization
    #[arg(long = "kv-cache-reject-different-quant")]
    pub kv_cache_reject_different_quant: bool,

    /// Disable exact DSML tool replay
    #[arg(long = "disable-exact-dsml-tool-replay")]
    pub disable_exact_dsml_tool_replay: bool,

    /// Maximum exact tool-call IDs kept in RAM
    #[arg(long = "tool-memory-max-ids", default_value = "100000")]
    pub tool_memory_max_ids: usize,
}
