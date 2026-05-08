// ds4f — unified CLI for DeepSeek V4 Flash
//
//   ds4f run      interactive / one-shot chat
//   ds4f serve    OpenAI-compatible HTTP server
//   ds4f download fetch GGUF files from Hugging Face

use std::fs;
use std::io::{self, BufRead, Write};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};
use std::env;

use anyhow::{anyhow, bail, Context, Result};
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use clap::{Parser, Subcommand, ValueEnum};
use ds4_rust::{
    build_chat_generation_prompt, ensure_supported_role, generate_rust,
    is_rendered_chat_prompt, mib_string, Backend, ChatMessage, Engine, EngineOptions,
    GenerationOptions, RustSession, ThinkMode,
};
use futures_util::stream;
use hf_hub::api::sync::ApiBuilder;
use serde::Deserialize;
use serde_json::{json, Value};

// ---------------------------------------------------------------------------
// Top-level CLI
// ---------------------------------------------------------------------------

#[derive(Debug, Parser)]
#[command(
    name = "ds4f",
    about = "DeepSeek V4 Flash — run, serve, or download",
    subcommand_required = true,
    arg_required_else_help = true
)]
struct Cli {
    #[command(subcommand)]
    command: Subcommands,
}

#[derive(Debug, Subcommand)]
enum Subcommands {
    /// Interactive / one-shot chat
    Run(RunArgs),
    /// OpenAI-compatible HTTP server
    Serve(ServeArgs),
    /// Download GGUF model files from Hugging Face
    Download(DownloadArgs),
}

// ---------------------------------------------------------------------------
// Shared backend arg (Metal for inference; Cpu retained for correctness checks)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum BackendArg {
    Metal,
    Cpu,
}

impl From<BackendArg> for Backend {
    fn from(value: BackendArg) -> Self {
        match value {
            BackendArg::Metal => Backend::Metal,
            BackendArg::Cpu => Backend::Cpu,
        }
    }
}

// ---------------------------------------------------------------------------
// `ds4f run` args
// ---------------------------------------------------------------------------

#[derive(Debug, Parser)]
#[command(name = "run")]
struct RunArgs {
    #[arg(short = 'm', long = "model", default_value = "ds4flash.gguf")]
    model: String,
    #[arg(long = "mtp")]
    mtp: Option<String>,
    #[arg(long = "mtp-draft", default_value_t = 1)]
    mtp_draft_tokens: i32,
    #[arg(long = "mtp-margin", default_value_t = 3.0)]
    mtp_margin: f32,
    #[arg(short = 'c', long = "ctx", default_value_t = 32768)]
    ctx_size: i32,
    #[arg(short = 't', long = "threads", default_value_t = 0)]
    threads: i32,
    #[arg(long = "backend", value_enum, default_value_t = BackendArg::Metal)]
    backend: BackendArg,
    #[arg(long = "quality", default_value_t = false)]
    quality: bool,
    #[arg(long = "warm-weights", default_value_t = false)]
    warm_weights: bool,
    #[arg(short = 'p', long = "prompt")]
    prompt: Option<String>,
    #[arg(long = "prompt-file")]
    prompt_file: Option<String>,
    #[arg(short = 's', long = "system", default_value = "You are a helpful assistant")]
    system: String,
    #[arg(short = 'n', long = "tokens", default_value_t = 50_000)]
    max_tokens: i32,
    #[arg(long = "temp", default_value_t = 1.0)]
    temperature: f32,
    #[arg(long = "top-p", default_value_t = 1.0)]
    top_p: f32,
    #[arg(long = "top-k", default_value_t = 0)]
    top_k: i32,
    #[arg(long = "min-p", default_value_t = 0.0)]
    min_p: f32,
    #[arg(long = "seed")]
    seed: Option<u64>,
    #[arg(long = "think", default_value_t = false, conflicts_with_all = ["nothink", "think_max"])]
    think: bool,
    #[arg(long = "think-max", default_value_t = false, conflicts_with_all = ["think", "nothink"])]
    think_max: bool,
    #[arg(long = "nothink", default_value_t = false, conflicts_with_all = ["think", "think_max"])]
    nothink: bool,
    #[arg(long = "inspect", default_value_t = false)]
    inspect: bool,
    #[arg(long = "dump-tokens", default_value_t = false)]
    dump_tokens: bool,
}

// ---------------------------------------------------------------------------
// `ds4f serve` args
// ---------------------------------------------------------------------------

#[derive(Debug, Parser)]
#[command(name = "serve")]
struct ServeArgs {
    #[arg(short = 'm', long = "model", default_value = "ds4flash.gguf")]
    model: String,
    #[arg(long = "mtp")]
    mtp: Option<String>,
    #[arg(long = "mtp-draft", default_value_t = 1)]
    mtp_draft_tokens: i32,
    #[arg(long = "mtp-margin", default_value_t = 3.0)]
    mtp_margin: f32,
    #[arg(short = 'c', long = "ctx", default_value_t = 32768)]
    ctx_size: i32,
    #[arg(short = 'n', long = "tokens", default_value_t = 393_216)]
    default_tokens: i32,
    #[arg(short = 't', long = "threads", default_value_t = 0)]
    threads: i32,
    #[arg(long = "host", default_value = "127.0.0.1")]
    host: String,
    #[arg(long = "port", default_value_t = 8000)]
    port: u16,
    #[arg(long = "quality", default_value_t = false)]
    quality: bool,
    #[arg(long = "warm-weights", default_value_t = false)]
    warm_weights: bool,
}

// ---------------------------------------------------------------------------
// `ds4f download` args
// ---------------------------------------------------------------------------

const REPO: &str = "antirez/deepseek-v4-gguf";
const Q2_FILE: &str = "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf";
const Q4_FILE: &str =
    "DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2.gguf";
const MTP_FILE: &str = "DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf";

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum ModelArg {
    Q2,
    Q4,
    Mtp,
}

impl ModelArg {
    fn filename(self) -> &'static str {
        match self {
            Self::Q2 => Q2_FILE,
            Self::Q4 => Q4_FILE,
            Self::Mtp => MTP_FILE,
        }
    }
}

#[derive(Debug, Parser)]
#[command(
    name = "download",
    about = "DeepSeek V4 Flash GGUF downloader",
    after_help = "Targets:\n  q2   2-bit routed experts, about 81 GB on disk.\n       Main model for 128 GB RAM machines.\n\n  q4   4-bit routed experts, about 153 GB on disk.\n       Main model for machines with 256 GB RAM or more.\n\n  mtp  Optional speculative decoding component, about 3.5 GB on disk.\n       Useful with both q2 and q4; enable with --mtp when running.\n\nAfter q2/q4 the tool updates:\n  ./ds4flash.gguf -> gguf/<model>"
)]
struct DownloadArgs {
    #[arg(value_enum)]
    model: ModelArg,
    #[arg(long = "token")]
    token: Option<String>,
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Subcommands::Run(args) => run_cmd(args),
        Subcommands::Serve(args) => serve_cmd(args).await,
        Subcommands::Download(args) => download_cmd(args),
    }
}

// ===========================================================================
// run subcommand
// ===========================================================================

#[derive(Clone, Debug)]
struct ReplState {
    system: String,
    messages: Vec<ChatMessage>,
    think_mode: ThinkMode,
    ctx_size: i32,
}

fn run_cmd(args: RunArgs) -> Result<()> {
    let backend: Backend = args.backend.into();
    let think_mode = if args.nothink {
        ThinkMode::None
    } else if args.think_max {
        ThinkMode::Max
    } else {
        ThinkMode::High
    };

    let memory = Engine::context_memory_estimate(backend, args.ctx_size);
    if !args.inspect {
        eprintln!(
            "ds4f run: context buffers {:.2} MiB (ctx={}, backend={}, prefill_chunk={}, raw_kv_rows={}, compressed_kv_rows={})",
            mib_string(memory.total_bytes),
            args.ctx_size,
            backend.as_str(),
            memory.prefill_cap,
            memory.raw_cap,
            memory.comp_cap
        );
    }

    let engine = Engine::open(&EngineOptions {
        model_path: args.model.clone(),
        mtp_path: args.mtp.clone(),
        backend,
        n_threads: args.threads,
        mtp_draft_tokens: args.mtp_draft_tokens,
        mtp_margin: args.mtp_margin,
        warm_weights: args.warm_weights,
        quality: args.quality,
    })?;

    if args.inspect {
        engine.summary()?;
        return Ok(());
    }

    let prompt = load_run_prompt(&args)?;
    if let Some(prompt) = prompt {
        run_one_shot(&engine, &args, think_mode, &prompt)
    } else {
        run_repl(&engine, &args, think_mode)
    }
}

fn load_run_prompt(args: &RunArgs) -> Result<Option<String>> {
    match (&args.prompt, &args.prompt_file) {
        (Some(_), Some(_)) => bail!("specify only one prompt source"),
        (Some(prompt), None) => Ok(Some(prompt.clone())),
        (None, Some(path)) => Ok(Some(
            fs::read_to_string(path).with_context(|| format!("failed to read {path}"))?,
        )),
        (None, None) => Ok(None),
    }
}

fn run_generation_options(args: &RunArgs) -> GenerationOptions {
    GenerationOptions {
        max_tokens: args.max_tokens,
        temperature: args.temperature,
        top_k: args.top_k,
        top_p: args.top_p,
        min_p: args.min_p,
        seed: args.seed,
    }
}

fn run_one_shot(engine: &Engine, args: &RunArgs, think_mode: ThinkMode, prompt: &str) -> Result<()> {
    let backend: Backend = args.backend.into();
    let prompt_tokens = if is_rendered_chat_prompt(prompt) {
        engine.tokenize_rendered_chat(prompt)?
    } else {
        engine.encode_chat_prompt(Some(&args.system), prompt, think_mode.for_context(args.ctx_size))?
    };
    if args.dump_tokens {
        engine.dump_tokens(&prompt_tokens);
    }
    let memory = Engine::context_memory_estimate(backend, args.ctx_size);
    let mut session = engine.create_rust_session(args.ctx_size as u32, memory.raw_cap);
    let result = generate_rust(engine, &mut session, &prompt_tokens, run_generation_options(args))?;
    io::stdout().write_all(&result.bytes)?;
    if !result.bytes.ends_with(b"\n") {
        println!();
    }
    log_run_stats(prompt_tokens.len(), &result);
    Ok(())
}

fn run_repl<'a>(engine: &'a Engine, args: &RunArgs, think_mode: ThinkMode) -> Result<()> {
    let interrupted = Arc::new(AtomicBool::new(false));
    {
        let interrupted = interrupted.clone();
        ctrlc::set_handler(move || {
            interrupted.store(true, Ordering::SeqCst);
        })
        .context("failed to install Ctrl+C handler")?;
    }

    let mut state = ReplState {
        system: args.system.clone(),
        messages: Vec::new(),
        think_mode,
        ctx_size: args.ctx_size,
    };
    let backend: Backend = args.backend.into();
    let memory = Engine::context_memory_estimate(backend, args.ctx_size);
    let mut session = engine.create_rust_session(args.ctx_size as u32, memory.raw_cap);
    let stdin = io::stdin();
    let mut locked = stdin.lock();

    println!("ds4f>");
    loop {
        print!("ds4f> ");
        io::stdout().flush()?;
        let mut line = String::new();
        if locked.read_line(&mut line)? == 0 {
            break;
        }
        let line = line.trim_end().to_owned();
        if line.is_empty() {
            continue;
        }

        if handle_repl_command(engine, &mut session, &mut state, &line, run_generation_options(args), args, &interrupted)? {
            continue;
        }
        if line == "/quit" || line == "/exit" {
            break;
        }

        interrupted.store(false, Ordering::SeqCst);
        let response = run_chat_turn(engine, &mut session, &state, run_generation_options(args), &line, &interrupted)?;
        state.messages.push(ChatMessage { role: "user".to_owned(), content: line });
        state.messages.push(ChatMessage { role: "assistant".to_owned(), content: response });
    }

    Ok(())
}

fn handle_repl_command<'a>(
    engine: &'a Engine,
    session: &mut RustSession<'a>,
    state: &mut ReplState,
    line: &str,
    options: GenerationOptions,
    args: &RunArgs,
    interrupted: &AtomicBool,
) -> Result<bool> {
    if line == "/help" {
        println!("/help /think /think-max /nothink /ctx N /read FILE /quit /exit");
        return Ok(true);
    }
    if line == "/think" {
        state.think_mode = ThinkMode::High;
        println!("thinking mode: high");
        return Ok(true);
    }
    if line == "/think-max" {
        state.think_mode = ThinkMode::Max;
        println!("thinking mode: max");
        return Ok(true);
    }
    if line == "/nothink" {
        state.think_mode = ThinkMode::None;
        println!("thinking mode: none");
        return Ok(true);
    }
    if let Some(rest) = line.strip_prefix("/ctx ") {
        let ctx_size: i32 = rest.trim().parse().context("invalid /ctx value")?;
        let backend: Backend = args.backend.into();
        let memory = Engine::context_memory_estimate(backend, ctx_size);
        *session = engine.create_rust_session(ctx_size as u32, memory.raw_cap);
        state.ctx_size = ctx_size;
        println!("context size: {ctx_size}");
        return Ok(true);
    }
    if let Some(path) = line.strip_prefix("/read ") {
        let contents = fs::read_to_string(path.trim())
            .with_context(|| format!("failed to read {}", path.trim()))?;
        let response = run_chat_turn(engine, session, state, options, &contents, interrupted)?;
        state.messages.push(ChatMessage { role: "user".to_owned(), content: contents });
        state.messages.push(ChatMessage { role: "assistant".to_owned(), content: response });
        return Ok(true);
    }
    Ok(false)
}

fn run_chat_turn(
    engine: &Engine,
    session: &mut RustSession<'_>,
    state: &ReplState,
    options: GenerationOptions,
    user_text: &str,
    _interrupted: &AtomicBool,
) -> Result<String> {
    let mut messages = state.messages.clone();
    messages.push(ChatMessage { role: "user".to_owned(), content: user_text.to_owned() });
    let prompt = build_chat_generation_prompt(
        engine,
        Some(&state.system),
        &messages,
        state.think_mode.for_context(state.ctx_size),
    )?;
    let result = generate_rust(engine, session, &prompt, options)?;
    io::stdout().write_all(&result.bytes)?;
    if !result.bytes.ends_with(b"\n") {
        println!();
    }
    log_run_stats(prompt.len(), &result);
    Ok(result.text_lossy())
}

fn log_run_stats(prompt_len: i32, result: &ds4_rust::GenerationResult) {
    let prefill_tps = if result.prefill_time.as_secs_f64() > 0.0 {
        prompt_len as f64 / result.prefill_time.as_secs_f64()
    } else {
        0.0
    };
    let decode_tps = if result.decode_time.as_secs_f64() > 0.0 {
        result.completion_tokens as f64 / result.decode_time.as_secs_f64()
    } else {
        0.0
    };
    eprintln!("ds4f run: prefill: {:.2} t/s, generation: {:.2} t/s", prefill_tps, decode_tps);
}

// ===========================================================================
// serve subcommand
// ===========================================================================

#[derive(Debug)]
struct ServeRuntime {
    engine: Engine,
    ctx_size: i32,
    raw_cap: u32,
    default_tokens: i32,
}

unsafe impl Send for ServeRuntime {}

type SharedRuntime = Arc<Mutex<ServeRuntime>>;

#[derive(Debug, Deserialize)]
struct OpenAiMessage {
    role: String,
    content: Value,
}

#[derive(Debug, Deserialize)]
struct OpenAiChatRequest {
    model: Option<String>,
    messages: Vec<OpenAiMessage>,
    max_tokens: Option<i32>,
    max_completion_tokens: Option<i32>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<i32>,
    min_p: Option<f32>,
    seed: Option<u64>,
    stream: Option<bool>,
    tools: Option<Value>,
    tool_choice: Option<Value>,
    thinking: Option<Value>,
}

#[derive(Debug, Deserialize)]
struct CompletionRequest {
    model: Option<String>,
    prompt: String,
    max_tokens: Option<i32>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<i32>,
    min_p: Option<f32>,
    seed: Option<u64>,
    stream: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct AnthropicMessage {
    role: String,
    content: Value,
}

#[derive(Debug, Deserialize)]
struct AnthropicRequest {
    model: Option<String>,
    system: Option<Value>,
    messages: Vec<AnthropicMessage>,
    max_tokens: i32,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<i32>,
    stream: Option<bool>,
    tools: Option<Value>,
    tool_choice: Option<Value>,
    thinking: Option<Value>,
}

async fn serve_cmd(args: ServeArgs) -> Result<()> {
    let backend = Backend::Metal;
    let memory = Engine::context_memory_estimate(backend, args.ctx_size);
    eprintln!(
        "ds4f serve: context buffers {:.2} MiB (ctx={}, backend={}, prefill_chunk={}, raw_kv_rows={}, compressed_kv_rows={})",
        mib_string(memory.total_bytes),
        args.ctx_size,
        backend.as_str(),
        memory.prefill_cap,
        memory.raw_cap,
        memory.comp_cap
    );

    let engine = Engine::open(&EngineOptions {
        model_path: args.model.clone(),
        mtp_path: args.mtp.clone(),
        backend,
        n_threads: args.threads,
        mtp_draft_tokens: args.mtp_draft_tokens,
        mtp_margin: args.mtp_margin,
        warm_weights: args.warm_weights,
        quality: args.quality,
    })?;

    let state = Arc::new(Mutex::new(ServeRuntime {
        engine,
        ctx_size: args.ctx_size,
        raw_cap: memory.raw_cap,
        default_tokens: args.default_tokens,
    }));

    let app = Router::new()
        .route("/v1/models", get(list_models))
        .route("/v1/models/deepseek-v4-flash", get(get_model))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .route("/v1/messages", post(messages))
        .with_state(state);

    let addr: SocketAddr = format!("{}:{}", args.host, args.port)
        .parse()
        .context("invalid listen address")?;
    eprintln!("ds4f serve: listening on http://{}", addr);
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

async fn list_models() -> Json<Value> {
    Json(json!({ "object": "list", "data": [model_json()] }))
}

async fn get_model() -> Json<Value> {
    Json(model_json())
}

async fn chat_completions(
    State(state): State<SharedRuntime>,
    Json(request): Json<OpenAiChatRequest>,
) -> Response {
    if request.tools.is_some() || request.tool_choice.is_some() {
        return error_response(StatusCode::BAD_REQUEST, "tool schemas not supported yet");
    }
    let stream = request.stream.unwrap_or(false);
    let result = match tokio::task::spawn_blocking(move || generate_openai_chat(state, request)).await {
        Ok(result) => result,
        Err(err) => Err(anyhow!(err)),
    };
    match result {
        Ok(payload) if !stream => Json(payload).into_response(),
        Ok(payload) => openai_stream_response(payload),
        Err(err) => error_response(StatusCode::BAD_REQUEST, &err.to_string()),
    }
}

async fn completions(
    State(state): State<SharedRuntime>,
    Json(request): Json<CompletionRequest>,
) -> Response {
    let stream = request.stream.unwrap_or(false);
    let result = match tokio::task::spawn_blocking(move || generate_completion(state, request)).await {
        Ok(result) => result,
        Err(err) => Err(anyhow!(err)),
    };
    match result {
        Ok(payload) if !stream => Json(payload).into_response(),
        Ok(payload) => openai_stream_response(payload),
        Err(err) => error_response(StatusCode::BAD_REQUEST, &err.to_string()),
    }
}

async fn messages(
    State(state): State<SharedRuntime>,
    Json(request): Json<AnthropicRequest>,
) -> Response {
    if request.tools.is_some() || request.tool_choice.is_some() {
        return error_response(StatusCode::BAD_REQUEST, "tool schemas not supported yet");
    }
    let stream = request.stream.unwrap_or(false);
    let result = match tokio::task::spawn_blocking(move || generate_anthropic(state, request)).await {
        Ok(result) => result,
        Err(err) => Err(anyhow!(err)),
    };
    match result {
        Ok(payload) if !stream => Json(payload).into_response(),
        Ok(payload) => anthropic_stream_response(payload),
        Err(err) => error_response(StatusCode::BAD_REQUEST, &err.to_string()),
    }
}

fn generate_openai_chat(state: SharedRuntime, request: OpenAiChatRequest) -> Result<Value> {
    let model = request.model.unwrap_or_else(|| "deepseek-v4-flash".to_owned());
    let think_mode = think_mode_from_value(request.thinking.as_ref());
    let messages = request.messages.into_iter()
        .map(|m| { ensure_supported_role(&m.role)?; Ok(ChatMessage { role: m.role, content: content_to_text(&m.content)? }) })
        .collect::<Result<Vec<_>>>()?;
    let max_tokens = request.max_completion_tokens.or(request.max_tokens)
        .unwrap_or_else(|| state.lock().expect("runtime poisoned").default_tokens);
    let generated = with_runtime(&state, |rt| {
        let prompt = build_chat_generation_prompt(&rt.engine, None, &messages, think_mode.for_context(rt.ctx_size))?;
        let options = GenerationOptions {
            max_tokens,
            temperature: request.temperature.unwrap_or(1.0),
            top_k: request.top_k.unwrap_or(0),
            top_p: request.top_p.unwrap_or(1.0),
            min_p: request.min_p.unwrap_or(0.0),
            seed: request.seed,
        };
        let result = serve_generate(rt, &prompt, options)?;
        Ok((prompt.len(), result.text_lossy(), result.completion_tokens))
    })?;
    Ok(json!({
        "id": request_id("chatcmpl"), "object": "chat.completion",
        "created": unix_time(), "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": generated.1}, "finish_reason": "stop"}],
        "usage": usage_json(generated.0, generated.2),
    }))
}

fn generate_completion(state: SharedRuntime, request: CompletionRequest) -> Result<Value> {
    let model = request.model.unwrap_or_else(|| "deepseek-v4-flash".to_owned());
    let generated = with_runtime(&state, |rt| {
        let prompt = if is_rendered_chat_prompt(&request.prompt) {
            rt.engine.tokenize_rendered_chat(&request.prompt)?
        } else {
            rt.engine.tokenize_text(&request.prompt)?
        };
        let options = GenerationOptions {
            max_tokens: request.max_tokens.unwrap_or(rt.default_tokens),
            temperature: request.temperature.unwrap_or(1.0),
            top_k: request.top_k.unwrap_or(0),
            top_p: request.top_p.unwrap_or(1.0),
            min_p: request.min_p.unwrap_or(0.0),
            seed: request.seed,
        };
        let result = serve_generate(rt, &prompt, options)?;
        Ok((prompt.len(), result.text_lossy(), result.completion_tokens))
    })?;
    Ok(json!({
        "id": request_id("cmpl"), "object": "text_completion",
        "created": unix_time(), "model": model,
        "choices": [{"index": 0, "text": generated.1, "finish_reason": "stop"}],
        "usage": usage_json(generated.0, generated.2),
    }))
}

fn generate_anthropic(state: SharedRuntime, request: AnthropicRequest) -> Result<Value> {
    let model = request.model.unwrap_or_else(|| "deepseek-v4-flash".to_owned());
    let think_mode = think_mode_from_value(request.thinking.as_ref());
    let system = request.system.as_ref().map(content_to_text).transpose()?;
    let messages = request.messages.into_iter()
        .map(|m| { ensure_supported_role(&m.role)?; Ok(ChatMessage { role: m.role, content: content_to_text(&m.content)? }) })
        .collect::<Result<Vec<_>>>()?;
    let generated = with_runtime(&state, |rt| {
        let prompt = build_chat_generation_prompt(&rt.engine, system.as_deref(), &messages, think_mode.for_context(rt.ctx_size))?;
        let options = GenerationOptions {
            max_tokens: request.max_tokens,
            temperature: request.temperature.unwrap_or(1.0),
            top_k: request.top_k.unwrap_or(0),
            top_p: request.top_p.unwrap_or(1.0),
            min_p: 0.0, seed: None,
        };
        let result = serve_generate(rt, &prompt, options)?;
        Ok((prompt.len(), result.text_lossy(), result.completion_tokens))
    })?;
    Ok(json!({
        "id": request_id("msg"), "type": "message", "role": "assistant", "model": model,
        "content": [{"type": "text", "text": generated.1}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": generated.0, "output_tokens": generated.2},
    }))
}

fn with_runtime<T>(state: &SharedRuntime, f: impl FnOnce(&mut ServeRuntime) -> Result<T>) -> Result<T> {
    let mut rt = state.lock().map_err(|_| anyhow!("runtime poisoned"))?;
    f(&mut rt)
}

fn serve_generate(rt: &mut ServeRuntime, prompt: &ds4_rust::Tokens, options: GenerationOptions) -> Result<ds4_rust::GenerationResult> {
    let mut session = rt.engine.create_rust_session(rt.ctx_size as u32, rt.raw_cap);
    generate_rust(&rt.engine, &mut session, prompt, options)
}

fn think_mode_from_value(value: Option<&Value>) -> ThinkMode {
    match value {
        Some(Value::Bool(false)) => ThinkMode::None,
        Some(Value::Object(map)) if map.get("enabled") == Some(&Value::Bool(false)) => ThinkMode::None,
        _ => ThinkMode::High,
    }
}

fn content_to_text(value: &Value) -> Result<String> {
    match value {
        Value::String(text) => Ok(text.clone()),
        Value::Array(blocks) => {
            let mut parts = Vec::new();
            for block in blocks {
                if let Some(text) = block.get("text").and_then(Value::as_str) {
                    parts.push(text.to_owned()); continue;
                }
                if let Some(text) = block.get("content").and_then(Value::as_str) {
                    parts.push(text.to_owned()); continue;
                }
                bail!("unsupported content block")
            }
            Ok(parts.join("\n\n"))
        }
        Value::Null => Ok(String::new()),
        _ => bail!("unsupported content shape"),
    }
}

fn model_json() -> Value {
    json!({ "id": "deepseek-v4-flash", "object": "model", "owned_by": "ds4f" })
}

fn usage_json(prompt_tokens: i32, completion_tokens: i32) -> Value {
    json!({ "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": prompt_tokens + completion_tokens })
}

fn error_response(status: StatusCode, message: &str) -> Response {
    (status, Json(json!({"error": {"message": message}}))).into_response()
}

fn openai_stream_response(payload: Value) -> Response {
    let model = payload.get("model").and_then(Value::as_str).unwrap_or("deepseek-v4-flash").to_owned();
    let id = payload.get("id").and_then(Value::as_str).unwrap_or("chatcmpl-rs").to_owned();
    let text = payload["choices"][0]["message"]["content"].as_str().unwrap_or("").to_owned();
    let created = payload.get("created").and_then(Value::as_i64).unwrap_or(unix_time() as i64);
    let events: Vec<Result<Event, std::convert::Infallible>> = vec![
        Ok(Event::default().data(json!({"id": id, "object": "chat.completion.chunk", "created": created, "model": model, "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": null}]}).to_string())),
        Ok(Event::default().data(json!({"id": payload["id"], "object": "chat.completion.chunk", "created": created, "model": payload["model"], "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": null}]}).to_string())),
        Ok(Event::default().data(json!({"id": payload["id"], "object": "chat.completion.chunk", "created": created, "model": payload["model"], "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}).to_string())),
        Ok(Event::default().data("[DONE]")),
    ];
    Sse::new(stream::iter(events)).keep_alive(KeepAlive::default()).into_response()
}

fn anthropic_stream_response(payload: Value) -> Response {
    let id = payload.get("id").and_then(Value::as_str).unwrap_or("msg-rs").to_owned();
    let model = payload.get("model").and_then(Value::as_str).unwrap_or("deepseek-v4-flash").to_owned();
    let text = payload["content"][0]["text"].as_str().unwrap_or("").to_owned();
    let events: Vec<Result<Event, std::convert::Infallible>> = vec![
        Ok(Event::default().event("message_start").data(json!({"type": "message_start", "message": {"id": id, "type": "message", "role": "assistant", "model": model, "content": [], "stop_reason": null, "stop_sequence": null, "usage": {"input_tokens": payload["usage"]["input_tokens"], "output_tokens": 0}}}).to_string())),
        Ok(Event::default().event("content_block_start").data(json!({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}).to_string())),
        Ok(Event::default().event("content_block_delta").data(json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}}).to_string())),
        Ok(Event::default().event("content_block_stop").data(json!({"type": "content_block_stop", "index": 0}).to_string())),
        Ok(Event::default().event("message_delta").data(json!({"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": null}, "usage": {"output_tokens": payload["usage"]["output_tokens"]}}).to_string())),
        Ok(Event::default().event("message_stop").data(json!({"type": "message_stop"}).to_string())),
    ];
    Sse::new(stream::iter(events)).keep_alive(KeepAlive::default()).into_response()
}

fn unix_time() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

fn request_id(prefix: &str) -> String {
    format!("{}-{}", prefix, unix_time())
}

// ===========================================================================
// download subcommand
// ===========================================================================

fn download_cmd(args: DownloadArgs) -> Result<()> {
    let token = args.token.or_else(hf_token_from_env).or_else(read_local_hf_token);
    let api = build_hf_api(token)?;
    let repo = api.model(REPO.to_owned());
    let filename = args.model.filename();

    println!("Downloading {filename}");
    println!("from https://huggingface.co/{REPO}");
    let cached_path = repo.get(filename)
        .with_context(|| format!("failed to fetch {filename} from Hugging Face"))?;

    let repo_entry = ensure_repo_entry(filename, &cached_path)?;

    if args.model == ModelArg::Mtp {
        println!();
        println!("MTP is an optional component for both q2 and q4.");
        println!("Enable it explicitly, for example:");
        println!("  ds4f run --mtp gguf/{MTP_FILE} --mtp-draft 2");
    } else {
        update_default_model_link(filename)?;
        println!("Linked ./ds4flash.gguf -> gguf/{filename}");
    }

    println!();
    println!("Ready: {}", repo_entry.display());
    println!("Done.");
    Ok(())
}

fn build_hf_api(token: Option<String>) -> Result<hf_hub::api::sync::Api> {
    let mut builder = ApiBuilder::from_env()
        .with_progress(true)
        .with_retries(3)
        .with_user_agent("ds4f", env!("CARGO_PKG_VERSION"));
    if let Some(token) = token {
        builder = builder.with_token(Some(token));
    }
    builder.build().context("failed to initialize Hugging Face API client")
}

fn hf_token_from_env() -> Option<String> {
    env::var("HF_TOKEN").ok().map(|t| t.trim().to_owned()).filter(|t| !t.is_empty())
}

fn read_local_hf_token() -> Option<String> {
    let home = env::var_os("HOME")?;
    let token_path = PathBuf::from(home).join(".cache/huggingface/token");
    let token = fs::read_to_string(token_path).ok()?;
    let token = token.trim();
    if token.is_empty() { None } else { Some(token.to_owned()) }
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn ensure_repo_entry(filename: &str, cached_path: &Path) -> Result<PathBuf> {
    let repo_entry = repo_root().join("gguf").join(filename);
    fs::create_dir_all(repo_entry.parent().expect("gguf output dir"))
        .with_context(|| format!("failed to create {}", repo_entry.parent().unwrap().display()))?;

    if path_present(&repo_entry) {
        if repo_entry.exists() {
            println!("Already available: {}", repo_entry.display());
            return Ok(repo_entry);
        }
        fs::remove_file(&repo_entry)
            .with_context(|| format!("failed to remove broken link {}", repo_entry.display()))?;
    }

    let cached_path = fs::canonicalize(cached_path)
        .with_context(|| format!("failed to resolve Hugging Face cache path {}", cached_path.display()))?;
    create_symlink(&cached_path, &repo_entry)
        .with_context(|| format!("failed to link {} to {}", repo_entry.display(), cached_path.display()))?;
    Ok(repo_entry)
}

fn update_default_model_link(filename: &str) -> Result<()> {
    let link_path = repo_root().join("ds4flash.gguf");
    if path_present(&link_path) {
        fs::remove_file(&link_path)
            .with_context(|| format!("failed to replace {}", link_path.display()))?;
    }
    let relative_target = PathBuf::from("gguf").join(filename);
    create_symlink(&relative_target, &link_path)
        .with_context(|| format!("failed to link {} to {}", link_path.display(), relative_target.display()))
}

fn path_present(path: &Path) -> bool {
    fs::symlink_metadata(path).is_ok()
}

#[cfg(target_family = "unix")]
fn create_symlink(target: &Path, link: &Path) -> std::io::Result<()> {
    std::os::unix::fs::symlink(target, link)
}

#[cfg(target_family = "windows")]
fn create_symlink(target: &Path, link: &Path) -> std::io::Result<()> {
    std::os::windows::fs::symlink_file(target, link)
}

#[cfg(not(any(target_family = "unix", target_family = "windows")))]
fn create_symlink(target: &Path, link: &Path) -> std::io::Result<()> {
    fs::hard_link(target, link)
}
