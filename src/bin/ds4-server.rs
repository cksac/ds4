use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, bail, Context, Result};
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use clap::Parser;
use ds4_rust::{
    build_chat_generation_prompt, ensure_supported_role, generate, generate_rust, is_rendered_chat_prompt,
    mib_string, Backend, ChatMessage, Engine, EngineOptions, GenerationOptions, Session,
    ThinkMode,
};
use futures_util::stream;
use serde::Deserialize;
use serde_json::{json, Value};

fn warn_rust_session_backend() {
    eprintln!(
        "ds4-server-rs: --rust-session uses the Rust Metal decode path.",
    );
}

fn effective_server_backend(use_rust_session: bool) -> Backend {
    let _ = use_rust_session;
    Backend::Metal
}

#[derive(Debug, Parser)]
#[command(name = "ds4-server-rs")]
struct Cli {
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
    #[arg(long = "rust-session", default_value_t = false)]
    rust_session: bool,
}

#[derive(Debug)]
struct Runtime {
    session: Option<Session>,
    engine: Engine,
    ctx_size: i32,
    raw_cap: u32,
    default_tokens: i32,
    rust_session: bool,
}

unsafe impl Send for Runtime {}

type SharedRuntime = Arc<Mutex<Runtime>>;

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

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let backend = effective_server_backend(cli.rust_session);
    let memory = Engine::context_memory_estimate(backend, cli.ctx_size);
    eprintln!(
        "ds4-server-rs: context buffers {:.2} MiB (ctx={}, backend={}, prefill_chunk={}, raw_kv_rows={}, compressed_kv_rows={})",
        mib_string(memory.total_bytes),
        cli.ctx_size,
        backend.as_str(),
        memory.prefill_cap,
        memory.raw_cap,
        memory.comp_cap
    );

    let engine = Engine::open(&EngineOptions {
        model_path: cli.model.clone(),
        mtp_path: cli.mtp.clone(),
        backend,
        n_threads: cli.threads,
        mtp_draft_tokens: cli.mtp_draft_tokens,
        mtp_margin: cli.mtp_margin,
        warm_weights: cli.warm_weights,
        quality: cli.quality,
    })?;
    if cli.rust_session {
        warn_rust_session_backend();
    }
    let session = if cli.rust_session {
        None
    } else {
        Some(Session::create(&engine, cli.ctx_size)?)
    };
    let state = Arc::new(Mutex::new(Runtime {
        session,
        engine,
        ctx_size: cli.ctx_size,
        raw_cap: memory.raw_cap,
        default_tokens: cli.default_tokens,
        rust_session: cli.rust_session,
    }));

    let app = Router::new()
        .route("/v1/models", get(list_models))
        .route("/v1/models/deepseek-v4-flash", get(get_model))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .route("/v1/messages", post(messages))
        .with_state(state);

    let addr: SocketAddr = format!("{}:{}", cli.host, cli.port)
        .parse()
        .context("invalid listen address")?;
    eprintln!("ds4-server-rs: listening on http://{}", addr);
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

async fn list_models() -> Json<Value> {
    Json(json!({
        "object": "list",
        "data": [model_json()],
    }))
}

async fn get_model() -> Json<Value> {
    Json(model_json())
}

async fn chat_completions(
    State(state): State<SharedRuntime>,
    Json(request): Json<OpenAiChatRequest>,
) -> Response {
    if request.tools.is_some() || request.tool_choice.is_some() {
        return error_response(StatusCode::BAD_REQUEST, "Rust server does not support tool schemas yet");
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
        return error_response(StatusCode::BAD_REQUEST, "Rust server does not support tool schemas yet");
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
    let messages = request
        .messages
        .into_iter()
        .map(|message| {
            ensure_supported_role(&message.role)?;
            Ok(ChatMessage {
                role: message.role,
                content: content_to_text(&message.content)?,
            })
        })
        .collect::<Result<Vec<_>>>()?;

    let max_tokens = request
        .max_completion_tokens
        .or(request.max_tokens)
        .unwrap_or_else(|| state.lock().expect("runtime poisoned").default_tokens);
    let generated = with_runtime(&state, |runtime| {
        let prompt = build_chat_generation_prompt(
            &runtime.engine,
            None,
            &messages,
            think_mode.for_context(runtime.ctx_size),
        )?;
        let options = GenerationOptions {
            max_tokens,
            temperature: request.temperature.unwrap_or(1.0),
            top_k: request.top_k.unwrap_or(0),
            top_p: request.top_p.unwrap_or(1.0),
            min_p: request.min_p.unwrap_or(0.0),
            seed: request.seed,
        };
        let result = generate_runtime(runtime, &prompt, options)?;
        Ok((prompt.len(), result.text_lossy(), result.completion_tokens))
    })?;

    Ok(json!({
        "id": request_id("chatcmpl"),
        "object": "chat.completion",
        "created": unix_time(),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": generated.1,
            },
            "finish_reason": "stop"
        }],
        "usage": usage_json(generated.0, generated.2),
    }))
}

fn generate_completion(state: SharedRuntime, request: CompletionRequest) -> Result<Value> {
    let model = request.model.unwrap_or_else(|| "deepseek-v4-flash".to_owned());
    let generated = with_runtime(&state, |runtime| {
        let prompt = if is_rendered_chat_prompt(&request.prompt) {
            runtime.engine.tokenize_rendered_chat(&request.prompt)?
        } else {
            runtime.engine.tokenize_text(&request.prompt)?
        };
        let options = GenerationOptions {
            max_tokens: request.max_tokens.unwrap_or(runtime.default_tokens),
            temperature: request.temperature.unwrap_or(1.0),
            top_k: request.top_k.unwrap_or(0),
            top_p: request.top_p.unwrap_or(1.0),
            min_p: request.min_p.unwrap_or(0.0),
            seed: request.seed,
        };
        let result = generate_runtime(runtime, &prompt, options)?;
        Ok((prompt.len(), result.text_lossy(), result.completion_tokens))
    })?;

    Ok(json!({
        "id": request_id("cmpl"),
        "object": "text_completion",
        "created": unix_time(),
        "model": model,
        "choices": [{
            "index": 0,
            "text": generated.1,
            "finish_reason": "stop"
        }],
        "usage": usage_json(generated.0, generated.2),
    }))
}

fn generate_anthropic(state: SharedRuntime, request: AnthropicRequest) -> Result<Value> {
    let model = request.model.unwrap_or_else(|| "deepseek-v4-flash".to_owned());
    let think_mode = think_mode_from_value(request.thinking.as_ref());
    let system = request
        .system
        .as_ref()
        .map(content_to_text)
        .transpose()?;
    let messages = request
        .messages
        .into_iter()
        .map(|message| {
            ensure_supported_role(&message.role)?;
            Ok(ChatMessage {
                role: message.role,
                content: content_to_text(&message.content)?,
            })
        })
        .collect::<Result<Vec<_>>>()?;

    let generated = with_runtime(&state, |runtime| {
        let prompt = build_chat_generation_prompt(
            &runtime.engine,
            system.as_deref(),
            &messages,
            think_mode.for_context(runtime.ctx_size),
        )?;
        let options = GenerationOptions {
            max_tokens: request.max_tokens,
            temperature: request.temperature.unwrap_or(1.0),
            top_k: request.top_k.unwrap_or(0),
            top_p: request.top_p.unwrap_or(1.0),
            min_p: 0.0,
            seed: None,
        };
        let result = generate_runtime(runtime, &prompt, options)?;
        Ok((prompt.len(), result.text_lossy(), result.completion_tokens))
    })?;

    Ok(json!({
        "id": request_id("msg"),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{
            "type": "text",
            "text": generated.1,
        }],
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": generated.0,
            "output_tokens": generated.2,
        }
    }))
}

fn with_runtime<T>(state: &SharedRuntime, f: impl FnOnce(&mut Runtime) -> Result<T>) -> Result<T> {
    let mut runtime = state.lock().map_err(|_| anyhow!("runtime poisoned"))?;
    f(&mut runtime)
}

fn generate_runtime(
    runtime: &mut Runtime,
    prompt: &ds4_rust::Tokens,
    options: GenerationOptions,
) -> Result<ds4_rust::GenerationResult> {
    if runtime.rust_session {
        let mut session = runtime
            .engine
            .create_rust_session(runtime.ctx_size as u32, runtime.raw_cap);
        generate_rust(&runtime.engine, &mut session, prompt, options)
    } else {
        let session = runtime.session.as_mut().context("missing FFI session")?;
        generate(&runtime.engine, session, prompt, options)
    }
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
                    parts.push(text.to_owned());
                    continue;
                }
                if let Some(text) = block.get("content").and_then(Value::as_str) {
                    parts.push(text.to_owned());
                    continue;
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
    json!({
        "id": "deepseek-v4-flash",
        "object": "model",
        "owned_by": "ds4-rust"
    })
}

fn usage_json(prompt_tokens: i32, completion_tokens: i32) -> Value {
    json!({
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    })
}

fn error_response(status: StatusCode, message: &str) -> Response {
    (
        status,
        Json(json!({
            "error": {
                "message": message,
            }
        })),
    )
        .into_response()
}

fn openai_stream_response(payload: Value) -> Response {
    let model = payload
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or("deepseek-v4-flash")
        .to_owned();
    let id = payload
        .get("id")
        .and_then(Value::as_str)
        .unwrap_or("chatcmpl-rs")
        .to_owned();
    let text = payload["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("")
        .to_owned();
    let created = payload.get("created").and_then(Value::as_i64).unwrap_or(unix_time() as i64);
    let events: Vec<Result<Event, std::convert::Infallible>> = vec![
        Ok(Event::default().data(json!({
            "id": id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": null}],
        }).to_string())),
        Ok(Event::default().data(json!({
            "id": payload["id"],
            "object": "chat.completion.chunk",
            "created": created,
            "model": payload["model"],
            "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": null}],
        }).to_string())),
        Ok(Event::default().data(json!({
            "id": payload["id"],
            "object": "chat.completion.chunk",
            "created": created,
            "model": payload["model"],
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }).to_string())),
        Ok(Event::default().data("[DONE]")),
    ];
    Sse::new(stream::iter(events))
        .keep_alive(KeepAlive::default())
        .into_response()
}

fn anthropic_stream_response(payload: Value) -> Response {
    let id = payload.get("id").and_then(Value::as_str).unwrap_or("msg-rs").to_owned();
    let model = payload
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or("deepseek-v4-flash")
        .to_owned();
    let text = payload["content"][0]["text"].as_str().unwrap_or("").to_owned();
    let events: Vec<Result<Event, std::convert::Infallible>> = vec![
        Ok(Event::default().event("message_start").data(json!({
            "type": "message_start",
            "message": {
                "id": id,
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [],
                "stop_reason": null,
                "stop_sequence": null,
                "usage": {"input_tokens": payload["usage"]["input_tokens"], "output_tokens": 0}
            }
        }).to_string())),
        Ok(Event::default().event("content_block_start").data(json!({
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""}
        }).to_string())),
        Ok(Event::default().event("content_block_delta").data(json!({
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": text}
        }).to_string())),
        Ok(Event::default().event("content_block_stop").data(json!({
            "type": "content_block_stop",
            "index": 0
        }).to_string())),
        Ok(Event::default().event("message_delta").data(json!({
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": null},
            "usage": {"output_tokens": payload["usage"]["output_tokens"]}
        }).to_string())),
        Ok(Event::default().event("message_stop").data(json!({
            "type": "message_stop"
        }).to_string())),
    ];
    Sse::new(stream::iter(events))
        .keep_alive(KeepAlive::default())
        .into_response()
}

fn unix_time() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn request_id(prefix: &str) -> String {
    format!("{}-{}", prefix, unix_time())
}