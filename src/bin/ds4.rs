use std::fs;
use std::io::{self, BufRead, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use clap::{Parser, ValueEnum};
use ds4_rust::{
    build_chat_generation_prompt, generate, generate_rust, is_rendered_chat_prompt, mib_string,
    Backend, ChatMessage, Engine, EngineOptions, GenerationOptions, RustSession, Session,
    ThinkMode,
};

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

#[derive(Debug, Parser)]
#[command(name = "ds4-rs")]
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
    #[arg(long = "rust-session", default_value_t = false)]
    rust_session: bool,
}

#[derive(Clone, Debug)]
struct ReplState {
    system: String,
    messages: Vec<ChatMessage>,
    think_mode: ThinkMode,
    ctx_size: i32,
}

enum CliSession<'a> {
    Ffi(Session),
    Rust(RustSession<'a>),
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let requested_backend: Backend = cli.backend.into();
    let backend = effective_session_backend(requested_backend, cli.rust_session);
    let think_mode = if cli.nothink {
        ThinkMode::None
    } else if cli.think_max {
        ThinkMode::Max
    } else {
        ThinkMode::High
    };

    let memory = Engine::context_memory_estimate(backend, cli.ctx_size);
    if !cli.inspect {
        eprintln!(
            "ds4-rs: context buffers {:.2} MiB (ctx={}, backend={}, prefill_chunk={}, raw_kv_rows={}, compressed_kv_rows={})",
            mib_string(memory.total_bytes),
            cli.ctx_size,
            backend.as_str(),
            memory.prefill_cap,
            memory.raw_cap,
            memory.comp_cap
        );
    }

    let engine_options = EngineOptions {
        model_path: cli.model.clone(),
        mtp_path: cli.mtp.clone(),
        backend,
        n_threads: cli.threads,
        mtp_draft_tokens: cli.mtp_draft_tokens,
        mtp_margin: cli.mtp_margin,
        warm_weights: cli.warm_weights,
        quality: cli.quality,
    };
    let engine = Engine::open(&engine_options)?;

    if cli.rust_session {
        warn_rust_session_backend(requested_backend, backend);
    }

    if cli.inspect {
        engine.summary()?;
        return Ok(());
    }

    let prompt = load_prompt(&cli)?;
    if let Some(prompt) = prompt {
        run_one_shot(&engine, &cli, think_mode, &prompt)
    } else {
        run_repl(&engine, &cli, think_mode)
    }
}

fn load_prompt(cli: &Cli) -> Result<Option<String>> {
    match (&cli.prompt, &cli.prompt_file) {
        (Some(_), Some(_)) => bail!("specify only one prompt source"),
        (Some(prompt), None) => Ok(Some(prompt.clone())),
        (None, Some(path)) => Ok(Some(fs::read_to_string(path).with_context(|| format!("failed to read {path}"))?)),
        (None, None) => Ok(None),
    }
}

fn generation_options(cli: &Cli) -> GenerationOptions {
    GenerationOptions {
        max_tokens: cli.max_tokens,
        temperature: cli.temperature,
        top_k: cli.top_k,
        top_p: cli.top_p,
        min_p: cli.min_p,
        seed: cli.seed,
    }
}

fn effective_session_backend(requested_backend: Backend, use_rust_session: bool) -> Backend {
    let _ = use_rust_session;
    requested_backend
}

fn warn_rust_session_backend(requested_backend: Backend, effective_backend: Backend) {
    if requested_backend != effective_backend {
        eprintln!(
            "ds4-rs: --rust-session: requested backend '{}' but using '{}' backend setup.",
            requested_backend.as_str(),
            effective_backend.as_str()
        );
    }
}

fn effective_think_mode(think_mode: ThinkMode, ctx_size: i32) -> ThinkMode {
    think_mode.for_context(ctx_size)
}

fn run_one_shot(engine: &Engine, cli: &Cli, think_mode: ThinkMode, prompt: &str) -> Result<()> {
    let prompt_tokens = if is_rendered_chat_prompt(prompt) {
        engine.tokenize_rendered_chat(prompt)?
    } else {
        engine.encode_chat_prompt(Some(&cli.system), prompt, effective_think_mode(think_mode, cli.ctx_size))?
    };

    if cli.dump_tokens {
        engine.dump_tokens(&prompt_tokens);
    }

    let mut session = create_cli_session(
        engine,
        cli.ctx_size,
        effective_session_backend(cli.backend.into(), cli.rust_session),
        cli.rust_session,
    )?;
    let result = run_generation(engine, &mut session, &prompt_tokens, generation_options(cli))?;
    io::stdout().write_all(&result.bytes)?;
    if !result.bytes.ends_with(b"\n") {
        println!();
    }
    log_generation_stats(prompt_tokens.len(), &result);
    Ok(())
}

fn run_repl<'a>(engine: &'a Engine, cli: &Cli, think_mode: ThinkMode) -> Result<()> {
    let interrupted = Arc::new(AtomicBool::new(false));
    {
        let interrupted = interrupted.clone();
        ctrlc::set_handler(move || {
            interrupted.store(true, Ordering::SeqCst);
        })
        .context("failed to install Ctrl+C handler")?;
    }

    let mut state = ReplState {
        system: cli.system.clone(),
        messages: Vec::new(),
        think_mode,
        ctx_size: cli.ctx_size,
    };
    let backend = effective_session_backend(cli.backend.into(), cli.rust_session);
    let mut session = create_cli_session(engine, state.ctx_size, backend, cli.rust_session)?;
    let stdin = io::stdin();
    let mut locked = stdin.lock();

    println!("ds4>");
    loop {
        print!("ds4> ");
        io::stdout().flush()?;
        let mut line = String::new();
        if locked.read_line(&mut line)? == 0 {
            break;
        }
        let line = line.trim_end().to_owned();
        if line.is_empty() {
            continue;
        }

        if handle_repl_command(
            engine,
            &mut session,
            &mut state,
            &line,
            generation_options(cli),
            backend,
            cli.rust_session,
            interrupted.as_ref(),
        )? {
            continue;
        }
        if line == "/quit" || line == "/exit" {
            break;
        }

        interrupted.store(false, Ordering::SeqCst);
        let response = run_chat_turn(engine, &mut session, &state, generation_options(cli), &line, &interrupted)?;
        state.messages.push(ChatMessage {
            role: "user".to_owned(),
            content: line,
        });
        state.messages.push(ChatMessage {
            role: "assistant".to_owned(),
            content: response,
        });
    }

    Ok(())
}

fn handle_repl_command<'a>(
    engine: &'a Engine,
    session: &mut CliSession<'a>,
    state: &mut ReplState,
    line: &str,
    options: GenerationOptions,
    backend: Backend,
    use_rust_session: bool,
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
        *session = create_cli_session(engine, ctx_size, backend, use_rust_session)?;
        state.ctx_size = ctx_size;
        println!("context size: {ctx_size}");
        return Ok(true);
    }
    if let Some(path) = line.strip_prefix("/read ") {
        let contents = fs::read_to_string(path.trim())
            .with_context(|| format!("failed to read {}", path.trim()))?;
        let response = run_chat_turn(engine, session, state, options, &contents, interrupted)?;
        state.messages.push(ChatMessage {
            role: "user".to_owned(),
            content: contents,
        });
        state.messages.push(ChatMessage {
            role: "assistant".to_owned(),
            content: response,
        });
        return Ok(true);
    }
    Ok(false)
}

fn run_chat_turn(
    engine: &Engine,
    session: &mut CliSession<'_>,
    state: &ReplState,
    options: GenerationOptions,
    user_text: &str,
    interrupted: &AtomicBool,
) -> Result<String> {
    let mut messages = state.messages.clone();
    messages.push(ChatMessage {
        role: "user".to_owned(),
        content: user_text.to_owned(),
    });
    let prompt = build_chat_generation_prompt(
        engine,
        Some(&state.system),
        &messages,
        effective_think_mode(state.think_mode, state.ctx_size),
    )?;
    let result = generate_interruptible(engine, session, &prompt, options, interrupted)?;
    io::stdout().write_all(&result.bytes)?;
    if !result.bytes.ends_with(b"\n") {
        println!();
    }
    log_generation_stats(prompt.len(), &result);
    Ok(result.text_lossy())
}

fn generate_interruptible(
    engine: &Engine,
    session: &mut CliSession<'_>,
    prompt: &ds4_rust::Tokens,
    options: GenerationOptions,
    interrupted: &AtomicBool,
) -> Result<ds4_rust::GenerationResult> {
    if !interrupted.load(Ordering::SeqCst) {
        return run_generation(engine, session, prompt, options);
    }
    interrupted.store(false, Ordering::SeqCst);
    run_generation(engine, session, prompt, options)
}

fn create_cli_session<'a>(
    engine: &'a Engine,
    ctx_size: i32,
    backend: Backend,
    use_rust_session: bool,
) -> Result<CliSession<'a>> {
    if use_rust_session {
        let memory = Engine::context_memory_estimate(backend, ctx_size);
        Ok(CliSession::Rust(
            engine.create_rust_session(ctx_size as u32, memory.raw_cap),
        ))
    } else {
        Ok(CliSession::Ffi(Session::create(engine, ctx_size)?))
    }
}

fn run_generation(
    engine: &Engine,
    session: &mut CliSession<'_>,
    prompt: &ds4_rust::Tokens,
    options: GenerationOptions,
) -> Result<ds4_rust::GenerationResult> {
    match session {
        CliSession::Ffi(session) => generate(engine, session, prompt, options),
        CliSession::Rust(session) => generate_rust(engine, session, prompt, options),
    }
}

fn log_generation_stats(prompt_len: i32, result: &ds4_rust::GenerationResult) {
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
    eprintln!(
        "ds4-rs: prefill: {:.2} t/s, generation: {:.2} t/s",
        prefill_tps, decode_tps
    );
}