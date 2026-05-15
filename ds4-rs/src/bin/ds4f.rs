//! ds4f - Unified DS4 CLI: run and serve

use std::path::Path;
use std::sync::Mutex;
use clap::{Parser, Subcommand, Args};

const DEFAULT_MODEL: &str = "ds4flash.gguf";

#[derive(Parser)]
#[command(name = "ds4f", about = "DS4 inference engine CLI", version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run inference with a prompt
    Run(RunArgs),
    /// Start HTTP server
    Serve(ServeArgs),
}

#[derive(Args)]
struct RunArgs {
    /// Model GGUF path
    #[arg(short = 'm', long = "model", default_value = DEFAULT_MODEL)]
    model: String,

    /// Context size
    #[arg(short = 'c', long = "ctx-size", default_value_t = 32768u32)]
    ctx_size: u32,

    /// Maximum tokens to generate
    #[arg(short = 'n', long = "n-predict", default_value_t = 50000usize)]
    n_predict: usize,

    /// Sampling temperature (0 = greedy)
    #[arg(short = 't', long = "temperature", default_value_t = 1.0)]
    temperature: f32,

    /// Top-p sampling
    #[arg(long = "top-p", default_value_t = 1.0)]
    top_p: f32,

    /// Top-k sampling (0 = disabled)
    #[arg(long = "top-k", default_value_t = 0usize)]
    top_k: usize,

    /// Random seed (0 = random)
    #[arg(long = "seed", default_value_t = 0u64)]
    seed: u64,

    /// System prompt
    #[arg(long = "system", default_value = "You are a helpful assistant")]
    system: String,

    /// Prompt text
    #[arg(short = 'p', long = "prompt")]
    prompt: Option<String>,

    /// Read prompt from file
    #[arg(long = "prompt-file")]
    prompt_file: Option<String>,

    /// Enable thinking mode
    #[arg(long = "think", default_value_t = true)]
    think: bool,

    /// Disable thinking mode
    #[arg(long = "nothink", overrides_with = "think")]
    nothink: bool,

    /// Print model info and exit
    #[arg(long = "inspect")]
    inspect: bool,

    /// Interactive REPL mode
    #[arg(short = 'i', long = "interactive")]
    interactive: bool,

    /// Prompt (positional)
    #[arg()]
    positional_prompt: Option<String>,
}

#[derive(Args)]
struct ServeArgs {
    /// Model GGUF path
    #[arg(short = 'm', long = "model", default_value = DEFAULT_MODEL)]
    model: String,

    /// HTTP port
    #[arg(short = 'p', long = "port", default_value_t = 8080u16)]
    port: u16,

    /// Context size
    #[arg(short = 'c', long = "ctx-size", default_value_t = 65536u32)]
    ctx_size: u32,

    /// Bind address
    #[arg(long = "host", default_value = "0.0.0.0")]
    host: String,

    /// Disable thinking globally
    #[arg(long = "nothink")]
    nothink: bool,
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Commands::Run(args) => cmd_run(args),
        Commands::Serve(args) => cmd_serve(args),
    }
}

// ── Model loading ─────────────────────────────────────────────────────────

fn init_metal() {
    if let Err(e) = ds4_rs::bridge::init() {
        eprintln!("ERROR: Metal init failed: {}", e);
        eprintln!("  Try setting DS4_NO_GPU=1 for CPU mode.");
        std::process::exit(1);
    }
    let ram_gib = std::process::Command::new("sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.trim().parse::<u64>().ok())
        .map(|b| b as f64 / (1024.0 * 1024.0 * 1024.0))
        .unwrap_or(0.0);
    let device_name = ds4_rs::bridge::device_name();
    eprintln!("ds4: Metal device {}, {:.2} GiB RAM", device_name, ram_gib);
}

fn load_model(model_path: &str, ctx_size: u32, quiet: bool) -> ds4_rs::session::SessionState {
    if !quiet { eprintln!("ds4: loading model: {}", model_path); }
    if Path::new(model_path).exists() {
        let t0 = std::time::Instant::now();
        match ds4_rs::gguf::GgufModel::open(Path::new(model_path)) {
            Ok(model) => {
                let t_map_ms = t0.elapsed().as_secs_f64() * 1000.0;
                let size_mib = model.size as f64 / (1024.0 * 1024.0);
                let offset_mib = model.tensor_data_pos as f64 / (1024.0 * 1024.0);
                if !quiet {
                    eprintln!("ds4: model views created in {:.3} ms (mapped {:.2} MiB from offset {:.2} MiB, tensors={} kv={})",
                        t_map_ms, size_mib, offset_mib, model.n_tensors, model.n_kv);
                }
                match ds4_rs::session::SessionState::from_model(&model) {
                    Ok(s) => {
                        if !quiet {
                            eprintln!("ds4: context buffers (ctx={}, backend=metal, prefill_chunk={}, raw_kv_rows={}, compressed_kv_rows={})",
                                ctx_size, s.prefill_cap, s.raw_cap, s.comp_cap);
                            eprintln!("ds4: metal backend initialized");
                        }
                        s
                    }
                    Err(e) => {
                        eprintln!("WARN: Session init: {} (stub mode)", e);
                        ds4_rs::session::SessionState::new(ctx_size)
                    }
                }
            }
            Err(e) => {
                eprintln!("WARN: Model loading failed: {}. Stub mode.", e);
                ds4_rs::session::SessionState::new(ctx_size)
            }
        }
    } else {
        eprintln!("WARN: Model file '{}' not found. Using stub.", model_path);
        ds4_rs::session::SessionState::new(ctx_size)
    }
}

// ── `ds4f run` ────────────────────────────────────────────────────────────

fn cmd_run(opts: RunArgs) {
    let think = opts.think && !opts.nothink;

    init_metal();
    let session = load_model(&opts.model, opts.ctx_size, false);

    if opts.inspect {
        eprintln!("Model: {}", opts.model);
        eprintln!("Context size: {}", opts.ctx_size);
        let has_vocab = session.vocab.is_some();
        eprintln!("Vocab loaded: {}", has_vocab);
        eprintln!("GPU graph: {}", session.graph.is_some());
        if has_vocab { eprintln!("Vocabulary size: {}", ds4_rs::gguf::N_VOCAB); }
        return;
    }

    let prompt = opts.prompt.clone()
        .or_else(|| opts.prompt_file.as_ref().and_then(|p| std::fs::read_to_string(p).ok()))
        .or_else(|| opts.positional_prompt.clone());

    if opts.interactive {
        run_repl(session, &opts, think);
    } else if let Some(ref prompt) = prompt {
        run_one_shot(session, &opts, prompt, think);
    } else {
        eprintln!("No prompt. Use -p <text>, --prompt-file <path>, positional arg, or -i.");
        std::process::exit(1);
    }
}

// ── One-shot generation ───────────────────────────────────────────────────

fn run_one_shot(mut sess: ds4_rs::session::SessionState, opts: &RunArgs, prompt: &str, think: bool) {
    let rendered = render_prompt(prompt, &opts.system, think);

    let tokens = match sess.vocab.as_ref() {
        Some(vocab) => ds4_rs::tokenizer::bpe_tokenize(vocab, &rendered),
        None => { eprintln!("No vocab loaded."); std::process::exit(1); }
    };
    let n_prompt = tokens.len();

    // Time prefill separately
    let t_prefill_start = std::time::Instant::now();
    eprintln!("ds4: prefilling {} tokens...", n_prompt);
    if let Err(e) = sess.prefill(&tokens) {
        eprintln!("Prefill error: {}", e);
        std::process::exit(1);
    }
    let t_prefill = t_prefill_start.elapsed();
    eprintln!("ds4: prefill done in {:.3}s", t_prefill.as_secs_f64());

    // Time generation separately, stream tokens as they are produced
    let t_gen_start = std::time::Instant::now();
    let mut result_tokens = Vec::new();
    let mut dbg_count = 0usize;
    for _ in 0..opts.n_predict {
        let (token, score) = sess.argmax();
        let token = if opts.temperature < 0.01 { token } else { sess.sample(opts.temperature, opts.top_k) };
        if dbg_count < 5 { eprintln!("ds4 [DBG] gen token={} score={:.4}", token, score); dbg_count += 1; }
        if sess.is_stop_token(token) { eprintln!("ds4 [DBG] stop token {}", token); break; }
        result_tokens.push(token);
        if let Some(ref vocab) = sess.vocab {
            let txt = ds4_rs::tokenizer::token_decode(vocab, &[token]);
            print!("{}", txt);
            let _ = std::io::Write::flush(&mut std::io::stdout());
        }
        if let Err(e) = sess.decode(token) {
            eprintln!("\nDecode error: {}", e);
            break;
        }
    }
    let t_gen = t_gen_start.elapsed();
    println!();

    let n_gen = result_tokens.len();
    let prefill_tps = if t_prefill.as_secs_f64() > 0.0 { n_prompt as f64 / t_prefill.as_secs_f64() } else { 0.0 };
    let gen_tps = if t_gen.as_secs_f64() > 0.0 { n_gen as f64 / t_gen.as_secs_f64() } else { 0.0 };
    eprintln!("ds4: prefill: {:.2} t/s, generation: {:.2} t/s", prefill_tps, gen_tps);
}

// ── Interactive REPL ──────────────────────────────────────────────────────

fn run_repl(mut sess: ds4_rs::session::SessionState, opts: &RunArgs, mut think: bool) {
    eprintln!("Interactive mode. Type /help for commands, /quit to exit.");
    use std::io::{stdin, BufRead};

    let mut transcript: Vec<(String, String)> = Vec::new();
    push_system(&mut transcript, &opts.system);

    for line in stdin().lock().lines() {
        let line = match line { Ok(l) => l, Err(_) => break };
        let line = line.trim().to_string();
        if line.is_empty() { eprint!("ds4> "); continue; }

        if line.starts_with('/') {
            match line.as_str() {
                "/quit" | "/exit" => break,
                "/help" => {
                    eprintln!("Commands:");
                    eprintln!("  /help           Show this help");
                    eprintln!("  /think          Enable thinking");
                    eprintln!("  /nothink        Disable thinking");
                    eprintln!("  /ctx <N>        Change context size");
                    eprintln!("  /read <FILE>    Read prompt from file");
                    eprintln!("  /history        Show history");
                    eprintln!("  /quit, /exit    Exit");
                }
                "/think" => { think = true; eprintln!("Thinking enabled"); }
                "/nothink" => { think = false; eprintln!("Thinking disabled"); }
                "/history" => {
                    let text: Vec<&str> = transcript.iter().map(|(_, c)| c.as_str()).collect();
                    let t = text.join("\n");
                    if t.is_empty() { eprintln!("(empty)"); }
                    else { println!("{}", t); }
                }
                cmd if cmd.starts_with("/ctx ") => {
                    if let Ok(n) = cmd[5..].trim().parse::<u32>() {
                        eprintln!("Recreating session with ctx={}...", n);
                        sess = load_model(&opts.model, n, false);
                        transcript.clear();
                        push_system(&mut transcript, &opts.system);
                    } else { eprintln!("Usage: /ctx <N>"); }
                }
                cmd if cmd.starts_with("/read ") => {
                    let path = cmd[6..].trim();
                    match std::fs::read_to_string(path) {
                        Ok(t) => do_chat_turn(&mut sess, &mut transcript, &t, opts.n_predict, opts.temperature, opts.top_k, think),
                        Err(e) => eprintln!("Error: {}", e),
                    }
                }
                _ => eprintln!("Unknown: {}", line),
            }
            eprint!("ds4> ");
            continue;
        }

        do_chat_turn(&mut sess, &mut transcript, &line, opts.n_predict, opts.temperature, opts.top_k, think);
        eprint!("ds4> ");
    }
    println!();
}

fn push_system(transcript: &mut Vec<(String, String)>, system: &str) {
    if !system.is_empty() {
        transcript.push(("system".to_string(), system.to_string()));
    }
}

fn build_chat_prompt(transcript: &[(String, String)], think: bool) -> String {
    let mut out = String::from("<｜begin▁of▁sentence｜>");
    for (role, content) in transcript {
        match role.as_str() {
            "system" => out.push_str(content),
            "user" => { out.push_str("<｜User｜>"); out.push_str(content); }
            "assistant" => {
                out.push_str("<｜Assistant｜>");
                out.push_str(if think { "<think>" } else { "</think>" });
                out.push_str(content);
                out.push_str("<｜end▁of▁sentence｜>");
            }
            _ => {}
        }
    }
    out.push_str("<｜Assistant｜>");
    out.push_str(if think { "<think>" } else { "</think>" });
    out
}

fn do_chat_turn(
    sess: &mut ds4_rs::session::SessionState,
    transcript: &mut Vec<(String, String)>,
    user_msg: &str,
    max_tokens: usize,
    temperature: f32,
    top_k: usize,
    think: bool,
) {
    transcript.push(("user".to_string(), user_msg.to_string()));
    let prompt_text = build_chat_prompt(transcript, think);

    let tokens = match sess.vocab.as_ref() {
        Some(v) => ds4_rs::tokenizer::bpe_tokenize(v, &prompt_text),
        None => { eprintln!("No vocab."); return; }
    };
    let n_prompt = tokens.len();

    let t_prefill_start = std::time::Instant::now();
    if let Err(e) = sess.prefill(&tokens) {
        eprintln!("Prefill error: {}", e);
        return;
    }
    let t_prefill = t_prefill_start.elapsed();

    let t_gen_start = std::time::Instant::now();
    let mut out = Vec::new();
    for _ in 0..max_tokens {
        let token = if temperature < 0.01 { sess.argmax().0 } else { sess.sample(temperature, top_k) };
        if sess.is_stop_token(token) { break; }
        out.push(token);

        if let Some(ref vocab) = sess.vocab {
            let txt = ds4_rs::tokenizer::token_decode(vocab, &[token]);
            print!("{}", txt);
            let _ = std::io::Write::flush(&mut std::io::stdout());
        }

        if let Err(e) = sess.decode(token) {
            eprintln!("\nDecode error: {}", e);
            break;
        }
    }
    let t_gen = t_gen_start.elapsed();

    let text = sess.vocab.as_ref()
        .map(|v| ds4_rs::tokenizer::token_decode(v, &out))
        .unwrap_or_default();
    transcript.push(("assistant".to_string(), text));

    let n_gen = out.len();
    let prefill_tps = if t_prefill.as_secs_f64() > 0.0 { n_prompt as f64 / t_prefill.as_secs_f64() } else { 0.0 };
    let gen_tps = if t_gen.as_secs_f64() > 0.0 { n_gen as f64 / t_gen.as_secs_f64() } else { 0.0 };
    eprintln!("\nds4: prefill: {:.2} t/s, generation: {:.2} t/s", prefill_tps, gen_tps);
}

fn render_prompt(prompt: &str, system: &str, think: bool) -> String {
    let mut out = String::from("<｜begin▁of▁sentence｜>");
    if !system.is_empty() { out.push_str(system); }
    out.push_str("<｜User｜>");
    out.push_str(prompt);
    out.push_str("<｜Assistant｜>");
    out.push_str(if think { "<think>" } else { "</think>" });
    out
}

// ── `ds4f serve` ──────────────────────────────────────────────────────────

fn cmd_serve(opts: ServeArgs) {
    let model = std::env::var("DS4_MODEL").unwrap_or(opts.model);

    init_metal();
    let session = load_model(&model, opts.ctx_size, false);
    let _ = ds4_rs::session::SESSION.set(Mutex::new(session));

    eprintln!();
    eprintln!("ds4f serve: listening on http://{}:{}", opts.host, opts.port);
    eprintln!("  OpenAI-compatible:  POST /v1/chat/completions, /v1/completions");
    eprintln!("  Anthropic-compatible: POST /v1/messages");
    eprintln!("  Health: GET /health");
    eprintln!("  Models: GET /v1/models");
    if opts.nothink { eprintln!("  Thinking: DISABLED (global)"); }
    eprintln!();

    if let Err(e) = ds4_rs::server::serve(opts.port) {
        eprintln!("Server error: {}", e);
    }
}
