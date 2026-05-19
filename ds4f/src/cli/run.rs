//! Entry point for `ds4f run` subcommand.

use std::ffi::CString;
use std::io::IsTerminal;

use super::args::RunArgs;
use crate::ffi;

pub fn main(args: RunArgs) -> anyhow::Result<()> {
    // Build engine options
    let model_path = CString::new(args.model_path.as_str())?;
    let mtp_path = args.mtp_path.as_deref().map(CString::new).transpose()?;
    let dir_steering_file = args
        .directional_steering_file
        .as_deref()
        .map(CString::new)
        .transpose()?;

    let backend = match args.backend.as_str() {
        "metal" => ffi::ds4_backend::DS4_BACKEND_METAL,
        "cuda" => ffi::ds4_backend::DS4_BACKEND_CUDA,
        "cpu" => ffi::ds4_backend::DS4_BACKEND_CPU,
        other => anyhow::bail!("invalid backend: {}", other),
    };

    let think_mode = match args.think_mode.as_str() {
        "nothink" => ffi::ds4_think_mode::DS4_THINK_NONE,
        "think" => ffi::ds4_think_mode::DS4_THINK_HIGH,
        "think-max" => ffi::ds4_think_mode::DS4_THINK_MAX,
        other => anyhow::bail!("invalid think-mode: {}", other),
    };

    let dir_steering_ffn = args.directional_steering_ffn.unwrap_or(0.0);
    let dir_steering_attn = args.directional_steering_attn.unwrap_or(0.0);

    let engine_opts = ffi::ds4_engine_options {
        model_path: model_path.as_ptr(),
        mtp_path: mtp_path.as_ref().map(|s| s.as_ptr()).unwrap_or(std::ptr::null()),
        backend,
        n_threads: args.n_threads,
        mtp_draft_tokens: args.mtp_draft_tokens,
        mtp_margin: args.mtp_margin,
        directional_steering_file: dir_steering_file
            .as_ref()
            .map(|s| s.as_ptr())
            .unwrap_or(std::ptr::null()),
        directional_steering_attn: dir_steering_attn,
        directional_steering_ffn: dir_steering_ffn,
        warm_weights: args.warm_weights,
        quality: args.quality,
    };

    // Print context memory estimate
    if !args.inspect {
        let mem = unsafe { ffi::ds4_context_memory_estimate(backend, args.ctx_size) };
        let backend_name = unsafe {
            std::ffi::CStr::from_ptr(ffi::ds4_backend_name(backend))
                .to_string_lossy()
        };
        eprintln!(
            "ds4: context buffers {:.2} MiB (ctx={}, backend={}, prefill_chunk={}, raw_kv_rows={}, compressed_kv_rows={})",
            mem.total_bytes as f64 / (1024.0 * 1024.0),
            args.ctx_size,
            backend_name,
            mem.prefill_cap,
            mem.raw_cap,
            mem.comp_cap,
        );

        // Warn if think-max downgraded
        if think_mode == ffi::ds4_think_mode::DS4_THINK_MAX {
            let effective = unsafe {
                ffi::ds4_think_mode_for_context(think_mode, args.ctx_size)
            };
            if effective != ffi::ds4_think_mode::DS4_THINK_MAX {
                let min_ctx = unsafe { ffi::ds4_think_max_min_context() };
                eprintln!(
                    "ds4: warning: --think-max needs --ctx >= {}; ctx={} uses normal thinking instead",
                    min_ctx, args.ctx_size,
                );
            }
        }
    }

    // Open engine
    let mut engine: *mut ffi::ds4_engine = std::ptr::null_mut();
    let rc = unsafe { ffi::ds4_engine_open(&mut engine, &engine_opts) };
    if rc != 0 {
        anyhow::bail!("failed to open model: {}", args.model_path);
    }

    if args.inspect {
        unsafe { ffi::ds4_engine_summary(engine) };
        unsafe { ffi::ds4_engine_close(engine) };
        return Ok(());
    }

    if args.dump_tokens {
        let prompt = args.prompt.as_deref().or(args.prompt_file.as_deref());
        if let Some(text) = prompt {
            let c_text = CString::new(text)?;
            let rc = unsafe {
                ffi::ds4_dump_text_tokenization(
                    model_path.as_ptr(),
                    c_text.as_ptr(),
                    std::ptr::null_mut(), // stdout not easily passed as FILE*
                )
            };
            if rc != 0 {
                anyhow::bail!("--dump-tokens failed");
            }
        }
        unsafe { ffi::ds4_engine_close(engine) };
        return Ok(());
    }

    // Initialize native Rust Metal session if --native flag
    let mut native_session: Option<super::native::NativeSession> = None;
    if args.native {
        eprintln!("ds4: --native flag set, initializing Rust Metal session");

        // mmap model independently (avoids sharing MTLBuffer with C engine)
        let model_file = std::fs::File::open(&args.model_path)?;
        let mmap = unsafe { memmap2::Mmap::map(&model_file)? };
        let map = mmap.as_ptr() as *const std::ffi::c_void;
        let size = mmap.len() as u64;

        // Use hardcoded DS4 Flash geometry
        let n_layer = 43u32; let n_embd = 4096u32; let n_hc = 4u32;
        let n_head = 64u32; let head_dim = 512u32; let n_rot = 64u32;
        let n_vocab = unsafe { ffi::ds4_bridge_n_vocab(engine) } as u32;

        // Tensor data starts after GGUF metadata (~5 MiB)
        let td_off = 5u64 * 1024 * 1024 + 80 * 1024;
        let td_size = size - td_off;

        eprintln!("ds4: model map={:p} size={:.2} GiB n_vocab={}",
            map, size as f64 / (1024.0*1024.0*1024.0), n_vocab);

        match super::native::NativeSession::init_from_engine(
            engine, map, size, td_off, td_size, args.ctx_size,
            n_layer, n_embd, n_hc, n_head, head_dim, n_rot, n_vocab,
        ) {
            Ok(ns) => {
                eprintln!("ds4: Rust Metal session created — using native inference path");
                std::mem::forget(mmap); // keep mmap alive
                native_session = Some(ns);
            }
            Err(e) => {
                eprintln!("ds4: Rust session init failed: {} — falling back to C engine", e);
                drop(mmap);
            }
        }
    }

    if args.prompt.is_none() && args.prompt_file.is_none() {
        super::repl::run_repl(engine, &args, native_session.as_mut())?;
    } else {
        run_generation(engine, &args, native_session.as_mut())?;
    }

    unsafe { ffi::ds4_engine_close(engine) };
    Ok(())
}

fn run_generation(engine: *mut ffi::ds4_engine, args: &RunArgs, native: Option<&mut super::native::NativeSession>) -> anyhow::Result<()> {
    if let Some(ns) = native {
        return run_generation_native(engine, args, ns);
    }
    run_generation_ffi(engine, args)
}

fn run_generation_native(engine: *mut ffi::ds4_engine, args: &RunArgs, ns: &mut super::native::NativeSession) -> anyhow::Result<()> {
    let think_mode = match args.think_mode.as_str() {
        "nothink" => ffi::ds4_think_mode::DS4_THINK_NONE,
        "think" => ffi::ds4_think_mode::DS4_THINK_HIGH,
        "think-max" => ffi::ds4_think_mode::DS4_THINK_MAX,
        _ => ffi::ds4_think_mode::DS4_THINK_HIGH,
    };
    let effective_think = unsafe { ffi::ds4_think_mode_for_context(think_mode, args.ctx_size) };

    let mut tokens: ffi::ds4_tokens = ffi::ds4_tokens { v: std::ptr::null_mut(), len: 0, cap: 0 };
    let system = CString::new(args.system.as_str())?;
    let prompt_text = args.prompt.as_deref().or(args.prompt_file.as_deref()).unwrap_or("");
    let prompt_c = CString::new(prompt_text)?;
    unsafe { ffi::ds4_encode_chat_prompt(engine, system.as_ptr(), prompt_c.as_ptr(), effective_think, &mut tokens); }

    // Sync: eval all prompt tokens into the Rust session
    let n_prompt = tokens.len as usize;
    let mut logits: &[f32] = &[];
    for i in 0..n_prompt {
        let token = unsafe { *tokens.v.add(i) };
        logits = ns.decode(token)?;
    }

    let eos_token = unsafe { ffi::ds4_token_eos(engine) };
    let mut rng_state: u64 = args.seed.unwrap_or_else(|| {
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos() as u64
    });
    let use_color = std::io::stdout().is_terminal();
    let mut printer = super::printer::TokenPrinter::new(Box::new(std::io::stdout()), use_color);

    for _ in 0..args.n_predict {
        let token = if args.temperature < 0.001 {
            super::native::NativeSession::argmax(logits)
        } else {
            super::native::NativeSession::sample(logits, args.temperature, args.top_p, args.min_p, &mut rng_state)
        };

        if token == eos_token { break; }

        let mut text_len: usize = 0;
        let text_ptr = unsafe { ffi::ds4_token_text(engine, token, &mut text_len) };
        if !text_ptr.is_null() && text_len > 0 {
            let text = unsafe { std::slice::from_raw_parts(text_ptr as *const u8, text_len) };
            printer.process(text, false)?;
        }

        logits = ns.decode(token)?;
    }
    printer.finish()?;
    unsafe { ffi::ds4_tokens_free(&mut tokens) };
    Ok(())
}

fn run_generation_ffi(engine: *mut ffi::ds4_engine, args: &RunArgs) -> anyhow::Result<()> {
    let think_mode = match args.think_mode.as_str() {
        "nothink" => ffi::ds4_think_mode::DS4_THINK_NONE,
        "think" => ffi::ds4_think_mode::DS4_THINK_HIGH,
        "think-max" => ffi::ds4_think_mode::DS4_THINK_MAX,
        _ => ffi::ds4_think_mode::DS4_THINK_HIGH,
    };
    let effective_think = unsafe { ffi::ds4_think_mode_for_context(think_mode, args.ctx_size) };

    // Build chat prompt
    let mut tokens: ffi::ds4_tokens = ffi::ds4_tokens {
        v: std::ptr::null_mut(),
        len: 0,
        cap: 0,
    };

    let system = CString::new(args.system.as_str())?;
    let prompt_text = args
        .prompt
        .as_deref()
        .or(args.prompt_file.as_deref())
        .unwrap_or("");
    let prompt_c = CString::new(prompt_text)?;

    unsafe {
        ffi::ds4_encode_chat_prompt(engine, system.as_ptr(), prompt_c.as_ptr(), effective_think, &mut tokens);
    }

    // Create session
    let mut session: *mut ffi::ds4_session = std::ptr::null_mut();
    let rc = unsafe { ffi::ds4_session_create(&mut session, engine, args.ctx_size) };
    if rc != 0 {
        unsafe { ffi::ds4_tokens_free(&mut tokens) };
        anyhow::bail!("failed to create session");
    }

    // Sync session to prompt
    let mut err_buf = vec![0u8; 256];
    let rc = unsafe {
        ffi::ds4_session_sync(
            session,
            &tokens,
            err_buf.as_mut_ptr() as *mut _,
            err_buf.len(),
        )
    };
    if rc != 0 {
        let err_msg = String::from_utf8_lossy(&err_buf);
        unsafe {
            ffi::ds4_session_free(session);
            ffi::ds4_tokens_free(&mut tokens);
        }
        anyhow::bail!("session sync failed: {}", err_msg);
    }

    // Generation loop with think-tag-aware token printer
    let eos_token = unsafe { ffi::ds4_token_eos(engine) };
    let mut rng_state: u64 = args.seed.unwrap_or_else(|| {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64
    });

    let use_greedy = args.temperature < 0.001;
    let use_color = std::io::stdout().is_terminal();
    let mut printer = super::printer::TokenPrinter::new(Box::new(std::io::stdout()), use_color);

    for _ in 0..args.n_predict {
        let token = if use_greedy {
            unsafe { ffi::ds4_session_argmax(session) }
        } else {
            unsafe {
                ffi::ds4_session_sample(
                    session,
                    args.temperature,
                    0,
                    args.top_p,
                    args.min_p,
                    &mut rng_state,
                )
            }
        };

        if token == eos_token {
            break;
        }

        let mut text_len: usize = 0;
        let text_ptr = unsafe { ffi::ds4_token_text(engine, token, &mut text_len) };
        if !text_ptr.is_null() && text_len > 0 {
            let text = unsafe { std::slice::from_raw_parts(text_ptr as *const u8, text_len) };
            printer.process(text, false)?;
        }

        let rc = unsafe {
            ffi::ds4_session_eval(
                session,
                token,
                err_buf.as_mut_ptr() as *mut _,
                err_buf.len(),
            )
        };
        if rc != 0 {
            eprintln!("\nds4: eval error: {}", String::from_utf8_lossy(&err_buf));
            break;
        }
    }

    printer.finish()?;

    unsafe {
        ffi::ds4_session_free(session);
        ffi::ds4_tokens_free(&mut tokens);
    }

    Ok(())
}
