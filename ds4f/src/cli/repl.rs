//! Interactive REPL with session reuse.
//!
//! Mirrors the `run_repl()` function in ds4_cli.c.

use std::ffi::CString;
use std::io::IsTerminal;

use crate::ffi;
use super::args::RunArgs;

pub fn run_repl(engine: *mut ffi::ds4_engine, args: &RunArgs, _native: Option<&mut super::native::NativeSession>) -> anyhow::Result<()> {
    let think_mode = match args.think_mode.as_str() {
        "nothink" => ffi::ds4_think_mode::DS4_THINK_NONE,
        "think" => ffi::ds4_think_mode::DS4_THINK_HIGH,
        "think-max" => ffi::ds4_think_mode::DS4_THINK_MAX,
        _ => ffi::ds4_think_mode::DS4_THINK_HIGH,
    };
    let mut effective_think =
        unsafe { ffi::ds4_think_mode_for_context(think_mode, args.ctx_size) };

    let _system = CString::new(args.system.as_str())?;
    let eos_token = unsafe { ffi::ds4_token_eos(engine) };

    // Initialize chat token transcript
    let mut transcript: ffi::ds4_tokens = ffi::ds4_tokens {
        v: std::ptr::null_mut(),
        len: 0,
        cap: 0,
    };
    unsafe { ffi::ds4_chat_begin(engine, &mut transcript) };

    // Create session
    let mut session: *mut ffi::ds4_session = std::ptr::null_mut();
    let rc = unsafe { ffi::ds4_session_create(&mut session, engine, args.ctx_size) };
    if rc != 0 {
        unsafe { ffi::ds4_tokens_free(&mut transcript) };
        anyhow::bail!("failed to create REPL session");
    }
    let mut current_ctx = args.ctx_size;

    let mut rng_state: u64 = args.seed.unwrap_or_else(|| {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64
    });

    let use_greedy = args.temperature < 0.001;
    let mut err_buf = vec![0u8; 256];

    let mut rl: rustyline::Editor<(), rustyline::history::DefaultHistory> = rustyline::Editor::new()?;

    println!("ds4> interactive chat (type /help for commands, /quit to exit)");

    loop {
        let line = rl.readline("ds4> ");
        match line {
            Ok(line) => {
                let line: String = line.trim().to_string();
                if line.is_empty() {
                    continue;
                }

                // Handle /commands
                if line.starts_with('/') {
                    let parts: Vec<&str> = line.splitn(2, ' ').collect();
                    match parts[0] {
                        "/quit" | "/exit" => break,
                        "/help" => {
                            println!("Commands:");
                            println!("  /help          Show this help");
                            println!("  /think         Use normal thinking mode");
                            println!("  /think-max     Use Think Max (needs large --ctx)");
                            println!("  /nothink       Disable thinking");
                            println!("  /ctx N         Recreate session with new context size");
                            println!("  /quit, /exit   Exit");
                            continue;
                        }
                        "/think" => {
                            effective_think = ffi::ds4_think_mode::DS4_THINK_HIGH;
                            println!("ds4: thinking mode = think");
                            continue;
                        }
                        "/think-max" => {
                            let mode = ffi::ds4_think_mode::DS4_THINK_MAX;
                            effective_think =
                                unsafe { ffi::ds4_think_mode_for_context(mode, args.ctx_size) };
                            if effective_think != ffi::ds4_think_mode::DS4_THINK_MAX {
                                println!("ds4: Think Max needs larger --ctx, using normal thinking");
                            } else {
                                println!("ds4: thinking mode = think-max");
                            }
                            continue;
                        }
                        "/nothink" => {
                            effective_think = ffi::ds4_think_mode::DS4_THINK_NONE;
                            println!("ds4: thinking mode = none");
                            continue;
                        }
                        "/ctx" => {
                            if let Some(n_str) = parts.get(1) {
                                if let Ok(n) = n_str.trim().parse::<i32>() {
                                    if n > 0 {
                                        // Recreate session with new context
                                        unsafe { ffi::ds4_session_free(session) };
                                        let rc = unsafe {
                                            ffi::ds4_session_create(&mut session, engine, n)
                                        };
                                        if rc != 0 {
                                            eprintln!("ds4: failed to create session with ctx={}", n);
                                            // Try to recreate with old size
                                            let _ = unsafe {
                                                ffi::ds4_session_create(
                                                    &mut session,
                                                    engine,
                                                    current_ctx,
                                                )
                                            };
                                        } else {
                                            current_ctx = n;
                                            println!("ds4: context size = {}", n);
                                        }
                                    }
                                }
                            }
                            continue;
                        }
                        _ => {
                            println!("ds4: unknown command: {}", parts[0]);
                            continue;
                        }
                    }
                }

                let _ = rl.add_history_entry(&line);

                // Build user message tokens
                let line_c = CString::new(line.as_str())?;
                let role_c = CString::new("user")?;

                unsafe {
                    ffi::ds4_chat_append_message(
                        engine,
                        &mut transcript,
                        role_c.as_ptr(),
                        line_c.as_ptr(),
                    );
                    ffi::ds4_chat_append_assistant_prefix(engine, &mut transcript, effective_think);
                }

                // Sync session
                let rc = unsafe {
                    ffi::ds4_session_sync(
                        session,
                        &transcript,
                        err_buf.as_mut_ptr() as *mut _,
                        err_buf.len(),
                    )
                };
                if rc != 0 {
                    let err_msg = String::from_utf8_lossy(&err_buf);
                    eprintln!("ds4: sync error: {}", err_msg);
                    continue;
                }

                // Generation loop with think-tag-aware token printer
                let use_color = std::io::stdout().is_terminal();
                let mut printer = super::printer::TokenPrinter::new(
                    Box::new(std::io::stdout()),
                    use_color,
                );

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
                    let text_ptr =
                        unsafe { ffi::ds4_token_text(engine, token, &mut text_len) };
                    if !text_ptr.is_null() && text_len > 0 {
                        let text = unsafe {
                            std::slice::from_raw_parts(text_ptr as *const u8, text_len)
                        };
                        let _ = printer.process(text, false);
                    }

                    // Add to transcript
                    unsafe { ffi::ds4_tokens_push(&mut transcript, token) };

                    let rc = unsafe {
                        ffi::ds4_session_eval(
                            session,
                            token,
                            err_buf.as_mut_ptr() as *mut _,
                            err_buf.len(),
                        )
                    };
                    if rc != 0 {
                        break;
                    }
                }

                let _ = printer.finish();
            }
            Err(rustyline::error::ReadlineError::Interrupted) => {
                println!("^C");
                continue;
            }
            Err(rustyline::error::ReadlineError::Eof) => {
                break;
            }
            Err(e) => {
                eprintln!("ds4: readline error: {}", e);
                break;
            }
        }
    }

    unsafe {
        ffi::ds4_session_free(session);
        ffi::ds4_tokens_free(&mut transcript);
    }

    Ok(())
}
