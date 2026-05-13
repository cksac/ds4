//! ds4f — DeepSeek V4 Flash CLI.

use anyhow::{bail, Result};
use std::path::PathBuf;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        print_usage();
        return Ok(());
    }

    match args[1].as_str() {
        "run" => cmd_run(&args[2..]),
        "info" => cmd_info(&args[2..]),
        "--help" | "-h" | "help" => {
            print_usage();
            Ok(())
        }
        other => bail!("unknown command: {other}"),
    }
}

fn print_usage() {
    eprintln!(
        "Usage: ds4f <command> [options]

Commands:
  run   --model <path>  [--prompt <text>]  [--ctx <size>]
  info  --model <path>
  help
"
    );
}

fn cmd_info(args: &[String]) -> Result<()> {
    let model_path = parse_model_path(args)?;
    let mut file = std::fs::File::open(&model_path)?;
    let gguf = ds4::gguf::parse(&mut file)?;

    println!("GGUF file: {}", model_path.display());
    println!("Tensors:   {}", gguf.tensors.len());
    println!("Metadata:  {} keys", gguf.metadata.len());
    println!("Data offset: {:#x}", gguf.data_offset);
    println!("File size:   {} MB", gguf.file_size / (1024 * 1024));

    // Print model architecture metadata.
    for key in &[
        "general.architecture",
        "general.name",
        "deepseek4.block_count",
        "deepseek4.embedding_length",
        "deepseek4.vocab_size",
        "deepseek4.attention.head_count",
        "deepseek4.attention.key_length",
        "deepseek4.expert_count",
    ] {
        if let Some(val) = gguf.metadata.get(*key) {
            println!("  {key} = {val:?}");
        }
    }
    Ok(())
}

fn cmd_run(args: &[String]) -> Result<()> {
    let _model_path = parse_model_path(args)?;
    eprintln!("ds4f: run command not yet implemented");
    Ok(())
}

fn parse_model_path(args: &[String]) -> Result<PathBuf> {
    for (i, arg) in args.iter().enumerate() {
        if (arg == "--model" || arg == "-m") && i + 1 < args.len() {
            return Ok(PathBuf::from(&args[i + 1]));
        }
    }
    bail!("--model <path> is required")
}
