use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use hf_hub::api::sync::ApiBuilder;

const REPO: &str = "antirez/deepseek-v4-gguf";
const Q2_FILE: &str = "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf";
const Q4_FILE: &str = "DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2.gguf";
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
    name = "download-model-rs",
    about = "DeepSeek V4 Flash GGUF downloader",
    after_help = "Targets:\n  q2   2-bit routed experts, about 81 GB on disk.\n       Main model for 128 GB RAM machines.\n\n  q4   4-bit routed experts, about 153 GB on disk.\n       Main model for machines with 256 GB RAM or more.\n\n  mtp  Optional speculative decoding component, about 3.5 GB on disk.\n       It is useful with both q2 and q4, but must be enabled explicitly\n       with --mtp when running ds4 or ds4-server.\n\nOptions:\n  --token TOKEN  Hugging Face token. Otherwise HF_TOKEN or the local HF token\n                 cache is used if present.\n\nAfter q2/q4 downloads the tool updates:\n  ./ds4flash.gguf -> gguf/<selected model>\n\nThen the default commands work:\n  ./ds4 -p \"Hello\"\n  ./ds4-server --ctx 100000\n\nAfter downloading mtp, enable it explicitly, for example:\n  ./ds4 --mtp gguf/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf --mtp-draft 2"
)]
struct Cli {
    #[arg(value_enum)]
    model: ModelArg,
    #[arg(long = "token")]
    token: Option<String>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let token = cli
        .token
        .or_else(hf_token_from_env)
        .or_else(read_local_hf_token);
    let api = build_api(token)?;
    let repo = api.model(REPO.to_owned());
    let filename = cli.model.filename();

    println!("Downloading {filename}");
    println!("from https://huggingface.co/{REPO}");
    let cached_path = repo
        .get(filename)
        .with_context(|| format!("failed to fetch {filename} from Hugging Face"))?;

    let repo_entry = ensure_repo_entry(filename, &cached_path)?;

    if cli.model == ModelArg::Mtp {
        println!();
        println!("MTP is an optional component for both q2 and q4.");
        println!("Enable it explicitly, for example:");
        println!("  ./ds4 --mtp gguf/{MTP_FILE} --mtp-draft 2");
    } else {
        update_default_model_link(filename)?;
        println!("Linked ./ds4flash.gguf -> gguf/{filename}");
    }

    println!();
    println!("Ready: {}", repo_entry.display());
    println!("Done.");
    Ok(())
}

fn build_api(token: Option<String>) -> Result<hf_hub::api::sync::Api> {
    let mut builder = ApiBuilder::from_env()
        .with_progress(true)
        .with_retries(3)
        .with_user_agent("ds4", env!("CARGO_PKG_VERSION"));
    if let Some(token) = token {
        builder = builder.with_token(Some(token));
    }
    builder.build().context("failed to initialize Hugging Face API client")
}

fn hf_token_from_env() -> Option<String> {
    env::var("HF_TOKEN")
        .ok()
        .map(|token| token.trim().to_owned())
        .filter(|token| !token.is_empty())
}

fn read_local_hf_token() -> Option<String> {
    let home = env::var_os("HOME")?;
    let token_path = PathBuf::from(home).join(".cache/huggingface/token");
    let token = fs::read_to_string(token_path).ok()?;
    let token = token.trim();
    if token.is_empty() {
        None
    } else {
        Some(token.to_owned())
    }
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
    create_symlink(&cached_path, &repo_entry).with_context(|| {
        format!(
            "failed to link {} to {}",
            repo_entry.display(),
            cached_path.display()
        )
    })?;
    Ok(repo_entry)
}

fn update_default_model_link(filename: &str) -> Result<()> {
    let link_path = repo_root().join("ds4flash.gguf");
    if path_present(&link_path) {
        fs::remove_file(&link_path)
            .with_context(|| format!("failed to replace {}", link_path.display()))?;
    }

    let relative_target = PathBuf::from("gguf").join(filename);
    create_symlink(&relative_target, &link_path).with_context(|| {
        format!(
            "failed to link {} to {}",
            link_path.display(),
            relative_target.display()
        )
    })
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
    let _ = target;
    fs::hard_link(target, link)
}