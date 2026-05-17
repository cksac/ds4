use clap::{Parser, Subcommand};

mod cli;
mod ffi;
mod metal;
mod server;

#[derive(Parser)]
#[command(name = "ds4f", about = "DeepSeek V4 Flash inference engine (Rust port)")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run interactive chat or one-shot generation
    Run(cli::args::RunArgs),
    /// Start the HTTP API server
    Serve(server::args::ServeArgs),
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Run(args) => cli::run::main(args),
        Commands::Serve(args) => server::server::main(args),
    }
}
