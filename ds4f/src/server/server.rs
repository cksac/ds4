//! HTTP server entry point.
//!
//! Mirrors `main()` in ds4_server.c: engine open, session create, worker
//! thread, accept loop with per-client threads.

use super::args::ServeArgs;

pub fn main(args: ServeArgs) -> anyhow::Result<()> {
    eprintln!(
        "ds4f: starting server on http://{}:{}",
        args.host, args.port
    );
    eprintln!("ds4f: model: {}, ctx: {}", args.model_path, args.ctx_size);
    eprintln!("ds4f: server not yet fully implemented");

    anyhow::bail!("server not yet fully implemented");
}
