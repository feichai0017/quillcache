use clap::{Parser, Subcommand};
use quillcache_gateway::run_from_config_path;
use quillcache_sim::{run_synthetic, SyntheticWorkloadConfig};

#[derive(Debug, Parser)]
#[command(name = "quillcache")]
#[command(about = "Research CLI for QuillCache inference-state experiments")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Run the OpenAI-compatible QuillCache gateway.
    Gateway {
        #[arg(long)]
        config: String,
    },
    /// Print the current research plan and build order.
    Plan,
    /// Run a synthetic KV cache routing simulation.
    Simulate {
        #[arg(long, default_value_t = 32)]
        requests: u32,
        #[arg(long, default_value_t = 4)]
        workers: u32,
        #[arg(long, default_value_t = 8)]
        shared_prefix_blocks: u32,
        #[arg(long, default_value_t = 2)]
        unique_blocks: u32,
        #[arg(long, default_value_t = 64)]
        block_tokens: u32,
        #[arg(long, default_value_t = 4 * 1024 * 1024)]
        block_bytes: u64,
        #[arg(long)]
        json: bool,
    },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "quillcache=info".to_string()),
        )
        .init();

    let cli = Cli::parse();

    match cli.command {
        Command::Gateway { config } => run_from_config_path(config).await?,
        Command::Plan => print_plan(),
        Command::Simulate {
            requests,
            workers,
            shared_prefix_blocks,
            unique_blocks,
            block_tokens,
            block_bytes,
            json,
        } => {
            let report = run_synthetic(SyntheticWorkloadConfig {
                requests,
                workers,
                shared_prefix_blocks,
                unique_blocks_per_request: unique_blocks,
                block_tokens,
                block_bytes,
            })?;

            if json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                println!("QuillCache synthetic simulation");
                println!("requests: {}", report.total_requests);
                println!("workers: {}", report.workers);
                println!("reusable blocks: {}", report.cache_reusable_blocks);
                println!("transfer blocks: {}", report.transfer_blocks);
                println!("recompute blocks: {}", report.recompute_blocks);
                println!("avg estimated TTFT: {:.2} ms", report.avg_estimated_ttft_ms);
            }
        }
    }

    Ok(())
}

fn print_plan() {
    println!("QuillCache research plan");
    println!("1. Make KV block identity explicit across model/tokenizer/adapter/tenant.");
    println!("2. Build trace simulators for chat, RAG, and agentic workflows.");
    println!("3. Compare round-robin, cache-aware, SLO-aware, and network-aware routing.");
    println!("4. Add tiered placement and eviction across HBM, DRAM, SSD, and remote pools.");
    println!(
        "5. Run the gateway against vLLM/SGLang and ingest KV events through connector bridges."
    );
}
