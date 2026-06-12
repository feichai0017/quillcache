mod cluster;
mod gateway;
mod master_http;
mod node;

use crate::gateway::run_from_config_path;
use clap::{Parser, Subcommand};
use quillcache_core::bench::{bench_index, IndexBenchConfig};
use quillcache_core::MemoryIndex;

#[derive(Debug, Parser)]
#[command(name = "quillcache")]
#[command(about = "QuillCache: a Mooncake-style KV cache store + control plane, in Rust")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Run the OpenAI-compatible QuillCache gateway in front of real engines.
    Gateway {
        #[arg(long)]
        config: String,
    },
    /// Print the build order / roadmap.
    Plan,
    /// Benchmark an index backend (the ART-vs-LSM storage study): ingest
    /// throughput, prefix-scan latency, churn, write amplification, recovery.
    BenchIndex {
        #[arg(long, default_value = "memory")]
        backend: String,
        #[arg(long, default_value_t = 2000)]
        requests: u32,
        #[arg(long, default_value_t = 16)]
        shared_prefix_blocks: u32,
        #[arg(long, default_value_t = 4)]
        unique_blocks: u32,
        #[arg(long, default_value_t = 64)]
        block_tokens: u32,
        #[arg(long, default_value_t = 4 * 1024 * 1024)]
        block_bytes: u64,
        #[arg(long, default_value_t = 20000)]
        scan_queries: u32,
        #[arg(long, default_value_t = 0)]
        churn_ops: u32,
        #[arg(long)]
        json: bool,
    },
    /// Run a local multi-node cluster simulation over loopback TCP: a shared
    /// master + N nodes (each a byte pool + transfer server), a concurrent
    /// shared-prefix workload, cross-node fetch, and the identity guard.
    Cluster {
        #[arg(long, default_value_t = 3)]
        nodes: usize,
        #[arg(long, default_value_t = 12)]
        requests: usize,
    },
    /// Run the master metadata service (shared residency index + node registry)
    /// over HTTP, for out-of-process nodes or a real engine KV connector.
    Master {
        #[arg(long, default_value = "127.0.0.1:7777")]
        addr: String,
    },
    /// Run a standalone pool node: a local KV byte store + a transfer server,
    /// registered with the master. The target a real engine's KV connector
    /// (bridge/vllm_quillcache_connector.py) offloads to and fetches from.
    Node {
        #[arg(long, default_value = "127.0.0.1:7001")]
        addr: String,
        #[arg(long, default_value = "http://127.0.0.1:7777")]
        master: String,
        #[arg(long, default_value = "node-1")]
        id: String,
        #[arg(long, default_value = "./qc-node-data")]
        data_dir: String,
        #[arg(long, default_value_t = 256 * 1024 * 1024)]
        dram_bytes: u64,
        #[arg(long, default_value_t = 4 * 1024 * 1024 * 1024)]
        ssd_bytes: u64,
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
        Command::Cluster { nodes, requests } => cluster::run_cluster(nodes, requests).await?,
        Command::Master { addr } => master_http::run_master(addr).await?,
        Command::Node {
            addr,
            master,
            id,
            data_dir,
            dram_bytes,
            ssd_bytes,
        } => node::run_node(addr, master, id, data_dir, dram_bytes, ssd_bytes).await?,
        Command::BenchIndex {
            backend,
            requests,
            shared_prefix_blocks,
            unique_blocks,
            block_tokens,
            block_bytes,
            scan_queries,
            churn_ops,
            json,
        } => {
            let config = IndexBenchConfig {
                requests,
                shared_prefix_blocks,
                unique_blocks_per_request: unique_blocks,
                block_tokens,
                block_bytes,
                scan_queries,
                churn_ops,
            };
            let report = match backend.as_str() {
                "memory" => bench_index(&mut MemoryIndex::new(), config),
                #[cfg(feature = "rocksdb")]
                "rocksdb" => bench_rocksdb(config)?,
                #[cfg(feature = "holt")]
                "holt" => bench_holt(config)?,
                other => {
                    return Err(format!(
                        "unknown index backend '{other}' (available: memory, rocksdb, holt — build with --features rocksdb,holt)"
                    )
                    .into())
                }
            };
            if json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                println!("QuillCache index benchmark — backend: {}", report.backend);
                println!("persistent: {}", report.persistent);
                println!("blocks ingested: {}", report.blocks_ingested);
                println!(
                    "ingest: {:.0} puts/sec ({:.3}s)",
                    report.ingest_puts_per_sec, report.ingest_secs
                );
                println!(
                    "prefix_scan: p50 {:.2} · p99 {:.2} · p999 {:.2} · mean {:.2} us ({} queries)",
                    report.scan_p50_us,
                    report.scan_p99_us,
                    report.scan_p999_us,
                    report.scan_mean_us,
                    report.scan_queries
                );
                if report.churn_ops > 0 {
                    println!(
                        "churn: {:.0} ops/sec ({} cycles, {:.3}s)",
                        report.churn_ops_per_sec, report.churn_ops, report.churn_secs
                    );
                }
                println!(
                    "resident blocks: {} · on-disk: {} bytes",
                    report.metrics.resident_blocks, report.metrics.bytes_written
                );
                if report.write_amplification > 0.0 {
                    println!(
                        "write amplification: {:.2}× ({} physical bytes written{})",
                        report.write_amplification,
                        report.physical_bytes_written,
                        if report.write_amplification <= 1.01 {
                            ", append-only / write-once"
                        } else {
                            ", LSM compaction rewrites"
                        }
                    );
                }
                if let Some(ms) = report.recovery_ms {
                    println!("recovery (reopen from disk): {:.2} ms", ms);
                }
            }
        }
    }

    Ok(())
}

#[cfg(feature = "rocksdb")]
fn bench_rocksdb(
    config: IndexBenchConfig,
) -> Result<quillcache_core::bench::IndexBenchReport, Box<dyn std::error::Error>> {
    use quillcache_core::IndexBackend;
    use quillcache_index_rocksdb::RocksIndex;

    let dir = std::env::temp_dir().join(format!("quillcache-bench-rocksdb-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);

    let mut index = RocksIndex::open(&dir)?;
    let mut report = bench_index(&mut index, config);
    // Merge to one level so the reported on-disk size reflects the compacted state.
    index.compact();
    report.metrics = index.metrics();
    // Real LSM write amplification (flush + compaction bytes / user bytes).
    let (physical, amp) = index.write_amplification();
    report.physical_bytes_written = physical;
    report.write_amplification = amp;
    drop(index);

    // Recovery: reopen the index from disk and time it.
    let started = std::time::Instant::now();
    let reopened = RocksIndex::open(&dir)?;
    let recovery_ms = started.elapsed().as_secs_f64() * 1_000.0;
    let _ = reopened.len();
    drop(reopened);
    let _ = std::fs::remove_dir_all(&dir);

    report.recovery_ms = Some(recovery_ms);
    Ok(report)
}

#[cfg(feature = "holt")]
fn bench_holt(
    config: IndexBenchConfig,
) -> Result<quillcache_core::bench::IndexBenchReport, Box<dyn std::error::Error>> {
    use quillcache_core::IndexBackend;
    use quillcache_index_holt::HoltIndex;

    let dir = std::env::temp_dir().join(format!("quillcache-bench-holt-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);

    let mut index = HoltIndex::open(&dir).map_err(|e| format!("holt open: {e:?}"))?;
    let mut report = bench_index(&mut index, config);
    // Checkpoint the WAL so the reported on-disk size reflects all writes.
    index.flush();
    report.metrics = index.metrics();
    // Holt is append-only (WAL): each record is written once, no compaction
    // rewrite, so write amplification is ~1× by construction.
    report.physical_bytes_written = report.metrics.bytes_written;
    report.write_amplification = 1.0;
    drop(index);

    // Recovery: reopen the index from disk (WAL replay) and time it.
    let started = std::time::Instant::now();
    let reopened = HoltIndex::open(&dir).map_err(|e| format!("holt reopen: {e:?}"))?;
    let recovery_ms = started.elapsed().as_secs_f64() * 1_000.0;
    let _ = reopened.len();
    drop(reopened);
    let _ = std::fs::remove_dir_all(&dir);

    report.recovery_ms = Some(recovery_ms);
    Ok(report)
}

fn print_plan() {
    println!("QuillCache build order (Mooncake-style KV store + control plane, Rust)");
    println!("1. Real KV byte store: DRAM + SSD tiers, identity-guarded get (quillcache-store).");
    println!("2. Transfer engine seam: Local/TCP now, RDMA reserved (quillcache-transfer).");
    println!("3. CUDA device tier: HBM<->host copies + FP8 quantize-on-offload (quillcache-cuda).");
    println!(
        "4. ART-vs-LSM residency index study: prefix-scan, write-amp, recovery (bench-index)."
    );
    println!("5. Crash-consistent persistent KV pool + bounded-staleness routing metadata.");
}
