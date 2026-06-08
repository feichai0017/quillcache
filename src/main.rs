use clap::{Parser, Subcommand};
use quillcache_core::MemoryIndex;
use quillcache_gateway::run_from_config_path;
use quillcache_sim::{
    bench_index, run_safe_reuse, run_synthetic, run_tiered, IndexBenchConfig, SafeReuseConfig,
    SyntheticWorkloadConfig, TieredConfig,
};

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
    /// Benchmark an index backend: ingest throughput and prefix-scan latency.
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
    /// Tiered KV block management (KVBM-style): HBM/DRAM/SSD with promotion and
    /// eviction, vs an HBM-only baseline on the same trace.
    Tiered {
        #[arg(long, default_value_t = 4000)]
        blocks: u32,
        #[arg(long, default_value_t = 40000)]
        accesses: u32,
        #[arg(long, default_value_t = 200)]
        hbm: u32,
        #[arg(long, default_value_t = 800)]
        dram: u32,
        #[arg(long, default_value_t = 4000)]
        ssd: u32,
        #[arg(long, default_value_t = 10)]
        hot_percent: u32,
        #[arg(long, default_value_t = 256)]
        block_tokens: u32,
        #[arg(long)]
        json: bool,
    },
    /// Identity-governed safe-reuse experiment: naive content-hash reuse vs
    /// QuillCache's identity guard on a cross-identity workload.
    SafeReuse {
        #[arg(long, default_value_t = 50)]
        prefixes: u32,
        #[arg(long, default_value_t = 8)]
        prefix_blocks: u32,
        #[arg(long, default_value_t = 8)]
        tenants: u32,
        #[arg(long, default_value_t = 4)]
        adapters: u32,
        #[arg(long, default_value_t = 2)]
        models: u32,
        #[arg(long, default_value_t = 2)]
        tokenizers: u32,
        #[arg(long, default_value_t = 2)]
        repeats: u32,
        #[arg(long, default_value_t = 64)]
        block_tokens: u32,
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
                    "resident blocks: {} · bytes_written (on-disk): {}",
                    report.metrics.resident_blocks, report.metrics.bytes_written
                );
                if let Some(ms) = report.recovery_ms {
                    println!("recovery (reopen from disk): {:.2} ms", ms);
                }
            }
        }
        Command::Tiered {
            blocks,
            accesses,
            hbm,
            dram,
            ssd,
            hot_percent,
            block_tokens,
            json,
        } => {
            let report = run_tiered(TieredConfig {
                blocks,
                accesses,
                hbm_blocks: hbm,
                dram_blocks: dram,
                ssd_blocks: ssd,
                hot_percent,
                block_tokens,
                block_bytes: 2 * 1024 * 1024,
            });
            if json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                let total = report.accesses.max(1) as f64;
                let hit = |n: u64| 100.0 * n as f64 / total;
                println!("QuillCache tiered KV block management (KVBM-style)");
                println!(
                    "tiers: HBM {} · DRAM {} · SSD {} blocks (effective cache {})",
                    hbm, dram, ssd, report.effective_cache_blocks
                );
                println!("accesses: {}", report.accesses);
                println!(
                    "hits: HBM {} ({:.1}%) · DRAM {} ({:.1}%) · SSD {} ({:.1}%) · miss {} ({:.1}%)",
                    report.hbm_hits,
                    hit(report.hbm_hits),
                    report.dram_hits,
                    hit(report.dram_hits),
                    report.ssd_hits,
                    hit(report.ssd_hits),
                    report.misses,
                    hit(report.misses)
                );
                println!(
                    "movement: {} promotions · {} demotions · {} evictions",
                    report.promotions, report.demotions, report.evictions
                );
                println!(
                    "vs HBM-only: {} misses → {} ({} recomputes avoided)",
                    report.hbm_only_misses, report.misses, report.recomputes_avoided
                );
                println!(
                    "cost: {:.1} ms vs HBM-only {:.1} ms → {:.1}% saved",
                    report.total_cost_ms, report.hbm_only_cost_ms, report.cost_saved_pct
                );
            }
        }
        Command::SafeReuse {
            prefixes,
            prefix_blocks,
            tenants,
            adapters,
            models,
            tokenizers,
            repeats,
            block_tokens,
            json,
        } => {
            let report = run_safe_reuse(SafeReuseConfig {
                distinct_prefixes: prefixes,
                prefix_blocks,
                tenants,
                adapters,
                models,
                tokenizers,
                repeats,
                block_tokens,
            });
            if json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                let naive_unsafe_pct = if report.naive_reuses > 0 {
                    100.0 * report.naive_unsafe as f64 / report.naive_reuses as f64
                } else {
                    0.0
                };
                println!("QuillCache safe-reuse experiment");
                println!("identities sharing each prefix: {}", report.identities);
                println!("blocks evaluated: {}", report.blocks_evaluated);
                println!(
                    "naive content reuse: {} hits, of which {} UNSAFE ({:.1}%)",
                    report.naive_reuses, report.naive_unsafe, naive_unsafe_pct
                );
                println!(
                    "  unsafe: {} cross-tenant (privacy) · {} cross-adapter · {} cross-model/quant · {} cross-tokenizer (correctness)",
                    report.unsafe_cross_tenant,
                    report.unsafe_cross_adapter,
                    report.unsafe_cross_model,
                    report.unsafe_cross_tokenizer
                );
                println!(
                    "identity guard: {} unsafe served · {} safe reuses preserved · {} recomputes forced",
                    0, report.safe_reuses, report.guard_recomputes
                );
                println!(
                    "cost of safety: {:.1} ms prefill to avoid {} unsafe serves",
                    report.guard_recompute_ms, report.unsafe_blocks_avoided
                );
                println!(
                    "safety overhead: {:.1}% of reuse work (≈0 when same-identity reuse dominates)",
                    report.safety_overhead_pct
                );
            }
        }
    }

    Ok(())
}

#[cfg(feature = "rocksdb")]
fn bench_rocksdb(
    config: IndexBenchConfig,
) -> Result<quillcache_sim::IndexBenchReport, Box<dyn std::error::Error>> {
    use quillcache_core::IndexBackend;
    use quillcache_index_rocksdb::RocksIndex;

    let dir = std::env::temp_dir().join(format!("quillcache-bench-rocksdb-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);

    let mut index = RocksIndex::open(&dir)?;
    let mut report = bench_index(&mut index, config);
    // Merge to one level so the reported on-disk size reflects the compacted state.
    index.compact();
    report.metrics = index.metrics();
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
) -> Result<quillcache_sim::IndexBenchReport, Box<dyn std::error::Error>> {
    use quillcache_core::IndexBackend;
    use quillcache_index_holt::HoltIndex;

    let dir = std::env::temp_dir().join(format!("quillcache-bench-holt-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);

    let mut index = HoltIndex::open(&dir).map_err(|e| format!("holt open: {e:?}"))?;
    let mut report = bench_index(&mut index, config);
    // Checkpoint the WAL so the reported on-disk size reflects all writes.
    index.flush();
    report.metrics = index.metrics();
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
    println!("QuillCache research plan");
    println!("1. Make KV block identity explicit across model/tokenizer/adapter/tenant.");
    println!("2. Build trace simulators for chat, RAG, and agentic workflows.");
    println!("3. Compare round-robin, cache-aware, SLO-aware, and network-aware routing.");
    println!("4. Add tiered placement and eviction across HBM, DRAM, SSD, and remote pools.");
    println!(
        "5. Run the gateway against vLLM/SGLang and ingest KV events through connector bridges."
    );
}
