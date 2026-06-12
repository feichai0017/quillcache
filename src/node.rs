//! A standalone pool node: a local KV byte store (DRAM + SSD, crash-consistent)
//! plus a transfer server, registered with the master. This is the out-of-process
//! node that a real engine's KV connector (bridge/vllm_quillcache_connector.py)
//! offloads to and fetches from. Run it with `quillcache node --addr ...`.

use quillcache_store::{serve_listener, LocalKvStore, StoreBlockSource};
use std::sync::{Arc, Mutex};
use tokio::net::TcpListener;

#[allow(clippy::too_many_arguments)]
pub async fn run_node(
    addr: String,
    master_url: String,
    node_id: String,
    data_dir: String,
    dram_bytes: u64,
    ssd_bytes: u64,
) -> Result<(), Box<dyn std::error::Error>> {
    let store = Arc::new(Mutex::new(LocalKvStore::new(
        &data_dir, dram_bytes, ssd_bytes,
    )?));
    let listener = TcpListener::bind(&addr).await?;
    let bound = listener.local_addr()?.to_string();

    // Register with the master so peers can locate + fetch our blocks. The master
    // must already be running (`quillcache master`); a real engine's connector
    // then locates blocks via the master and fetches them from this transfer addr.
    let master_base = master_url.trim_end_matches('/').to_string();
    reqwest::Client::new()
        .post(format!("{master_base}/v1/register"))
        .json(&serde_json::json!({"node_id": node_id, "transfer_addr": bound}))
        .send()
        .await?
        .error_for_status()?;

    println!("QuillCache node '{node_id}' — transfer server on {bound}, data in {data_dir}");
    println!("  registered with master {master_base}; serving block reads/writes until killed");
    serve_listener(listener, Arc::new(StoreBlockSource::new(store))).await?;
    Ok(())
}
