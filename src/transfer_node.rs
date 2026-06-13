//! A standalone Transfer Engine storage node: serves one named RAM segment over
//! TCP (the `(segment, offset)` wire). This is the target a store client / a real
//! engine's KV connector (`bridge/`) writes KV bytes to and reads them from — the
//! Mooncake-faithful replacement for the old key-oriented pool node. Run it with
//! `quillcache transfer-node --addr ... --segment ...`.

use quillcache_transfer_engine::{InMemoryMetadata, MetadataBackend, TransferEngine};
use std::sync::Arc;

pub async fn run_transfer_node(
    addr: String,
    segment: String,
) -> Result<(), Box<dyn std::error::Error>> {
    // A standalone node uses in-memory metadata (the connector addresses it
    // directly by endpoint); init binds `addr` and serves the segment over TCP.
    let metadata: Arc<dyn MetadataBackend> = Arc::new(InMemoryMetadata::new());
    let _engine = TransferEngine::init(segment.clone(), metadata, &addr).await?;
    println!(
        "QuillCache transfer node — segment '{segment}' serving the (segment, offset) wire on {addr}"
    );
    println!(
        "  a store client writes/reads KV bytes here at master-allocated offsets; killed to stop"
    );
    // The serve loop runs as a task spawned by init; keep the process alive.
    std::future::pending::<()>().await;
    Ok(())
}
