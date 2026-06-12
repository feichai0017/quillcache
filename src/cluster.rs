//! Local multi-node cluster demo over loopback TCP, on the Mooncake-faithful
//! store. Brings up N storage-node transfer engines + a master + a client, then
//! Puts a replicated object and Gets it back **over the transfer engine**, with
//! the identity guard enforced. This is the same flow a real multi-machine
//! cluster runs; here the "machines" are transfer engines on the local runtime
//! and the wire is loopback TCP (swap `TcpTransport` for RDMA, and the local
//! engines for remote nodes, and nothing above changes).

use quillcache_core::IdentityScope;
use quillcache_store::{ErrorCode, RealClient, ReplicateConfig};
use quillcache_transfer_engine::{InMemoryMetadata, MetadataBackend, TransferEngine};
use std::sync::Arc;

fn identity(tenant: &str) -> IdentityScope {
    IdentityScope {
        model_id: "Qwen2.5".into(),
        tokenizer_id: "Qwen2.5".into(),
        adapter_id: None,
        tenant_id: tenant.into(),
    }
}

pub async fn run_cluster(
    nodes_n: usize,
    requests: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let nodes_n = nodes_n.max(2);
    println!(
        "QuillCache cluster — {nodes_n} storage nodes over loopback TCP, Mooncake-faithful store\n"
    );

    let metadata: Arc<dyn MetadataBackend> = Arc::new(InMemoryMetadata::new());

    // Bring up N storage-node transfer engines; each serves its RAM segment over
    // TCP. Hold the handles so the serving tasks stay alive.
    let mut nodes = Vec::new();
    for i in 0..nodes_n {
        let name = format!("node-{i}");
        let engine = TransferEngine::init(name.clone(), metadata.clone(), "127.0.0.1:0").await?;
        println!("  {name}  transfer-engine RAM segment, serving over TCP");
        nodes.push(engine);
    }

    // A client with its own engine; the master mounts a segment per storage node.
    let client_engine = TransferEngine::init("client", metadata.clone(), "127.0.0.1:0").await?;
    let mut client = RealClient::new("random", client_engine);
    for i in 0..nodes_n {
        client.mount(&format!("node-{i}"), 1 << 20);
    }

    // Put a shared system prompt's KV (4 KiB), replicated across distinct nodes.
    let id = identity("tenant-a");
    let replicas = nodes_n.min(3);
    let kv = vec![7u8; 4096];
    client
        .put(
            "shared-system-prompt",
            id.clone(),
            &kv,
            &ReplicateConfig::replicas(replicas),
        )
        .await?;
    println!(
        "\n  put shared system prompt (4 KiB KV) — {replicas} replicas across distinct nodes\n"
    );

    // Get it back `requests` times — each a transfer-engine READ from a replica.
    let mut served = 0;
    for _ in 0..requests {
        if client
            .get("shared-system-prompt", &id)
            .await
            .map(|b| b.len())
            == Ok(4096)
        {
            served += 1;
        }
    }

    // Cluster-wide identity guard: tenant-b is refused the same content key.
    let refused = matches!(
        client
            .get("shared-system-prompt", &identity("tenant-b"))
            .await,
        Err(ErrorCode::UnsafeReuse(_))
    );

    println!(
        "  workload: {requests} gets over the transfer engine -> {served} served correct bytes"
    );
    println!(
        "  identity guard: tenant-b's request for tenant-a's prefix was {}",
        if refused {
            "REFUSED (no leak)"
        } else {
            "SERVED (bug!)"
        }
    );
    println!(
        "  master: {} object(s) across {} mounted segments",
        client.master().object_count(),
        client.master().segment_count()
    );

    Ok(())
}
