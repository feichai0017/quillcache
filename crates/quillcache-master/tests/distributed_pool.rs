//! End-to-end test of the distributed KV pool, exercising every component
//! together over real TCP (in-process, two "nodes"):
//!
//!   master (shared index + registry)
//!     ├─ node B: engine offloads a prefix's KV -> reports placement -> serves it
//!     └─ node A: engine needs the prefix -> locate via master -> fetch from B
//!
//! This is the flow a real multi-machine cluster runs; here both nodes are in one
//! process and the transfer is loopback TCP. The remaining gap to a deployed
//! cluster is the live gateway HTTP wiring + a real vLLM/SGLang connector (the
//! `EngineConnector` here is the in-process stand-in).

use bytes::Bytes;
use quillcache_core::{CacheResidency, CacheTier, KvBlockKey};
use quillcache_master::Master;
use quillcache_store::{
    EngineConnector, LocalKvStore, PooledStore, StaticRegistry, StoreBlockSource, StoreError,
};
use quillcache_transfer::{serve_listener, TcpTransfer};
use std::sync::{Arc, Mutex};
use tokio::net::TcpListener;

fn tmp(name: &str) -> std::path::PathBuf {
    let mut dir = std::env::temp_dir();
    dir.push(format!("qc-dist-{}-{}", name, std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

fn key(tenant: &str, hash: &str) -> KvBlockKey {
    KvBlockKey::new("m", "t", tenant, "p", hash, 0, 64)
}

#[tokio::test]
async fn distributed_pool_end_to_end_over_tcp() {
    // The master holds the shared residency index + node registry.
    let mut master = Master::new();

    // ---- Node B: a peer whose engine offloaded a block; it serves the pool. ----
    let dir_b = tmp("b");
    let store_b = Arc::new(Mutex::new(
        LocalKvStore::new(&dir_b, 1 << 20, 1 << 20).unwrap(),
    ));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr_b = listener.local_addr().unwrap().to_string();
    tokio::spawn(serve_listener(
        listener,
        Arc::new(StoreBlockSource::new(store_b.clone())),
    ));
    master.register("node-b", &addr_b);
    // B's engine offloads a shared system prompt's KV into B's pool and reports it.
    let shared = key("ten-a", "sysprompt");
    store_b
        .lock()
        .unwrap()
        .put(shared.clone(), Bytes::from_static(b"sysprompt-KV-on-B"))
        .unwrap();
    master.placed(CacheResidency {
        key: shared.clone(),
        worker_id: "node-b".into(),
        tier: CacheTier::CpuDram,
        bytes: 17,
        last_access_ms: 0,
        ref_count: 0,
        pinned: false,
    });

    // ---- Node A: an empty pool + an engine connector. ----
    let dir_a = tmp("a");
    let local_a = LocalKvStore::new(&dir_a, 1 << 20, 1 << 20).unwrap();
    // A loads the cluster's node -> addr map from the master into its registry.
    let mut registry = StaticRegistry::new("node-a");
    for (node, addr) in master.nodes() {
        registry = registry.with_node(node.clone(), addr.clone());
    }
    let pool_a = PooledStore::new(local_a, Arc::new(TcpTransfer), Arc::new(registry));
    let mut engine_a = EngineConnector::new("node-a", pool_a);

    // 1) A's engine offloads its own block; reloading it is a local hit.
    let local_block = key("ten-a", "A-only");
    let residency = engine_a
        .offload(local_block.clone(), Bytes::from_static(b"local-KV-on-A"))
        .unwrap();
    master.placed(residency);
    assert_eq!(
        &engine_a.reload(&local_block, &[]).await.unwrap()[..],
        b"local-KV-on-A"
    );

    // 2) A's engine needs the shared prefix: locate via master -> fetch from B over TCP.
    let located = master.locate_nodes(&shared);
    assert_eq!(located, vec!["node-b".to_string()]);
    let bytes = engine_a.reload(&shared, &located).await.unwrap();
    assert_eq!(&bytes[..], b"sysprompt-KV-on-B");
    // It is now resident locally on A too (cached after the cross-node fetch).
    assert_eq!(
        &engine_a.reload(&shared, &[]).await.unwrap()[..],
        b"sysprompt-KV-on-B"
    );

    // 3) Cluster-wide identity guard: tenant-b cannot reload tenant-a's block from
    //    A's pool (same content hash, different identity) — refused, not leaked.
    let cross = key("ten-b", "sysprompt");
    assert!(matches!(
        engine_a.reload(&cross, &[]).await,
        Err(StoreError::Unsafe(_))
    ));

    let _ = std::fs::remove_dir_all(&dir_a);
    let _ = std::fs::remove_dir_all(&dir_b);
}
