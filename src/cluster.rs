//! Local multi-node cluster simulation.
//!
//! Brings up a shared master and N nodes as Tokio tasks over loopback TCP — each
//! node a real byte pool plus a transfer server — then runs a concurrent
//! shared-prefix workload so you can watch the distributed pool work: one node's
//! engine offloads a system prompt, the others locate it via the master and
//! fetch it cross-node, and the identity guard refuses a cross-tenant request.
//!
//! This is the same flow a real multi-machine cluster runs; here the "machines"
//! are tasks on the multi-thread runtime and the wire is loopback TCP. Swapping
//! `TcpTransfer` for an RDMA backend and the in-process `EngineConnector` for a
//! real vLLM/SGLang connector is the only difference to a deployed cluster.

use bytes::Bytes;
use quillcache_core::{CacheResidency, CacheTier, KvBlockKey, Master};
use quillcache_store::{
    serve_listener, CountingTransfer, EngineConnector, LocalKvStore, PooledStore, StaticRegistry,
    StoreBlockSource, StoreError, TcpTransfer,
};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use tokio::net::TcpListener;

#[derive(Clone)]
struct Node {
    id: String,
    store: Arc<Mutex<LocalKvStore>>,
    connector: EngineConnector,
}

fn shared_block(tenant: &str) -> KvBlockKey {
    KvBlockKey::new(
        "Qwen2.5",
        "Qwen2.5",
        tenant,
        "sys",
        "shared-system-prompt",
        0,
        64,
    )
}

pub async fn run_cluster(
    nodes_n: usize,
    requests: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let nodes_n = nodes_n.max(2);
    println!("QuillCache local cluster — {nodes_n} nodes over loopback TCP, one shared master\n");

    let master = Arc::new(Mutex::new(Master::new()));
    let base = std::env::temp_dir().join(format!("qc-cluster-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&base);

    // Bring up each node: a byte pool + a TCP transfer server + master registration.
    let mut stores: Vec<Arc<Mutex<LocalKvStore>>> = Vec::new();
    for i in 0..nodes_n {
        let id = format!("node-{i}");
        let store = Arc::new(Mutex::new(LocalKvStore::new(
            base.join(&id),
            1 << 30,
            1 << 30,
        )?));
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let addr = listener.local_addr()?.to_string();
        tokio::spawn(serve_listener(
            listener,
            Arc::new(StoreBlockSource::new(store.clone())),
        ));
        master.lock().unwrap().register(&id, &addr);
        println!("  {id}  byte pool + transfer server @ {addr}");
        stores.push(store);
    }

    // Each node's connector gets the cluster's node→addr map from the master.
    let node_map = master.lock().unwrap().nodes().clone();
    // One shared transfer engine, wrapped to count ACTUAL cross-node fetches, so
    // we can show single-flight coalescing the concurrent duplicate reads.
    let transfer = Arc::new(CountingTransfer::new(Arc::new(TcpTransfer)));
    let nodes: Vec<Node> = (0..nodes_n)
        .map(|i| {
            let id = format!("node-{i}");
            let mut registry = StaticRegistry::new(id.clone());
            for (n, a) in &node_map {
                registry = registry.with_node(n.clone(), a.clone());
            }
            let pool = PooledStore::new(stores[i].clone(), transfer.clone(), Arc::new(registry));
            Node {
                id: id.clone(),
                store: stores[i].clone(),
                connector: EngineConnector::new(id, pool),
            }
        })
        .collect();

    // node-0's engine computes + offloads a shared system prompt's KV (4 KiB).
    let shared = shared_block("tenant-a");
    let residency = nodes[0]
        .connector
        .offload(shared.clone(), Bytes::from(vec![7u8; 4096]))?;
    master.lock().unwrap().placed(residency);
    println!("\n  node-0 offloaded the shared system prompt (4 KiB KV) to its pool; reported to master.\n");

    // Concurrent workload: `requests` requests round-robin across nodes, all
    // needing the shared prefix. A node that doesn't have it locates it via the
    // master and fetches it from a peer over TCP, then caches it.
    let local_hits = Arc::new(AtomicUsize::new(0));
    let cold_arrivals = Arc::new(AtomicUsize::new(0));
    let mut handles = Vec::new();
    for r in 0..requests {
        let node = nodes[r % nodes_n].clone();
        let master = master.clone();
        let shared = shared.clone();
        let (lh, cf) = (local_hits.clone(), cold_arrivals.clone());
        handles.push(tokio::spawn(async move {
            let was_local = node.store.lock().unwrap().tier_of(&shared).is_some();
            let located = master.lock().unwrap().locate_nodes(&shared);
            let _ = node.connector.reload(&shared, &located).await;
            if was_local {
                lh.fetch_add(1, Ordering::Relaxed);
            } else {
                cf.fetch_add(1, Ordering::Relaxed);
                master.lock().unwrap().placed(CacheResidency {
                    key: shared.clone(),
                    worker_id: node.id.clone(),
                    tier: CacheTier::CpuDram,
                    bytes: 4096,
                    last_access_ms: 0,
                    ref_count: 0,
                    pinned: false,
                });
            }
        }));
    }
    for handle in handles {
        let _ = handle.await;
    }

    // Cluster-wide identity guard: tenant-b asks node-0 for the same content.
    let cross = shared_block("tenant-b");
    let guarded = nodes[0].connector.reload(&cross, &[]).await;
    let refused = matches!(guarded, Err(StoreError::Unsafe(_)));

    let (lh, cold) = (
        local_hits.load(Ordering::Relaxed),
        cold_arrivals.load(Ordering::Relaxed),
    );
    let actual = transfer.reads();
    println!(
        "  workload: {requests} requests -> {lh} local hits, {cold} arrived cold -> \
         {actual} actual cross-node fetches over TCP (single-flight coalesced {cold}->{actual})"
    );
    println!(
        "  identity guard: tenant-b's request for tenant-a's prefix was {}",
        if refused {
            "REFUSED (no leak)"
        } else {
            "SERVED (bug!)"
        }
    );
    {
        let master = master.lock().unwrap();
        println!(
            "  master: {} residency records across {} registered nodes",
            master.resident_blocks(),
            master.node_count()
        );
    }

    let _ = std::fs::remove_dir_all(&base);
    Ok(())
}
