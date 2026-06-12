//! HTTP front for the master metadata service.
//!
//! Exposes `quillcache_core::Master` over HTTP so out-of-process clients — a real
//! vLLM/SGLang KV connector, or other gateway nodes — can register, report
//! placements, and locate blocks. Run it with `quillcache master --addr ...`.
//!
//! Endpoints:
//! - `POST /v1/register` `{node_id, transfer_addr}`  — a node joins the pool.
//! - `POST /v1/placed`   `[CacheResidency, ...]`      — a node block-reports.
//! - `POST /v1/locate`   `{key}`                      — which nodes/tiers hold it.
//! - `GET  /v1/nodes`                                  — node id → transfer addr.
//! - `GET  /v1/state`                                  — nodes + resident count.

use axum::extract::State;
use axum::routing::{get, post};
use axum::{Json, Router};
use quillcache_core::{CacheResidency, KvBlockKey, Master};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use tokio::net::TcpListener;

type Shared = Arc<Mutex<Master>>;

#[derive(Deserialize)]
struct RegisterReq {
    node_id: String,
    transfer_addr: String,
}

#[derive(Deserialize)]
struct LocateReq {
    key: KvBlockKey,
}

#[derive(Serialize, Deserialize)]
struct LocateResp {
    nodes: Vec<String>,
    residencies: Vec<CacheResidency>,
}

#[derive(Serialize, Deserialize)]
struct StateResp {
    nodes: HashMap<String, String>,
    node_count: usize,
    resident_blocks: usize,
}

async fn register(State(master): State<Shared>, Json(req): Json<RegisterReq>) -> Json<bool> {
    master
        .lock()
        .unwrap()
        .register(req.node_id, req.transfer_addr);
    Json(true)
}

async fn placed(
    State(master): State<Shared>,
    Json(residencies): Json<Vec<CacheResidency>>,
) -> Json<usize> {
    let n = residencies.len();
    master.lock().unwrap().placed_batch(residencies);
    Json(n)
}

async fn locate(State(master): State<Shared>, Json(req): Json<LocateReq>) -> Json<LocateResp> {
    let master = master.lock().unwrap();
    Json(LocateResp {
        nodes: master.locate_nodes(&req.key),
        residencies: master.locate(&req.key),
    })
}

async fn nodes(State(master): State<Shared>) -> Json<HashMap<String, String>> {
    Json(master.lock().unwrap().nodes().clone())
}

async fn state(State(master): State<Shared>) -> Json<StateResp> {
    let master = master.lock().unwrap();
    Json(StateResp {
        nodes: master.nodes().clone(),
        node_count: master.node_count(),
        resident_blocks: master.resident_blocks(),
    })
}

fn router(shared: Shared) -> Router {
    Router::new()
        .route("/v1/register", post(register))
        .route("/v1/placed", post(placed))
        .route("/v1/locate", post(locate))
        .route("/v1/nodes", get(nodes))
        .route("/v1/state", get(state))
        .with_state(shared)
}

pub async fn run_master(addr: String) -> Result<(), Box<dyn std::error::Error>> {
    let shared: Shared = Arc::new(Mutex::new(Master::new()));
    let socket: SocketAddr = addr.parse()?;
    let listener = TcpListener::bind(socket).await?;
    println!("QuillCache master metadata service on http://{socket}");
    println!(
        "  POST /v1/register · POST /v1/placed · POST /v1/locate · GET /v1/nodes · GET /v1/state"
    );
    axum::serve(listener, router(shared)).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn register_place_locate_over_http() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let shared: Shared = Arc::new(Mutex::new(Master::new()));
        tokio::spawn(async move { axum::serve(listener, router(shared)).await.unwrap() });
        let base = format!("http://{addr}");
        let http = reqwest::Client::new();

        // A node registers.
        http.post(format!("{base}/v1/register"))
            .json(&serde_json::json!({"node_id":"node-b","transfer_addr":"127.0.0.1:7000"}))
            .send()
            .await
            .unwrap();

        // It reports a placement.
        let key = KvBlockKey::new("m", "t", "ten-a", "p", "blk", 0, 64);
        let residency = CacheResidency {
            key: key.clone(),
            worker_id: "node-b".into(),
            tier: quillcache_core::CacheTier::CpuDram,
            bytes: 16,
            last_access_ms: 0,
            ref_count: 0,
            pinned: false,
        };
        http.post(format!("{base}/v1/placed"))
            .json(&vec![residency])
            .send()
            .await
            .unwrap();

        // Another node locates it.
        let resp: LocateResp = http
            .post(format!("{base}/v1/locate"))
            .json(&serde_json::json!({ "key": key }))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(resp.nodes, vec!["node-b".to_string()]);
        assert_eq!(resp.residencies.len(), 1);

        // State reflects it.
        let st: StateResp = http
            .get(format!("{base}/v1/state"))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(st.node_count, 1);
        assert_eq!(st.resident_blocks, 1);
    }
}
