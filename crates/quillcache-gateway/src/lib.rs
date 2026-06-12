use axum::body::Bytes;
use axum::extract::State;
use axum::http::{HeaderMap, HeaderName, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures_util::StreamExt;
use quillcache_control::{ControlPlane, IngestSummary, PlanAction, RequestPlan, ServingMode};
use quillcache_core::{
    DataPlane, DataPlaneAction, EngineEndpoint, ExternalKvBlockKey, IndexBackend, KvBlockKey,
    KvEventBatch, MemoryIndex, NoDataPlane, RequestKvHints, RequestShape, SloTarget,
};
use quillcache_router::{
    DynamoCostRouter, GreedyStatePlaneRouter, LeastLoadedRouter, PrefixAffinityRouter,
    RoundRobinRouter, RoutingPolicy, SessionAffinityRouter, SloAwareRouter,
};
use quillcache_store::{StoreDataPlane, StoreTierConfig};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::hash_map::DefaultHasher;
use std::fs;
use std::hash::{Hash, Hasher};
use std::net::SocketAddr;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use thiserror::Error;
use tokio::net::TcpListener;
use tokio::sync::RwLock;

#[derive(Debug, Error)]
pub enum GatewayError {
    #[error("failed to read config {path}: {source}")]
    ReadConfig {
        path: String,
        source: std::io::Error,
    },
    #[error("failed to parse config {path}: {source}")]
    ParseConfig {
        path: String,
        source: serde_yaml::Error,
    },
    #[error("gateway config must include at least one engine")]
    NoEngines,
    #[error("failed to bind gateway on {addr}: {source}")]
    Bind {
        addr: SocketAddr,
        source: std::io::Error,
    },
    #[error("action_sink kind=http requires url")]
    ActionSinkMissingUrl,
    #[error("gateway server failed: {0}")]
    Serve(std::io::Error),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GatewayConfig {
    pub bind: SocketAddr,
    pub engines: Vec<EngineEndpoint>,
    /// Routing policy: "prefix-affinity" (cache-affine across the fleet),
    /// "round-robin" (spread baseline), "least-loaded", "slo-aware" (SLO as a
    /// near-hard constraint), "session-affinity", "dynamo-cost" (mirrors NVIDIA
    /// Dynamo's KV-router cost function), or "greedy" (default).
    #[serde(default)]
    pub policy: Option<String>,
    /// Residency index backend: "memory" (default, ephemeral), "holt"
    /// (persistent ART), or "rocksdb" (persistent LSM). The persistent backends
    /// need the matching build feature; otherwise the gateway warns and uses
    /// memory. A persistent index keeps fleet residency across restarts.
    #[serde(default)]
    pub index: Option<String>,
    /// On-disk path for a persistent index backend (default
    /// `quillcache-residency`).
    #[serde(default)]
    pub index_path: Option<String>,
    /// Runtime KV tensor data-plane adapter. `none` keeps the previous inferred
    /// placement behavior; `tiered` enables an in-process HBM/DRAM/SSD control
    /// plane that performs real admission, promotion, demotion, and eviction.
    #[serde(default)]
    pub data_plane: Option<DataPlaneConfig>,
    /// Optional synchronous action sink. `http` posts planner/data-plane actions
    /// to an external adapter, for example vLLM kv_transfer, SGLang/LMCache, or
    /// a Dynamo KVBM bridge.
    #[serde(default)]
    pub action_sink: Option<ActionSinkConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataPlaneConfig {
    #[serde(default = "default_data_plane_kind")]
    pub kind: String,
    #[serde(default)]
    pub hbm_capacity_bytes: Option<u64>,
    #[serde(default)]
    pub cpu_dram_capacity_bytes: Option<u64>,
    #[serde(default)]
    pub local_ssd_capacity_bytes: Option<u64>,
}

fn default_data_plane_kind() -> String {
    "none".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionSinkConfig {
    #[serde(default = "default_action_sink_kind")]
    pub kind: String,
    #[serde(default)]
    pub url: Option<String>,
    #[serde(default = "default_action_sink_fail_open")]
    pub fail_open: bool,
    #[serde(default = "default_action_sink_timeout_ms")]
    pub timeout_ms: u64,
}

fn default_action_sink_kind() -> String {
    "none".to_string()
}

fn default_action_sink_fail_open() -> bool {
    true
}

fn default_action_sink_timeout_ms() -> u64 {
    250
}

impl GatewayConfig {
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, GatewayError> {
        let path_ref = path.as_ref();
        let raw = fs::read_to_string(path_ref).map_err(|source| GatewayError::ReadConfig {
            path: path_ref.display().to_string(),
            source,
        })?;
        let config: Self =
            serde_yaml::from_str(&raw).map_err(|source| GatewayError::ParseConfig {
                path: path_ref.display().to_string(),
                source,
            })?;
        if config.engines.is_empty() {
            return Err(GatewayError::NoEngines);
        }
        Ok(config)
    }
}

/// Live SLO goodput: of the requests served, how many produced their first token
/// within the request's TTFT SLO budget. Measured from the gateway's own clock
/// (arrival → first streamed chunk), not the cost model — a real online metric.
#[derive(Debug, Default)]
struct SloGoodput {
    served: AtomicU64,
    met: AtomicU64,
    ttft_ms_sum: AtomicU64,
}

impl SloGoodput {
    fn record(&self, ttft_ms: u64, budget_ms: u64) {
        self.served.fetch_add(1, Ordering::Relaxed);
        self.ttft_ms_sum.fetch_add(ttft_ms, Ordering::Relaxed);
        if ttft_ms <= budget_ms {
            self.met.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn snapshot(&self) -> serde_json::Value {
        let served = self.served.load(Ordering::Relaxed);
        let met = self.met.load(Ordering::Relaxed);
        let sum = self.ttft_ms_sum.load(Ordering::Relaxed);
        json!({
            "served": served,
            "met_slo": met,
            "goodput_pct": if served > 0 { 100.0 * met as f64 / served as f64 } else { 0.0 },
            "mean_ttft_ms": if served > 0 { sum as f64 / served as f64 } else { 0.0 },
        })
    }
}

#[derive(Debug, Clone)]
struct GatewayState {
    control: Arc<RwLock<ControlPlane>>,
    client: Client,
    action_sink: Option<ActionSink>,
    slo: Arc<SloGoodput>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GatewayRouteTrace {
    pub request_id: String,
    pub mode: ServingMode,
    pub engine_id: String,
    pub prefill_engine_id: Option<String>,
    pub decode_engine_id: String,
    pub planner_actions: usize,
    pub reusable_blocks: usize,
    pub local_hits: usize,
    pub transfer_blocks: usize,
    pub recompute_blocks: usize,
    /// Content-matching blocks the identity guard refused (resident only under a
    /// different identity — a naive content cache would have served them).
    pub reuse_refused: usize,
    pub estimated_ttft_us: u64,
    pub estimated_tpot_us: u64,
}

#[derive(Debug, Clone)]
struct ActionSink {
    url: String,
    fail_open: bool,
    timeout: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ActionSinkSnapshot {
    kind: String,
    url: Option<String>,
    fail_open: bool,
    timeout_ms: u64,
}

impl ActionSinkSnapshot {
    fn disabled() -> Self {
        Self {
            kind: "none".to_string(),
            url: None,
            fail_open: true,
            timeout_ms: 0,
        }
    }
}

impl ActionSink {
    fn snapshot(&self) -> ActionSinkSnapshot {
        ActionSinkSnapshot {
            kind: "http".to_string(),
            url: Some(self.url.clone()),
            fail_open: self.fail_open,
            timeout_ms: self.timeout.as_millis() as u64,
        }
    }

    async fn publish(
        &self,
        client: &Client,
        event: &ActionSinkEvent,
    ) -> Result<(), reqwest::Error> {
        client
            .post(&self.url)
            .timeout(self.timeout)
            .json(event)
            .send()
            .await?
            .error_for_status()?;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActionSinkPhase {
    Planned,
    Committed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActionSinkPlan {
    pub mode: ServingMode,
    pub execution_worker_id: String,
    pub prefill_worker_id: Option<String>,
    pub decode_worker_id: String,
    pub actions: Vec<PlanAction>,
}

impl From<&RequestPlan> for ActionSinkPlan {
    fn from(plan: &RequestPlan) -> Self {
        Self {
            mode: plan.mode,
            execution_worker_id: plan.execution_worker_id.clone(),
            prefill_worker_id: plan.prefill_worker_id.clone(),
            decode_worker_id: plan.decode_worker_id.clone(),
            actions: plan.actions.clone(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActionSinkEvent {
    pub schema_version: u32,
    pub phase: ActionSinkPhase,
    pub openai_path: String,
    pub request: RequestShape,
    pub route: GatewayRouteTrace,
    pub plan: ActionSinkPlan,
    pub cache_actions: Vec<DataPlaneAction>,
}

impl ActionSinkEvent {
    pub fn new(
        phase: ActionSinkPhase,
        openai_path: &str,
        request: &RequestShape,
        route: &GatewayRouteTrace,
        plan: ActionSinkPlan,
        cache_actions: Vec<DataPlaneAction>,
    ) -> Self {
        Self {
            schema_version: 1,
            phase,
            openai_path: openai_path.to_string(),
            request: request.clone(),
            route: route.clone(),
            plan,
            cache_actions,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ActionSinkDelivery {
    Disabled,
    Sent,
    Failed,
}

impl ActionSinkDelivery {
    fn as_header(self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
            Self::Sent => "sent",
            Self::Failed => "failed",
        }
    }

    fn merge(self, other: Self) -> Self {
        match (self, other) {
            (Self::Failed, _) | (_, Self::Failed) => Self::Failed,
            (Self::Sent, _) | (_, Self::Sent) => Self::Sent,
            _ => Self::Disabled,
        }
    }
}

pub async fn run_from_config_path(path: impl AsRef<Path>) -> Result<(), GatewayError> {
    let config = GatewayConfig::from_path(path)?;
    run(config).await
}

pub async fn run(config: GatewayConfig) -> Result<(), GatewayError> {
    let policy = build_policy(config.policy.as_deref());
    let policy_name = policy.name().to_string();
    let index = build_index(&config);
    let index_name = index.name().to_string();
    let data_plane = build_data_plane(config.data_plane.as_ref());
    let data_plane_name = data_plane.name().to_string();
    let action_sink = build_action_sink(config.action_sink.as_ref())?;
    let action_sink_name = action_sink
        .as_ref()
        .map(|sink| sink.snapshot().kind)
        .unwrap_or_else(|| "none".to_string());
    let control = ControlPlane::with_index_and_policy(config.engines, index, policy)
        .with_data_plane(data_plane);
    tracing::info!(
        policy = %policy_name,
        index = %index_name,
        data_plane = %data_plane_name,
        action_sink = %action_sink_name,
        "control plane configured"
    );
    let state = GatewayState {
        control: Arc::new(RwLock::new(control)),
        client: Client::new(),
        action_sink,
        slo: Arc::new(SloGoodput::default()),
    };
    let control = state.control.clone();
    let app = router(state);
    let listener = TcpListener::bind(config.bind)
        .await
        .map_err(|source| GatewayError::Bind {
            addr: config.bind,
            source,
        })?;

    tracing::info!(addr = %config.bind, "starting QuillCache gateway");
    // Persist the residency index on shutdown so a persistent backend survives a
    // restart (in-memory no-ops).
    axum::serve(listener, app)
        .with_graceful_shutdown(async move {
            shutdown_signal().await;
            control.read().await.flush();
            tracing::info!("flushed residency index on shutdown");
        })
        .await
        .map_err(GatewayError::Serve)
}

/// Resolve when the process receives Ctrl-C or (on Unix) SIGTERM.
async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut stream) => {
                stream.recv().await;
            }
            Err(_) => std::future::pending::<()>().await,
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    tokio::select! {
        _ = ctrl_c => {}
        _ = terminate => {}
    }
}

/// Build the residency index backend from config. Persistent backends are
/// feature-gated; if the requested one is not compiled in (or fails to open),
/// the gateway warns and falls back to the in-memory reference index, so a
/// misconfigured backend degrades instead of failing to start.
fn build_index(config: &GatewayConfig) -> Box<dyn IndexBackend> {
    match config.index.as_deref().unwrap_or("memory") {
        "memory" => Box::new(MemoryIndex::new()),
        #[cfg(feature = "holt")]
        "holt" => {
            let path = config
                .index_path
                .clone()
                .unwrap_or_else(|| "quillcache-residency".to_string());
            match quillcache_index_holt::HoltIndex::open(&path) {
                Ok(index) => {
                    tracing::info!(path = %path, "persistent ART (Holt) residency index");
                    Box::new(index)
                }
                Err(error) => {
                    tracing::error!(?error, "failed to open Holt index; using in-memory");
                    Box::new(MemoryIndex::new())
                }
            }
        }
        #[cfg(feature = "rocksdb")]
        "rocksdb" => {
            let path = config
                .index_path
                .clone()
                .unwrap_or_else(|| "quillcache-residency".to_string());
            match quillcache_index_rocksdb::RocksIndex::open(&path) {
                Ok(index) => {
                    tracing::info!(path = %path, "persistent LSM (RocksDB) residency index");
                    Box::new(index)
                }
                Err(error) => {
                    tracing::error!(?error, "failed to open RocksDB index; using in-memory");
                    Box::new(MemoryIndex::new())
                }
            }
        }
        other => {
            tracing::warn!(
                backend = other,
                "index backend unavailable (needs a build feature); using in-memory"
            );
            Box::new(MemoryIndex::new())
        }
    }
}

fn build_data_plane(config: Option<&DataPlaneConfig>) -> Box<dyn DataPlane> {
    let Some(config) = config else {
        return Box::new(NoDataPlane);
    };
    match config.kind.as_str() {
        "none" => Box::new(NoDataPlane),
        "tiered" | "store" => {
            let defaults = StoreTierConfig::default();
            Box::new(StoreDataPlane::new(StoreTierConfig {
                hbm_capacity_bytes: config
                    .hbm_capacity_bytes
                    .unwrap_or(defaults.hbm_capacity_bytes),
                cpu_dram_capacity_bytes: config
                    .cpu_dram_capacity_bytes
                    .unwrap_or(defaults.cpu_dram_capacity_bytes),
                local_ssd_capacity_bytes: config
                    .local_ssd_capacity_bytes
                    .unwrap_or(defaults.local_ssd_capacity_bytes),
            }))
        }
        other => {
            tracing::warn!(data_plane = other, "unknown data plane; using none");
            Box::new(NoDataPlane)
        }
    }
}

fn build_action_sink(
    config: Option<&ActionSinkConfig>,
) -> Result<Option<ActionSink>, GatewayError> {
    let Some(config) = config else {
        return Ok(None);
    };
    match config.kind.as_str() {
        "none" => Ok(None),
        "http" => {
            let url = config
                .url
                .clone()
                .ok_or(GatewayError::ActionSinkMissingUrl)?;
            Ok(Some(ActionSink {
                url,
                fail_open: config.fail_open,
                timeout: Duration::from_millis(config.timeout_ms),
            }))
        }
        other => {
            tracing::warn!(action_sink = other, "unknown action sink; disabling");
            Ok(None)
        }
    }
}

/// Build a routing policy from its config name (default: cache-aware greedy).
fn build_policy(name: Option<&str>) -> Box<dyn RoutingPolicy> {
    match name.unwrap_or("greedy") {
        "prefix-affinity" | "affinity" => Box::new(PrefixAffinityRouter::default()),
        "round-robin" | "roundrobin" => Box::new(RoundRobinRouter::default()),
        "least-loaded" | "load" => Box::new(LeastLoadedRouter::default()),
        "slo-aware" | "slo" => Box::new(SloAwareRouter::default()),
        "session-affinity" | "session" => Box::new(SessionAffinityRouter::default()),
        "dynamo-cost" | "dynamo" | "kv-router" => Box::new(DynamoCostRouter::default()),
        _ => Box::new(GreedyStatePlaneRouter::default()),
    }
}

fn router(state: GatewayState) -> Router {
    Router::new()
        .route("/healthz", get(healthz))
        .route("/v1/state", get(state_snapshot))
        .route("/v1/kv-events", post(ingest_kv_events))
        .route("/v1/chat/completions", post(proxy_chat_completions))
        .route("/v1/completions", post(proxy_completions))
        .with_state(state)
}

async fn healthz() -> impl IntoResponse {
    Json(json!({ "status": "ok" }))
}

async fn state_snapshot(State(state): State<GatewayState>) -> impl IntoResponse {
    let control = state.control.read().await;
    Json(json!({
        "engines": control.engines(),
        "workers": control.workers(),
        "index": control.residency().metrics(),
        "index_backend": control.residency().name(),
        "data_plane": control.data_plane().name(),
        "data_plane_metrics": control.data_plane().metrics(),
        "data_plane_residency": control.data_plane().snapshot(),
        "action_sink": state.action_sink.as_ref().map(ActionSink::snapshot).unwrap_or_else(ActionSinkSnapshot::disabled),
        "slo_goodput": state.slo.snapshot(),
        "resident_blocks": control.residency().len(),
        "residency": control.residency().snapshot(),
    }))
}

async fn ingest_kv_events(
    State(state): State<GatewayState>,
    Json(batch): Json<KvEventBatch>,
) -> Result<Json<IngestSummary>, GatewayHttpError> {
    let mut control = state.control.write().await;
    let summary = control.ingest(batch)?;
    Ok(Json(summary))
}

async fn proxy_chat_completions(
    State(state): State<GatewayState>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, GatewayHttpError> {
    proxy_openai_path(state, headers, body, "/v1/chat/completions").await
}

async fn proxy_completions(
    State(state): State<GatewayState>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, GatewayHttpError> {
    proxy_openai_path(state, headers, body, "/v1/completions").await
}

async fn proxy_openai_path(
    state: GatewayState,
    headers: HeaderMap,
    body: Bytes,
    path: &str,
) -> Result<Response, GatewayHttpError> {
    // Arrival clock for the SLO-goodput measurement (arrival → first token).
    let request_start = Instant::now();
    let mut payload: Value = serde_json::from_slice(&body)?;
    let request_shape = request_shape_from_payload(&mut payload);
    let ttft_budget_ms = request_shape.slo.ttft_ms;
    let clean_body = serde_json::to_vec(&payload)?;

    let (engine, trace, action_plan) = {
        let control = state.control.read().await;
        let plan = control.plan(&request_shape)?;
        let decision = &plan.route;
        // Identity guard: how many content-matching blocks we refused to reuse
        // because they belong to another identity (the safety property, made
        // observable on the live path).
        let audit = control.audit_reuse(&request_shape);
        let engine = control
            .engine(&plan.execution_worker_id)
            .cloned()
            .ok_or_else(|| GatewayHttpError::MissingEngine(plan.execution_worker_id.clone()))?;
        let trace = GatewayRouteTrace {
            request_id: decision.request_id.clone(),
            mode: plan.mode,
            engine_id: plan.execution_worker_id.clone(),
            prefill_engine_id: plan.prefill_worker_id.clone(),
            decode_engine_id: plan.decode_worker_id.clone(),
            planner_actions: plan.actions.len(),
            reusable_blocks: decision.reusable_blocks(),
            local_hits: decision.local_hits.len(),
            transfer_blocks: decision.transfers.len(),
            recompute_blocks: decision.recomputes.len(),
            reuse_refused: audit.refused_unsafe,
            estimated_ttft_us: decision.estimated_ttft_us,
            estimated_tpot_us: decision.estimated_tpot_us,
        };
        let action_plan = ActionSinkPlan::from(&plan);
        (engine, trace, action_plan)
    };

    if trace.reuse_refused > 0 {
        tracing::warn!(
            request_id = %trace.request_id,
            reuse_refused = trace.reuse_refused,
            "identity guard refused unsafe cross-identity reuse"
        );
    }

    let planned_delivery = dispatch_action_sink_event(
        &state,
        ActionSinkEvent::new(
            ActionSinkPhase::Planned,
            path,
            &request_shape,
            &trace,
            action_plan.clone(),
            Vec::new(),
        ),
    )
    .await?;

    let target_url = format!("{}{}", engine.base_url.trim_end_matches('/'), path);
    tracing::info!(
        engine_id = %trace.engine_id,
        request_id = %trace.request_id,
        reusable_blocks = trace.reusable_blocks,
        recompute_blocks = trace.recompute_blocks,
        "proxying request"
    );

    let mut request = state.client.post(target_url).body(clean_body);
    for (name, value) in headers.iter() {
        if should_forward_header(name) {
            request = request.header(name, value);
        }
    }
    request = request.header("x-quillcache-engine-id", trace.engine_id.as_str());
    request = request.header("x-quillcache-request-id", trace.request_id.as_str());
    request = request.header("x-quillcache-mode", serving_mode_header(trace.mode));
    request = request.header(
        "x-quillcache-decode-engine-id",
        trace.decode_engine_id.as_str(),
    );
    if let Some(prefill_engine_id) = trace.prefill_engine_id.as_deref() {
        request = request.header("x-quillcache-prefill-engine-id", prefill_engine_id);
    }
    request = request.header(
        "x-quillcache-reusable-blocks",
        trace.reusable_blocks.to_string(),
    );

    // Count this request as in flight on the chosen engine for the prefill
    // window (dispatch → response headers), so concurrent requests see the load
    // and a cache-aware policy spreads instead of dog-piling the one cache-hot
    // engine. This is the gateway's own load signal feeding back into routing.
    state.control.write().await.begin_request(&trace.engine_id);
    let send_result = request.send().await;
    state.control.write().await.end_request(&trace.engine_id);
    let upstream = send_result?;
    let status = StatusCode::from_u16(upstream.status().as_u16())
        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

    // Close the residency loop: record where we placed this request's prefix
    // blocks, so the next request for the same prefix sees them resident on this
    // engine — cache-aware routing now works end-to-end without a KV-events
    // bridge. (Tier 2 events later correct this inference on eviction.)
    let mut cache_action_list = Vec::new();
    if status.is_success() {
        let mut control = state.control.write().await;
        cache_action_list =
            control.observe_placement(&trace.engine_id, &request_shape, DEFAULT_BLOCK_BYTES);
    }
    let committed_delivery = if status.is_success() {
        dispatch_action_sink_event(
            &state,
            ActionSinkEvent::new(
                ActionSinkPhase::Committed,
                path,
                &request_shape,
                &trace,
                action_plan,
                cache_action_list.clone(),
            ),
        )
        .await?
    } else {
        ActionSinkDelivery::Disabled
    };
    let action_sink_delivery = planned_delivery.merge(committed_delivery);

    let mut response = Response::builder().status(status);
    for (name, value) in upstream.headers() {
        if should_return_header(name) {
            response = response.header(name, value);
        }
    }
    response = response
        .header("x-quillcache-engine-id", trace.engine_id)
        .header("x-quillcache-mode", serving_mode_header(trace.mode))
        .header("x-quillcache-decode-engine-id", trace.decode_engine_id)
        .header(
            "x-quillcache-prefill-engine-id",
            trace.prefill_engine_id.unwrap_or_default(),
        )
        .header(
            "x-quillcache-planner-actions",
            trace.planner_actions.to_string(),
        )
        .header(
            "x-quillcache-cache-actions",
            cache_action_list.len().to_string(),
        )
        .header("x-quillcache-action-sink", action_sink_delivery.as_header())
        .header("x-quillcache-request-id", trace.request_id)
        .header("x-quillcache-local-hits", trace.local_hits.to_string())
        .header(
            "x-quillcache-transfer-blocks",
            trace.transfer_blocks.to_string(),
        )
        .header(
            "x-quillcache-recompute-blocks",
            trace.recompute_blocks.to_string(),
        )
        .header(
            "x-quillcache-reuse-refused",
            trace.reuse_refused.to_string(),
        )
        .header(
            "x-quillcache-estimated-ttft-us",
            trace.estimated_ttft_us.to_string(),
        );
    // Stream the upstream body straight through (SSE chunks forwarded as they
    // arrive) instead of buffering it, so the client's time-to-first-token
    // reflects the real engine — QuillCache's decision headers are already set
    // above and flush with the response head, before the first token. On the
    // first chunk we record real TTFT against the SLO budget for live goodput.
    let slo = state.slo.clone();
    let mut recorded = false;
    let stream = upstream.bytes_stream().inspect(move |_chunk| {
        if !recorded {
            recorded = true;
            let ttft_ms = request_start.elapsed().as_millis() as u64;
            slo.record(ttft_ms, ttft_budget_ms);
        }
    });
    response
        .body(axum::body::Body::from_stream(stream))
        .map_err(GatewayHttpError::BuildResponse)
}

async fn dispatch_action_sink_event(
    state: &GatewayState,
    event: ActionSinkEvent,
) -> Result<ActionSinkDelivery, GatewayHttpError> {
    let Some(sink) = state.action_sink.as_ref() else {
        return Ok(ActionSinkDelivery::Disabled);
    };
    match sink.publish(&state.client, &event).await {
        Ok(()) => Ok(ActionSinkDelivery::Sent),
        Err(error) if sink.fail_open => {
            tracing::warn!(
                ?error,
                phase = ?event.phase,
                request_id = %event.route.request_id,
                "action sink delivery failed; continuing because fail_open=true"
            );
            Ok(ActionSinkDelivery::Failed)
        }
        Err(error) => Err(GatewayHttpError::ActionSink(error.to_string())),
    }
}

fn serving_mode_header(mode: ServingMode) -> &'static str {
    match mode {
        ServingMode::Aggregated => "aggregated",
        ServingMode::Disaggregated => "disaggregated",
    }
}

fn request_shape_from_payload(payload: &mut Value) -> RequestShape {
    let hints = payload
        .as_object_mut()
        .and_then(|object| object.remove("quillcache"))
        .and_then(|value| serde_json::from_value::<RequestKvHints>(value).ok());

    let model_id = payload
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or("unknown-model")
        .to_string();
    let tokenizer_id = hints
        .as_ref()
        .and_then(|hints| hints.tokenizer_id.clone())
        .unwrap_or_else(|| model_id.clone());
    let tenant_id = hints
        .as_ref()
        .and_then(|hints| hints.tenant_id.clone())
        .unwrap_or_else(|| "default".to_string());
    let estimated_decode_tokens = hints
        .as_ref()
        .and_then(|hints| hints.estimated_decode_tokens)
        .or_else(|| {
            payload
                .get("max_tokens")
                .and_then(Value::as_u64)
                .map(|v| v as u32)
        })
        .unwrap_or(128);
    let id = hints
        .as_ref()
        .and_then(|hints| hints.request_id.clone())
        .unwrap_or_else(fallback_request_id);
    let session_id = hints.as_ref().and_then(|hints| hints.session_id.clone());

    let blocks = hints
        .as_ref()
        .filter(|hints| !hints.block_hashes.is_empty())
        .map(|hints| hints.to_blocks(&model_id, &tokenizer_id, &tenant_id))
        .unwrap_or_else(|| fallback_blocks(payload, &model_id, &tokenizer_id, &tenant_id));

    RequestShape {
        id,
        model_id,
        tokenizer_id,
        adapter_id: hints.and_then(|hints| hints.adapter_id),
        tenant_id,
        session_id,
        blocks,
        estimated_decode_tokens,
        slo: SloTarget::default(),
    }
}

/// Inferred bytes per KV block when recording placement (no engine event yet to
/// give the real size). 4 MiB ≈ a 64-token block for a mid-size model.
const DEFAULT_BLOCK_BYTES: u64 = 4 * 1024 * 1024;

/// Approx. characters per fallback block (no tokenizer in the gateway, so we
/// chunk prompt text). ~4 chars/token ⇒ ~64 tokens/block.
const FALLBACK_BLOCK_CHARS: usize = 256;
/// Cap fallback blocks per request so a huge prompt can't explode the index.
const FALLBACK_MAX_BLOCKS: usize = 64;

/// Derive prefix blocks from the request itself when the client sends no
/// `quillcache` hints. Each block hash is **prefix-inclusive** (a hash of all
/// prompt text up to and including the block), so two requests that share a
/// leading prefix — e.g. the same system prompt or RAG context — produce the
/// same leading block hashes and route cache-affinely. The diverging suffix (the
/// user's question) only changes the trailing blocks. This is a tokenizer-free
/// approximation of how engines hash KV blocks; precise hashes arrive via
/// `quillcache` hints or `/v1/kv-events`.
fn fallback_blocks(
    payload: &Value,
    model_id: &str,
    tokenizer_id: &str,
    tenant_id: &str,
) -> Vec<KvBlockKey> {
    let text = prompt_text(payload);
    let chars: Vec<char> = text.chars().collect();
    if chars.is_empty() {
        let mut hasher = DefaultHasher::new();
        payload.to_string().hash(&mut hasher);
        return vec![KvBlockKey::external_hash(ExternalKvBlockKey {
            model_id: model_id.to_string(),
            tokenizer_id: tokenizer_id.to_string(),
            adapter_id: None,
            tenant_id: tenant_id.to_string(),
            prefix_hash: "root".to_string(),
            block_hash: format!("pfx-{:016x}", hasher.finish()),
            block_index: 0,
            token_count: 64,
        })];
    }

    let mut blocks = Vec::new();
    let mut parent = "root".to_string();
    let mut start = 0usize;
    let mut idx = 0u32;
    while start < chars.len() && blocks.len() < FALLBACK_MAX_BLOCKS {
        let end = (start + FALLBACK_BLOCK_CHARS).min(chars.len());
        // Prefix-inclusive content hash: bind the whole chain up to `end`.
        let prefix_text: String = chars[..end].iter().collect();
        let mut hasher = DefaultHasher::new();
        prefix_text.hash(&mut hasher);
        let block_hash = format!("pfx-{:016x}", hasher.finish());
        blocks.push(KvBlockKey::external_hash(ExternalKvBlockKey {
            model_id: model_id.to_string(),
            tokenizer_id: tokenizer_id.to_string(),
            adapter_id: None,
            tenant_id: tenant_id.to_string(),
            prefix_hash: parent.clone(),
            block_hash: block_hash.clone(),
            block_index: idx,
            token_count: ((end - start) as u32).div_ceil(4).max(1),
        }));
        parent = block_hash;
        start = end;
        idx += 1;
    }
    blocks
}

/// Flatten the request's prompt to text for fallback block hashing: chat
/// `messages` become `role:content` lines; a completion `prompt` is used as-is.
fn prompt_text(payload: &Value) -> String {
    if let Some(messages) = payload.get("messages").and_then(Value::as_array) {
        let mut text = String::new();
        for message in messages {
            if let Some(role) = message.get("role").and_then(Value::as_str) {
                text.push_str(role);
                text.push(':');
            }
            if let Some(content) = message.get("content").and_then(Value::as_str) {
                text.push_str(content);
                text.push('\n');
            }
        }
        text
    } else if let Some(prompt) = payload.get("prompt").and_then(Value::as_str) {
        prompt.to_string()
    } else {
        payload.to_string()
    }
}

fn fallback_request_id() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    format!("req-{nanos}")
}

fn should_forward_header(name: &HeaderName) -> bool {
    !matches!(
        name.as_str(),
        "host"
            | "content-length"
            | "connection"
            | "upgrade"
            | "proxy-authorization"
            | "proxy-authenticate"
            | "te"
            | "trailer"
            | "transfer-encoding"
    )
}

fn should_return_header(name: &HeaderName) -> bool {
    should_forward_header(name) && name != HeaderName::from_static("content-length")
}

#[derive(Debug, Error)]
enum GatewayHttpError {
    #[error("invalid JSON request body: {0}")]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Control(#[from] quillcache_control::ControlError),
    #[error("routed to unknown engine: {0}")]
    MissingEngine(String),
    #[error("upstream request failed: {0}")]
    Upstream(#[from] reqwest::Error),
    #[error("action sink delivery failed: {0}")]
    ActionSink(String),
    #[error("failed to build response: {0}")]
    BuildResponse(axum::http::Error),
}

impl IntoResponse for GatewayHttpError {
    fn into_response(self) -> Response {
        let message = self.to_string();
        let status = match self {
            Self::Json(_) => StatusCode::BAD_REQUEST,
            Self::MissingEngine(_) => StatusCode::BAD_GATEWAY,
            Self::Control(_) | Self::Upstream(_) | Self::ActionSink(_) | Self::BuildResponse(_) => {
                StatusCode::BAD_GATEWAY
            }
        };
        let body = Json(json!({
            "error": {
                "message": message,
                "type": "quillcache_gateway_error"
            }
        }));
        (status, body).into_response()
    }
}

fn _assert_header_value_send_sync(_: HeaderValue) {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_quillcache_hints_before_forwarding() {
        let mut value = json!({
            "model": "Qwen/Qwen3-0.6B",
            "messages": [{"role": "user", "content": "hello"}],
            "quillcache": {
                "request_id": "req-a",
                "block_hashes": ["h0"],
                "block_tokens": 16
            }
        });
        let shape = request_shape_from_payload(&mut value);

        assert!(value.get("quillcache").is_none());
        assert_eq!(shape.id, "req-a");
        assert_eq!(shape.blocks[0].block_hash, "h0");
    }

    #[test]
    fn build_index_defaults_to_memory_and_degrades_gracefully() {
        let base = GatewayConfig {
            bind: "127.0.0.1:0".parse().unwrap(),
            engines: vec![],
            policy: None,
            index: None,
            index_path: None,
            data_plane: None,
            action_sink: None,
        };
        // No backend configured -> in-memory reference.
        assert_eq!(build_index(&base).name(), "memory");
        // An unavailable / uncompiled backend falls back to memory, not a panic.
        let unknown = GatewayConfig {
            index: Some("not-a-backend".to_string()),
            ..base.clone()
        };
        assert_eq!(build_index(&unknown).name(), "memory");
    }

    #[test]
    fn build_data_plane_supports_tiered_runtime_backend() {
        assert_eq!(build_data_plane(None).name(), "none");
        let config = DataPlaneConfig {
            kind: "tiered".to_string(),
            hbm_capacity_bytes: Some(1024),
            cpu_dram_capacity_bytes: Some(2048),
            local_ssd_capacity_bytes: Some(4096),
        };
        let data_plane = build_data_plane(Some(&config));
        // The "tiered" config kind now builds the real store data plane.
        assert_eq!(data_plane.name(), "store");
        assert_eq!(data_plane.metrics().resident_blocks, 0);
    }

    #[test]
    fn build_action_sink_defaults_to_none_and_supports_http() {
        assert!(build_action_sink(None).unwrap().is_none());
        let sink = build_action_sink(Some(&ActionSinkConfig {
            kind: "http".to_string(),
            url: Some("http://127.0.0.1:9090/v1/quillcache/actions".to_string()),
            fail_open: false,
            timeout_ms: 500,
        }))
        .unwrap()
        .unwrap();

        let snapshot = sink.snapshot();
        assert_eq!(snapshot.kind, "http");
        assert!(!snapshot.fail_open);
        assert_eq!(snapshot.timeout_ms, 500);
    }

    #[test]
    fn action_sink_event_carries_request_plan_and_cache_actions() {
        let request = RequestShape {
            id: "req-a".to_string(),
            model_id: "model".to_string(),
            tokenizer_id: "tok".to_string(),
            adapter_id: None,
            tenant_id: "tenant".to_string(),
            session_id: None,
            blocks: vec![],
            estimated_decode_tokens: 1,
            slo: SloTarget::default(),
        };
        let route = GatewayRouteTrace {
            request_id: "req-a".to_string(),
            mode: ServingMode::Aggregated,
            engine_id: "vllm-a".to_string(),
            prefill_engine_id: None,
            decode_engine_id: "vllm-a".to_string(),
            planner_actions: 1,
            reusable_blocks: 0,
            local_hits: 0,
            transfer_blocks: 0,
            recompute_blocks: 0,
            reuse_refused: 0,
            estimated_ttft_us: 10,
            estimated_tpot_us: 20,
        };
        let plan = ActionSinkPlan {
            mode: ServingMode::Aggregated,
            execution_worker_id: "vllm-a".to_string(),
            prefill_worker_id: None,
            decode_worker_id: "vllm-a".to_string(),
            actions: vec![],
        };
        let event = ActionSinkEvent::new(
            ActionSinkPhase::Planned,
            "/v1/chat/completions",
            &request,
            &route,
            plan,
            vec![],
        );

        assert_eq!(event.schema_version, 1);
        assert_eq!(event.phase, ActionSinkPhase::Planned);
        assert_eq!(event.request.id, "req-a");
        assert_eq!(event.route.engine_id, "vllm-a");
    }

    #[test]
    fn builds_fallback_block_when_no_hints_exist() {
        let mut value = json!({
            "model": "Qwen/Qwen3-0.6B",
            "prompt": "hello",
            "max_tokens": 8
        });
        let shape = request_shape_from_payload(&mut value);

        assert_eq!(shape.model_id, "Qwen/Qwen3-0.6B");
        assert_eq!(shape.estimated_decode_tokens, 8);
        // "hello" is one short block.
        assert_eq!(shape.blocks.len(), 1);
        assert!(shape.blocks[0].block_hash.starts_with("pfx-"));
    }

    #[test]
    fn shared_system_prompt_yields_shared_prefix_blocks() {
        // A long shared system prompt (spans several fallback blocks) followed by
        // a per-request user turn — the multi-tenant shared-prompt case.
        let system = "You are a careful assistant. ".repeat(40);
        let make = |question: &str| {
            json!({
                "model": "m",
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": question}
                ]
            })
        };
        let mut a = make("What is 2 + 2?");
        let mut b = make("Name a primary color.");
        let sa = request_shape_from_payload(&mut a);
        let sb = request_shape_from_payload(&mut b);

        // The shared system prefix yields identical leading block hashes (the
        // cache-affinity signal)...
        assert!(sa.blocks.len() >= 2 && sb.blocks.len() >= 2);
        assert_eq!(sa.blocks[0].block_hash, sb.blocks[0].block_hash);
        // ...while the diverging user turn changes the trailing block.
        assert_ne!(
            sa.blocks.last().unwrap().block_hash,
            sb.blocks.last().unwrap().block_hash
        );
    }
}
