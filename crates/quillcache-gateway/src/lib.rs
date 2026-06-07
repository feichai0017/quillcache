use axum::body::Bytes;
use axum::extract::State;
use axum::http::{HeaderMap, HeaderName, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use quillcache_control::{ControlPlane, IngestSummary};
use quillcache_core::{
    EngineEndpoint, ExternalKvBlockKey, KvBlockKey, KvEventBatch, RequestKvHints, RequestShape,
    SloTarget,
};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::hash_map::DefaultHasher;
use std::fs;
use std::hash::{Hash, Hasher};
use std::net::SocketAddr;
use std::path::Path;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
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
    #[error("gateway server failed: {0}")]
    Serve(std::io::Error),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GatewayConfig {
    pub bind: SocketAddr,
    pub engines: Vec<EngineEndpoint>,
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

#[derive(Debug, Clone)]
struct GatewayState {
    control: Arc<RwLock<ControlPlane>>,
    client: Client,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GatewayRouteTrace {
    request_id: String,
    engine_id: String,
    reusable_blocks: usize,
    local_hits: usize,
    transfer_blocks: usize,
    recompute_blocks: usize,
    estimated_ttft_us: u64,
    estimated_tpot_us: u64,
}

pub async fn run_from_config_path(path: impl AsRef<Path>) -> Result<(), GatewayError> {
    let config = GatewayConfig::from_path(path)?;
    run(config).await
}

pub async fn run(config: GatewayConfig) -> Result<(), GatewayError> {
    let state = GatewayState {
        control: Arc::new(RwLock::new(ControlPlane::new(config.engines))),
        client: Client::new(),
    };
    let app = router(state);
    let listener = TcpListener::bind(config.bind)
        .await
        .map_err(|source| GatewayError::Bind {
            addr: config.bind,
            source,
        })?;

    tracing::info!(addr = %config.bind, "starting QuillCache gateway");
    axum::serve(listener, app)
        .await
        .map_err(GatewayError::Serve)
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
    let mut payload: Value = serde_json::from_slice(&body)?;
    let request_shape = request_shape_from_payload(&mut payload);
    let clean_body = serde_json::to_vec(&payload)?;

    let (engine, trace) = {
        let control = state.control.read().await;
        let decision = control.route(&request_shape)?;
        let engine = control
            .engine(&decision.worker_id)
            .cloned()
            .ok_or_else(|| GatewayHttpError::MissingEngine(decision.worker_id.clone()))?;
        let trace = GatewayRouteTrace {
            request_id: decision.request_id.clone(),
            engine_id: decision.worker_id.clone(),
            reusable_blocks: decision.reusable_blocks(),
            local_hits: decision.local_hits.len(),
            transfer_blocks: decision.transfers.len(),
            recompute_blocks: decision.recomputes.len(),
            estimated_ttft_us: decision.estimated_ttft_us,
            estimated_tpot_us: decision.estimated_tpot_us,
        };
        (engine, trace)
    };

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
    request = request.header(
        "x-quillcache-reusable-blocks",
        trace.reusable_blocks.to_string(),
    );

    let upstream = request.send().await?;
    let status = StatusCode::from_u16(upstream.status().as_u16())
        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    let mut response = Response::builder().status(status);
    for (name, value) in upstream.headers() {
        if should_return_header(name) {
            response = response.header(name, value);
        }
    }
    response = response
        .header("x-quillcache-engine-id", trace.engine_id)
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
            "x-quillcache-estimated-ttft-us",
            trace.estimated_ttft_us.to_string(),
        );
    let bytes = upstream.bytes().await?;
    response
        .body(axum::body::Body::from(bytes))
        .map_err(GatewayHttpError::BuildResponse)
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
        blocks,
        estimated_decode_tokens,
        slo: SloTarget::default(),
    }
}

fn fallback_blocks(
    payload: &Value,
    model_id: &str,
    tokenizer_id: &str,
    tenant_id: &str,
) -> Vec<KvBlockKey> {
    let mut hasher = DefaultHasher::new();
    payload.to_string().hash(&mut hasher);
    let block_hash = format!("fallback-{:016x}", hasher.finish());
    vec![KvBlockKey::external_hash(ExternalKvBlockKey {
        model_id: model_id.to_string(),
        tokenizer_id: tokenizer_id.to_string(),
        adapter_id: None,
        tenant_id: tenant_id.to_string(),
        prefix_hash: "root".to_string(),
        block_hash,
        block_index: 0,
        token_count: 64,
    })]
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
    #[error("failed to build response: {0}")]
    BuildResponse(axum::http::Error),
}

impl IntoResponse for GatewayHttpError {
    fn into_response(self) -> Response {
        let message = self.to_string();
        let status = match self {
            Self::Json(_) => StatusCode::BAD_REQUEST,
            Self::MissingEngine(_) => StatusCode::BAD_GATEWAY,
            Self::Control(_) | Self::Upstream(_) | Self::BuildResponse(_) => {
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
    fn builds_fallback_block_when_no_hints_exist() {
        let mut value = json!({
            "model": "Qwen/Qwen3-0.6B",
            "prompt": "hello",
            "max_tokens": 8
        });
        let shape = request_shape_from_payload(&mut value);

        assert_eq!(shape.model_id, "Qwen/Qwen3-0.6B");
        assert_eq!(shape.estimated_decode_tokens, 8);
        assert_eq!(shape.blocks.len(), 1);
        assert!(shape.blocks[0].block_hash.starts_with("fallback-"));
    }
}
