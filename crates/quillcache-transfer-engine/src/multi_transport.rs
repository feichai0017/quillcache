//! MultiTransport (Mooncake's `multi_transport.h`) — the registry of installed
//! transports plus per-request backend selection. The engine installs one
//! backend per protocol (`"tcp"`, `"rdma"`, …) and selects by the target
//! segment's protocol. Mooncake's `MultiTransport::selectTransport` also weighs
//! topology; that hook lands with the RDMA backend.

use crate::transport::Transport;
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Debug, Default)]
pub struct MultiTransport {
    transports: HashMap<String, Arc<dyn Transport>>,
}

impl MultiTransport {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn install(&mut self, protocol: impl Into<String>, transport: Arc<dyn Transport>) {
        self.transports.insert(protocol.into(), transport);
    }

    /// Pick the backend for a target segment's protocol.
    pub fn select(&self, protocol: &str) -> Option<Arc<dyn Transport>> {
        self.transports.get(protocol).cloned()
    }

    pub fn installed(&self) -> Vec<String> {
        self.transports.keys().cloned().collect()
    }
}
