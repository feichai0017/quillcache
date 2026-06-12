//! TransferMetadata (Mooncake's `transfer_metadata.h`) — segment discovery and
//! handshake. A node publishes its [`SegmentDesc`] (name, protocol, endpoint,
//! buffers); peers look it up to `open_segment`. The backend is pluggable
//! (Mooncake: `etcd://` / `redis://` / `http://` / `P2PHANDSHAKE`). Here an
//! in-memory backend is the P2PHANDSHAKE / single-process analogue; etcd / redis
//! / http are the reserved seams (need a metadata server or cluster) and plug in
//! behind [`MetadataBackend`].

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Mutex;

/// A registered buffer within a segment (Mooncake's `BufferDesc`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BufferDesc {
    pub offset: u64,
    pub length: u64,
}

/// A node's RAM segment descriptor (Mooncake's `SegmentDesc`): how to reach it
/// and what it exposes. The minimal fields needed to discover + open a segment;
/// the topology / device list is added when an RDMA backend needs path selection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentDesc {
    pub name: String,
    pub protocol: String,
    pub endpoint: String,
    pub buffers: Vec<BufferDesc>,
}

/// The metadata storage seam (Mooncake's `MetadataStoragePlugin`): resolve a
/// segment name → its descriptor. etcd / redis / http backends implement this.
pub trait MetadataBackend: Send + Sync + std::fmt::Debug {
    fn put_segment(&self, desc: SegmentDesc);
    fn get_segment(&self, name: &str) -> Option<SegmentDesc>;
    fn remove_segment(&self, name: &str);
    fn segment_names(&self) -> Vec<String>;
}

/// In-memory metadata — the P2PHANDSHAKE / single-process backend. Two engines
/// sharing one `Arc<InMemoryMetadata>` discover each other with no external store.
#[derive(Debug, Default)]
pub struct InMemoryMetadata {
    segments: Mutex<HashMap<String, SegmentDesc>>,
}

impl InMemoryMetadata {
    pub fn new() -> Self {
        Self::default()
    }
}

impl MetadataBackend for InMemoryMetadata {
    fn put_segment(&self, desc: SegmentDesc) {
        self.segments
            .lock()
            .unwrap()
            .insert(desc.name.clone(), desc);
    }

    fn get_segment(&self, name: &str) -> Option<SegmentDesc> {
        self.segments.lock().unwrap().get(name).cloned()
    }

    fn remove_segment(&self, name: &str) {
        self.segments.lock().unwrap().remove(name);
    }

    fn segment_names(&self) -> Vec<String> {
        self.segments.lock().unwrap().keys().cloned().collect()
    }
}
