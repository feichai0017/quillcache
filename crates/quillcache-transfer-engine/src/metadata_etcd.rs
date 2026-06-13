//! etcd-backed metadata (Mooncake's `etcd://` `MetadataStoragePlugin`) — the
//! production distributed metadata path. Segment descriptors live in etcd under a
//! key prefix; a background **watch** keeps a local cache fresh so the sync
//! [`MetadataBackend`] reads stay fast (this is how a real distributed metadata
//! cache works — Mooncake caches segment metadata locally and syncs). Behind the
//! `etcd` feature; needs a running etcd cluster (like the RDMA backend needs a
//! NIC).

use crate::metadata::{MetadataBackend, SegmentDesc};
use etcd_client::{Client, EventType, GetOptions, WatchOptions};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// etcd metadata backend: a local cache fed by an etcd watch, with writes pushed
/// to etcd so peers' watches pick them up.
pub struct EtcdMetadata {
    client: Client,
    prefix: String,
    cache: Arc<Mutex<HashMap<String, SegmentDesc>>>,
}

impl std::fmt::Debug for EtcdMetadata {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EtcdMetadata")
            .field("prefix", &self.prefix)
            .field("cached_segments", &self.cache.lock().unwrap().len())
            .finish()
    }
}

impl EtcdMetadata {
    /// Connect to etcd, load existing segments under `prefix`, and start a watch
    /// that keeps the local cache in sync. Must be called from within a Tokio
    /// runtime (the watch + writes run as spawned tasks).
    pub async fn connect(
        endpoints: Vec<String>,
        prefix: impl Into<String>,
    ) -> Result<Self, etcd_client::Error> {
        let prefix = prefix.into();
        let mut client = Client::connect(endpoints, None).await?;
        let cache: Arc<Mutex<HashMap<String, SegmentDesc>>> = Arc::new(Mutex::new(HashMap::new()));

        // Initial load: every descriptor already under the prefix.
        let resp = client
            .get(prefix.clone(), Some(GetOptions::new().with_prefix()))
            .await?;
        {
            let mut guard = cache.lock().unwrap();
            for kv in resp.kvs() {
                if let Ok(desc) = serde_json::from_slice::<SegmentDesc>(kv.value()) {
                    guard.insert(desc.name.clone(), desc);
                }
            }
        }

        // Background watch keeps the cache fresh as peers publish / remove segments.
        let (watcher, mut stream) = client
            .watch(prefix.clone(), Some(WatchOptions::new().with_prefix()))
            .await?;
        let watch_cache = cache.clone();
        let watch_prefix = prefix.clone();
        tokio::spawn(async move {
            let _watcher = watcher; // dropping it cancels the watch — keep it alive
            while let Ok(Some(resp)) = stream.message().await {
                for event in resp.events() {
                    let Some(kv) = event.kv() else { continue };
                    match event.event_type() {
                        EventType::Put => {
                            if let Ok(desc) = serde_json::from_slice::<SegmentDesc>(kv.value()) {
                                watch_cache.lock().unwrap().insert(desc.name.clone(), desc);
                            }
                        }
                        EventType::Delete => {
                            if let Ok(key) = std::str::from_utf8(kv.key()) {
                                if let Some(name) = key.strip_prefix(watch_prefix.as_str()) {
                                    watch_cache.lock().unwrap().remove(name);
                                }
                            }
                        }
                    }
                }
            }
        });

        Ok(Self {
            client,
            prefix,
            cache,
        })
    }

    fn key(&self, name: &str) -> String {
        format!("{}{}", self.prefix, name)
    }
}

impl MetadataBackend for EtcdMetadata {
    fn put_segment(&self, desc: SegmentDesc) {
        // Update the local cache now; push to etcd async so peers' watches see it.
        self.cache
            .lock()
            .unwrap()
            .insert(desc.name.clone(), desc.clone());
        let mut client = self.client.clone();
        let key = self.key(&desc.name);
        let value = serde_json::to_vec(&desc).unwrap_or_default();
        tokio::spawn(async move {
            let _ = client.put(key, value, None).await;
        });
    }

    fn get_segment(&self, name: &str) -> Option<SegmentDesc> {
        self.cache.lock().unwrap().get(name).cloned()
    }

    fn remove_segment(&self, name: &str) {
        self.cache.lock().unwrap().remove(name);
        let mut client = self.client.clone();
        let key = self.key(name);
        tokio::spawn(async move {
            let _ = client.delete(key, None).await;
        });
    }

    fn segment_names(&self) -> Vec<String> {
        self.cache.lock().unwrap().keys().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metadata::BufferDesc;
    use std::time::Duration;

    // Run a local etcd, then: cargo test -p quillcache-transfer-engine \
    //   --features etcd -- --ignored
    //   (e.g. docker run -d -p 2379:2379 quay.io/coreos/etcd:v3.5.13 \
    //         /usr/local/bin/etcd --advertise-client-urls http://0.0.0.0:2379 \
    //         --listen-client-urls http://0.0.0.0:2379)
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    #[ignore = "needs a running etcd on 127.0.0.1:2379"]
    async fn segment_published_on_one_backend_is_discovered_on_another_via_etcd() {
        let endpoints = vec!["http://127.0.0.1:2379".to_string()];
        let prefix = format!("/quillcache-test/{}/", std::process::id());

        // Two independent backends (separate caches) against the same etcd —
        // like two nodes' transfer engines.
        let node_a = EtcdMetadata::connect(endpoints.clone(), prefix.clone())
            .await
            .unwrap();
        let node_b = EtcdMetadata::connect(endpoints, prefix).await.unwrap();

        // A publishes its segment; B must discover it via the etcd watch.
        node_a.put_segment(SegmentDesc {
            name: "node-a".into(),
            protocol: "tcp".into(),
            endpoint: "127.0.0.1:9000".into(),
            buffers: vec![BufferDesc {
                offset: 0,
                length: 4096,
            }],
        });

        let mut discovered = None;
        for _ in 0..40 {
            if let Some(desc) = node_b.get_segment("node-a") {
                discovered = Some(desc);
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        let desc = discovered.expect("node-b should discover node-a's segment via etcd");
        assert_eq!(desc.endpoint, "127.0.0.1:9000");

        // And a remove on A propagates to B too.
        node_a.remove_segment("node-a");
        let mut gone = false;
        for _ in 0..40 {
            if node_b.get_segment("node-a").is_none() {
                gone = true;
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        assert!(gone, "node-b should see node-a's segment removed via etcd");
    }
}
