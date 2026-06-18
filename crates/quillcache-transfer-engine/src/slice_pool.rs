//! Slice pipelining (Mooncake's `worker_pool`) — split a transfer into fixed-size
//! slices and run them with a **bounded in-flight depth**, instead of one big
//! synchronous round-trip. Mooncake's worker pool posts many RDMA slices across
//! QPs and polls completions; this is the same idea expressed with async tasks.
//!
//! Generic over the per-slice op so the slicing + bounded-concurrency logic is
//! tested without a NIC; the `rdma` transport plugs in its one-sided read/write
//! per slice (the byte movement, which is what only Linux CI can exercise).

use crate::transport::TransferError;
use std::future::Future;

/// The `(offset, len)` slices covering `[0, total_len)` at `slice_size`
/// granularity (the last slice carries the remainder).
pub fn slices(total_len: u64, slice_size: u64) -> impl Iterator<Item = (u64, u64)> {
    let step = slice_size.max(1);
    (0..)
        .map(move |i| i * step)
        .take_while(move |&off| off < total_len)
        .map(move |off| (off, step.min(total_len - off)))
}

/// Run every slice of a `total_len`-byte transfer through `op`, with at most
/// `max_inflight` slices in flight at once (Mooncake's worker-pool depth). Refills
/// as slices complete. Returns the first error; in-flight slices still drain.
pub async fn run_slices<F, Fut>(
    total_len: u64,
    slice_size: u64,
    max_inflight: usize,
    op: F,
) -> Result<(), TransferError>
where
    F: Fn(u64, u64) -> Fut,
    Fut: Future<Output = Result<(), TransferError>> + Send + 'static,
{
    use tokio::task::JoinSet;
    let max = max_inflight.max(1);
    let mut iter = slices(total_len, slice_size);
    let mut set: JoinSet<Result<(), TransferError>> = JoinSet::new();
    // Prime up to `max` slices.
    for _ in 0..max {
        match iter.next() {
            Some((off, len)) => {
                set.spawn(op(off, len));
            }
            None => break,
        }
    }
    let mut first_err: Option<TransferError> = None;
    while let Some(joined) = set.join_next().await {
        match joined {
            Ok(Ok(())) => {}
            Ok(Err(e)) => {
                first_err.get_or_insert(e);
            }
            Err(e) => {
                first_err.get_or_insert(TransferError::Io(e.to_string()));
            }
        }
        // Keep the pipeline full while healthy; stop launching new on first error.
        if first_err.is_none() {
            if let Some((off, len)) = iter.next() {
                set.spawn(op(off, len));
            }
        }
    }
    match first_err {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

/// Per-layer `(layer_index, offset, len)` slices for a KV blob whose layers have
/// sizes `layer_sizes` (in transformer order). Offsets are the running prefix sum,
/// so the slices tile `[0, sum(layer_sizes))` exactly — one slice per layer, at the
/// **semantic layer boundary** rather than a fixed byte size. This is the unit the
/// layer-wise overlap pipeline moves and signals on.
pub fn layer_slices(layer_sizes: &[u64]) -> Vec<(usize, u64, u64)> {
    let mut out = Vec::with_capacity(layer_sizes.len());
    let mut off = 0u64;
    for (i, &len) in layer_sizes.iter().enumerate() {
        out.push((i, off, len));
        off += len;
    }
    out
}

/// Transfer every layer of a KV blob with at most `max_inflight` layers in flight,
/// and deliver `on_layer_ready(i)` **strictly in layer order** (0, 1, 2, …) as soon
/// as layer `i` *and every layer before it* has landed — even when the underlying
/// transfers complete out of order. A reorder gate (the `watermark`) holds back a
/// completed-but-early layer until its predecessors arrive.
///
/// This is the layer-wise overlap primitive: the consumer (decode) starts consuming
/// layer 0 the instant it is ready while later layers are still in flight, so KV
/// transfer hides behind compute instead of being one monolithic barrier — the
/// consumer-start latency becomes time-to-first-layer, not time-to-all-layers.
///
/// `op(layer_index, offset, len)` moves one layer's bytes (backend-agnostic: TCP
/// today, RDMA / GPUDirect later). Returns the first error; in-flight layers still
/// drain, and no notify is delivered past the first gap.
pub async fn run_layers_with_notify<F, Fut, N>(
    layer_sizes: &[u64],
    max_inflight: usize,
    op: F,
    mut on_layer_ready: N,
) -> Result<(), TransferError>
where
    F: Fn(usize, u64, u64) -> Fut,
    Fut: Future<Output = Result<(), TransferError>> + Send + 'static,
    N: FnMut(usize),
{
    use std::collections::BTreeSet;
    use tokio::task::JoinSet;

    let layers = layer_slices(layer_sizes);
    let n = layers.len();
    let max = max_inflight.max(1);
    let mut next_idx = 0usize;
    let mut set: JoinSet<Result<usize, TransferError>> = JoinSet::new();

    // Prime up to `max` layers; each task reports back its own layer index.
    while next_idx < n && set.len() < max {
        let (i, off, len) = layers[next_idx];
        let fut = op(i, off, len);
        set.spawn(async move { fut.await.map(|()| i) });
        next_idx += 1;
    }

    let mut first_err: Option<TransferError> = None;
    let mut done: BTreeSet<usize> = BTreeSet::new();
    let mut watermark = 0usize; // next layer index eligible to notify

    while let Some(joined) = set.join_next().await {
        match joined {
            Ok(Ok(i)) => {
                done.insert(i);
                // Release in-order notifications for every consecutive landed layer.
                if first_err.is_none() {
                    while done.remove(&watermark) {
                        on_layer_ready(watermark);
                        watermark += 1;
                    }
                }
            }
            Ok(Err(e)) => {
                first_err.get_or_insert(e);
            }
            Err(e) => {
                first_err.get_or_insert(TransferError::Io(e.to_string()));
            }
        }
        // Keep the pipeline full while healthy; stop launching new on first error.
        if first_err.is_none() && next_idx < n {
            let (i, off, len) = layers[next_idx];
            let fut = op(i, off, len);
            set.spawn(async move { fut.await.map(|()| i) });
            next_idx += 1;
        }
    }

    match first_err {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    #[test]
    fn slices_cover_the_range_exactly() {
        let s: Vec<(u64, u64)> = slices(10, 4).collect();
        assert_eq!(s, vec![(0, 4), (4, 4), (8, 2)]);
        // Sum of lengths == total, offsets contiguous.
        assert_eq!(s.iter().map(|(_, l)| l).sum::<u64>(), 10);
        assert!(slices(0, 4).next().is_none());
        // Exact multiple: no zero-length remainder slice.
        assert_eq!(slices(8, 4).collect::<Vec<_>>(), vec![(0, 4), (4, 4)]);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn run_slices_runs_all_within_the_inflight_bound() {
        let total = 100u64;
        let slice = 10u64; // 10 slices
        let max = 3usize;
        let cur = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let done = Arc::new(AtomicUsize::new(0));
        let (c, p, d) = (cur.clone(), peak.clone(), done.clone());
        run_slices(total, slice, max, move |_off, _len| {
            let (c, p, d) = (c.clone(), p.clone(), d.clone());
            async move {
                let now = c.fetch_add(1, Ordering::SeqCst) + 1;
                p.fetch_max(now, Ordering::SeqCst);
                tokio::time::sleep(std::time::Duration::from_millis(5)).await;
                c.fetch_sub(1, Ordering::SeqCst);
                d.fetch_add(1, Ordering::SeqCst);
                Ok(())
            }
        })
        .await
        .unwrap();
        assert_eq!(done.load(Ordering::SeqCst), 10, "every slice ran");
        assert!(
            peak.load(Ordering::SeqCst) <= max,
            "never more than max_inflight slices at once (peak {})",
            peak.load(Ordering::SeqCst)
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn run_slices_propagates_an_error() {
        let result = run_slices(100, 10, 4, move |off, _len| async move {
            if off == 50 {
                Err(TransferError::Io("boom".into()))
            } else {
                Ok(())
            }
        })
        .await;
        assert!(matches!(result, Err(TransferError::Io(_))));
    }

    #[test]
    fn layer_slices_tile_the_blob_exactly() {
        let s = layer_slices(&[3, 5, 2]);
        assert_eq!(s, vec![(0, 0, 3), (1, 3, 5), (2, 8, 2)]);
        assert_eq!(s.iter().map(|(_, _, l)| l).sum::<u64>(), 10);
        assert!(layer_slices(&[]).is_empty());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn run_layers_notifies_in_order_despite_out_of_order_completion() {
        // Layer 0's transfer is the slowest, so it completes LAST — but it must be
        // notified FIRST, and the rest strictly in order, proving the reorder gate
        // (and that layers 1..3 overlapped while layer 0 was still in flight).
        let sizes = [10u64, 10, 10, 10];
        let ran = Arc::new(AtomicUsize::new(0));
        let notified = Arc::new(std::sync::Mutex::new(Vec::<usize>::new()));
        let r = ran.clone();
        let nf = notified.clone();
        run_layers_with_notify(
            &sizes,
            4, // all in flight at once
            move |i, _off, _len| {
                let r = r.clone();
                async move {
                    let ms = if i == 0 { 40 } else { 5 };
                    tokio::time::sleep(std::time::Duration::from_millis(ms)).await;
                    r.fetch_add(1, Ordering::SeqCst);
                    Ok(())
                }
            },
            move |i| nf.lock().unwrap().push(i),
        )
        .await
        .unwrap();
        assert_eq!(ran.load(Ordering::SeqCst), 4, "every layer transferred");
        assert_eq!(
            *notified.lock().unwrap(),
            vec![0, 1, 2, 3],
            "notifications delivered strictly in layer order"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn run_layers_stops_notifying_at_an_error_gap() {
        let notified = Arc::new(std::sync::Mutex::new(Vec::<usize>::new()));
        let nf = notified.clone();
        let result = run_layers_with_notify(
            &[10, 10, 10, 10],
            4,
            move |i, _off, _len| async move {
                if i == 1 {
                    Err(TransferError::Io("boom".into()))
                } else {
                    Ok(())
                }
            },
            move |i| nf.lock().unwrap().push(i),
        )
        .await;
        assert!(matches!(result, Err(TransferError::Io(_))));
        let got = notified.lock().unwrap().clone();
        assert!(
            !got.contains(&2) && !got.contains(&3),
            "no notify past the error gap (got {got:?})"
        );
    }

    // A coarse wall-clock benchmark (not a microbench) demonstrating the core thesis:
    // with layer-wise overlap, KV transfer hides behind the consumer's compute
    // instead of blocking it — the consumer starts far earlier and total wall-clock
    // drops. Transfer runs on the tokio runtime; the consumer runs on its own thread,
    // gated in-order by the per-layer notify — the real two-pipeline shape.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn layer_wise_overlap_hides_transfer_behind_compute() {
        use std::sync::mpsc;
        use std::time::{Duration, Instant};

        const N: usize = 6;
        const XFER_MS: u64 = 20; // transfer cost per layer
        const COMPUTE_MS: u64 = 20; // consumer (decode) compute per layer
        const DEPTH: usize = 2; // bandwidth-limited: 2 layers move at once
        let sizes = [1u64; N];

        // Zero-capture closure → Copy, so it can drive both scenarios.
        let xfer = |_: usize, _: u64, _: u64| async {
            tokio::time::sleep(Duration::from_millis(XFER_MS)).await;
            Ok(())
        };

        // ---- Monolithic: a barrier — transfer ALL layers, THEN consume ALL. ----
        let t0 = Instant::now();
        run_layers_with_notify(&sizes, DEPTH, xfer, |_| {})
            .await
            .unwrap();
        let mono_consumer_start = t0.elapsed();
        for _ in 0..N {
            std::thread::sleep(Duration::from_millis(COMPUTE_MS));
        }
        let mono_total = t0.elapsed();

        // ---- Layer-wise overlap: consume each layer the instant it lands. ----
        let (tx, rx) = mpsc::channel::<usize>();
        let consumer = std::thread::spawn(move || {
            while rx.recv().is_ok() {
                std::thread::sleep(Duration::from_millis(COMPUTE_MS));
            }
        });
        let first_ready = Arc::new(std::sync::Mutex::new(None));
        let fr = first_ready.clone();
        let t1 = Instant::now();
        run_layers_with_notify(&sizes, DEPTH, xfer, move |i| {
            if i == 0 {
                *fr.lock().unwrap() = Some(t1.elapsed());
            }
            tx.send(i).unwrap();
        })
        .await
        .unwrap();
        consumer.join().unwrap();
        let overlap_total = t1.elapsed();
        let overlap_consumer_start = first_ready.lock().unwrap().unwrap();

        println!(
            "\nlayer-wise overlap (N={N}, xfer={XFER_MS}ms/layer, compute={COMPUTE_MS}ms/layer, depth={DEPTH}):\n  \
             consumer-start : monolithic {mono_consumer_start:?}  vs  overlap {overlap_consumer_start:?}\n  \
             total wall     : monolithic {mono_total:?}  vs  overlap {overlap_total:?}\n"
        );

        assert!(
            overlap_consumer_start * 2 < mono_consumer_start,
            "overlap must start the consumer far earlier ({overlap_consumer_start:?} vs {mono_consumer_start:?})"
        );
        assert!(
            overlap_total + Duration::from_millis(XFER_MS) < mono_total,
            "overlap must hide transfer behind compute ({overlap_total:?} vs {mono_total:?})"
        );
    }
}
