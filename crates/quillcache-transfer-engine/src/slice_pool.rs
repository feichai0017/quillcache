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
}
