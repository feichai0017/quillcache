//! RDMA backend (Mooncake's `RdmaTransport`) — **one-sided RDMA**, Mooncake's
//! primary data path and core IP: RDMA READ/WRITE move bytes directly between
//! registered memory by `(remote_addr, rkey)`, with no remote CPU in the loop —
//! exactly QuillCache's `(segment, offset)` model, kernel-bypass.
//!
//! The real verbs path lives behind `--features rdma` (raw `ibverbs-sys`; needs
//! libibverbs at build and an RDMA NIC *or* SoftRoCE/rxe to run). [`one_sided_roundtrip`]
//! is the verified core: register memory, connect an RC queue-pair, and do an
//! RDMA WRITE then READ against a remote MR — proven over SoftRoCE on commodity
//! hardware (see the `#[ignore]` test). The `RdmaTransport` Transport-trait
//! wiring (TCP-side-channel handshake + a per-endpoint QP pool) builds on this
//! proven mechanism and is the remaining increment; without the feature it stays
//! an `Unsupported` stub so the default build needs no NIC.

use super::{LinkClass, TransferError, Transport};
use async_trait::async_trait;
use bytes::Bytes;

#[derive(Debug, Default)]
pub struct RdmaTransport;

const NEEDS_WIRING: &str = "RDMA Transport wiring (QP handshake/pool) is the next step; the one-sided verbs core is verified — see rdma::one_sided_roundtrip (build --features rdma)";

#[async_trait]
impl Transport for RdmaTransport {
    fn name(&self) -> &str {
        "rdma"
    }

    fn link_class(&self) -> LinkClass {
        LinkClass::RdmaRoce
    }

    async fn read_remote(
        &self,
        _endpoint: &str,
        _offset: u64,
        _length: u64,
    ) -> Result<Bytes, TransferError> {
        Err(TransferError::Unsupported(NEEDS_WIRING))
    }

    async fn write_remote(
        &self,
        _endpoint: &str,
        _offset: u64,
        _data: Bytes,
    ) -> Result<(), TransferError> {
        Err(TransferError::Unsupported(NEEDS_WIRING))
    }
}

// =============================================================================
// Real one-sided RDMA over ibverbs (feature `rdma`). Verified over SoftRoCE.
// =============================================================================

#[cfg(feature = "rdma")]
pub use verbs::one_sided_roundtrip;

#[cfg(feature = "rdma")]
mod verbs {
    use ibverbs_sys as ffi;
    use std::ffi::{c_int, c_void, CStr};
    use std::ptr;

    /// One queue-pair's wire identity, exchanged to connect two RC QPs.
    #[derive(Clone, Copy)]
    struct QpEndpoint {
        qpn: u32,
        psn: u32,
        gid: ffi::ibv_gid,
    }

    fn errno() -> i32 {
        std::io::Error::last_os_error().raw_os_error().unwrap_or(-1)
    }

    /// Open an ibverbs device by name (e.g. "rxe0"); first device if `None`.
    unsafe fn open_device(want: Option<&str>) -> Result<*mut ffi::ibv_context, String> {
        let mut num: c_int = 0;
        let list = ffi::ibv_get_device_list(&mut num as *mut _);
        if list.is_null() || num == 0 {
            return Err("no RDMA devices (is rxe0 up? `rdma link show`)".into());
        }
        let mut chosen: *mut ffi::ibv_device = ptr::null_mut();
        for i in 0..num as isize {
            let dev = *list.offset(i);
            match want {
                None => {
                    chosen = dev;
                    break;
                }
                Some(name) => {
                    let dname = CStr::from_ptr(ffi::ibv_get_device_name(dev));
                    if dname.to_string_lossy() == name {
                        chosen = dev;
                        break;
                    }
                }
            }
        }
        if chosen.is_null() {
            ffi::ibv_free_device_list(list);
            return Err(format!("RDMA device {want:?} not found"));
        }
        let ctx = ffi::ibv_open_device(chosen);
        ffi::ibv_free_device_list(list);
        if ctx.is_null() {
            return Err("ibv_open_device failed".into());
        }
        Ok(ctx)
    }

    /// The RoCEv2 GID at `gid_index` on `port` (index 1 = RoCEv2 IPv4 on rxe).
    unsafe fn query_gid(
        ctx: *mut ffi::ibv_context,
        port: u8,
        gid_index: c_int,
    ) -> Result<ffi::ibv_gid, String> {
        let mut gid: ffi::ibv_gid = std::mem::zeroed();
        if ffi::ibv_query_gid(ctx, port, gid_index, &mut gid as *mut _) != 0 {
            return Err(format!(
                "ibv_query_gid(port={port}, index={gid_index}) failed"
            ));
        }
        Ok(gid)
    }

    /// Post a single one-sided RDMA WR (WRITE or READ) and wait for its completion.
    #[allow(clippy::too_many_arguments)]
    unsafe fn post_and_wait(
        qp: *mut ffi::ibv_qp,
        cq: *mut ffi::ibv_cq,
        opcode: ffi::ibv_wr_opcode::Type,
        local_addr: u64,
        lkey: u32,
        len: u32,
        remote_addr: u64,
        rkey: u32,
    ) -> Result<(), String> {
        let mut sge = ffi::ibv_sge {
            addr: local_addr,
            length: len,
            lkey,
        };
        let mut wr: ffi::ibv_send_wr = std::mem::zeroed();
        wr.wr_id = 1;
        wr.next = ptr::null_mut();
        wr.sg_list = &mut sge as *mut _;
        wr.num_sge = 1;
        wr.opcode = opcode;
        wr.send_flags = ffi::ibv_send_flags::IBV_SEND_SIGNALED.0;
        wr.wr.rdma.remote_addr = remote_addr;
        wr.wr.rdma.rkey = rkey;

        let mut bad: *mut ffi::ibv_send_wr = ptr::null_mut();
        let ctx = (*qp).context;
        let ops = &mut (*ctx).ops;
        let rc = ops.post_send.as_mut().unwrap()(qp, &mut wr as *mut _, &mut bad as *mut _);
        if rc != 0 {
            return Err(format!("ibv_post_send rc={rc} errno={}", errno()));
        }

        // Poll the CQ until the completion lands.
        let cqctx = (*cq).context;
        let cqops = &mut (*cqctx).ops;
        let mut wc: ffi::ibv_wc = std::mem::zeroed();
        loop {
            let n = cqops.poll_cq.as_mut().unwrap()(cq, 1, &mut wc as *mut _);
            if n < 0 {
                return Err("ibv_poll_cq failed".into());
            }
            if n == 0 {
                continue;
            }
            if let Some((status, vendor_err)) = wc.error() {
                return Err(format!(
                    "RDMA op failed: wc.status={status} vendor_err={vendor_err}"
                ));
            }
            return Ok(());
        }
    }

    /// Bring an RC QP INIT → RTR → RTS, connected to `remote` over RoCEv2.
    unsafe fn connect_qp(
        qp: *mut ffi::ibv_qp,
        local_psn: u32,
        remote: &QpEndpoint,
        port: u8,
        gid_index: u8,
    ) -> Result<(), String> {
        use ffi::ibv_qp_attr_mask as M;

        // INIT
        let mut attr: ffi::ibv_qp_attr = std::mem::zeroed();
        attr.qp_state = ffi::ibv_qp_state::IBV_QPS_INIT;
        attr.pkey_index = 0;
        attr.port_num = port;
        attr.qp_access_flags = ffi::ibv_access_flags::IBV_ACCESS_LOCAL_WRITE.0
            | ffi::ibv_access_flags::IBV_ACCESS_REMOTE_WRITE.0
            | ffi::ibv_access_flags::IBV_ACCESS_REMOTE_READ.0;
        let mask = (M::IBV_QP_STATE.0
            | M::IBV_QP_PKEY_INDEX.0
            | M::IBV_QP_PORT.0
            | M::IBV_QP_ACCESS_FLAGS.0) as c_int;
        if ffi::ibv_modify_qp(qp, &mut attr as *mut _, mask) != 0 {
            return Err(format!("modify_qp INIT failed errno={}", errno()));
        }

        // RTR — RoCE requires global routing (GRH) with the remote's GID.
        let mut attr: ffi::ibv_qp_attr = std::mem::zeroed();
        attr.qp_state = ffi::ibv_qp_state::IBV_QPS_RTR;
        attr.path_mtu = ffi::IBV_MTU_1024;
        attr.dest_qp_num = remote.qpn;
        attr.rq_psn = remote.psn;
        attr.max_dest_rd_atomic = 1;
        attr.min_rnr_timer = 12;
        attr.ah_attr.is_global = 1;
        attr.ah_attr.dlid = 0;
        attr.ah_attr.sl = 0;
        attr.ah_attr.src_path_bits = 0;
        attr.ah_attr.port_num = port;
        attr.ah_attr.grh.dgid = remote.gid;
        attr.ah_attr.grh.sgid_index = gid_index;
        attr.ah_attr.grh.hop_limit = 1;
        attr.ah_attr.grh.traffic_class = 0;
        attr.ah_attr.grh.flow_label = 0;
        let mask = (M::IBV_QP_STATE.0
            | M::IBV_QP_AV.0
            | M::IBV_QP_PATH_MTU.0
            | M::IBV_QP_DEST_QPN.0
            | M::IBV_QP_RQ_PSN.0
            | M::IBV_QP_MAX_DEST_RD_ATOMIC.0
            | M::IBV_QP_MIN_RNR_TIMER.0) as c_int;
        if ffi::ibv_modify_qp(qp, &mut attr as *mut _, mask) != 0 {
            return Err(format!("modify_qp RTR failed errno={}", errno()));
        }

        // RTS
        let mut attr: ffi::ibv_qp_attr = std::mem::zeroed();
        attr.qp_state = ffi::ibv_qp_state::IBV_QPS_RTS;
        attr.timeout = 14;
        attr.retry_cnt = 7;
        attr.rnr_retry = 7;
        attr.sq_psn = local_psn;
        attr.max_rd_atomic = 1;
        let mask = (M::IBV_QP_STATE.0
            | M::IBV_QP_TIMEOUT.0
            | M::IBV_QP_RETRY_CNT.0
            | M::IBV_QP_RNR_RETRY.0
            | M::IBV_QP_SQ_PSN.0
            | M::IBV_QP_MAX_QP_RD_ATOMIC.0) as c_int;
        if ffi::ibv_modify_qp(qp, &mut attr as *mut _, mask) != 0 {
            return Err(format!("modify_qp RTS failed errno={}", errno()));
        }
        Ok(())
    }

    /// Real one-sided RDMA round-trip over the given device (e.g. "rxe0"): two RC
    /// queue-pairs in one process, connected RoCEv2. RDMA-WRITE `payload` from a
    /// source MR into a destination MR (`(dst.addr, dst.rkey)`), then RDMA-READ it
    /// back from the destination into a third MR — proving both one-sided verbs.
    /// Returns the read-back bytes (the caller asserts they equal `payload`).
    ///
    /// This is the exact mechanism QuillCache's transfer engine moves KV with on
    /// real RDMA — no remote CPU touches the data; the HCA does the copy by
    /// `(remote_addr, rkey)`. Verified over SoftRoCE; a real NIC is a drop-in.
    pub fn one_sided_roundtrip(device: &str, payload: &[u8]) -> Result<Vec<u8>, String> {
        const PORT: u8 = 1;
        const GID_INDEX: u8 = 1; // RoCEv2 IPv4 on rxe (see `rdma link`/gid_attrs)
        const PSN_A: u32 = 0;
        const PSN_B: u32 = 0;
        let len = payload.len();
        if len == 0 {
            return Ok(Vec::new());
        }

        unsafe {
            let ctx = open_device(Some(device))?;
            let pd = ffi::ibv_alloc_pd(ctx);
            if pd.is_null() {
                return Err("ibv_alloc_pd failed".into());
            }
            let gid = query_gid(ctx, PORT, GID_INDEX as c_int)?;

            // Three host buffers + MRs in the one PD: source, destination, readback.
            let mut src = payload.to_vec();
            let mut dst = vec![0u8; len];
            let mut back = vec![0u8; len];
            let access = (ffi::ibv_access_flags::IBV_ACCESS_LOCAL_WRITE.0
                | ffi::ibv_access_flags::IBV_ACCESS_REMOTE_WRITE.0
                | ffi::ibv_access_flags::IBV_ACCESS_REMOTE_READ.0)
                as c_int;
            let reg = |buf: &mut [u8]| -> *mut ffi::ibv_mr {
                ffi::ibv_reg_mr(pd, buf.as_mut_ptr() as *mut c_void, buf.len(), access)
            };
            let mr_src = reg(&mut src);
            let mr_dst = reg(&mut dst);
            let mr_back = reg(&mut back);
            if mr_src.is_null() || mr_dst.is_null() || mr_back.is_null() {
                return Err("ibv_reg_mr failed".into());
            }

            // Two RC queue-pairs (A active, B passive), each with its own CQ.
            let cq_a = ffi::ibv_create_cq(ctx, 16, ptr::null_mut(), ptr::null_mut(), 0);
            let cq_b = ffi::ibv_create_cq(ctx, 16, ptr::null_mut(), ptr::null_mut(), 0);
            if cq_a.is_null() || cq_b.is_null() {
                return Err("ibv_create_cq failed".into());
            }
            let mk_qp = |cq: *mut ffi::ibv_cq| -> *mut ffi::ibv_qp {
                let mut ia: ffi::ibv_qp_init_attr = std::mem::zeroed();
                ia.send_cq = cq;
                ia.recv_cq = cq;
                ia.qp_type = ffi::ibv_qp_type::IBV_QPT_RC;
                ia.cap.max_send_wr = 16;
                ia.cap.max_recv_wr = 16;
                ia.cap.max_send_sge = 1;
                ia.cap.max_recv_sge = 1;
                ffi::ibv_create_qp(pd, &mut ia as *mut _)
            };
            let qp_a = mk_qp(cq_a);
            let qp_b = mk_qp(cq_b);
            if qp_a.is_null() || qp_b.is_null() {
                return Err("ibv_create_qp failed".into());
            }

            let ep_a = QpEndpoint {
                qpn: (*qp_a).qp_num,
                psn: PSN_A,
                gid,
            };
            let ep_b = QpEndpoint {
                qpn: (*qp_b).qp_num,
                psn: PSN_B,
                gid,
            };
            connect_qp(qp_a, PSN_A, &ep_b, PORT, GID_INDEX)?;
            connect_qp(qp_b, PSN_B, &ep_a, PORT, GID_INDEX)?;

            // 1) RDMA WRITE: src -> dst (one-sided; qp_b's CPU is not involved).
            post_and_wait(
                qp_a,
                cq_a,
                ffi::ibv_wr_opcode::IBV_WR_RDMA_WRITE,
                (*mr_src).addr as u64,
                (*mr_src).lkey,
                len as u32,
                (*mr_dst).addr as u64,
                (*mr_dst).rkey,
            )?;
            if dst != payload {
                return Err("RDMA WRITE did not land the payload in dst".into());
            }

            // 2) RDMA READ: dst -> back (read remote memory by (addr, rkey)).
            post_and_wait(
                qp_a,
                cq_a,
                ffi::ibv_wr_opcode::IBV_WR_RDMA_READ,
                (*mr_back).addr as u64,
                (*mr_back).lkey,
                len as u32,
                (*mr_dst).addr as u64,
                (*mr_dst).rkey,
            )?;

            // Teardown (best-effort).
            ffi::ibv_destroy_qp(qp_a);
            ffi::ibv_destroy_qp(qp_b);
            ffi::ibv_destroy_cq(cq_a);
            ffi::ibv_destroy_cq(cq_b);
            ffi::ibv_dereg_mr(mr_src);
            ffi::ibv_dereg_mr(mr_dst);
            ffi::ibv_dereg_mr(mr_back);
            ffi::ibv_dealloc_pd(pd);
            ffi::ibv_close_device(ctx);

            Ok(back)
        }
    }
}

#[cfg(all(test, feature = "rdma"))]
mod tests {
    use super::*;

    // Needs an ibverbs device — a real RDMA NIC or SoftRoCE (rxe). Set it up with
    // `sudo modprobe rdma_rxe && sudo rdma link add rxe0 type rxe netdev <iface>`,
    // then: `cargo test -p quillcache-transfer-engine --features rdma -- --ignored`.
    #[test]
    #[ignore = "requires an ibverbs device (real NIC or SoftRoCE/rxe)"]
    fn one_sided_rdma_write_then_read_roundtrip() {
        let payload: Vec<u8> = (0..4096).map(|i| (i % 251) as u8).collect();
        let back = one_sided_roundtrip("rxe0", &payload).expect("one-sided RDMA round-trip");
        assert_eq!(
            back, payload,
            "RDMA WRITE then READ must round-trip the bytes"
        );
    }
}
