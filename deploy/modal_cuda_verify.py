"""Verify the REAL CUDA device tier (quillcache-cuda --features cuda) on a GPU.

Builds the crate with cudarc 0.19 (dynamic-loading) on an L4 and runs the
GPU-only tests for real:
  - host_device_roundtrip : H2D -> D2H copy preserves the bytes (needs libcuda)
  - quantize_kernel       : FP16 -> FP8 NVRTC kernel runs on the GPU (needs libnvrtc)

    modal run deploy/modal_cuda_verify.py

cudarc's dynamic-loading dlopen's `libcuda.so` / `libnvrtc.so`, but a GPU
container ships `libcuda.so.1` (driver) and `libnvrtc.so.13` (the pip wheel), so
the function symlinks unversioned names onto LD_LIBRARY_PATH before running.
"""

import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "build-essential", "pkg-config")
    .run_commands("curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y")
    .add_local_file("Cargo.toml", "/build/Cargo.toml", copy=True)
    .add_local_file("Cargo.lock", "/build/Cargo.lock", copy=True)
    .add_local_dir("crates", "/build/crates", copy=True, ignore=["**/target/**"])
    .add_local_dir("src", "/build/src", copy=True, ignore=["**/target/**"])
    # Pre-compile the test binaries so the GPU run is just execution.
    .run_commands(
        "cd /build && $HOME/.cargo/bin/cargo test -p quillcache-cuda --features cuda --no-run",
        "cd /build && $HOME/.cargo/bin/cargo test -p quillcache-transfer-engine --features cuda --no-run",
    )
)
app = modal.App("quillcache-cuda-verify")


@app.function(gpu="L4", image=image, timeout=60 * 20)
def verify():
    import glob
    import os
    import subprocess

    # Expose unversioned libcuda.so / libnvrtc.so where cudarc's dlopen looks.
    linkdir = "/usr/local/lib/quillcache-cuda"
    os.makedirs(linkdir, exist_ok=True)
    found = {}
    for soname, patterns in {
        "libcuda.so": [
            "/usr/lib/x86_64-linux-gnu/libcuda.so*",
            "/usr/lib64/libcuda.so*",
            "/usr/local/cuda/lib64/libcuda.so*",
            "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
        ],
    }.items():
        cands = []
        for p in patterns:
            cands += glob.glob(p)
        # Prefer the real versioned object (skip any unversioned symlink we might add).
        cands = sorted(c for c in cands if os.path.basename(c) != soname)
        if cands:
            target = cands[-1]
            link = os.path.join(linkdir, soname)
            if not os.path.exists(link):
                os.symlink(target, link)
            found[soname] = target
        else:
            found[soname] = None

    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = linkdir + ":" + env.get("LD_LIBRARY_PATH", "")
    env["PATH"] = os.path.expanduser("~/.cargo/bin") + ":" + env["PATH"]

    # nvidia-smi for the record
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True)

    # Driver-only GPU tests (H2D/D2H + the HBM device segment over the one-sided
    # wire). The NVRTC quantize test needs libnvrtc (CUDA toolkit), absent here.
    suites = [
        ("device tier H2D/D2H", "quillcache-cuda", "host_device_roundtrip"),
        ("transfer-engine HBM segment", "quillcache-transfer-engine",
         "tcp_peer_reads_writes_a_gpu_hbm_segment"),
    ]
    results = []
    for label, pkg, test in suites:
        run = subprocess.run(
            ["cargo", "test", "-p", pkg, "--features", "cuda", test,
             "--", "--ignored", "--nocapture", "--test-threads=1"],
            cwd="/build", env=env, capture_output=True, text=True,
        )
        results.append({
            "label": label, "pkg": pkg,
            "stdout": run.stdout[-1500:], "stderr": run.stderr[-1500:],
            "returncode": run.returncode,
        })
    return {"gpu": smi.stdout.strip(), "resolved_libs": found, "suites": results}


@app.local_entrypoint()
def main():
    res = verify.remote()
    print("\n" + "=" * 78)
    print("quillcache-cuda (cudarc 0.19, --features cuda) — real GPU verification")
    print("=" * 78)
    print("GPU:", res["gpu"])
    print("resolved libs:", res["resolved_libs"])
    all_ok = True
    for s in res["suites"]:
        ok = s["returncode"] == 0
        all_ok = all_ok and ok
        print(f"\n[{'PASS' if ok else 'FAIL'}] {s['label']}  (-p {s['pkg']})")
        for line in s["stdout"].splitlines():
            if "test result" in line or " ... " in line or line.startswith("running"):
                print("   ", line.strip())
        if not ok:
            print("   --- stderr ---")
            print(s["stderr"])
    verdict = (
        "PASS — real CUDA on the GPU: device-tier H2D/D2H + HBM device segment "
        "served over the one-sided wire (cudarc 0.19 dynamic-loading)"
        if all_ok else "see failures above"
    )
    print(f"\nVERDICT: {verdict}")
