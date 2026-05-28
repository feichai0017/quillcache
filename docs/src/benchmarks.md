# Benchmarking

QuillSQL uses benchmarks to separate three different claims:

- JIT lowering cost: how much time QuillSQL spends building the JIT expression
  graph and MLIR.
- DataFusion baseline cost: how the pure DataFusion path behaves on Arrow
  batches or Parquet datasets.
- MLIR JIT cost: how rewritten record and aggregate pipelines behave when
  executable MLIR function pointers are enabled.

The current code has a real `CompiledPipelineExec` node in the DataFusion hot
path for both record and scalar-sum pipelines, and its execution body uses
QuillSQL's fixed-width Arrow batch kernels. With `QUILL_JIT=mlir`, the
multi-column fixed-width record pipeline and the f64 and
TPC-H Q6-style decimal filter/sum paths can dispatch to executable MLIR kernels for
null-free, offset-free fixed-width batches. Other compiled MLIR kernel speedups
are intentionally not claimed as end-to-end query speedups yet.

## Microbenchmarks

`benches/jit_micro.rs` isolates small, repeatable paths:

```bash
cargo bench --bench jit_micro -- --sample-size 10
```

Benchmarks:

| Name | Measures |
| ---- | -------- |
| `lowering/filter_project_graph` | Exact `PipelineGraph` lowering into the record pipeline shape. |
| `compile/mlir_filter_text` | JIT expression to MLIR module generation for a filter. |
| `compile/mlir_filter_project_text` | Fused filter/project MLIR module generation. |
| `compile/mlir_i64_filter` | MLIR parse/lower/JIT cost for the first compiled fixed-width filter kernel. |
| `compile/mlir_record_pipeline` | MLIR parse/lower/JIT cost for the fixed-width record pipeline. |
| `compile/mlir_f64_plain_sum` | MLIR parse/lower/JIT cost for the first compiled fixed-width plain SUM kernel. |
| `compile/mlir_decimal_plain_sum` | MLIR parse/lower/JIT cost for a fixed-width `Date32`/`Decimal128` plain SUM kernel. |
| `kernel/i64_filter_64k` | Compiled MLIR i64 filter execution over a 64K-row values vector, writing a byte selection mask. |
| `kernel/record_pipeline_64k` | Compiled MLIR record pipeline execution over 64K rows, compacting projected fixed-width columns. |
| `kernel/f64_plain_sum_64k` | Compiled MLIR f64 filter/plain-SUM execution over 64K rows. |
| `kernel/decimal_plain_sum_64k` | Compiled MLIR decimal filter/plain-SUM execution over 64K fixed-width column slices. |
| `sql/df/filter_project_64k` | DataFusion SQL planning/execution over a 64K-row in-memory Arrow table, including `CompiledPipelineExec` when the pattern matches. |
| `sql/df/filter_sum_64k` | DataFusion SQL planning/execution over a 64K-row in-memory Arrow table, including `CompiledPipelineExec` when the pattern matches. |
| `sql/df/prepared_filter_sum_64k` | Prepared-plan filter/sum execution that removes SQL parsing and logical-plan construction from the timed loop while still using DataFusion physical planning and execution. |

The benchmark includes `melior` parse and verifier cost, so LLVM/MLIR 22 must
be available:

```bash
MLIR_SYS_220_PREFIX=/opt/homebrew/opt/llvm \
LLVM_SYS_220_PREFIX=/opt/homebrew/opt/llvm \
cargo bench --bench jit_micro -- --sample-size 10
```

## TPC-H

`benches/tpch.rs` is the analytical benchmark harness. It expects Parquet data.
By default it uses the pure Rust `tpchgen-cli` generator to create SF1 data
inside the repository under `benchdata/tpch-sf1`. Each query reports two
measurements:

- `sql/<mode>/<query>` runs through `Database::run`, including SQL parsing,
  logical optimization, physical planning, and execution.
- `prepared/<mode>/<query>` runs a `PreparedQuery`, reusing the SQL/logical plan while
  letting DataFusion create a fresh physical plan for each execution. DataFusion
  physical plans are not assumed to be reentrant.

```bash
cargo bench --bench tpch -- --sample-size 10
scripts/bench_tpch.sh
QUILL_JIT=mlir scripts/bench_tpch.sh
```

Generated data is outside version control. Useful knobs:

| Variable | Meaning |
| -------- | ------- |
| `QUILL_TPCH_SF` | Scale factor for generated data, default `1`. |
| `QUILL_TPCH_GEN_THREADS` | Number of generator threads. |
| `QUILL_TPCH_REGENERATE=1` | Delete and rebuild the generated data directory. |
| `QUILL_TPCH_DIR` | Use an existing Parquet dataset instead of generating one. |
| `QUILL_JIT=off` | Keep the pure DataFusion physical plan for baseline measurements. |
| `QUILL_JIT=mlir` | Use executable MLIR dispatch. This is the default. |

TPC-H mode names are reported as `datafusion/native` and `quill/mlir-jit`.
The primary comparison is the pure DataFusion baseline versus executable MLIR
JIT. Host-runtime ablation is intentionally not part of the benchmark report.

When `QUILL_TPCH_DIR` is set, the directory can contain either
`<table>.parquet` files or table directories:

```text
tpch-parquet/
  lineitem.parquet
  orders.parquet
  customer.parquet
```

or:

```text
tpch-parquet/
  lineitem/
  orders/
  customer/
```

Current query ladder:

| Query | Family | Why it is included |
| ----- | ------ | ------------------ |
| Q6 | scan/filter/plain aggregate | First fixed-width baseline for date/decimal filtering and plain sum. |
| Q1 | scan/filter/grouped aggregate/sort | Exercises aggregate state and result materialization. |
| Q3 | join-heavy aggregate | Adds multi-table build/probe pressure. |

## Reporting Rules

When reporting results, include:

- exact git commit
- command line and feature flags
- CPU, memory, OS, Rust version, and LLVM/MLIR version
- TPC-H scale factor and data format
- whether file-system cache was warm
- whether the number is JIT lowering, DataFusion end-to-end, or compiled
  kernel execution
