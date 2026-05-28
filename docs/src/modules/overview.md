# Module Overview

The current codebase is a small Cargo workspace. QuillSQL's core JIT path is
frontend-neutral; DataFusion is the first complete frontend adapter used by the
CLI, server, tests, and benchmarks.

## Workspace Packages

| Package | Role |
| ------- | ---- |
| `quill-sql` | CLI/server binaries, benchmarks, and release metadata. |
| `quill-core` | Public `Database` API, query execution wrapper, Parquet registration, and debug traces. |
| `quill-plan` | Frontend-neutral `PipelineGraph`, expression, type, source, stage, and sink model. |
| `quill-runtime` | Arrow batch binding, fixed-width kernels, aggregate state, and result materialization. |
| `quill-jit` | Compiler orchestration, frontend adapter trait, Quill dialect emission, and MLIR lowering. |
| `quill-df` | DataFusion frontend adapter, physical optimizer rule, and `CompiledPipelineExec`. |
| `quill-mlir` | C++/TableGen Quill dialect registration and MLIR pass extension points. |

## Database (`crates/quill-core/src/database.rs`)

`Database::run` is the interactive SQL entry point. It asks DataFusion to create
the logical plan, captures a debug snapshot of the logical/physical plan,
executes through DataFusion, and returns Arrow `RecordBatch` output.

`Database::prepare` creates a reusable logical plan wrapped in `PreparedQuery`.
Benchmark code uses this path to reduce parsing and logical planning noise while
still giving DataFusion a fresh physical plan for each execution.

`Database::register_parquet` exposes the durable storage path by registering a
Parquet dataset as a DataFusion table.

## JIT Workspace

| Package / Directory | Role |
| ------------------- | ---- |
| `quill-plan` | Neutral graph and expression model shared by all frontends. |
| `quill-runtime` | DataFusion-free Arrow runtime kernels and pipeline specs. |
| `quill-df` | DataFusion expression lowering, pipeline extraction, optimizer rule, and execution wrapper. |
| `quill-jit/src/dialect` | Quill pipeline dialect text model used as the explicit lowering boundary. |
| `quill-jit/src/lower` | Exact graph pattern lowering and JIT options. |
| `quill-jit/src/mlir` | MLIR emission, formal Quill dialect verification, and compiled ExecutionEngine invocation. |

The JIT subdirectories have stricter internal boundaries:

- `quill-df/src/expr.rs`: lowers supported DataFusion physical expressions into
  the neutral JIT expression model.
- `quill-plan/src/graph.rs`: defines the semantic `PipelineGraph` shape extracted
  from frontend physical plans.
- `quill-df/src/extract.rs`: extracts recognizable physical-plan pipelines such as
  `filter -> projection` and `filter -> plain SUM`.
- `quill-df/src/rule.rs`: physical optimizer rule that delegates supported pipeline
  rewrites to the compiler.
- `quill-df/src/compiler.rs`: compiles recognized `PipelineGraph` shapes into DataFusion
  execution nodes.
- `quill-jit/src/mlir/mod.rs`: public backend surface and `KernelBackend` implementation.
- `quill-jit/src/mlir/lower.rs`: lowers supported Quill dialect modules to executable MLIR;
  currently covers the Q6-shaped decimal filter/sum path.
- `quill-jit/src/mlir/emit.rs`: textual MLIR emission helpers for QuillSQL JIT expressions
  and fixed-width kernels.
- `quill-jit/src/mlir/verify.rs`: MLIR parser/verifier setup.
- `quill-jit/src/mlir/compiled.rs`: `ExecutionEngine` invocation artifacts.
- `quill-df/src/exec.rs`: unified DataFusion physical execution node for compiled
  record and scalar aggregate pipelines.
- `quill-runtime/src/kernel.rs`: `PipelineSpec` and compiled pipeline
  metadata.
- `quill-runtime/src/record.rs`: fixed-width filter/project record-batch runtime.
- `quill-runtime/src/sum.rs`: fixed-width plain `SUM` runtime and Q6-shaped decimal
  filter/sum specialization.
- `quill-runtime/src/array.rs`: Arrow array views and output builders.
- `quill-runtime/src/eval.rs`: expression evaluation and SQL boolean/null semantics.
- `quill-runtime/src/value.rs`: scalar value representation.

The JIT package is not a storage adapter and not a second SQL engine. It is the
research boundary for replacing selected DataFusion physical operators with
compiled kernels.

## Front Ends (`src/bin`)

`client` is the interactive SQL shell. `server` exposes HTTP endpoints for SQL
execution plus lightweight debug endpoints for the last DataFusion plan.

## Removed Layers

The project no longer contains the old teaching database stack:

- custom SQL AST/planner/optimizer/executor
- custom system tables
- custom row heap, buffer manager, legacy index manager, WAL, or recovery manager
- external KV storage adapter
- SQL-layer transaction manager

Those topics are useful, but they are no longer the theme of this repository.
