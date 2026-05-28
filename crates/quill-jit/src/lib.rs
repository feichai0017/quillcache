mod dialect;
mod frontend;
mod lower;
mod mlir;

pub use dialect::{QuillDialectModule, QuillDialectOp, QuillDialectSink, QuillDialectSource};
pub use frontend::{CompiledPipeline, FrontendAdapter};
pub use lower::{JitOptions, PipelineLowering};
pub use mlir::{
    CompiledI64Filter, CompiledPlainSum, CompiledRecordPipeline, FixedColumnInput, MlirBackend,
    MlirColumn, MlirModule, RecordPipelineOutput,
};
pub use quill_plan::{
    AggregateFunc, GroupAggregate, JitBinaryOp, JitError, JitExpr, JitProjection, JitResult,
    JitScalar, JitType, PipelineGraph, PipelineKind, PipelineSink, PipelineSource, PipelineStage,
};
pub use quill_runtime::{
    CompiledKernel, FilterProjectKernel, FilterSumKernel, FilterSumValue, FixedColumn,
    KernelBackend, KernelKind, PipelineSpec,
};

pub use mlir::{execute_filter_project, execute_filter_sum};
