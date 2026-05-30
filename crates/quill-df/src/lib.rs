mod compiler;
mod exec;
mod expr;
mod extract;
mod rule;

pub use exec::{CompiledGlobalGroupAggregateExec, CompiledPipelineExec, PipelineRuntime};
pub use extract::PipelineCandidate;
pub(crate) use extract::{extract_pipeline_from_node, pipeline_from_node};
pub use rule::{DataFusionAdapter, JitCandidate, MlirJitRule};

fn map_jit_err(err: quill_plan::JitError) -> datafusion::common::DataFusionError {
    datafusion::common::DataFusionError::Execution(err.to_string())
}
