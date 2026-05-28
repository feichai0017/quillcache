use std::sync::Arc;

use datafusion::common::Result;
use datafusion::physical_plan::repartition::RepartitionExec;
use datafusion::physical_plan::ExecutionPlan;

use quill_jit::{JitOptions, MlirBackend, PipelineLowering};
use quill_runtime::{
    CompiledKernel, FilterProjectKernel, FilterSumKernel, KernelBackend, KernelKind, PipelineSpec,
};

use crate::extract::{OutputAdapter, PhysicalPipeline};
use crate::{CompiledPipelineExec, PipelineRuntime};

#[derive(Debug)]
pub(crate) struct PipelineCompiler<'a> {
    backend: &'a MlirBackend,
    options: &'a JitOptions,
}

impl<'a> PipelineCompiler<'a> {
    pub fn new(backend: &'a MlirBackend, options: &'a JitOptions) -> Self {
        Self { backend, options }
    }

    pub fn compile(&self, pipeline: PhysicalPipeline) -> Result<Option<Arc<dyn ExecutionPlan>>> {
        let PhysicalPipeline {
            input,
            output_schema,
            graph,
            output_adapter,
        } = pipeline;

        match PipelineLowering::from_graph(&graph) {
            Some(PipelineLowering::Record {
                predicate,
                projections,
            }) => {
                let runtime = match FilterProjectKernel::try_new(
                    predicate.clone(),
                    projections.clone(),
                    Arc::clone(&output_schema),
                ) {
                    Ok(runtime) => runtime,
                    Err(_) => return Ok(None),
                };
                let kernel = self.filter_project_kernel(&runtime);
                let exec = CompiledPipelineExec::try_new(
                    input,
                    PipelineRuntime::RecordBatch(runtime),
                    output_schema,
                    kernel,
                )?;
                let exec = Arc::new(exec) as Arc<dyn ExecutionPlan>;
                Self::apply_output_adapter(exec, output_adapter).map(Some)
            }
            Some(PipelineLowering::PlainSum { predicate, measure }) => {
                let runtime = match FilterSumKernel::try_new(predicate.clone(), measure.clone()) {
                    Ok(runtime) => runtime,
                    Err(_) => return Ok(None),
                };
                let kernel = self.filter_sum_kernel(&runtime);
                let exec = CompiledPipelineExec::try_new(
                    input,
                    PipelineRuntime::ScalarSum(runtime),
                    output_schema,
                    kernel,
                )?;
                Ok(Some(Arc::new(exec) as Arc<dyn ExecutionPlan>))
            }
            _ => Ok(None),
        }
    }

    fn apply_output_adapter(
        exec: Arc<dyn ExecutionPlan>,
        adapter: Option<OutputAdapter>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let Some(adapter) = adapter else {
            return Ok(exec);
        };

        let mut repartition = RepartitionExec::try_new(exec, adapter.partitioning)?;
        if adapter.preserve_order {
            repartition = repartition.with_preserve_order();
        }
        Ok(Arc::new(repartition) as Arc<dyn ExecutionPlan>)
    }

    fn filter_project_kernel(&self, runtime: &FilterProjectKernel) -> CompiledKernel {
        let spec = runtime
            .spec()
            .cloned()
            .unwrap_or_else(|| PipelineSpec::generic(KernelKind::FilterProject));
        let executable = self.options.mlir_execution_enabled() && runtime.spec().is_some();
        CompiledKernel::with_spec(
            "record_filter_project",
            spec,
            self.kernel_backend_name(executable),
            executable,
        )
    }

    fn filter_sum_kernel(&self, runtime: &FilterSumKernel) -> CompiledKernel {
        let spec = runtime
            .spec()
            .cloned()
            .unwrap_or_else(|| PipelineSpec::generic(KernelKind::FilterSum));
        let executable = self.options.mlir_execution_enabled() && runtime.spec().is_some();
        CompiledKernel::with_spec(
            "filter_plain_sum",
            spec,
            self.kernel_backend_name(executable),
            executable,
        )
    }

    fn kernel_backend_name(&self, executable: bool) -> &str {
        if executable {
            self.backend.name()
        } else {
            "quill-runtime"
        }
    }
}
