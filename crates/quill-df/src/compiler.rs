use std::sync::Arc;

use datafusion::arrow::datatypes::Schema as ArrowSchema;
use datafusion::common::Result;
use datafusion::physical_plan::repartition::RepartitionExec;
use datafusion::physical_plan::ExecutionPlan;

use quill_jit::{JitOptions, MlirBackend, PipelineLowering};
use quill_plan::{JitExpr, JitProjection};
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
                let kernel = self.filter_project_kernel(&runtime, &predicate, &projections);
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
                let kernel = self.filter_sum_kernel(&runtime, &predicate, &measure);
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

    fn filter_project_kernel(
        &self,
        runtime: &FilterProjectKernel,
        predicate: &JitExpr,
        projections: &[JitProjection],
    ) -> CompiledKernel {
        if let Ok(module) = self.backend.lower_record_pipeline(predicate, projections) {
            let executable = self.backend.verify_module(&module).is_ok()
                && self.options.mlir_execution_enabled();
            let spec = runtime
                .spec()
                .cloned()
                .unwrap_or_else(|| PipelineSpec::generic(KernelKind::FilterProject));
            return CompiledKernel::with_spec(module.symbol, spec, self.backend.name(), executable);
        }

        match self.backend.compile_filter_project(
            Arc::new(ArrowSchema::empty()),
            predicate,
            projections,
        ) {
            Ok(kernel) => kernel,
            Err(_) => CompiledKernel::new(
                "filter_project_runtime",
                KernelKind::FilterProject,
                "fixed-width-runtime",
                false,
            ),
        }
    }

    fn filter_sum_kernel(
        &self,
        runtime: &FilterSumKernel,
        predicate: &JitExpr,
        measure: &JitExpr,
    ) -> CompiledKernel {
        let spec = || {
            runtime
                .spec()
                .cloned()
                .unwrap_or_else(|| PipelineSpec::generic(KernelKind::FilterSum))
        };
        if let Ok(module) = self.backend.lower_plain_sum(predicate, measure) {
            return CompiledKernel::with_spec(
                module.symbol,
                spec(),
                self.backend.name(),
                self.options.mlir_execution_enabled(),
            );
        }

        CompiledKernel::new(
            "fixed_width_filter_sum",
            KernelKind::FilterSum,
            "fixed-width-runtime",
            false,
        )
    }
}
