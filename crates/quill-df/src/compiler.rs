use std::sync::Arc;

use datafusion::common::Result;
use datafusion::physical_plan::repartition::RepartitionExec;
use datafusion::physical_plan::ExecutionPlan;

use quill_jit::{
    CompiledKernel, JitOptions, KernelKind, MlirBackend, PipelineLowering, PipelineSpec,
};
use quill_plan::GroupAggregateOutputMode;
use quill_runtime::{FilterProjectKernel, FilterSumKernel, GroupAggregateKernel};

use crate::extract::{OutputAdapter, PhysicalPipeline, PipelineExecution};
use crate::{CompiledGlobalGroupAggregateExec, CompiledPipelineExec, PipelineRuntime};

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
            execution,
        } = pipeline;

        match PipelineLowering::from_graph(&graph) {
            Some(PipelineLowering::Record {
                predicate,
                projections,
            }) => {
                let spec = PipelineSpec::record_project(&predicate, &projections);
                let runtime = match FilterProjectKernel::try_new(
                    predicate.clone(),
                    projections.clone(),
                    Arc::clone(&output_schema),
                ) {
                    Ok(runtime) => runtime,
                    Err(_) => return Ok(None),
                };
                let kernel = self.filter_project_kernel(spec);
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
                let spec = PipelineSpec::filter_sum(&predicate, &measure);
                let runtime = match FilterSumKernel::try_new(predicate.clone(), measure.clone()) {
                    Ok(runtime) => runtime,
                    Err(_) => return Ok(None),
                };
                let kernel = self.filter_sum_kernel(spec);
                let exec = CompiledPipelineExec::try_new(
                    input,
                    PipelineRuntime::ScalarSum(runtime),
                    output_schema,
                    kernel,
                )?;
                Ok(Some(Arc::new(exec) as Arc<dyn ExecutionPlan>))
            }
            Some(PipelineLowering::GroupAggregate {
                predicate,
                keys,
                aggregates,
            }) => {
                let stages = predicate
                    .clone()
                    .map(quill_plan::PipelineStage::Filter)
                    .into_iter()
                    .collect::<Vec<_>>();
                let spec = PipelineSpec::group_aggregate(predicate.as_ref(), &keys, &aggregates);
                let output_mode = match execution {
                    PipelineExecution::PartitionLocal => GroupAggregateOutputMode::PartialState,
                    PipelineExecution::Global => GroupAggregateOutputMode::FinalValues,
                };
                let runtime = match GroupAggregateKernel::try_new_with_output(
                    &stages,
                    keys,
                    aggregates,
                    Arc::clone(&output_schema),
                    output_mode,
                ) {
                    Ok(runtime) => runtime,
                    Err(_) => return Ok(None),
                };
                let kernel = self.group_aggregate_kernel(spec);
                if execution == PipelineExecution::Global {
                    let exec = CompiledGlobalGroupAggregateExec::try_new(
                        input,
                        runtime,
                        output_schema,
                        kernel,
                    )?;
                    return Ok(Some(Arc::new(exec) as Arc<dyn ExecutionPlan>));
                }
                let exec = CompiledPipelineExec::try_new(
                    input,
                    PipelineRuntime::GroupAggregate(runtime),
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

    fn filter_project_kernel(&self, spec: Option<PipelineSpec>) -> CompiledKernel {
        let executable = self.options.mlir_execution_enabled() && spec.is_some();
        let spec = spec.unwrap_or_else(|| PipelineSpec::generic(KernelKind::FilterProject));
        CompiledKernel::with_spec(
            "record_filter_project",
            spec,
            self.kernel_backend_name(executable),
            executable,
        )
    }

    fn filter_sum_kernel(&self, spec: Option<PipelineSpec>) -> CompiledKernel {
        let executable = self.options.mlir_execution_enabled() && spec.is_some();
        let spec = spec.unwrap_or_else(|| PipelineSpec::generic(KernelKind::FilterSum));
        CompiledKernel::with_spec(
            "filter_plain_aggregate",
            spec,
            self.kernel_backend_name(executable),
            executable,
        )
    }

    fn group_aggregate_kernel(&self, spec: Option<PipelineSpec>) -> CompiledKernel {
        let executable = self.options.mlir_execution_enabled() && spec.is_some();
        let spec = spec.unwrap_or_else(|| PipelineSpec::generic(KernelKind::GroupAggregate));
        CompiledKernel::with_spec(
            "group_aggregate",
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
