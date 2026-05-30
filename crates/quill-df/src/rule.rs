use std::sync::Arc;

use datafusion::common::config::ConfigOptions;
use datafusion::common::tree_node::{Transformed, TreeNode};
use datafusion::common::Result;
use datafusion::physical_optimizer::PhysicalOptimizerRule;
use datafusion::physical_plan::ExecutionPlan;
use serde::Serialize;

use quill_jit::{FrontendAdapter, JitOptions, KernelKind, MlirBackend};

use crate::compiler::PipelineCompiler;
use crate::{
    extract_pipeline_from_node, pipeline_from_node, CompiledGlobalGroupAggregateExec,
    CompiledPipelineExec, PipelineCandidate,
};

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct JitCandidate {
    pub node: &'static str,
    pub kernel: KernelKind,
    pub backend: String,
    pub executable: bool,
}

#[derive(Debug, Default)]
pub struct MlirJitRule {
    backend: MlirBackend,
    options: JitOptions,
}

#[derive(Debug, Default)]
pub struct DataFusionAdapter {
    rule: MlirJitRule,
}

impl DataFusionAdapter {
    pub fn with_options(options: JitOptions) -> Self {
        Self {
            rule: MlirJitRule::with_options(options),
        }
    }
}

impl MlirJitRule {
    pub fn new() -> Self {
        Self::with_options(JitOptions::default())
    }

    pub fn with_options(options: JitOptions) -> Self {
        Self {
            backend: MlirBackend::new(),
            options,
        }
    }

    pub fn enabled() -> bool {
        true
    }

    pub fn inspect_plan(&self, plan: Arc<dyn ExecutionPlan>) -> Vec<JitCandidate> {
        let mut candidates = Vec::new();
        let _ = plan.transform_down(|plan| {
            if let Some(candidate) = self.inspect_node(&plan) {
                candidates.push(candidate);
            }
            Ok(Transformed::no(plan))
        });
        candidates
    }

    pub fn inspect_pipelines(&self, plan: Arc<dyn ExecutionPlan>) -> Vec<PipelineCandidate> {
        let mut candidates = Vec::new();
        let _ = plan.transform_down(|plan| {
            if let Some(candidate) =
                pipeline_from_node(&plan).and_then(|pipeline| pipeline.candidate())
            {
                candidates.push(candidate);
            }
            Ok(Transformed::no(plan))
        });
        candidates
    }

    fn inspect_node(&self, plan: &Arc<dyn ExecutionPlan>) -> Option<JitCandidate> {
        if let Some(compiled) = plan.as_any().downcast_ref::<CompiledPipelineExec>() {
            return Some(JitCandidate {
                node: "CompiledPipelineExec",
                kernel: compiled.kernel().kind,
                backend: compiled.kernel().backend.clone(),
                executable: compiled.kernel().executable,
            });
        }
        if let Some(compiled) = plan
            .as_any()
            .downcast_ref::<CompiledGlobalGroupAggregateExec>()
        {
            return Some(JitCandidate {
                node: "CompiledGlobalGroupAggregateExec",
                kernel: compiled.kernel().kind,
                backend: compiled.kernel().backend.clone(),
                executable: compiled.kernel().executable,
            });
        }

        None
    }
}

impl FrontendAdapter for DataFusionAdapter {
    type Plan = Arc<dyn ExecutionPlan>;
    type Candidate = PipelineCandidate;
    type Compiled = Arc<dyn ExecutionPlan>;
    type Error = datafusion::common::DataFusionError;

    fn extract(&self, plan: &Self::Plan) -> Vec<Self::Candidate> {
        self.rule.inspect_pipelines(Arc::clone(plan))
    }

    fn replace(&self, plan: Self::Plan, compiled: Vec<Self::Compiled>) -> Result<Self::Plan> {
        if !compiled.is_empty() {
            return Err(datafusion::common::DataFusionError::Internal(
                "DataFusionAdapter compiles replacements during physical optimization".to_string(),
            ));
        }
        self.rule.optimize(plan, &ConfigOptions::new())
    }
}

impl PhysicalOptimizerRule for MlirJitRule {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _config: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if !self.options.enabled() {
            return Ok(plan);
        }

        plan.transform_up(|plan| self.try_compile_node(plan))
            .map(|transformed| transformed.data)
    }

    fn name(&self) -> &str {
        "mlir_jit"
    }

    fn schema_check(&self) -> bool {
        true
    }
}

impl MlirJitRule {
    fn try_compile_node(
        &self,
        plan: Arc<dyn ExecutionPlan>,
    ) -> Result<Transformed<Arc<dyn ExecutionPlan>>> {
        let Some(pipeline) = extract_pipeline_from_node(&plan) else {
            return Ok(Transformed::no(plan));
        };

        match PipelineCompiler::new(&self.backend, &self.options).compile(pipeline)? {
            Some(compiled) => Ok(Transformed::yes(compiled)),
            None => Ok(Transformed::no(plan)),
        }
    }
}
