mod compiled;
mod dispatch;
mod emit;
mod lower;
#[cfg(test)]
mod tests;
mod verify;

use crate::{
    FixedColumn, GroupAggregate, JitExpr, JitProjection, JitResult, PipelineGraph,
    QuillDialectModule,
};

#[derive(Debug, Clone)]
pub struct MlirModule {
    pub symbol: String,
    pub text: String,
}

pub type MlirColumn = FixedColumn;

#[derive(Debug, Default)]
pub struct MlirBackend;

pub use compiled::{
    CompiledGroupAggregateUpdate, CompiledPlainSum, CompiledRecordPipeline, FixedColumnInput,
    RecordPipelineOutput,
};
pub use dispatch::{execute_filter_project, execute_filter_sum, execute_group_aggregate_update};

impl MlirBackend {
    pub fn new() -> Self {
        Self
    }

    pub fn is_available(&self) -> bool {
        true
    }

    pub fn lower_record_pipeline(
        &self,
        predicate: &JitExpr,
        projections: &[JitProjection],
    ) -> JitResult<MlirModule> {
        let pipeline = PipelineGraph::record(vec![
            crate::PipelineStage::Filter(predicate.clone()),
            crate::PipelineStage::Projection(projections.to_vec()),
        ]);
        let dialect =
            self.emit_quill_dialect(emit::next_symbol("quill_record_pipeline"), &pipeline);
        lower::lower_quill_dialect(&dialect)
    }

    pub fn lower_plain_sum(&self, predicate: &JitExpr, measure: &JitExpr) -> JitResult<MlirModule> {
        let pipeline = PipelineGraph::filter_sum(predicate.clone(), measure.clone());
        let dialect = self.emit_quill_dialect(emit::next_symbol("quill_plain_sum"), &pipeline);
        lower::lower_quill_dialect(&dialect)
    }

    pub fn lower_group_aggregate_update(
        &self,
        predicate: Option<&JitExpr>,
        keys: &[JitExpr],
        aggregates: &[GroupAggregate],
    ) -> JitResult<MlirModule> {
        let stages = predicate
            .cloned()
            .map(crate::PipelineStage::Filter)
            .into_iter()
            .collect();
        let pipeline = PipelineGraph::group_aggregate(stages, keys.to_vec(), aggregates.to_vec());
        let dialect =
            self.emit_quill_dialect(emit::next_symbol("quill_group_aggregate"), &pipeline);
        lower::lower_quill_dialect(&dialect)
    }

    pub fn lower_quill_dialect(&self, module: &QuillDialectModule) -> JitResult<MlirModule> {
        lower::lower_quill_dialect(module)
    }

    pub fn lower_graph_to_quill_mlir(
        &self,
        symbol: impl Into<String>,
        graph: &PipelineGraph,
    ) -> JitResult<MlirModule> {
        let dialect = self.emit_quill_dialect(symbol, graph);
        Ok(MlirModule {
            symbol: dialect.symbol.clone(),
            text: dialect.to_mlir_text()?,
        })
    }

    pub fn emit_quill_dialect(
        &self,
        symbol: impl Into<String>,
        graph: &PipelineGraph,
    ) -> QuillDialectModule {
        QuillDialectModule::from_graph(symbol, graph)
    }

    pub fn compile_record_pipeline(
        &self,
        predicate: &JitExpr,
        projections: &[JitProjection],
    ) -> JitResult<CompiledRecordPipeline> {
        let spec =
            crate::PipelineSpec::record_project(predicate, projections).ok_or_else(|| {
                crate::JitError::UnsupportedExpr(
                    "record pipeline requires fixed-width filter/project expressions".to_string(),
                )
            })?;
        let crate::PipelineSpec::RecordProject {
            columns,
            output_types,
        } = spec
        else {
            unreachable!("record_project returned another spec")
        };
        let module = self.lower_record_pipeline(predicate, projections)?;
        self.verify_module(&module)?;
        compiled::compile_record_pipeline(&module, columns, output_types)
    }

    pub fn compile_plain_sum(
        &self,
        predicate: &JitExpr,
        measure: &JitExpr,
    ) -> JitResult<CompiledPlainSum> {
        let spec = crate::PipelineSpec::filter_sum(predicate, measure).ok_or_else(|| {
            crate::JitError::UnsupportedExpr(
                "plain SUM requires fixed-width filter and aggregate expressions".to_string(),
            )
        })?;
        let crate::PipelineSpec::PlainSum {
            columns,
            output_type,
        } = spec
        else {
            unreachable!("filter_sum returned another spec")
        };
        let module = self.lower_plain_sum(predicate, measure)?;
        self.verify_module(&module)?;
        compiled::compile_plain_sum(&module, columns, output_type)
    }

    pub fn compile_group_aggregate_update(
        &self,
        predicate: Option<&JitExpr>,
        keys: &[JitExpr],
        aggregates: &[GroupAggregate],
    ) -> JitResult<CompiledGroupAggregateUpdate> {
        let spec =
            crate::PipelineSpec::group_aggregate(predicate, keys, aggregates).ok_or_else(|| {
                crate::JitError::UnsupportedExpr(
                    "group aggregate update requires fixed-width aggregate state".to_string(),
                )
            })?;
        let crate::PipelineSpec::GroupAggregate {
            columns,
            aggregate_funcs,
            state_types,
            ..
        } = spec
        else {
            unreachable!("group_aggregate returned another spec")
        };
        let module = self.lower_group_aggregate_update(predicate, keys, aggregates)?;
        self.verify_module(&module)?;
        compiled::compile_group_aggregate_update(&module, columns, aggregate_funcs, state_types)
    }

    pub fn verify_module(&self, module: &MlirModule) -> JitResult<()> {
        verify::verify_module(module)
    }
}

impl MlirBackend {
    pub fn name(&self) -> &str {
        "mlir"
    }
}
