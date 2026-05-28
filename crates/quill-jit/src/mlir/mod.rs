mod compiled;
mod dispatch;
mod emit;
mod lower;
#[cfg(test)]
mod tests;
mod verify;

use arrow::datatypes::SchemaRef as ArrowSchemaRef;

use crate::{
    CompiledKernel, JitExpr, JitProjection, JitResult, KernelBackend, KernelKind, PipelineGraph,
    QuillDialectModule,
};

#[derive(Debug, Clone)]
pub struct MlirModule {
    pub symbol: String,
    pub text: String,
}

pub type MlirColumn = quill_runtime::FixedColumn;

#[derive(Debug, Default)]
pub struct MlirBackend;

pub use compiled::{
    CompiledI64Filter, CompiledPlainSum, CompiledRecordPipeline, FixedColumnInput,
    RecordPipelineOutput,
};
pub use dispatch::{execute_filter_project, execute_filter_sum};

impl MlirBackend {
    pub fn new() -> Self {
        Self
    }

    pub fn is_available(&self) -> bool {
        true
    }

    pub fn lower_filter(&self, predicate: &JitExpr) -> JitResult<MlirModule> {
        emit::lower_filter(predicate)
    }

    pub fn lower_projection(&self, projections: &[JitProjection]) -> JitResult<MlirModule> {
        emit::lower_projection(projections)
    }

    pub fn lower_filter_project(
        &self,
        predicate: &JitExpr,
        projections: &[JitProjection],
    ) -> JitResult<MlirModule> {
        emit::lower_filter_project(predicate, projections)
    }

    pub fn lower_i64_predicate(&self, predicate: &JitExpr) -> JitResult<MlirModule> {
        emit::lower_i64_predicate(predicate)
    }

    pub fn lower_i64_filter(&self, predicate: &JitExpr) -> JitResult<MlirModule> {
        emit::lower_i64_filter(predicate)
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

    pub fn invoke_i64_predicate(&self, predicate: &JitExpr, value: i64) -> JitResult<bool> {
        let module = self.lower_i64_predicate(predicate)?;
        self.verify_module(&module)?;
        compiled::invoke_i64_predicate(&module, value)
    }

    pub fn compile_i64_filter(&self, predicate: &JitExpr) -> JitResult<CompiledI64Filter> {
        let module = self.lower_i64_filter(predicate)?;
        self.verify_module(&module)?;
        compiled::compile_i64_filter(&module)
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

    pub fn verify_module(&self, module: &MlirModule) -> JitResult<()> {
        verify::verify_module(module)
    }
}

impl KernelBackend for MlirBackend {
    fn name(&self) -> &str {
        "mlir"
    }

    fn compile_filter(
        &self,
        _input_schema: ArrowSchemaRef,
        predicate: &JitExpr,
    ) -> JitResult<CompiledKernel> {
        let module = self.lower_filter(predicate)?;
        self.verify_module(&module)?;
        Ok(CompiledKernel::new(
            module.symbol,
            KernelKind::Filter,
            self.name(),
            false,
        ))
    }

    fn compile_projection(
        &self,
        _input_schema: ArrowSchemaRef,
        projections: &[JitProjection],
    ) -> JitResult<CompiledKernel> {
        let module = self.lower_projection(projections)?;
        self.verify_module(&module)?;
        Ok(CompiledKernel::new(
            module.symbol,
            KernelKind::Projection,
            self.name(),
            false,
        ))
    }

    fn compile_filter_project(
        &self,
        _input_schema: ArrowSchemaRef,
        predicate: &JitExpr,
        projections: &[JitProjection],
    ) -> JitResult<CompiledKernel> {
        let module = self.lower_filter_project(predicate, projections)?;
        self.verify_module(&module)?;
        Ok(CompiledKernel::new(
            module.symbol,
            KernelKind::FilterProject,
            self.name(),
            false,
        ))
    }
}
