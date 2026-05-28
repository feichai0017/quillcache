use crate::{JitError, JitResult, PipelineSpec, QuillDialectModule};

use super::MlirModule;

pub(super) fn lower_quill_dialect(module: &QuillDialectModule) -> JitResult<MlirModule> {
    verify_formal_quill_module(module)?;
    match module.pipeline_spec() {
        Some(PipelineSpec::RecordProject { .. }) => lower_filter_project(module),
        Some(PipelineSpec::PlainSum { .. }) => lower_with_quill_pass(module),
        Some(spec) => Err(JitError::UnsupportedExpr(format!(
            "quill dialect lowering does not yet support {}",
            spec.name()
        ))),
        None => Err(JitError::UnsupportedExpr(
            "quill dialect lowering requires a supported pipeline spec".to_string(),
        )),
    }
}

fn lower_with_quill_pass(module: &QuillDialectModule) -> JitResult<MlirModule> {
    use melior::{ir::Module, pass, utility};

    let context = super::verify::mlir_context();
    let mut parsed = Module::parse(&context, &module.to_mlir_text()?).ok_or_else(|| {
        JitError::Backend("MLIR parser rejected Quill dialect module".to_string())
    })?;
    let pass_manager = pass::PassManager::new(&context);
    utility::parse_pass_pipeline(
        pass_manager.as_operation_pass_manager(),
        "builtin.module(convert-quill-to-loops)",
    )
    .map_err(|err| JitError::Backend(format!("Quill pass pipeline parse failed: {err:?}")))?;
    pass_manager
        .run(&mut parsed)
        .map_err(|err| JitError::Backend(format!("Quill to loops lowering failed: {err:?}")))?;
    let text = parsed.as_operation().to_string();
    if text.contains("quill.") {
        return Err(JitError::Backend(
            "Quill to loops lowering left Quill dialect operations in the module".to_string(),
        ));
    }
    Ok(MlirModule {
        symbol: module.symbol.clone(),
        text,
    })
}

fn lower_filter_project(module: &QuillDialectModule) -> JitResult<MlirModule> {
    lower_with_quill_pass(module)
}

fn verify_formal_quill_module(module: &QuillDialectModule) -> JitResult<()> {
    super::verify::verify_module(&MlirModule {
        symbol: module.symbol.clone(),
        text: module.to_mlir_text()?,
    })
}
