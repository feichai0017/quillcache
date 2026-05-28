use crate::{FilterSumValue, JitError, JitResult, JitType, MlirColumn, MlirModule};

use melior::{ir::Module, pass, ExecutionEngine};

pub(super) struct CompiledI64Predicate {
    symbol: String,
    engine: ExecutionEngine,
}

pub struct CompiledI64Filter {
    symbol: String,
    engine: ExecutionEngine,
}

pub struct CompiledRecordPipeline {
    symbol: String,
    engine: ExecutionEngine,
    columns: Vec<MlirColumn>,
    output_types: Vec<JitType>,
}

pub struct CompiledPlainSum {
    symbol: String,
    engine: ExecutionEngine,
    columns: Vec<MlirColumn>,
    output_type: JitType,
}

#[derive(Debug, Clone, Copy)]
pub enum FixedColumnInput<'a> {
    Date32 { index: usize, values: &'a [i32] },
    Int64 { index: usize, values: &'a [i64] },
    Float64 { index: usize, values: &'a [f64] },
    Decimal128 { index: usize, values: &'a [i128] },
}

pub enum RecordPipelineOutput<'a> {
    Date32 { values: &'a mut [i32] },
    Int64 { values: &'a mut [i64] },
    Float64 { values: &'a mut [f64] },
    Decimal128 { values: &'a mut [i128] },
}

impl CompiledI64Predicate {
    pub(super) fn invoke(&self, value: i64) -> JitResult<bool> {
        let mut argument = value;
        let mut result = -1_i32;
        unsafe {
            self.engine
                .invoke_packed(
                    &self.symbol,
                    &mut [
                        &mut argument as *mut i64 as *mut (),
                        &mut result as *mut i32 as *mut (),
                    ],
                )
                .map_err(|err| JitError::Backend(format!("MLIR invocation failed: {err:?}")))?;
        }
        Ok(result != 0)
    }
}

impl CompiledI64Filter {
    pub fn invoke(&self, values: &[i64], output: &mut [u8]) -> JitResult<()> {
        if output.len() < values.len() {
            return Err(JitError::Backend(format!(
                "compiled filter output len {} is smaller than input len {}",
                output.len(),
                values.len()
            )));
        }

        let mut len = values.len() as i64;
        let mut values_ptr = values.as_ptr();
        let mut output_ptr = output.as_mut_ptr();
        let mut result = -1_i32;
        unsafe {
            self.engine
                .invoke_packed(
                    &self.symbol,
                    &mut [
                        &mut len as *mut i64 as *mut (),
                        &mut values_ptr as *mut *const i64 as *mut (),
                        &mut output_ptr as *mut *mut u8 as *mut (),
                        &mut result as *mut i32 as *mut (),
                    ],
                )
                .map_err(|err| JitError::Backend(format!("MLIR invocation failed: {err:?}")))?;
        }
        if result == 0 {
            Ok(())
        } else {
            Err(JitError::Backend(format!(
                "compiled filter returned status {result}"
            )))
        }
    }
}

impl CompiledRecordPipeline {
    pub fn invoke(
        &self,
        inputs: &[FixedColumnInput<'_>],
        outputs: &mut [RecordPipelineOutput<'_>],
    ) -> JitResult<usize> {
        if outputs.len() != self.output_types.len() {
            return Err(JitError::Backend(format!(
                "compiled record pipeline expected {} output columns, got {}",
                self.output_types.len(),
                outputs.len()
            )));
        }
        for (index, (output, ty)) in outputs.iter().zip(&self.output_types).enumerate() {
            if !output.matches_type(*ty) {
                return Err(JitError::Backend(format!(
                    "compiled record pipeline output column {index} has incompatible type"
                )));
            }
        }

        let mut input_len = None;
        let mut input_ptrs = Vec::with_capacity(self.columns.len());
        for column in &self.columns {
            let input = inputs
                .iter()
                .find(|input| input.index() == column.index)
                .ok_or_else(|| {
                    JitError::Backend(format!(
                        "compiled record pipeline missing input column {}",
                        column.index
                    ))
                })?;
            if !input.matches_type(column.ty) {
                return Err(JitError::Backend(format!(
                    "compiled record pipeline input column {} has incompatible type",
                    column.index
                )));
            }
            match input_len {
                Some(expected) if expected != input.len() => {
                    return Err(JitError::Backend(format!(
                        "compiled record pipeline input column {} len {} does not match len {}",
                        column.index,
                        input.len(),
                        expected
                    )));
                }
                Some(_) => {}
                None => input_len = Some(input.len()),
            }
            input_ptrs.push(input.ptr());
        }

        let input_len = input_len.unwrap_or(0);
        for (index, output) in outputs.iter().enumerate() {
            if output.len() < input_len {
                return Err(JitError::Backend(format!(
                    "compiled record pipeline output column {index} len {} is smaller than input len {input_len}",
                    output.len()
                )));
            }
        }

        let mut len = input_len as i64;
        let mut output_ptrs = outputs
            .iter_mut()
            .map(RecordPipelineOutput::ptr)
            .collect::<Vec<_>>();
        let mut output_len = -1_i64;
        let mut output_len_ptr = &mut output_len as *mut i64;
        let mut result = -1_i32;
        let mut packed_args = Vec::with_capacity(self.columns.len() + self.output_types.len() + 3);
        packed_args.push(&mut len as *mut i64 as *mut ());
        for ptr in &mut input_ptrs {
            packed_args.push(ptr as *mut *const () as *mut ());
        }
        for ptr in &mut output_ptrs {
            packed_args.push(ptr as *mut *mut () as *mut ());
        }
        packed_args.push(&mut output_len_ptr as *mut *mut i64 as *mut ());
        packed_args.push(&mut result as *mut i32 as *mut ());

        unsafe {
            self.engine
                .invoke_packed(&self.symbol, &mut packed_args)
                .map_err(|err| JitError::Backend(format!("MLIR invocation failed: {err:?}")))?;
        }
        if result != 0 {
            return Err(JitError::Backend(format!(
                "compiled record pipeline returned status {result}"
            )));
        }
        if output_len < 0 || output_len as usize > input_len {
            return Err(JitError::Backend(format!(
                "compiled record pipeline returned invalid output len {output_len}"
            )));
        }
        Ok(output_len as usize)
    }
}

impl CompiledPlainSum {
    pub fn invoke(&self, inputs: &[FixedColumnInput<'_>]) -> JitResult<FilterSumValue> {
        match self.output_type {
            JitType::Float64 => self.invoke_f64(inputs),
            JitType::Decimal128 { scale, .. } => self.invoke_decimal(inputs, scale),
            other => Err(JitError::Backend(format!(
                "compiled plain SUM output type {other:?} is not supported"
            ))),
        }
    }

    fn invoke_f64(&self, inputs: &[FixedColumnInput<'_>]) -> JitResult<FilterSumValue> {
        let mut sum = 0.0_f64;
        let sum_ptr = &mut sum as *mut f64;
        let count = self.invoke_raw(inputs, sum_ptr.cast())?;
        Ok(FilterSumValue::Float64((count > 0).then_some(sum)))
    }

    fn invoke_decimal(
        &self,
        inputs: &[FixedColumnInput<'_>],
        scale: i8,
    ) -> JitResult<FilterSumValue> {
        let mut sum = 0_i128;
        let sum_ptr = &mut sum as *mut i128;
        let count = self.invoke_raw(inputs, sum_ptr.cast())?;
        Ok(FilterSumValue::Decimal128 {
            value: (count > 0).then_some(sum),
            scale,
        })
    }

    fn invoke_raw(&self, inputs: &[FixedColumnInput<'_>], output_sum: *mut ()) -> JitResult<i64> {
        let mut output_count = 0_i64;
        let mut output_count_ptr = &mut output_count as *mut i64;
        invoke_plain_sum(
            &self.engine,
            &self.symbol,
            &self.columns,
            inputs,
            output_sum,
            &mut output_count_ptr,
        )?;
        Ok(output_count)
    }
}

impl FixedColumnInput<'_> {
    fn index(&self) -> usize {
        match self {
            Self::Date32 { index, .. }
            | Self::Int64 { index, .. }
            | Self::Float64 { index, .. }
            | Self::Decimal128 { index, .. } => *index,
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::Date32 { values, .. } => values.len(),
            Self::Int64 { values, .. } => values.len(),
            Self::Float64 { values, .. } => values.len(),
            Self::Decimal128 { values, .. } => values.len(),
        }
    }

    fn ptr(&self) -> *const () {
        match self {
            Self::Date32 { values, .. } => values.as_ptr().cast(),
            Self::Int64 { values, .. } => values.as_ptr().cast(),
            Self::Float64 { values, .. } => values.as_ptr().cast(),
            Self::Decimal128 { values, .. } => values.as_ptr().cast(),
        }
    }

    fn matches_type(&self, ty: JitType) -> bool {
        matches!(
            (self, ty),
            (Self::Date32 { .. }, JitType::Date32)
                | (Self::Int64 { .. }, JitType::Int64)
                | (Self::Float64 { .. }, JitType::Float64)
                | (Self::Decimal128 { .. }, JitType::Decimal128 { .. })
        )
    }
}

impl RecordPipelineOutput<'_> {
    fn len(&self) -> usize {
        match self {
            Self::Date32 { values } => values.len(),
            Self::Int64 { values } => values.len(),
            Self::Float64 { values } => values.len(),
            Self::Decimal128 { values } => values.len(),
        }
    }

    fn ptr(&mut self) -> *mut () {
        match self {
            Self::Date32 { values } => values.as_mut_ptr().cast(),
            Self::Int64 { values } => values.as_mut_ptr().cast(),
            Self::Float64 { values } => values.as_mut_ptr().cast(),
            Self::Decimal128 { values } => values.as_mut_ptr().cast(),
        }
    }

    fn matches_type(&self, ty: JitType) -> bool {
        matches!(
            (self, ty),
            (Self::Date32 { .. }, JitType::Date32)
                | (Self::Int64 { .. }, JitType::Int64)
                | (Self::Float64 { .. }, JitType::Float64)
                | (Self::Decimal128 { .. }, JitType::Decimal128 { .. })
        )
    }
}

fn invoke_plain_sum(
    engine: &ExecutionEngine,
    symbol: &str,
    columns: &[MlirColumn],
    inputs: &[FixedColumnInput<'_>],
    output_sum: *mut (),
    output_count: &mut *mut i64,
) -> JitResult<()> {
    let mut input_len = None;
    let mut ptr_values = Vec::with_capacity(columns.len());

    for column in columns {
        let input = inputs
            .iter()
            .find(|input| input.index() == column.index)
            .ok_or_else(|| {
                JitError::Backend(format!(
                    "compiled plain SUM missing input column {}",
                    column.index
                ))
            })?;
        if !input.matches_type(column.ty) {
            return Err(JitError::Backend(format!(
                "compiled plain SUM input column {} has incompatible type",
                column.index
            )));
        }

        match input_len {
            Some(expected) if expected != input.len() => {
                return Err(JitError::Backend(format!(
                    "compiled plain SUM input column {} len {} does not match len {}",
                    column.index,
                    input.len(),
                    expected
                )));
            }
            Some(_) => {}
            None => input_len = Some(input.len()),
        }
        ptr_values.push(input.ptr());
    }

    let mut len = input_len.unwrap_or(0) as i64;
    let mut result = -1_i32;
    let mut packed_args = Vec::with_capacity(columns.len() + 4);
    packed_args.push(&mut len as *mut i64 as *mut ());
    for ptr in &mut ptr_values {
        packed_args.push(ptr as *mut *const () as *mut ());
    }
    let mut output_sum = output_sum;
    packed_args.push(&mut output_sum as *mut *mut () as *mut ());
    packed_args.push(output_count as *mut *mut i64 as *mut ());
    packed_args.push(&mut result as *mut i32 as *mut ());

    unsafe {
        engine
            .invoke_packed(symbol, &mut packed_args)
            .map_err(|err| JitError::Backend(format!("MLIR invocation failed: {err:?}")))?;
    }
    if result == 0 {
        Ok(())
    } else {
        Err(JitError::Backend(format!(
            "compiled plain SUM returned status {result}"
        )))
    }
}

pub(super) fn compile_i64_predicate(compiled: &MlirModule) -> JitResult<CompiledI64Predicate> {
    Ok(CompiledI64Predicate {
        symbol: compiled.symbol.clone(),
        engine: compile_engine(compiled)?,
    })
}

pub(super) fn invoke_i64_predicate(compiled: &MlirModule, value: i64) -> JitResult<bool> {
    compile_i64_predicate(compiled)?.invoke(value)
}

pub fn compile_i64_filter(compiled: &MlirModule) -> JitResult<CompiledI64Filter> {
    Ok(CompiledI64Filter {
        symbol: compiled.symbol.clone(),
        engine: compile_engine(compiled)?,
    })
}

pub fn compile_record_pipeline(
    compiled: &MlirModule,
    columns: Vec<MlirColumn>,
    output_types: Vec<JitType>,
) -> JitResult<CompiledRecordPipeline> {
    Ok(CompiledRecordPipeline {
        symbol: compiled.symbol.clone(),
        engine: compile_engine(compiled)?,
        columns,
        output_types,
    })
}

pub fn compile_plain_sum(
    compiled: &MlirModule,
    columns: Vec<MlirColumn>,
    output_type: JitType,
) -> JitResult<CompiledPlainSum> {
    Ok(CompiledPlainSum {
        symbol: compiled.symbol.clone(),
        engine: compile_engine(compiled)?,
        columns,
        output_type,
    })
}

fn compile_engine(compiled: &MlirModule) -> JitResult<ExecutionEngine> {
    let context = super::verify::mlir_context();
    let mut module = Module::parse(&context, &compiled.text)
        .ok_or_else(|| JitError::Backend("MLIR parser rejected generated module".to_string()))?;

    let pass_manager = pass::PassManager::new(&context);
    pass_manager
        .nested_under("func.func")
        .add_pass(pass::conversion::create_arith_to_llvm());
    pass_manager
        .nested_under("func.func")
        .add_pass(pass::conversion::create_index_to_llvm());
    pass_manager.add_pass(pass::conversion::create_scf_to_control_flow());
    pass_manager
        .nested_under("func.func")
        .add_pass(pass::conversion::create_arith_to_llvm());
    pass_manager.add_pass(pass::conversion::create_control_flow_to_llvm());
    pass_manager.add_pass(pass::conversion::create_func_to_llvm());
    pass_manager.add_pass(pass::conversion::create_finalize_mem_ref_to_llvm());
    pass_manager
        .run(&mut module)
        .map_err(|err| JitError::Backend(format!("MLIR to LLVM lowering failed: {err:?}")))?;

    Ok(ExecutionEngine::new(&module, 3, &[], false, false))
}
