use crate::{AggregateFunc, FilterSumValue, JitError, JitResult, JitType, MlirColumn, MlirModule};

use melior::{ir::Module, pass, ExecutionEngine};
use quill_runtime::GroupAggregateStateField;

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

pub struct CompiledGroupAggregateUpdate {
    symbol: String,
    engine: ExecutionEngine,
    columns: Vec<MlirColumn>,
    aggregate_funcs: Vec<AggregateFunc>,
    state_types: Vec<JitType>,
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

impl CompiledGroupAggregateUpdate {
    pub fn invoke(
        &self,
        group_ids: &[i64],
        inputs: &[FixedColumnInput<'_>],
        touched: &mut [u8],
        state_fields: &mut [GroupAggregateStateField],
    ) -> JitResult<()> {
        if state_fields.len() != self.state_types.len() {
            return Err(JitError::Backend(format!(
                "compiled group aggregate expected {} state fields, got {}",
                self.state_types.len(),
                state_fields.len()
            )));
        }
        for (index, (field, ty)) in state_fields.iter().zip(&self.state_types).enumerate() {
            if field.ty() != *ty {
                return Err(JitError::Backend(format!(
                    "compiled group aggregate state field {index} has incompatible type"
                )));
            }
        }

        let mut input_len = Some(group_ids.len());
        let mut input_ptrs = Vec::with_capacity(self.columns.len());
        for column in &self.columns {
            let input = inputs
                .iter()
                .find(|input| input.index() == column.index)
                .ok_or_else(|| {
                    JitError::Backend(format!(
                        "compiled group aggregate missing input column {}",
                        column.index
                    ))
                })?;
            if !input.matches_type(column.ty) {
                return Err(JitError::Backend(format!(
                    "compiled group aggregate input column {} has incompatible type",
                    column.index
                )));
            }
            match input_len {
                Some(expected) if expected != input.len() => {
                    return Err(JitError::Backend(format!(
                        "compiled group aggregate input column {} len {} does not match len {}",
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

        let max_group_id = group_ids
            .iter()
            .copied()
            .filter(|group_id| *group_id >= 0)
            .max();
        if let Some(max_group_id) = max_group_id {
            let max_group_id = usize::try_from(max_group_id)
                .map_err(|_| JitError::Backend("group id does not fit in usize".to_string()))?;
            if touched.len() <= max_group_id {
                return Err(JitError::Backend(format!(
                    "compiled group aggregate touched bitmap len {} cannot hold group id {max_group_id}",
                    touched.len()
                )));
            }
            for (index, field) in state_fields.iter().enumerate() {
                if field.len() <= max_group_id {
                    return Err(JitError::Backend(format!(
                        "compiled group aggregate state field {index} len {} cannot hold group id {max_group_id}",
                        field.len()
                    )));
                }
            }
        }

        let mut len = group_ids.len() as i64;
        let mut group_ids_ptr = group_ids.as_ptr();
        let mut touched_ptr = touched.as_mut_ptr();
        let mut state_ptrs = state_fields
            .iter_mut()
            .map(GroupAggregateStateField::values_ptr)
            .collect::<Vec<_>>();
        let mut valid_ptrs = state_fields
            .iter_mut()
            .map(GroupAggregateStateField::valid_ptr)
            .collect::<Vec<_>>();
        let mut result = -1_i32;
        let mut packed_args =
            Vec::with_capacity(3 + input_ptrs.len() + state_ptrs.len() + valid_ptrs.len() + 1);
        packed_args.push(&mut len as *mut i64 as *mut ());
        packed_args.push(&mut group_ids_ptr as *mut *const i64 as *mut ());
        packed_args.push(&mut touched_ptr as *mut *mut u8 as *mut ());
        for ptr in &mut input_ptrs {
            packed_args.push(ptr as *mut *const () as *mut ());
        }
        for ptr in &mut state_ptrs {
            packed_args.push(ptr as *mut *mut () as *mut ());
        }
        for ptr in &mut valid_ptrs {
            packed_args.push(ptr as *mut *mut u8 as *mut ());
        }
        packed_args.push(&mut result as *mut i32 as *mut ());

        unsafe {
            self.engine
                .invoke_packed(&self.symbol, &mut packed_args)
                .map_err(|err| JitError::Backend(format!("MLIR invocation failed: {err:?}")))?;
        }
        if result == 0 {
            Ok(())
        } else {
            Err(JitError::Backend(format!(
                "compiled group aggregate returned status {result}"
            )))
        }
    }

    pub fn aggregate_funcs(&self) -> &[AggregateFunc] {
        &self.aggregate_funcs
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

pub fn compile_group_aggregate_update(
    compiled: &MlirModule,
    columns: Vec<MlirColumn>,
    aggregate_funcs: Vec<AggregateFunc>,
    state_types: Vec<JitType>,
) -> JitResult<CompiledGroupAggregateUpdate> {
    Ok(CompiledGroupAggregateUpdate {
        symbol: compiled.symbol.clone(),
        engine: compile_engine(compiled)?,
        columns,
        aggregate_funcs,
        state_types,
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
