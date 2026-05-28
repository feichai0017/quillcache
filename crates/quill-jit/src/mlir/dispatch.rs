use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, Date32Array, Decimal128Array, Float64Array, Int64Array};
use arrow::datatypes::DataType as ArrowDataType;
use arrow::record_batch::RecordBatch;
use quill_plan::{JitError, JitResult};
type Result<T> = JitResult<T>;

use crate::{
    CompiledKernel, CompiledPlainSum, CompiledRecordPipeline, FilterProjectKernel, FilterSumKernel,
    FilterSumValue, FixedColumnInput, JitType, MlirBackend, MlirColumn, PipelineSpec,
    RecordPipelineOutput,
};

thread_local! {
    static RECORD_PIPELINE_CACHE: RefCell<HashMap<String, CompiledRecordPipeline>> =
        RefCell::new(HashMap::new());
    static PLAIN_SUM_CACHE: RefCell<HashMap<String, CompiledPlainSum>> =
        RefCell::new(HashMap::new());
}

pub fn execute_filter_project(
    kernel: &CompiledKernel,
    runtime: &FilterProjectKernel,
    batch: &RecordBatch,
) -> Result<Option<RecordBatch>> {
    if !kernel.executable || kernel.backend != "mlir" {
        return Ok(None);
    }

    let PipelineSpec::RecordProject {
        columns,
        output_types,
    } = &kernel.spec
    else {
        return Ok(None);
    };
    let Some(inputs) = fixed_inputs(batch, columns)? else {
        return Ok(None);
    };

    let mut buffers = output_types
        .iter()
        .map(|ty| OutputBuffer::with_capacity(*ty, batch.num_rows()))
        .collect::<Result<Vec<_>>>()?;
    let mut outputs = buffers
        .iter_mut()
        .map(OutputBuffer::as_output)
        .collect::<Vec<_>>();
    let cache_key = filter_project_cache_key(runtime);
    let output_len = RECORD_PIPELINE_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        if !cache.contains_key(&cache_key) {
            let compiled = MlirBackend::new()
                .compile_record_pipeline(runtime.predicate(), runtime.projections())?;
            cache.insert(cache_key.clone(), compiled);
        }
        cache
            .get(&cache_key)
            .expect("compiled kernel was inserted")
            .invoke(&inputs, &mut outputs)
    })?;

    let arrays = buffers
        .into_iter()
        .zip(runtime.schema().fields())
        .map(|(buffer, field)| buffer.finish(output_len, field.data_type()))
        .collect::<Result<Vec<_>>>()?;
    RecordBatch::try_new(runtime.schema(), arrays)
        .map(Some)
        .map_err(|err| JitError::Backend(err.to_string()))
}

pub fn execute_filter_sum(
    kernel: &CompiledKernel,
    runtime: &FilterSumKernel,
    batch: &RecordBatch,
) -> Result<Option<FilterSumValue>> {
    if !kernel.executable || kernel.backend != "mlir" {
        return Ok(None);
    }

    match &kernel.spec {
        PipelineSpec::PlainSum { columns, .. } => execute_plain_sum(runtime, batch, columns),
        _ => Ok(None),
    }
}

fn filter_project_cache_key(runtime: &FilterProjectKernel) -> String {
    format!("{:?}|{:?}", runtime.predicate(), runtime.projections())
}

fn execute_plain_sum(
    runtime: &FilterSumKernel,
    batch: &RecordBatch,
    columns: &[MlirColumn],
) -> Result<Option<FilterSumValue>> {
    let Some(inputs) = fixed_inputs(batch, columns)? else {
        return Ok(None);
    };

    let cache_key = filter_sum_cache_key(runtime);
    let output = PLAIN_SUM_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        if !cache.contains_key(&cache_key) {
            let compiled =
                MlirBackend::new().compile_plain_sum(runtime.predicate(), runtime.measure())?;
            cache.insert(cache_key.clone(), compiled);
        }
        cache
            .get(&cache_key)
            .expect("compiled kernel was inserted")
            .invoke(&inputs)
    })?;
    Ok(Some(output))
}

fn filter_sum_cache_key(runtime: &FilterSumKernel) -> String {
    format!("{:?}|{:?}", runtime.predicate(), runtime.measure())
}

fn fixed_inputs<'a>(
    batch: &'a RecordBatch,
    columns: &[MlirColumn],
) -> Result<Option<Vec<FixedColumnInput<'a>>>> {
    let mut inputs = Vec::with_capacity(columns.len());
    for column in columns {
        let Some(input) = fixed_input(batch, *column)? else {
            return Ok(None);
        };
        inputs.push(input);
    }
    Ok(Some(inputs))
}

fn fixed_input<'a>(
    batch: &'a RecordBatch,
    column: MlirColumn,
) -> Result<Option<FixedColumnInput<'a>>> {
    match column.ty {
        JitType::Date32 => {
            let array = date32_column(batch, column.index)?;
            Ok(date32_values(array).map(|values| FixedColumnInput::Date32 {
                index: column.index,
                values,
            }))
        }
        JitType::Int64 => {
            let array = int64_column(batch, column.index)?;
            Ok(int64_values(array).map(|values| FixedColumnInput::Int64 {
                index: column.index,
                values,
            }))
        }
        JitType::Float64 => {
            let array = float64_column(batch, column.index)?;
            Ok(
                float64_values(array).map(|values| FixedColumnInput::Float64 {
                    index: column.index,
                    values,
                }),
            )
        }
        JitType::Decimal128 { scale, .. } => {
            let array = decimal128_column(batch, column.index)?;
            if array.scale() != scale {
                return Err(JitError::Backend(format!(
                    "decimal input scale {} does not match column scale {}",
                    scale,
                    array.scale()
                )));
            }
            Ok(
                decimal128_values(array).map(|values| FixedColumnInput::Decimal128 {
                    index: column.index,
                    values,
                }),
            )
        }
        other => Err(JitError::Backend(format!(
            "unsupported fixed-width input type {other:?}"
        ))),
    }
}

enum OutputBuffer {
    Date32(Vec<i32>),
    Int64(Vec<i64>),
    Float64(Vec<f64>),
    Decimal128(Vec<i128>),
}

impl OutputBuffer {
    fn with_capacity(ty: JitType, capacity: usize) -> Result<Self> {
        match ty {
            JitType::Date32 => Ok(Self::Date32(vec![0; capacity])),
            JitType::Int64 => Ok(Self::Int64(vec![0; capacity])),
            JitType::Float64 => Ok(Self::Float64(vec![0.0; capacity])),
            JitType::Decimal128 { .. } => Ok(Self::Decimal128(vec![0; capacity])),
            other => Err(JitError::Backend(format!(
                "unsupported record output type {other:?}"
            ))),
        }
    }

    fn as_output(&mut self) -> RecordPipelineOutput<'_> {
        match self {
            Self::Date32(values) => RecordPipelineOutput::Date32 { values },
            Self::Int64(values) => RecordPipelineOutput::Int64 { values },
            Self::Float64(values) => RecordPipelineOutput::Float64 { values },
            Self::Decimal128(values) => RecordPipelineOutput::Decimal128 { values },
        }
    }

    fn finish(self, len: usize, data_type: &ArrowDataType) -> Result<ArrayRef> {
        let array = match (self, data_type) {
            (Self::Date32(values), ArrowDataType::Date32) => {
                Arc::new(Date32Array::from(values[..len].to_vec())) as ArrayRef
            }
            (Self::Int64(values), ArrowDataType::Int64) => {
                Arc::new(Int64Array::from(values[..len].to_vec())) as ArrayRef
            }
            (Self::Float64(values), ArrowDataType::Float64) => {
                Arc::new(Float64Array::from(values[..len].to_vec())) as ArrayRef
            }
            (Self::Decimal128(values), ArrowDataType::Decimal128(precision, scale)) => Arc::new(
                Decimal128Array::from(values[..len].to_vec())
                    .with_precision_and_scale(*precision, *scale)
                    .map_err(|err| JitError::Backend(err.to_string()))?,
            )
                as ArrayRef,
            (_, other) => {
                return Err(JitError::Backend(format!(
                    "record output buffer does not match schema type {other:?}"
                )));
            }
        };
        Ok(array)
    }
}

fn date32_values(array: &Date32Array) -> Option<&[i32]> {
    if array.null_count() != 0 || array.offset() != 0 {
        return None;
    }
    Some(array.values().as_ref())
}

fn int64_values(array: &Int64Array) -> Option<&[i64]> {
    if array.null_count() != 0 || array.offset() != 0 {
        return None;
    }
    Some(array.values().as_ref())
}

fn float64_values(array: &Float64Array) -> Option<&[f64]> {
    if array.null_count() != 0 || array.offset() != 0 {
        return None;
    }
    Some(array.values().as_ref())
}

fn decimal128_values(array: &Decimal128Array) -> Option<&[i128]> {
    if array.null_count() != 0 || array.offset() != 0 {
        return None;
    }
    Some(array.values().as_ref())
}

fn float64_column(batch: &RecordBatch, index: usize) -> Result<&Float64Array> {
    batch
        .column(index)
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| JitError::Backend(format!("column {index} is not Float64")))
}

fn date32_column(batch: &RecordBatch, index: usize) -> Result<&Date32Array> {
    batch
        .column(index)
        .as_any()
        .downcast_ref::<Date32Array>()
        .ok_or_else(|| JitError::Backend(format!("column {index} is not Date32")))
}

fn decimal128_column(batch: &RecordBatch, index: usize) -> Result<&Decimal128Array> {
    batch
        .column(index)
        .as_any()
        .downcast_ref::<Decimal128Array>()
        .ok_or_else(|| JitError::Backend(format!("column {index} is not Decimal128")))
}

fn int64_column(batch: &RecordBatch, index: usize) -> Result<&Int64Array> {
    batch
        .column(index)
        .as_any()
        .downcast_ref::<Int64Array>()
        .ok_or_else(|| JitError::Backend(format!("column {index} is not Int64")))
}
