use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, BooleanArray, BooleanBuilder, Date32Array, Date32Builder, Decimal128Array,
    Decimal128Builder, Float64Array, Float64Builder, Int32Array, Int32Builder, Int64Array,
    Int64Builder,
};
use arrow::datatypes::DataType as ArrowDataType;
use arrow::record_batch::RecordBatch;

use quill_plan::{JitError, JitResult, JitType};

use super::value::Scalar;

pub(super) struct BatchView<'a> {
    columns: Vec<ColumnView<'a>>,
}

impl<'a> BatchView<'a> {
    pub(super) fn try_new(batch: &'a RecordBatch) -> JitResult<Self> {
        let columns = batch
            .columns()
            .iter()
            .map(|array| ColumnView::try_new(array.as_ref()))
            .collect::<JitResult<Vec<_>>>()?;
        Ok(Self { columns })
    }

    pub(super) fn value(&self, index: usize, row: usize) -> JitResult<Scalar> {
        let column = self
            .columns
            .get(index)
            .ok_or_else(|| JitError::Backend(format!("column index {index} out of bounds")))?;
        column.value(row)
    }
}

enum ColumnView<'a> {
    Bool(&'a BooleanArray),
    Date32(&'a Date32Array),
    Int32(&'a Int32Array),
    Int64(&'a Int64Array),
    Float64(&'a Float64Array),
    Decimal128(&'a Decimal128Array),
}

impl<'a> ColumnView<'a> {
    fn try_new(array: &'a dyn Array) -> JitResult<Self> {
        match array.data_type() {
            ArrowDataType::Boolean => array
                .as_any()
                .downcast_ref::<BooleanArray>()
                .map(Self::Bool)
                .ok_or_else(|| JitError::UnsupportedType("Boolean".to_string())),
            ArrowDataType::Date32 => array
                .as_any()
                .downcast_ref::<Date32Array>()
                .map(Self::Date32)
                .ok_or_else(|| JitError::UnsupportedType("Date32".to_string())),
            ArrowDataType::Int32 => array
                .as_any()
                .downcast_ref::<Int32Array>()
                .map(Self::Int32)
                .ok_or_else(|| JitError::UnsupportedType("Int32".to_string())),
            ArrowDataType::Int64 => array
                .as_any()
                .downcast_ref::<Int64Array>()
                .map(Self::Int64)
                .ok_or_else(|| JitError::UnsupportedType("Int64".to_string())),
            ArrowDataType::Float64 => array
                .as_any()
                .downcast_ref::<Float64Array>()
                .map(Self::Float64)
                .ok_or_else(|| JitError::UnsupportedType("Float64".to_string())),
            ArrowDataType::Decimal128(_, _) => array
                .as_any()
                .downcast_ref::<Decimal128Array>()
                .map(Self::Decimal128)
                .ok_or_else(|| JitError::UnsupportedType("Decimal128".to_string())),
            other => Err(JitError::UnsupportedType(format!("{other:?}"))),
        }
    }

    fn value(&self, row: usize) -> JitResult<Scalar> {
        match self {
            Self::Bool(array) => Ok(Scalar::Bool(array.is_valid(row).then(|| array.value(row)))),
            Self::Date32(array) => Ok(Scalar::Date32(
                array.is_valid(row).then(|| array.value(row)),
            )),
            Self::Int32(array) => Ok(Scalar::Int32(array.is_valid(row).then(|| array.value(row)))),
            Self::Int64(array) => Ok(Scalar::Int64(array.is_valid(row).then(|| array.value(row)))),
            Self::Float64(array) => Ok(Scalar::Float64(
                array.is_valid(row).then(|| array.value(row)),
            )),
            Self::Decimal128(array) => Ok(Scalar::Decimal128 {
                value: array.is_valid(row).then(|| array.value(row)),
                precision: array.precision(),
                scale: array.scale(),
            }),
        }
    }
}

pub(super) enum OutputBuilder {
    Bool(BooleanBuilder),
    Date32(Date32Builder),
    Int32(Int32Builder),
    Int64(Int64Builder),
    Float64(Float64Builder),
    Decimal128(Decimal128Builder),
}

impl OutputBuilder {
    pub(super) fn with_capacity(ty: JitType, capacity: usize) -> Self {
        match ty {
            JitType::Bool => Self::Bool(BooleanBuilder::with_capacity(capacity)),
            JitType::Date32 => Self::Date32(Date32Builder::with_capacity(capacity)),
            JitType::Int32 => Self::Int32(Int32Builder::with_capacity(capacity)),
            JitType::Int64 => Self::Int64(Int64Builder::with_capacity(capacity)),
            JitType::Float64 => Self::Float64(Float64Builder::with_capacity(capacity)),
            JitType::Decimal128 { precision, scale } => Self::Decimal128(
                Decimal128Builder::with_capacity(capacity)
                    .with_data_type(ArrowDataType::Decimal128(precision, scale)),
            ),
        }
    }

    pub(super) fn with_arrow_type(data_type: &ArrowDataType, capacity: usize) -> JitResult<Self> {
        match data_type {
            ArrowDataType::Boolean => Ok(Self::Bool(BooleanBuilder::with_capacity(capacity))),
            ArrowDataType::Date32 => Ok(Self::Date32(Date32Builder::with_capacity(capacity))),
            ArrowDataType::Int32 => Ok(Self::Int32(Int32Builder::with_capacity(capacity))),
            ArrowDataType::Int64 => Ok(Self::Int64(Int64Builder::with_capacity(capacity))),
            ArrowDataType::Float64 => Ok(Self::Float64(Float64Builder::with_capacity(capacity))),
            ArrowDataType::Decimal128(precision, scale) => Ok(Self::Decimal128(
                Decimal128Builder::with_capacity(capacity)
                    .with_data_type(ArrowDataType::Decimal128(*precision, *scale)),
            )),
            other => Err(JitError::UnsupportedType(format!("{other:?}"))),
        }
    }

    pub(super) fn append(&mut self, value: Scalar) -> JitResult<()> {
        match (self, value) {
            (Self::Bool(builder), Scalar::Bool(value)) => builder.append_option(value),
            (Self::Date32(builder), Scalar::Date32(value)) => builder.append_option(value),
            (Self::Int32(builder), Scalar::Int32(value)) => builder.append_option(value),
            (Self::Int64(builder), Scalar::Int64(value)) => builder.append_option(value),
            (Self::Float64(builder), Scalar::Float64(value)) => builder.append_option(value),
            (
                Self::Decimal128(builder),
                Scalar::Decimal128 {
                    value,
                    precision: _,
                    scale: _,
                },
            ) => builder.append_option(value),
            (_, value) => {
                return Err(JitError::Backend(format!(
                    "cannot append {:?} into output builder",
                    value.ty()
                )));
            }
        }
        Ok(())
    }

    pub(super) fn finish(mut self) -> JitResult<ArrayRef> {
        let array = match &mut self {
            Self::Bool(builder) => Arc::new(builder.finish()) as ArrayRef,
            Self::Date32(builder) => Arc::new(builder.finish()) as ArrayRef,
            Self::Int32(builder) => Arc::new(builder.finish()) as ArrayRef,
            Self::Int64(builder) => Arc::new(builder.finish()) as ArrayRef,
            Self::Float64(builder) => Arc::new(builder.finish()) as ArrayRef,
            Self::Decimal128(builder) => Arc::new(builder.finish()) as ArrayRef,
        };
        Ok(array)
    }
}

pub(super) fn arrow_type(ty: JitType) -> ArrowDataType {
    match ty {
        JitType::Bool => ArrowDataType::Boolean,
        JitType::Date32 => ArrowDataType::Date32,
        JitType::Int32 => ArrowDataType::Int32,
        JitType::Int64 => ArrowDataType::Int64,
        JitType::Float64 => ArrowDataType::Float64,
        JitType::Decimal128 { precision, scale } => ArrowDataType::Decimal128(precision, scale),
    }
}
