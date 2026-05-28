use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::sync::Arc;

use arrow::datatypes::SchemaRef as ArrowSchemaRef;
use arrow::record_batch::RecordBatch;

use quill_plan::{
    AggregateFunc, GroupAggregate, JitError, JitExpr, JitResult, JitType, PipelineStage,
};

use super::array::{BatchView, OutputBuilder};
use super::eval::{ensure_supported_expr, eval_expr};
use super::value::Scalar;

#[derive(Debug, Clone)]
pub struct GroupAggregateKernel {
    predicate: Option<JitExpr>,
    keys: Vec<JitExpr>,
    aggregates: Vec<GroupAggregate>,
    schema: ArrowSchemaRef,
}

#[derive(Debug, Clone)]
pub struct GroupAggregateState {
    groups: BTreeMap<GroupKey, Vec<AggregateState>>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct GroupKey(Vec<KeyValue>);

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum KeyValue {
    Bool(Option<bool>),
    Date32(Option<i32>),
    Int32(Option<i32>),
    Int64(Option<i64>),
    Decimal128 {
        value: Option<i128>,
        precision: u8,
        scale: i8,
    },
}

#[derive(Debug, Clone)]
enum AggregateState {
    Sum(Option<Scalar>),
    Count(i64),
    Min(Option<Scalar>),
    Max(Option<Scalar>),
}

impl GroupAggregateKernel {
    pub fn try_new(
        stages: &[PipelineStage],
        keys: Vec<JitExpr>,
        aggregates: Vec<GroupAggregate>,
        schema: ArrowSchemaRef,
    ) -> JitResult<Self> {
        let predicate = match stages {
            [] => None,
            [PipelineStage::Filter(predicate)] => Some(predicate.clone()),
            _ => {
                return Err(JitError::UnsupportedExpr(
                    "group aggregate supports only an optional filter stage".to_string(),
                ));
            }
        };
        if keys.is_empty() || aggregates.is_empty() {
            return Err(JitError::UnsupportedExpr(
                "group aggregate requires at least one key and one aggregate".to_string(),
            ));
        }
        if schema.fields().len() != keys.len() + aggregates.len() {
            return Err(JitError::Backend(format!(
                "group aggregate output schema has {} fields, expected {}",
                schema.fields().len(),
                keys.len() + aggregates.len()
            )));
        }
        if let Some(predicate) = &predicate {
            if predicate.ty() != JitType::Bool {
                return Err(JitError::UnsupportedExpr(format!(
                    "group aggregate predicate must be bool, got {:?}",
                    predicate.ty()
                )));
            }
            ensure_supported_expr(predicate)?;
        }
        for key in &keys {
            ensure_group_key_type(key.ty())?;
            ensure_supported_expr(key)?;
        }
        for aggregate in &aggregates {
            ensure_aggregate_expr(aggregate)?;
        }

        Ok(Self {
            predicate,
            keys,
            aggregates,
            schema,
        })
    }

    pub fn predicate(&self) -> Option<&JitExpr> {
        self.predicate.as_ref()
    }

    pub fn keys(&self) -> &[JitExpr] {
        &self.keys
    }

    pub fn aggregates(&self) -> &[GroupAggregate] {
        &self.aggregates
    }

    pub fn stage_names(&self) -> &'static str {
        if self.predicate.is_some() {
            "filter"
        } else {
            ""
        }
    }

    pub fn new_state(&self) -> GroupAggregateState {
        GroupAggregateState {
            groups: BTreeMap::new(),
        }
    }

    pub fn accumulate(
        &self,
        state: &mut GroupAggregateState,
        batch: &RecordBatch,
    ) -> JitResult<()> {
        let view = BatchView::try_new(batch)?;
        for row in 0..batch.num_rows() {
            if let Some(predicate) = &self.predicate {
                if !eval_expr(predicate, &view, row)?.is_filter_true()? {
                    continue;
                }
            }
            let key = self.eval_key(&view, row)?;
            let aggregates = state.groups.entry(key).or_insert_with(|| {
                self.aggregates
                    .iter()
                    .map(AggregateState::empty)
                    .collect::<Vec<_>>()
            });
            for (aggregate, aggregate_state) in self.aggregates.iter().zip(aggregates) {
                let value = eval_expr(&aggregate.expr, &view, row)?;
                aggregate_state.update(aggregate.func, value)?;
            }
        }
        Ok(())
    }

    pub fn finish(&self, state: GroupAggregateState) -> JitResult<RecordBatch> {
        let mut builders = self
            .schema
            .fields()
            .iter()
            .map(|field| OutputBuilder::with_arrow_type(field.data_type(), state.groups.len()))
            .collect::<JitResult<Vec<_>>>()?;

        for (key, aggregates) in state.groups {
            for (value, builder) in key.0.into_iter().zip(&mut builders) {
                builder.append(value.into_scalar())?;
            }
            for (aggregate, (state, builder)) in self.aggregates.iter().zip(
                aggregates
                    .into_iter()
                    .zip(builders.iter_mut().skip(self.keys.len())),
            ) {
                builder.append(state.finish(aggregate)?)?;
            }
        }

        let arrays = builders
            .into_iter()
            .map(OutputBuilder::finish)
            .collect::<JitResult<Vec<_>>>()?;
        RecordBatch::try_new(Arc::clone(&self.schema), arrays)
            .map_err(|err| JitError::Backend(err.to_string()))
    }

    fn eval_key(&self, view: &BatchView<'_>, row: usize) -> JitResult<GroupKey> {
        self.keys
            .iter()
            .map(|expr| KeyValue::try_from_scalar(eval_expr(expr, view, row)?))
            .collect::<JitResult<Vec<_>>>()
            .map(GroupKey)
    }
}

impl AggregateState {
    fn empty(aggregate: &GroupAggregate) -> Self {
        match aggregate.func {
            AggregateFunc::Sum => Self::Sum(None),
            AggregateFunc::Count => Self::Count(0),
            AggregateFunc::Min => Self::Min(None),
            AggregateFunc::Max => Self::Max(None),
        }
    }

    fn update(&mut self, func: AggregateFunc, value: Scalar) -> JitResult<()> {
        match (self, func) {
            (Self::Sum(sum), AggregateFunc::Sum) => {
                if value.is_null() {
                    return Ok(());
                }
                *sum = Some(match *sum {
                    Some(current) => current.checked_add(value)?,
                    None => value,
                });
                Ok(())
            }
            (Self::Count(count), AggregateFunc::Count) => {
                if !value.is_null() {
                    *count += 1;
                }
                Ok(())
            }
            (Self::Min(min), AggregateFunc::Min) => update_minmax(min, value, Ordering::Less),
            (Self::Max(max), AggregateFunc::Max) => update_minmax(max, value, Ordering::Greater),
            (_, other) => Err(JitError::Backend(format!(
                "aggregate state does not match function {}",
                other.name()
            ))),
        }
    }

    fn finish(self, aggregate: &GroupAggregate) -> JitResult<Scalar> {
        match (self, aggregate.func) {
            (Self::Sum(value), AggregateFunc::Sum) => coerce_scalar(
                value.unwrap_or_else(|| null_scalar(aggregate.output_type)),
                aggregate.output_type,
            ),
            (Self::Count(value), AggregateFunc::Count) => Ok(Scalar::Int64(Some(value))),
            (Self::Min(value), AggregateFunc::Min) | (Self::Max(value), AggregateFunc::Max) => {
                coerce_scalar(
                    value.unwrap_or_else(|| null_scalar(aggregate.output_type)),
                    aggregate.output_type,
                )
            }
            (_, other) => Err(JitError::Backend(format!(
                "aggregate state does not match function {}",
                other.name()
            ))),
        }
    }
}

impl KeyValue {
    fn try_from_scalar(value: Scalar) -> JitResult<Self> {
        match value {
            Scalar::Bool(value) => Ok(Self::Bool(value)),
            Scalar::Date32(value) => Ok(Self::Date32(value)),
            Scalar::Int32(value) => Ok(Self::Int32(value)),
            Scalar::Int64(value) => Ok(Self::Int64(value)),
            Scalar::Decimal128 {
                value,
                precision,
                scale,
            } => Ok(Self::Decimal128 {
                value,
                precision,
                scale,
            }),
            Scalar::Float64(_) => Err(JitError::UnsupportedType(
                "Float64 group keys are not supported by the grouped aggregate runtime".to_string(),
            )),
        }
    }

    fn into_scalar(self) -> Scalar {
        match self {
            Self::Bool(value) => Scalar::Bool(value),
            Self::Date32(value) => Scalar::Date32(value),
            Self::Int32(value) => Scalar::Int32(value),
            Self::Int64(value) => Scalar::Int64(value),
            Self::Decimal128 {
                value,
                precision,
                scale,
            } => Scalar::Decimal128 {
                value,
                precision,
                scale,
            },
        }
    }
}

fn ensure_group_key_type(ty: JitType) -> JitResult<()> {
    match ty {
        JitType::Bool
        | JitType::Date32
        | JitType::Int32
        | JitType::Int64
        | JitType::Decimal128 { .. } => Ok(()),
        JitType::Float64 => Err(JitError::UnsupportedType(
            "Float64 group keys are not supported by the grouped aggregate runtime".to_string(),
        )),
    }
}

fn ensure_aggregate_expr(aggregate: &GroupAggregate) -> JitResult<()> {
    ensure_supported_expr(&aggregate.expr)?;
    match aggregate.func {
        AggregateFunc::Sum => match aggregate.expr.ty() {
            JitType::Int32 | JitType::Int64 | JitType::Float64 | JitType::Decimal128 { .. } => {
                Ok(())
            }
            other => Err(JitError::UnsupportedType(format!(
                "SUM does not support {other:?}"
            ))),
        },
        AggregateFunc::Count => Ok(()),
        AggregateFunc::Min | AggregateFunc::Max => match aggregate.expr.ty() {
            JitType::Bool
            | JitType::Date32
            | JitType::Int32
            | JitType::Int64
            | JitType::Float64
            | JitType::Decimal128 { .. } => Ok(()),
        },
    }
}

fn update_minmax(target: &mut Option<Scalar>, value: Scalar, ordering: Ordering) -> JitResult<()> {
    if value.is_null() {
        return Ok(());
    }
    let Some(current) = *target else {
        *target = Some(value);
        return Ok(());
    };
    if current.partial_cmp_value(value)? == Some(ordering) {
        return Ok(());
    }
    *target = Some(value);
    Ok(())
}

fn null_scalar(ty: JitType) -> Scalar {
    match ty {
        JitType::Bool => Scalar::Bool(None),
        JitType::Date32 => Scalar::Date32(None),
        JitType::Int32 => Scalar::Int32(None),
        JitType::Int64 => Scalar::Int64(None),
        JitType::Float64 => Scalar::Float64(None),
        JitType::Decimal128 { precision, scale } => Scalar::Decimal128 {
            value: None,
            precision,
            scale,
        },
    }
}

fn coerce_scalar(value: Scalar, ty: JitType) -> JitResult<Scalar> {
    match (value, ty) {
        (Scalar::Int32(value), JitType::Int64) => Ok(Scalar::Int64(value.map(i64::from))),
        (Scalar::Decimal128 { value, .. }, JitType::Decimal128 { precision, scale }) => {
            Ok(Scalar::Decimal128 {
                value,
                precision,
                scale,
            })
        }
        (value, ty) if value.ty() == ty => Ok(value),
        (value, ty) => Err(JitError::Backend(format!(
            "cannot coerce aggregate value {:?} to {:?}",
            value.ty(),
            ty
        ))),
    }
}
