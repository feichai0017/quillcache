use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap};
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
    group_ids: BTreeMap<GroupKey, usize>,
    fast_group_ids: BTreeMap<Vec<FastKeyValue>, usize>,
    string_key_ids: Vec<HashMap<Arc<str>, u32>>,
    groups: Vec<GroupState>,
    dense: Option<GroupAggregateDenseState>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GroupAggregateBatchBinding {
    group_ids: Vec<i64>,
    selected_rows: usize,
}

#[derive(Debug, Clone)]
pub struct GroupAggregateDenseState {
    fields: Vec<GroupAggregateStateField>,
    group_count: usize,
}

#[derive(Debug, Clone)]
pub enum GroupAggregateStateField {
    Int64 {
        values: Vec<i64>,
        valid: Vec<u8>,
    },
    UInt64 {
        values: Vec<u64>,
        valid: Vec<u8>,
    },
    Float64 {
        values: Vec<f64>,
        valid: Vec<u8>,
    },
    Decimal128 {
        values: Vec<i128>,
        valid: Vec<u8>,
        precision: u8,
        scale: i8,
    },
}

#[derive(Debug, Clone)]
struct GroupState {
    key: GroupKey,
    aggregates: Vec<AggregateState>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct GroupKey(Vec<KeyValue>);

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum KeyValue {
    Bool(Option<bool>),
    Date32(Option<i32>),
    Int32(Option<i32>),
    Int64(Option<i64>),
    UInt64(Option<u64>),
    Utf8(Option<Arc<str>>),
    Decimal128 {
        value: Option<i128>,
        precision: u8,
        scale: i8,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum FastKeyValue {
    Bool(Option<bool>),
    Date32(Option<i32>),
    Int32(Option<i32>),
    Int64(Option<i64>),
    UInt64(Option<u64>),
    Utf8(Option<u32>),
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
    Avg { sum: Option<Scalar>, count: u64 },
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
        let aggregate_fields = aggregates
            .iter()
            .map(|aggregate| aggregate.state_types.len())
            .sum::<usize>();
        if schema.fields().len() != keys.len() + aggregate_fields {
            return Err(JitError::Backend(format!(
                "group aggregate output schema has {} fields, expected {}",
                schema.fields().len(),
                keys.len() + aggregate_fields
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
            group_ids: BTreeMap::new(),
            fast_group_ids: BTreeMap::new(),
            string_key_ids: vec![HashMap::new(); self.keys.len()],
            groups: Vec::new(),
            dense: None,
        }
    }

    pub fn bind_batch(
        &self,
        state: &mut GroupAggregateState,
        batch: &RecordBatch,
    ) -> JitResult<GroupAggregateBatchBinding> {
        let view = BatchView::try_new(batch)?;
        let mut group_ids = Vec::with_capacity(batch.num_rows());
        let mut selected_rows = 0_usize;

        for row in 0..batch.num_rows() {
            if let Some(predicate) = &self.predicate {
                if !eval_expr(predicate, &view, row)?.is_filter_true()? {
                    group_ids.push(-1);
                    continue;
                }
            }

            let group_id = self.group_id_for_row(state, &view, row)?;
            let group_id = i64::try_from(group_id)
                .map_err(|_| JitError::Backend("group id does not fit in i64".to_string()))?;
            group_ids.push(group_id);
            selected_rows += 1;
        }

        Ok(GroupAggregateBatchBinding {
            group_ids,
            selected_rows,
        })
    }

    pub fn accumulate(
        &self,
        state: &mut GroupAggregateState,
        batch: &RecordBatch,
    ) -> JitResult<()> {
        let binding = self.bind_batch(state, batch)?;
        self.accumulate_bound(state, batch, &binding)
    }

    pub fn accumulate_bound(
        &self,
        state: &mut GroupAggregateState,
        batch: &RecordBatch,
        binding: &GroupAggregateBatchBinding,
    ) -> JitResult<()> {
        let view = BatchView::try_new(batch)?;
        for (row, group_id) in binding.group_ids.iter().copied().enumerate() {
            if group_id < 0 {
                continue;
            }
            let group_id = usize::try_from(group_id)
                .map_err(|_| JitError::Backend("negative group id".to_string()))?;
            let aggregates = &mut state.groups[group_id].aggregates;
            for (aggregate, aggregate_state) in self.aggregates.iter().zip(aggregates) {
                let value = eval_expr(&aggregate.expr, &view, row)?;
                aggregate_state.update(aggregate.func, value)?;
            }
        }
        Ok(())
    }

    pub fn dense_state_mut<'a>(
        &self,
        state: &'a mut GroupAggregateState,
    ) -> JitResult<&'a mut GroupAggregateDenseState> {
        state.dense_state_mut(&self.aggregates)
    }

    pub fn flush_dense_state(&self, state: &mut GroupAggregateState) -> JitResult<()> {
        state.flush_dense_state(&self.aggregates)
    }

    pub fn finish(&self, mut state: GroupAggregateState) -> JitResult<RecordBatch> {
        if let Some(dense) = state.dense.take() {
            return self.finish_dense_state(state, dense);
        }
        self.finish_sparse_state(state)
    }

    fn finish_sparse_state(&self, state: GroupAggregateState) -> JitResult<RecordBatch> {
        let mut builders = self.output_builders(state.groups.len())?;

        for group in state.sorted_groups() {
            for (value, builder) in group.key.0.into_iter().zip(&mut builders) {
                builder.append(value.into_scalar())?;
            }
            let mut builder_index = self.keys.len();
            for (aggregate, state) in self.aggregates.iter().zip(group.aggregates) {
                let values = state.finish_states(aggregate)?;
                for value in values {
                    let builder = builders.get_mut(builder_index).ok_or_else(|| {
                        JitError::Backend(format!(
                            "missing output builder for aggregate {}",
                            aggregate.alias
                        ))
                    })?;
                    builder.append(value)?;
                    builder_index += 1;
                }
            }
        }

        let arrays = builders
            .into_iter()
            .map(OutputBuilder::finish)
            .collect::<JitResult<Vec<_>>>()?;
        RecordBatch::try_new(Arc::clone(&self.schema), arrays)
            .map_err(|err| JitError::Backend(err.to_string()))
    }

    fn finish_dense_state(
        &self,
        state: GroupAggregateState,
        dense: GroupAggregateDenseState,
    ) -> JitResult<RecordBatch> {
        if dense.group_count != state.groups.len() {
            return Err(JitError::Backend(format!(
                "dense state has {} groups, runtime state has {}",
                dense.group_count,
                state.groups.len()
            )));
        }

        let mut builders = self.output_builders(state.groups.len())?;
        let mut sorted = state.group_ids.into_iter().collect::<Vec<_>>();
        sorted.sort_by(|(left, _), (right, _)| left.cmp(right));

        for (key, group_id) in sorted {
            if group_id >= state.groups.len() {
                return Err(JitError::Backend(format!(
                    "dense group id {group_id} out of bounds"
                )));
            }
            for (value, builder) in key.0.into_iter().zip(&mut builders) {
                builder.append(value.into_scalar())?;
            }
            let mut field_index = 0_usize;
            let mut builder_index = self.keys.len();
            for aggregate in &self.aggregates {
                for _ in &aggregate.state_types {
                    let field = dense.fields.get(field_index).ok_or_else(|| {
                        JitError::Backend("missing dense aggregate state field".to_string())
                    })?;
                    let builder = builders.get_mut(builder_index).ok_or_else(|| {
                        JitError::Backend(format!(
                            "missing output builder for aggregate {}",
                            aggregate.alias
                        ))
                    })?;
                    builder.append(field.scalar(group_id)?)?;
                    field_index += 1;
                    builder_index += 1;
                }
            }
            if field_index != dense.fields.len() {
                return Err(JitError::Backend(format!(
                    "dense state has {} fields, consumed {field_index}",
                    dense.fields.len()
                )));
            }
        }

        let arrays = builders
            .into_iter()
            .map(OutputBuilder::finish)
            .collect::<JitResult<Vec<_>>>()?;
        RecordBatch::try_new(Arc::clone(&self.schema), arrays)
            .map_err(|err| JitError::Backend(err.to_string()))
    }

    fn output_builders(&self, capacity: usize) -> JitResult<Vec<OutputBuilder>> {
        self.schema
            .fields()
            .iter()
            .map(|field| OutputBuilder::with_arrow_type(field.data_type(), capacity))
            .collect()
    }

    fn eval_key(&self, view: &BatchView<'_>, row: usize) -> JitResult<GroupKey> {
        self.keys
            .iter()
            .map(|expr| KeyValue::try_from_scalar(eval_expr(expr, view, row)?))
            .collect::<JitResult<Vec<_>>>()
            .map(GroupKey)
    }

    fn group_id_for_row(
        &self,
        state: &mut GroupAggregateState,
        view: &BatchView<'_>,
        row: usize,
    ) -> JitResult<usize> {
        if let Some(fast_key) = self.fast_key_for_row(state, view, row)? {
            if let Some(group_id) = state.fast_group_ids.get(fast_key.as_slice()) {
                return Ok(*group_id);
            }

            let key = self.eval_key(view, row)?;
            return Ok(state.group_id_with_fast_key(key, fast_key, &self.aggregates));
        }

        let key = self.eval_key(view, row)?;
        Ok(state.group_id(key, &self.aggregates))
    }

    fn fast_key_for_row(
        &self,
        state: &mut GroupAggregateState,
        view: &BatchView<'_>,
        row: usize,
    ) -> JitResult<Option<Vec<FastKeyValue>>> {
        let mut key = Vec::with_capacity(self.keys.len());
        for (key_index, expr) in self.keys.iter().enumerate() {
            match expr {
                JitExpr::Column {
                    index,
                    ty: JitType::Utf8,
                    ..
                } => {
                    let value = view
                        .utf8_value(*index, row)?
                        .map(|value| state.intern_string_key(key_index, value))
                        .transpose()?;
                    key.push(FastKeyValue::Utf8(value));
                }
                JitExpr::Column { .. } => {
                    key.push(FastKeyValue::try_from_scalar(eval_expr(expr, view, row)?)?);
                }
                _ => return Ok(None),
            }
        }
        Ok(Some(key))
    }
}

impl GroupAggregateBatchBinding {
    pub fn group_ids(&self) -> &[i64] {
        &self.group_ids
    }

    pub fn selected_rows(&self) -> usize {
        self.selected_rows
    }
}

impl GroupAggregateState {
    fn group_id(&mut self, key: GroupKey, aggregates: &[GroupAggregate]) -> usize {
        if let Some(group_id) = self.group_ids.get(&key) {
            return *group_id;
        }

        self.insert_group(key, aggregates)
    }

    fn group_id_with_fast_key(
        &mut self,
        key: GroupKey,
        fast_key: Vec<FastKeyValue>,
        aggregates: &[GroupAggregate],
    ) -> usize {
        if let Some(group_id) = self.fast_group_ids.get(fast_key.as_slice()) {
            return *group_id;
        }

        let group_id = self.insert_group(key, aggregates);
        self.fast_group_ids.insert(fast_key, group_id);
        group_id
    }

    fn insert_group(&mut self, key: GroupKey, aggregates: &[GroupAggregate]) -> usize {
        let group_id = self.groups.len();
        self.group_ids.insert(key.clone(), group_id);
        self.groups.push(GroupState {
            key,
            aggregates: aggregates.iter().map(AggregateState::empty).collect(),
        });
        if let Some(dense) = &mut self.dense {
            dense.push_empty(aggregates);
        }
        group_id
    }

    fn intern_string_key(&mut self, key_index: usize, value: &str) -> JitResult<u32> {
        let dictionary = self.string_key_ids.get_mut(key_index).ok_or_else(|| {
            JitError::Backend(format!("missing string dictionary for key {key_index}"))
        })?;
        if let Some(id) = dictionary.get(value) {
            return Ok(*id);
        }
        let id = u32::try_from(dictionary.len()).map_err(|_| {
            JitError::Backend("string group-key dictionary exceeded u32 ids".to_string())
        })?;
        dictionary.insert(Arc::from(value), id);
        Ok(id)
    }

    fn sorted_groups(self) -> Vec<GroupState> {
        let groups = self.groups;
        let mut sorted = self.group_ids.into_iter().collect::<Vec<_>>();
        sorted.sort_by(|(left, _), (right, _)| left.cmp(right));
        sorted
            .into_iter()
            .map(|(_, group_id)| groups[group_id].clone())
            .collect()
    }

    fn dense_state_mut(
        &mut self,
        aggregates: &[GroupAggregate],
    ) -> JitResult<&mut GroupAggregateDenseState> {
        if self.dense.is_none() {
            self.dense = Some(self.snapshot_dense_state(aggregates)?);
        }
        self.dense
            .as_mut()
            .ok_or_else(|| JitError::Backend("missing dense aggregate state".to_string()))
    }

    fn flush_dense_state(&mut self, aggregates: &[GroupAggregate]) -> JitResult<()> {
        let Some(dense) = self.dense.take() else {
            return Ok(());
        };
        self.apply_dense_state(aggregates, dense)
    }

    fn snapshot_dense_state(
        &self,
        aggregates: &[GroupAggregate],
    ) -> JitResult<GroupAggregateDenseState> {
        let group_count = self.groups.len();
        let mut fields = aggregates
            .iter()
            .flat_map(|aggregate| aggregate.state_types.iter().copied())
            .map(|ty| GroupAggregateStateField::with_len(ty, group_count))
            .collect::<JitResult<Vec<_>>>()?;

        for (group_id, group) in self.groups.iter().enumerate() {
            let mut field_index = 0_usize;
            for (aggregate, state) in aggregates.iter().zip(&group.aggregates) {
                for value in state.snapshot_states(aggregate)? {
                    let field = fields.get_mut(field_index).ok_or_else(|| {
                        JitError::Backend("missing dense aggregate state field".to_string())
                    })?;
                    field.set_scalar(group_id, value)?;
                    field_index += 1;
                }
            }
        }

        Ok(GroupAggregateDenseState {
            fields,
            group_count,
        })
    }

    fn apply_dense_state(
        &mut self,
        aggregates: &[GroupAggregate],
        dense: GroupAggregateDenseState,
    ) -> JitResult<()> {
        if dense.group_count != self.groups.len() {
            return Err(JitError::Backend(format!(
                "dense state has {} groups, runtime state has {}",
                dense.group_count,
                self.groups.len()
            )));
        }

        for (group_id, group) in self.groups.iter_mut().enumerate() {
            let mut field_index = 0_usize;
            for (aggregate, state) in aggregates.iter().zip(&mut group.aggregates) {
                let mut values = Vec::with_capacity(aggregate.state_types.len());
                for _ in &aggregate.state_types {
                    let field = dense.fields.get(field_index).ok_or_else(|| {
                        JitError::Backend("missing dense aggregate state field".to_string())
                    })?;
                    values.push(field.scalar(group_id)?);
                    field_index += 1;
                }
                state.replace_states(aggregate, values)?;
            }
        }

        Ok(())
    }
}

impl AggregateState {
    fn empty(aggregate: &GroupAggregate) -> Self {
        match aggregate.func {
            AggregateFunc::Sum => Self::Sum(None),
            AggregateFunc::Count => Self::Count(0),
            AggregateFunc::Avg => Self::Avg {
                sum: None,
                count: 0,
            },
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
                *sum = Some(match sum.take() {
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
            (Self::Avg { sum, count }, AggregateFunc::Avg) => {
                if value.is_null() {
                    return Ok(());
                }
                *sum = Some(match sum.take() {
                    Some(current) => current.checked_add(value)?,
                    None => value,
                });
                *count += 1;
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

    fn finish_states(self, aggregate: &GroupAggregate) -> JitResult<Vec<Scalar>> {
        match (self, aggregate.func) {
            (Self::Sum(value), AggregateFunc::Sum) => {
                ensure_state_len(aggregate, 1)?;
                let ty = aggregate.state_types[0];
                Ok(vec![coerce_scalar(
                    value.unwrap_or_else(|| null_scalar(ty)),
                    ty,
                )?])
            }
            (Self::Count(value), AggregateFunc::Count) => {
                ensure_state_len(aggregate, 1)?;
                let ty = aggregate.state_types[0];
                Ok(vec![coerce_scalar(Scalar::Int64(Some(value)), ty)?])
            }
            (Self::Avg { sum, count }, AggregateFunc::Avg) => {
                ensure_state_len(aggregate, 2)?;
                let count_ty = aggregate.state_types[0];
                let sum_ty = aggregate.state_types[1];
                Ok(vec![
                    coerce_scalar(Scalar::UInt64(Some(count)), count_ty)?,
                    coerce_scalar(sum.unwrap_or_else(|| null_scalar(sum_ty)), sum_ty)?,
                ])
            }
            (Self::Min(value), AggregateFunc::Min) | (Self::Max(value), AggregateFunc::Max) => {
                ensure_state_len(aggregate, 1)?;
                let ty = aggregate.state_types[0];
                Ok(vec![coerce_scalar(
                    value.unwrap_or_else(|| null_scalar(ty)),
                    ty,
                )?])
            }
            (_, other) => Err(JitError::Backend(format!(
                "aggregate state does not match function {}",
                other.name()
            ))),
        }
    }

    fn snapshot_states(&self, aggregate: &GroupAggregate) -> JitResult<Vec<Scalar>> {
        match (self, aggregate.func) {
            (Self::Sum(value), AggregateFunc::Sum) => {
                ensure_state_len(aggregate, 1)?;
                let ty = aggregate.state_types[0];
                Ok(vec![coerce_scalar(
                    value.clone().unwrap_or_else(|| null_scalar(ty)),
                    ty,
                )?])
            }
            (Self::Count(value), AggregateFunc::Count) => {
                ensure_state_len(aggregate, 1)?;
                let ty = aggregate.state_types[0];
                Ok(vec![coerce_scalar(Scalar::Int64(Some(*value)), ty)?])
            }
            (Self::Avg { sum, count }, AggregateFunc::Avg) => {
                ensure_state_len(aggregate, 2)?;
                let count_ty = aggregate.state_types[0];
                let sum_ty = aggregate.state_types[1];
                Ok(vec![
                    coerce_scalar(Scalar::UInt64(Some(*count)), count_ty)?,
                    coerce_scalar(sum.clone().unwrap_or_else(|| null_scalar(sum_ty)), sum_ty)?,
                ])
            }
            (Self::Min(value), AggregateFunc::Min) | (Self::Max(value), AggregateFunc::Max) => {
                ensure_state_len(aggregate, 1)?;
                let ty = aggregate.state_types[0];
                Ok(vec![coerce_scalar(
                    value.clone().unwrap_or_else(|| null_scalar(ty)),
                    ty,
                )?])
            }
            (_, other) => Err(JitError::Backend(format!(
                "aggregate state does not match function {}",
                other.name()
            ))),
        }
    }

    fn replace_states(&mut self, aggregate: &GroupAggregate, values: Vec<Scalar>) -> JitResult<()> {
        match (self, aggregate.func, values.as_slice()) {
            (Self::Sum(sum), AggregateFunc::Sum, [value]) => {
                ensure_state_len(aggregate, 1)?;
                *sum = (!value.is_null()).then(|| value.clone());
                Ok(())
            }
            (Self::Count(count), AggregateFunc::Count, [value]) => {
                ensure_state_len(aggregate, 1)?;
                *count = scalar_to_i64_count(value)?;
                Ok(())
            }
            (Self::Avg { sum, count }, AggregateFunc::Avg, [count_value, sum_value]) => {
                ensure_state_len(aggregate, 2)?;
                *count = scalar_to_u64_count(count_value)?;
                *sum = (!sum_value.is_null()).then(|| sum_value.clone());
                Ok(())
            }
            (Self::Min(min), AggregateFunc::Min, [value])
            | (Self::Max(min), AggregateFunc::Max, [value]) => {
                ensure_state_len(aggregate, 1)?;
                *min = (!value.is_null()).then(|| value.clone());
                Ok(())
            }
            (_, other, _) => Err(JitError::Backend(format!(
                "aggregate state does not match function {}",
                other.name()
            ))),
        }
    }
}

impl GroupAggregateDenseState {
    pub fn fields_mut(&mut self) -> &mut [GroupAggregateStateField] {
        &mut self.fields
    }

    pub fn group_count(&self) -> usize {
        self.group_count
    }

    fn push_empty(&mut self, aggregates: &[GroupAggregate]) {
        for (field, ty) in self.fields.iter_mut().zip(
            aggregates
                .iter()
                .flat_map(|aggregate| &aggregate.state_types),
        ) {
            debug_assert_eq!(field.ty(), *ty);
            field.push_null();
        }
        self.group_count += 1;
    }
}

impl GroupAggregateStateField {
    fn with_len(ty: JitType, len: usize) -> JitResult<Self> {
        match ty {
            JitType::Int64 => Ok(Self::Int64 {
                values: vec![0; len],
                valid: vec![0; len],
            }),
            JitType::UInt64 => Ok(Self::UInt64 {
                values: vec![0; len],
                valid: vec![0; len],
            }),
            JitType::Float64 => Ok(Self::Float64 {
                values: vec![0.0; len],
                valid: vec![0; len],
            }),
            JitType::Decimal128 { precision, scale } => Ok(Self::Decimal128 {
                values: vec![0; len],
                valid: vec![0; len],
                precision,
                scale,
            }),
            other => Err(JitError::UnsupportedType(format!(
                "MLIR dense group state does not support {other:?}"
            ))),
        }
    }

    pub fn ty(&self) -> JitType {
        match self {
            Self::Int64 { .. } => JitType::Int64,
            Self::UInt64 { .. } => JitType::UInt64,
            Self::Float64 { .. } => JitType::Float64,
            Self::Decimal128 {
                precision, scale, ..
            } => JitType::Decimal128 {
                precision: *precision,
                scale: *scale,
            },
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::Int64 { values, .. } => values.len(),
            Self::UInt64 { values, .. } => values.len(),
            Self::Float64 { values, .. } => values.len(),
            Self::Decimal128 { values, .. } => values.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn values_ptr(&mut self) -> *mut () {
        match self {
            Self::Int64 { values, .. } => values.as_mut_ptr().cast(),
            Self::UInt64 { values, .. } => values.as_mut_ptr().cast(),
            Self::Float64 { values, .. } => values.as_mut_ptr().cast(),
            Self::Decimal128 { values, .. } => values.as_mut_ptr().cast(),
        }
    }

    pub fn valid_ptr(&mut self) -> *mut u8 {
        match self {
            Self::Int64 { valid, .. }
            | Self::UInt64 { valid, .. }
            | Self::Float64 { valid, .. }
            | Self::Decimal128 { valid, .. } => valid.as_mut_ptr(),
        }
    }

    fn push_null(&mut self) {
        match self {
            Self::Int64 { values, valid } => {
                values.push(0);
                valid.push(0);
            }
            Self::UInt64 { values, valid } => {
                values.push(0);
                valid.push(0);
            }
            Self::Float64 { values, valid } => {
                values.push(0.0);
                valid.push(0);
            }
            Self::Decimal128 { values, valid, .. } => {
                values.push(0);
                valid.push(0);
            }
        }
    }

    fn set_scalar(&mut self, index: usize, value: Scalar) -> JitResult<()> {
        match (self, value) {
            (Self::Int64 { values, valid }, Scalar::Int64(value)) => {
                set_option(values, valid, index, value)
            }
            (Self::UInt64 { values, valid }, Scalar::UInt64(value)) => {
                set_option(values, valid, index, value)
            }
            (Self::Float64 { values, valid }, Scalar::Float64(value)) => {
                set_option(values, valid, index, value)
            }
            (
                Self::Decimal128 {
                    values,
                    valid,
                    scale,
                    ..
                },
                Scalar::Decimal128 {
                    value,
                    scale: value_scale,
                    ..
                },
            ) if *scale == value_scale => set_option(values, valid, index, value),
            (field, value) => Err(JitError::Backend(format!(
                "cannot store {:?} in dense {:?} state",
                value.ty(),
                field.ty()
            ))),
        }
    }

    fn scalar(&self, index: usize) -> JitResult<Scalar> {
        match self {
            Self::Int64 { values, valid } => Ok(Scalar::Int64(get_option(values, valid, index)?)),
            Self::UInt64 { values, valid } => Ok(Scalar::UInt64(get_option(values, valid, index)?)),
            Self::Float64 { values, valid } => {
                Ok(Scalar::Float64(get_option(values, valid, index)?))
            }
            Self::Decimal128 {
                values,
                valid,
                precision,
                scale,
            } => Ok(Scalar::Decimal128 {
                value: get_option(values, valid, index)?,
                precision: *precision,
                scale: *scale,
            }),
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
            Scalar::UInt64(value) => Ok(Self::UInt64(value)),
            Scalar::Utf8(value) => Ok(Self::Utf8(value)),
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
            Self::UInt64(value) => Scalar::UInt64(value),
            Self::Utf8(value) => Scalar::Utf8(value),
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

impl FastKeyValue {
    fn try_from_scalar(value: Scalar) -> JitResult<Self> {
        match value {
            Scalar::Bool(value) => Ok(Self::Bool(value)),
            Scalar::Date32(value) => Ok(Self::Date32(value)),
            Scalar::Int32(value) => Ok(Self::Int32(value)),
            Scalar::Int64(value) => Ok(Self::Int64(value)),
            Scalar::UInt64(value) => Ok(Self::UInt64(value)),
            Scalar::Utf8(_) => Err(JitError::Backend(
                "Utf8 fast keys must be interned from Arrow values".to_string(),
            )),
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
}

fn ensure_group_key_type(ty: JitType) -> JitResult<()> {
    match ty {
        JitType::Bool
        | JitType::Date32
        | JitType::Int32
        | JitType::Int64
        | JitType::UInt64
        | JitType::Utf8
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
        AggregateFunc::Avg => match aggregate.expr.ty() {
            JitType::Int32 | JitType::Int64 | JitType::Float64 | JitType::Decimal128 { .. } => {
                Ok(())
            }
            other => Err(JitError::UnsupportedType(format!(
                "AVG does not support {other:?}"
            ))),
        },
        AggregateFunc::Min | AggregateFunc::Max => match aggregate.expr.ty() {
            JitType::Bool
            | JitType::Date32
            | JitType::Int32
            | JitType::Int64
            | JitType::UInt64
            | JitType::Float64
            | JitType::Utf8
            | JitType::Decimal128 { .. } => Ok(()),
        },
    }
}

fn update_minmax(target: &mut Option<Scalar>, value: Scalar, ordering: Ordering) -> JitResult<()> {
    if value.is_null() {
        return Ok(());
    }
    let Some(current) = target.as_ref() else {
        *target = Some(value);
        return Ok(());
    };
    if current.clone().partial_cmp_value(value.clone())? == Some(ordering) {
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
        JitType::UInt64 => Scalar::UInt64(None),
        JitType::Float64 => Scalar::Float64(None),
        JitType::Utf8 => Scalar::Utf8(None),
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
        (Scalar::Int64(value), JitType::UInt64) => {
            let value = value.map(u64::try_from).transpose().map_err(|_| {
                JitError::Backend("negative count cannot coerce to UInt64".to_string())
            })?;
            Ok(Scalar::UInt64(value))
        }
        (Scalar::UInt64(value), JitType::Int64) => {
            let value = value
                .map(i64::try_from)
                .transpose()
                .map_err(|_| JitError::Backend("UInt64 count does not fit in Int64".to_string()))?;
            Ok(Scalar::Int64(value))
        }
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

fn set_option<T: Copy>(
    values: &mut [T],
    valid: &mut [u8],
    index: usize,
    value: Option<T>,
) -> JitResult<()> {
    let value_slot = values
        .get_mut(index)
        .ok_or_else(|| JitError::Backend(format!("dense state index {index} out of bounds")))?;
    let valid_slot = valid
        .get_mut(index)
        .ok_or_else(|| JitError::Backend(format!("dense validity index {index} out of bounds")))?;
    if let Some(value) = value {
        *value_slot = value;
        *valid_slot = 1;
    } else {
        *valid_slot = 0;
    }
    Ok(())
}

fn get_option<T: Copy>(values: &[T], valid: &[u8], index: usize) -> JitResult<Option<T>> {
    let value = values
        .get(index)
        .ok_or_else(|| JitError::Backend(format!("dense state index {index} out of bounds")))?;
    let valid = valid
        .get(index)
        .ok_or_else(|| JitError::Backend(format!("dense validity index {index} out of bounds")))?;
    Ok((*valid != 0).then_some(*value))
}

fn scalar_to_i64_count(value: &Scalar) -> JitResult<i64> {
    match value {
        Scalar::Int64(Some(value)) => Ok(*value),
        Scalar::UInt64(Some(value)) => i64::try_from(*value)
            .map_err(|_| JitError::Backend("UInt64 count does not fit in Int64".to_string())),
        Scalar::Int64(None) | Scalar::UInt64(None) => Ok(0),
        other => Err(JitError::Backend(format!(
            "count state cannot use {:?}",
            other.ty()
        ))),
    }
}

fn scalar_to_u64_count(value: &Scalar) -> JitResult<u64> {
    match value {
        Scalar::UInt64(Some(value)) => Ok(*value),
        Scalar::Int64(Some(value)) => u64::try_from(*value)
            .map_err(|_| JitError::Backend("negative count cannot coerce to UInt64".to_string())),
        Scalar::Int64(None) | Scalar::UInt64(None) => Ok(0),
        other => Err(JitError::Backend(format!(
            "AVG count state cannot use {:?}",
            other.ty()
        ))),
    }
}

fn ensure_state_len(aggregate: &GroupAggregate, expected: usize) -> JitResult<()> {
    if aggregate.state_types.len() == expected {
        return Ok(());
    }
    Err(JitError::Backend(format!(
        "aggregate {} expects {} state fields, got {}",
        aggregate.alias,
        expected,
        aggregate.state_types.len()
    )))
}
