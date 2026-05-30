use std::collections::BTreeMap;

use quill_plan::{AggregateFunc, GroupAggregate, JitExpr, JitProjection, JitType};
use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FixedColumn {
    pub index: usize,
    pub ty: JitType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum KernelKind {
    FilterProject,
    FilterSum,
    GroupAggregate,
}

#[derive(Debug, Clone, PartialEq)]
pub enum PipelineSpec {
    Generic {
        kind: KernelKind,
    },
    RecordProject {
        columns: Vec<FixedColumn>,
        output_types: Vec<JitType>,
    },
    PlainSum {
        columns: Vec<FixedColumn>,
        output_type: JitType,
    },
    GroupAggregate {
        columns: Vec<FixedColumn>,
        key_types: Vec<JitType>,
        aggregate_funcs: Vec<AggregateFunc>,
        state_types: Vec<JitType>,
    },
}

impl PipelineSpec {
    pub fn generic(kind: KernelKind) -> Self {
        Self::Generic { kind }
    }

    pub fn record_project(predicate: &JitExpr, projections: &[JitProjection]) -> Option<Self> {
        if predicate.ty() != JitType::Bool || projections.is_empty() {
            return None;
        }

        let mut columns = BTreeMap::new();
        collect_fixed_width_columns(predicate, &mut columns)?;
        let output_types = projections
            .iter()
            .map(|projection| {
                ensure_record_output_type(projection.expr.ty())?;
                collect_fixed_width_columns(&projection.expr, &mut columns)?;
                Some(projection.expr.ty())
            })
            .collect::<Option<Vec<_>>>()?;

        Some(Self::RecordProject {
            columns: columns
                .into_iter()
                .map(|(index, ty)| FixedColumn { index, ty })
                .collect(),
            output_types,
        })
    }

    pub fn filter_sum(predicate: &JitExpr, measure: &JitExpr) -> Option<Self> {
        if predicate.ty() != JitType::Bool || !is_plain_sum_output(measure.ty()) {
            return None;
        }

        let mut columns = BTreeMap::new();
        collect_fixed_width_columns(predicate, &mut columns)?;
        collect_fixed_width_columns(measure, &mut columns)?;
        Some(Self::PlainSum {
            columns: columns
                .into_iter()
                .map(|(index, ty)| FixedColumn { index, ty })
                .collect(),
            output_type: measure.ty(),
        })
    }

    pub fn group_aggregate(
        predicate: Option<&JitExpr>,
        keys: &[JitExpr],
        aggregates: &[GroupAggregate],
    ) -> Option<Self> {
        if keys.is_empty() || aggregates.is_empty() {
            return None;
        }

        if let Some(predicate) = predicate {
            if predicate.ty() != JitType::Bool {
                return None;
            }
        }

        let mut columns = BTreeMap::new();
        if let Some(predicate) = predicate {
            collect_fixed_width_columns(predicate, &mut columns)?;
        }
        for aggregate in aggregates {
            ensure_group_update_aggregate(aggregate)?;
            collect_fixed_width_columns(&aggregate.expr, &mut columns)?;
        }

        Some(Self::GroupAggregate {
            columns: columns
                .into_iter()
                .map(|(index, ty)| FixedColumn { index, ty })
                .collect(),
            key_types: keys.iter().map(JitExpr::ty).collect(),
            aggregate_funcs: aggregates.iter().map(|aggregate| aggregate.func).collect(),
            state_types: aggregates
                .iter()
                .flat_map(|aggregate| aggregate.state_types.iter().copied())
                .collect(),
        })
    }

    pub fn kind(&self) -> KernelKind {
        match self {
            Self::Generic { kind } => *kind,
            Self::RecordProject { .. } => KernelKind::FilterProject,
            Self::PlainSum { .. } => KernelKind::FilterSum,
            Self::GroupAggregate { .. } => KernelKind::GroupAggregate,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Generic { kind } => kind.name(),
            Self::RecordProject { .. } => "record_project",
            Self::PlainSum { .. } => "plain_sum",
            Self::GroupAggregate { .. } => "group_aggregate",
        }
    }
}

impl KernelKind {
    pub fn name(self) -> &'static str {
        match self {
            Self::FilterProject => "filter_project",
            Self::FilterSum => "filter_sum",
            Self::GroupAggregate => "group_aggregate",
        }
    }
}

#[derive(Debug, Clone)]
pub struct CompiledKernel {
    pub id: String,
    pub kind: KernelKind,
    pub spec: PipelineSpec,
    pub backend: String,
    pub executable: bool,
}

impl CompiledKernel {
    pub fn with_spec(
        id: impl Into<String>,
        spec: PipelineSpec,
        backend: impl Into<String>,
        executable: bool,
    ) -> Self {
        let kind = spec.kind();
        Self {
            id: id.into(),
            kind,
            spec,
            backend: backend.into(),
            executable,
        }
    }
}

fn collect_fixed_width_columns(
    expr: &JitExpr,
    columns: &mut BTreeMap<usize, JitType>,
) -> Option<()> {
    match expr {
        JitExpr::Column { index, ty, .. } => {
            ensure_fixed_width_type(*ty)?;
            match columns.get(index) {
                Some(existing) if *existing != *ty => return None,
                Some(_) => {}
                None => {
                    columns.insert(*index, *ty);
                }
            }
            Some(())
        }
        JitExpr::Literal(_) => Some(()),
        JitExpr::Binary { left, right, .. } => {
            collect_fixed_width_columns(left, columns)?;
            collect_fixed_width_columns(right, columns)
        }
        JitExpr::Cast { expr, ty, .. } => {
            ensure_fixed_width_type(*ty)?;
            collect_fixed_width_columns(expr, columns)
        }
        JitExpr::IsNull(_) => None,
    }
}

fn ensure_fixed_width_type(ty: JitType) -> Option<()> {
    match ty {
        JitType::Date32 | JitType::Int64 | JitType::Float64 | JitType::Decimal128 { .. } => {
            Some(())
        }
        JitType::Bool | JitType::Int32 | JitType::UInt64 | JitType::Utf8 => None,
    }
}

fn ensure_record_output_type(ty: JitType) -> Option<()> {
    match ty {
        JitType::Date32 | JitType::Int64 | JitType::Float64 | JitType::Decimal128 { .. } => {
            Some(())
        }
        JitType::Bool | JitType::Int32 | JitType::UInt64 | JitType::Utf8 => None,
    }
}

fn ensure_group_update_aggregate(aggregate: &GroupAggregate) -> Option<()> {
    match aggregate.func {
        AggregateFunc::Count => {
            let [count_ty] = aggregate.state_types.as_slice() else {
                return None;
            };
            ensure_group_update_count_type(*count_ty)
        }
        AggregateFunc::Avg => {
            let [count_ty, sum_ty] = aggregate.state_types.as_slice() else {
                return None;
            };
            ensure_group_update_count_type(*count_ty)?;
            ensure_group_update_state_type(*sum_ty)?;
            ensure_group_update_measure_type(aggregate.expr.ty())
        }
        AggregateFunc::Sum | AggregateFunc::Min | AggregateFunc::Max => {
            let [state_ty] = aggregate.state_types.as_slice() else {
                return None;
            };
            ensure_group_update_state_type(*state_ty)?;
            ensure_group_update_measure_type(aggregate.expr.ty())
        }
    }
}

fn ensure_group_update_count_type(ty: JitType) -> Option<()> {
    match ty {
        JitType::Int64 | JitType::UInt64 => Some(()),
        JitType::Bool
        | JitType::Date32
        | JitType::Int32
        | JitType::Float64
        | JitType::Utf8
        | JitType::Decimal128 { .. } => None,
    }
}

fn ensure_group_update_state_type(ty: JitType) -> Option<()> {
    match ty {
        JitType::Int64 | JitType::UInt64 | JitType::Float64 | JitType::Decimal128 { .. } => {
            Some(())
        }
        JitType::Bool | JitType::Date32 | JitType::Int32 | JitType::Utf8 => None,
    }
}

fn ensure_group_update_measure_type(ty: JitType) -> Option<()> {
    match ty {
        JitType::Int64 | JitType::Float64 | JitType::Decimal128 { .. } => Some(()),
        JitType::Bool | JitType::Date32 | JitType::Int32 | JitType::UInt64 | JitType::Utf8 => None,
    }
}

fn is_plain_sum_output(ty: JitType) -> bool {
    matches!(ty, JitType::Float64 | JitType::Decimal128 { .. })
}

#[cfg(test)]
mod tests {
    use super::*;
    use quill_plan::{JitBinaryOp, JitScalar};

    #[test]
    fn records_fixed_width_filter_project_spec() {
        let predicate = JitExpr::Binary {
            op: JitBinaryOp::Gt,
            left: Box::new(column(1, JitType::Int64)),
            right: Box::new(JitExpr::Literal(JitScalar::Int64(10))),
            ty: JitType::Bool,
            nullable: false,
        };
        let projection = JitProjection::new(column(0, JitType::Int64), "id");

        let spec = PipelineSpec::record_project(&predicate, &[projection]).unwrap();

        assert_eq!(spec.name(), "record_project");
        assert_eq!(spec.kind(), KernelKind::FilterProject);
    }

    #[test]
    fn rejects_variable_width_record_output_spec() {
        let predicate = JitExpr::Literal(JitScalar::Bool(true));
        let projection = JitProjection::new(column(0, JitType::Utf8), "name");

        assert!(PipelineSpec::record_project(&predicate, &[projection]).is_none());
    }

    #[test]
    fn records_decimal_plain_sum_spec() {
        let predicate = JitExpr::Literal(JitScalar::Bool(true));
        let measure = column(
            2,
            JitType::Decimal128 {
                precision: 30,
                scale: 4,
            },
        );

        let spec = PipelineSpec::filter_sum(&predicate, &measure).unwrap();

        assert_eq!(spec.name(), "plain_sum");
        assert_eq!(spec.kind(), KernelKind::FilterSum);
    }

    #[test]
    fn records_group_update_spec_with_utf8_key() {
        let key = column(0, JitType::Utf8);
        let aggregate = GroupAggregate::new(
            AggregateFunc::Sum,
            column(1, JitType::Int64),
            JitType::Int64,
            "sum_v",
        );

        let spec = PipelineSpec::group_aggregate(None, &[key], &[aggregate]).unwrap();

        assert_eq!(spec.name(), "group_aggregate");
        assert_eq!(spec.kind(), KernelKind::GroupAggregate);
    }

    fn column(index: usize, ty: JitType) -> JitExpr {
        JitExpr::Column {
            index,
            name: format!("c{index}"),
            ty,
            nullable: false,
        }
    }
}
