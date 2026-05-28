use arrow::datatypes::SchemaRef as ArrowSchemaRef;
use serde::Serialize;

use std::collections::BTreeMap;

use quill_plan::{JitExpr, JitProjection, JitResult, JitType};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FixedColumn {
    pub index: usize,
    pub ty: JitType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum KernelKind {
    Filter,
    Projection,
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

    pub fn kind(&self) -> KernelKind {
        match self {
            Self::Generic { kind } => *kind,
            Self::RecordProject { .. } => KernelKind::FilterProject,
            Self::PlainSum { .. } => KernelKind::FilterSum,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Generic { kind } => kind.name(),
            Self::RecordProject { .. } => "record_project",
            Self::PlainSum { .. } => "plain_sum",
        }
    }
}

impl KernelKind {
    pub fn name(self) -> &'static str {
        match self {
            Self::Filter => "filter",
            Self::Projection => "projection",
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
    pub fn new(
        id: impl Into<String>,
        kind: KernelKind,
        backend: impl Into<String>,
        executable: bool,
    ) -> Self {
        Self {
            id: id.into(),
            kind,
            spec: PipelineSpec::generic(kind),
            backend: backend.into(),
            executable,
        }
    }

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

pub trait KernelBackend: Send + Sync {
    fn name(&self) -> &str;

    fn compile_filter(
        &self,
        input_schema: ArrowSchemaRef,
        predicate: &JitExpr,
    ) -> JitResult<CompiledKernel>;

    fn compile_projection(
        &self,
        input_schema: ArrowSchemaRef,
        projections: &[JitProjection],
    ) -> JitResult<CompiledKernel>;

    fn compile_filter_project(
        &self,
        input_schema: ArrowSchemaRef,
        predicate: &JitExpr,
        projections: &[JitProjection],
    ) -> JitResult<CompiledKernel>;
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
        JitExpr::IsNull(_) => None,
    }
}

fn ensure_fixed_width_type(ty: JitType) -> Option<()> {
    match ty {
        JitType::Date32 | JitType::Int64 | JitType::Float64 | JitType::Decimal128 { .. } => {
            Some(())
        }
        JitType::Bool | JitType::Int32 => None,
    }
}

fn ensure_record_output_type(ty: JitType) -> Option<()> {
    match ty {
        JitType::Date32 | JitType::Int64 | JitType::Float64 | JitType::Decimal128 { .. } => {
            Some(())
        }
        JitType::Bool | JitType::Int32 => None,
    }
}

fn is_plain_sum_output(ty: JitType) -> bool {
    matches!(ty, JitType::Float64 | JitType::Decimal128 { .. })
}
