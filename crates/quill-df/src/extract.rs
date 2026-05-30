use std::sync::Arc;

use datafusion::arrow::datatypes::{
    DataType as ArrowDataType, Schema as ArrowSchema, SchemaRef as ArrowSchemaRef,
};
use datafusion::physical_expr::Partitioning;
use datafusion::physical_plan::aggregates::{AggregateExec, AggregateMode};
#[allow(deprecated)]
use datafusion::physical_plan::coalesce_batches::CoalesceBatchesExec;
use datafusion::physical_plan::coalesce_partitions::CoalescePartitionsExec;
use datafusion::physical_plan::filter::FilterExec;
use datafusion::physical_plan::projection::ProjectionExec;
use datafusion::physical_plan::repartition::RepartitionExec;
use datafusion::physical_plan::ExecutionPlan;
use serde::Serialize;

use quill_plan::{
    AggregateFunc, GroupAggregate, GroupAggregateOutputMode, JitExpr, JitProjection, JitResult,
    JitScalar, PipelineGraph, PipelineKind, PipelineStage,
};

use crate::{CompiledGlobalGroupAggregateExec, CompiledPipelineExec, PipelineRuntime};

#[derive(Debug, Clone)]
pub(crate) struct PipelineMatch {
    pub node: &'static str,
    pub graph: PipelineGraph,
    pub compiled: bool,
    pub backend: Option<String>,
    pub reason: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PipelineCandidate {
    pub node: &'static str,
    pub kind: PipelineKind,
    pub source: &'static str,
    pub stages: Vec<&'static str>,
    pub sink: &'static str,
    pub output_mode: Option<&'static str>,
    pub compiled: bool,
    pub backend: Option<String>,
    pub reason: &'static str,
}

#[derive(Debug, Clone)]
pub(crate) struct PhysicalPipeline {
    pub input: Arc<dyn ExecutionPlan>,
    pub output_schema: ArrowSchemaRef,
    pub graph: PipelineGraph,
    pub output_adapter: Option<OutputAdapter>,
    pub execution: PipelineExecution,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PipelineExecution {
    PartitionLocal,
    Global,
}

#[derive(Debug, Clone)]
pub(crate) struct OutputAdapter {
    pub partitioning: Partitioning,
    pub preserve_order: bool,
}

impl PipelineMatch {
    pub fn candidate(&self) -> Option<PipelineCandidate> {
        Some(PipelineCandidate {
            node: self.node,
            kind: self.graph.kind(),
            source: self.graph.source.name(),
            stages: self.graph.stage_names(),
            sink: self.graph.sink_name(),
            output_mode: group_output_mode(&self.graph),
            compiled: self.compiled,
            backend: self.backend.clone(),
            reason: self.reason,
        })
    }
}

impl PhysicalPipeline {
    pub fn filter_sum(
        input: Arc<dyn ExecutionPlan>,
        output_schema: ArrowSchemaRef,
        predicate: JitExpr,
        measure: JitExpr,
    ) -> Self {
        Self {
            input,
            output_schema,
            graph: PipelineGraph::filter_sum(predicate, measure),
            output_adapter: None,
            execution: PipelineExecution::PartitionLocal,
        }
    }

    pub fn filter_project(
        input: Arc<dyn ExecutionPlan>,
        output_schema: ArrowSchemaRef,
        predicate: JitExpr,
        projections: Vec<JitProjection>,
        output_adapter: Option<OutputAdapter>,
    ) -> Self {
        Self {
            input,
            output_schema,
            graph: PipelineGraph::record(vec![
                PipelineStage::Filter(predicate),
                PipelineStage::Projection(projections),
            ]),
            output_adapter,
            execution: PipelineExecution::PartitionLocal,
        }
    }

    pub fn group_aggregate(
        input: Arc<dyn ExecutionPlan>,
        output_schema: ArrowSchemaRef,
        graph: PipelineGraph,
    ) -> Self {
        Self {
            input,
            output_schema,
            graph,
            output_adapter: None,
            execution: PipelineExecution::PartitionLocal,
        }
    }

    pub fn global_group_aggregate(
        input: Arc<dyn ExecutionPlan>,
        output_schema: ArrowSchemaRef,
        graph: PipelineGraph,
    ) -> Self {
        Self {
            input,
            output_schema,
            graph,
            output_adapter: None,
            execution: PipelineExecution::Global,
        }
    }
}

pub(crate) fn extract_pipeline_from_node(
    plan: &Arc<dyn ExecutionPlan>,
) -> Option<PhysicalPipeline> {
    if let Some(aggregate) = plan.as_any().downcast_ref::<AggregateExec>() {
        return extract_global_group_aggregate_pipeline(aggregate)
            .or_else(|| extract_filter_sum_pipeline(aggregate))
            .or_else(|| extract_group_aggregate_pipeline(aggregate));
    }

    let projection = plan.as_any().downcast_ref::<ProjectionExec>()?;
    extract_filter_project_pipeline(projection)
}

pub(crate) fn pipeline_from_node(plan: &Arc<dyn ExecutionPlan>) -> Option<PipelineMatch> {
    if let Some(compiled) = plan
        .as_any()
        .downcast_ref::<CompiledGlobalGroupAggregateExec>()
    {
        let graph = PipelineGraph::group_aggregate_final(
            compiled
                .runtime()
                .predicate()
                .cloned()
                .map(PipelineStage::Filter)
                .into_iter()
                .collect(),
            compiled.runtime().keys().to_vec(),
            compiled.runtime().aggregates().to_vec(),
        );
        return Some(PipelineMatch {
            node: "CompiledGlobalGroupAggregateExec",
            graph,
            compiled: true,
            backend: Some(compiled.kernel().backend.clone()),
            reason: "compiled",
        });
    }

    if let Some(compiled) = plan.as_any().downcast_ref::<CompiledPipelineExec>() {
        let graph = match compiled.runtime() {
            PipelineRuntime::RecordBatch(runtime) => PipelineGraph::record(vec![
                PipelineStage::Filter(runtime.predicate().clone()),
                PipelineStage::Projection(runtime.projections().to_vec()),
            ]),
            PipelineRuntime::ScalarSum(runtime) => {
                PipelineGraph::filter_sum(runtime.predicate().clone(), runtime.measure().clone())
            }
            PipelineRuntime::GroupAggregate(runtime) => PipelineGraph::group_aggregate(
                runtime
                    .predicate()
                    .cloned()
                    .map(PipelineStage::Filter)
                    .into_iter()
                    .collect(),
                runtime.keys().to_vec(),
                runtime.aggregates().to_vec(),
            ),
        };
        return Some(PipelineMatch {
            node: "CompiledPipelineExec",
            graph,
            compiled: true,
            backend: Some(compiled.kernel().backend.clone()),
            reason: "compiled",
        });
    }

    let aggregate = plan.as_any().downcast_ref::<AggregateExec>()?;
    if let Some(pipeline) = extract_filter_sum_pipeline(aggregate) {
        return Some(PipelineMatch {
            node: "AggregateExec",
            graph: pipeline.graph,
            compiled: false,
            backend: None,
            reason: "candidate",
        });
    }

    extract_group_aggregate_pipeline(aggregate).map(|pipeline| PipelineMatch {
        node: "AggregateExec",
        graph: pipeline.graph,
        compiled: false,
        backend: None,
        reason: "candidate",
    })
}

fn group_output_mode(graph: &PipelineGraph) -> Option<&'static str> {
    match &graph.sink {
        quill_plan::PipelineSink::GroupAggregate { output, .. } => Some(output.name()),
        quill_plan::PipelineSink::RecordBatch | quill_plan::PipelineSink::Sum { .. } => None,
    }
}

fn extract_filter_project_pipeline(projection: &ProjectionExec) -> Option<PhysicalPipeline> {
    let input = projection.input();
    if let Some(filter) = input.as_any().downcast_ref::<FilterExec>() {
        return filter_project_pipeline_from_filter(filter, projection, None);
    }

    let repartition = input.as_any().downcast_ref::<RepartitionExec>()?;
    if !matches!(repartition.partitioning(), Partitioning::RoundRobinBatch(_)) {
        return None;
    }
    let filter = repartition.input().as_any().downcast_ref::<FilterExec>()?;
    filter_project_pipeline_from_filter(
        filter,
        projection,
        Some(OutputAdapter {
            partitioning: repartition.partitioning().clone(),
            preserve_order: repartition.preserve_order(),
        }),
    )
}

fn filter_project_pipeline_from_filter(
    filter: &FilterExec,
    projection: &ProjectionExec,
    output_adapter: Option<OutputAdapter>,
) -> Option<PhysicalPipeline> {
    if filter.projection().is_some() || filter.fetch().is_some() {
        return None;
    }

    let input_schema = filter.input().schema();
    let predicate = crate::expr::from_physical(filter.predicate(), input_schema.as_ref()).ok()?;
    let projections = lower_projection_exprs(projection, input_schema.as_ref())?;
    Some(PhysicalPipeline::filter_project(
        Arc::clone(filter.input()),
        projection.schema(),
        predicate,
        projections,
        output_adapter,
    ))
}

pub(crate) fn extract_filter_sum_pipeline(aggregate: &AggregateExec) -> Option<PhysicalPipeline> {
    if *aggregate.mode() != AggregateMode::Partial
        || !aggregate.group_expr().is_true_no_grouping()
        || aggregate.aggr_expr().len() != 1
        || aggregate.filter_expr().iter().any(Option::is_some)
        || aggregate.schema().fields().len() != 1
        || !is_supported_sum_output(aggregate.schema().field(0).data_type())
    {
        return None;
    }

    let input = strip_pipeline_adapters(aggregate.input());
    let filter = input.as_any().downcast_ref::<FilterExec>()?;
    if filter.fetch().is_some() {
        return None;
    }

    let predicate =
        crate::expr::from_physical(filter.predicate(), filter.input().schema().as_ref()).ok()?;
    let measure = lower_sum_measure(aggregate, aggregate.input().schema().as_ref())?;
    let measure = remap_projection_columns(
        &measure,
        filter.projection().as_ref().map(AsRef::as_ref),
        filter.input().schema().as_ref(),
    )?;

    Some(PhysicalPipeline::filter_sum(
        Arc::clone(filter.input()),
        aggregate.schema(),
        predicate,
        measure,
    ))
}

fn extract_group_aggregate_pipeline(aggregate: &AggregateExec) -> Option<PhysicalPipeline> {
    if *aggregate.mode() != AggregateMode::Partial
        || aggregate.group_expr().is_true_no_grouping()
        || !aggregate.group_expr().is_single()
        || aggregate.group_expr().groups().len() != 1
        || aggregate
            .group_expr()
            .groups()
            .first()
            .is_some_and(|group| group.iter().any(|is_null| *is_null))
        || aggregate.aggr_expr().is_empty()
        || aggregate.filter_expr().iter().any(Option::is_some)
    {
        return None;
    }

    let input = strip_pipeline_adapters(aggregate.input());
    let (input, stages, projection) =
        if let Some(filter) = input.as_any().downcast_ref::<FilterExec>() {
            if filter.fetch().is_some() {
                return None;
            }
            let predicate =
                crate::expr::from_physical(filter.predicate(), filter.input().schema().as_ref())
                    .ok()?;
            (
                filter.input(),
                vec![PipelineStage::Filter(predicate)],
                filter.projection().as_ref().map(AsRef::as_ref),
            )
        } else {
            (input, Vec::new(), None)
        };

    let aggregate_input_schema = aggregate.input().schema();
    let input_schema = input.schema();
    let keys = lower_group_keys(
        aggregate,
        aggregate_input_schema.as_ref(),
        projection,
        input_schema.as_ref(),
    )?;
    let aggregates = lower_group_aggregates(
        aggregate,
        aggregate_input_schema.as_ref(),
        projection,
        input_schema.as_ref(),
    )?;

    let graph = PipelineGraph::group_aggregate(stages, keys, aggregates);
    Some(PhysicalPipeline::group_aggregate(
        Arc::clone(input),
        aggregate.schema(),
        graph,
    ))
}

fn extract_global_group_aggregate_pipeline(aggregate: &AggregateExec) -> Option<PhysicalPipeline> {
    match aggregate.mode() {
        AggregateMode::Single | AggregateMode::SinglePartitioned => {
            extract_single_global_group_aggregate_pipeline(aggregate)
        }
        AggregateMode::Final | AggregateMode::FinalPartitioned => {
            extract_final_global_group_aggregate_pipeline(aggregate)
        }
        AggregateMode::Partial | AggregateMode::PartialReduce => None,
    }
}

fn extract_single_global_group_aggregate_pipeline(
    aggregate: &AggregateExec,
) -> Option<PhysicalPipeline> {
    if aggregate.group_expr().is_true_no_grouping()
        || !aggregate.group_expr().is_single()
        || aggregate.group_expr().groups().len() != 1
        || aggregate
            .group_expr()
            .groups()
            .first()
            .is_some_and(|group| group.iter().any(|is_null| *is_null))
        || aggregate.aggr_expr().is_empty()
        || aggregate.filter_expr().iter().any(Option::is_some)
    {
        return None;
    }

    let input = strip_pipeline_adapters(aggregate.input());
    let (input, stages, projection) =
        if let Some(filter) = input.as_any().downcast_ref::<FilterExec>() {
            if filter.fetch().is_some() {
                return None;
            }
            let predicate =
                crate::expr::from_physical(filter.predicate(), filter.input().schema().as_ref())
                    .ok()?;
            (
                filter.input(),
                vec![PipelineStage::Filter(predicate)],
                filter.projection().as_ref().map(AsRef::as_ref),
            )
        } else {
            (input, Vec::new(), None)
        };

    let aggregate_input_schema = aggregate.input().schema();
    let input_schema = input.schema();
    let keys = lower_group_keys(
        aggregate,
        aggregate_input_schema.as_ref(),
        projection,
        input_schema.as_ref(),
    )?;
    let aggregates = lower_group_aggregates_with_output(
        aggregate,
        aggregate_input_schema.as_ref(),
        projection,
        input_schema.as_ref(),
        GroupAggregateOutputMode::FinalValues,
    )?;

    let graph = PipelineGraph::group_aggregate_final(stages, keys, aggregates);
    Some(PhysicalPipeline::global_group_aggregate(
        Arc::clone(input),
        aggregate.schema(),
        graph,
    ))
}

fn extract_final_global_group_aggregate_pipeline(
    aggregate: &AggregateExec,
) -> Option<PhysicalPipeline> {
    if aggregate.group_expr().is_true_no_grouping()
        || aggregate.aggr_expr().is_empty()
        || aggregate.filter_expr().iter().any(Option::is_some)
    {
        return None;
    }
    if aggregate.aggr_expr().iter().any(|expr| {
        expr.is_distinct()
            || !expr.order_bys().is_empty()
            || aggregate_func(expr.fun().name()).is_none()
    }) {
        return None;
    }

    let compiled = strip_final_group_adapters(aggregate.input())?;
    let PipelineRuntime::GroupAggregate(runtime) = compiled.runtime() else {
        return None;
    };
    if runtime.output_mode() != GroupAggregateOutputMode::PartialState
        || aggregate.aggr_expr().len() != runtime.aggregates().len()
    {
        return None;
    }

    let mut aggregates = runtime.aggregates().to_vec();
    for (index, group_aggregate) in aggregates.iter_mut().enumerate() {
        group_aggregate.output_type =
            aggregate_output_type(&aggregate.schema(), runtime.keys().len(), index)?;
    }

    let stages = runtime
        .predicate()
        .cloned()
        .map(PipelineStage::Filter)
        .into_iter()
        .collect::<Vec<_>>();
    let graph = PipelineGraph::group_aggregate_final(stages, runtime.keys().to_vec(), aggregates);
    Some(PhysicalPipeline::global_group_aggregate(
        Arc::clone(compiled.input()),
        aggregate.schema(),
        graph,
    ))
}

fn lower_projection_exprs(
    projection: &ProjectionExec,
    input_schema: &ArrowSchema,
) -> Option<Vec<JitProjection>> {
    projection
        .expr()
        .iter()
        .map(|expr| {
            crate::expr::from_physical(&expr.expr, input_schema)
                .map(|jit_expr| JitProjection::new(jit_expr, expr.alias.clone()))
        })
        .collect::<JitResult<Vec<_>>>()
        .ok()
}

fn is_supported_sum_output(data_type: &ArrowDataType) -> bool {
    matches!(
        data_type,
        ArrowDataType::Float64 | ArrowDataType::Decimal128(_, _)
    )
}

#[allow(deprecated)]
fn strip_pipeline_adapters(input: &Arc<dyn ExecutionPlan>) -> &Arc<dyn ExecutionPlan> {
    if let Some(coalesce) = input.as_any().downcast_ref::<CoalesceBatchesExec>() {
        return strip_pipeline_adapters(coalesce.input());
    }
    if let Some(repartition) = input.as_any().downcast_ref::<RepartitionExec>() {
        if matches!(repartition.partitioning(), Partitioning::RoundRobinBatch(_)) {
            return strip_pipeline_adapters(repartition.input());
        }
    }
    input
}

#[allow(deprecated)]
fn strip_final_group_adapters(input: &Arc<dyn ExecutionPlan>) -> Option<&CompiledPipelineExec> {
    if let Some(compiled) = input.as_any().downcast_ref::<CompiledPipelineExec>() {
        return Some(compiled);
    }
    if let Some(coalesce) = input.as_any().downcast_ref::<CoalesceBatchesExec>() {
        return strip_final_group_adapters(coalesce.input());
    }
    if let Some(coalesce) = input.as_any().downcast_ref::<CoalescePartitionsExec>() {
        return strip_final_group_adapters(coalesce.input());
    }
    if let Some(repartition) = input.as_any().downcast_ref::<RepartitionExec>() {
        return strip_final_group_adapters(repartition.input());
    }
    None
}

fn lower_group_keys(
    aggregate: &AggregateExec,
    aggregate_input_schema: &ArrowSchema,
    projection: Option<&[usize]>,
    input_schema: &ArrowSchema,
) -> Option<Vec<JitExpr>> {
    aggregate
        .group_expr()
        .expr()
        .iter()
        .map(|(expr, _alias)| {
            let expr = crate::expr::from_physical(expr, aggregate_input_schema).ok()?;
            remap_projection_columns(&expr, projection, input_schema)
        })
        .collect()
}

fn lower_group_aggregates(
    aggregate: &AggregateExec,
    aggregate_input_schema: &ArrowSchema,
    projection: Option<&[usize]>,
    input_schema: &ArrowSchema,
) -> Option<Vec<GroupAggregate>> {
    lower_group_aggregates_with_output(
        aggregate,
        aggregate_input_schema,
        projection,
        input_schema,
        GroupAggregateOutputMode::PartialState,
    )
}

fn lower_group_aggregates_with_output(
    aggregate: &AggregateExec,
    aggregate_input_schema: &ArrowSchema,
    projection: Option<&[usize]>,
    input_schema: &ArrowSchema,
    output_mode: GroupAggregateOutputMode,
) -> Option<Vec<GroupAggregate>> {
    let key_count = aggregate.group_expr().expr().len();
    aggregate
        .aggr_expr()
        .iter()
        .enumerate()
        .map(|expr| {
            let (index, expr) = expr;
            if expr.is_distinct() || !expr.order_bys().is_empty() {
                return None;
            }
            let func = aggregate_func(expr.fun().name())?;
            let arguments = expr.expressions();
            let measure = match func {
                AggregateFunc::Count if arguments.is_empty() => {
                    JitExpr::Literal(JitScalar::Int64(1))
                }
                _ if arguments.len() == 1 => {
                    let lowered =
                        crate::expr::from_physical(&arguments[0], aggregate_input_schema).ok()?;
                    remap_projection_columns(&lowered, projection, input_schema)?
                }
                _ => return None,
            };
            let state_types = expr
                .state_fields()
                .ok()?
                .iter()
                .map(|field| crate::expr::jit_type(field.data_type()).ok())
                .collect::<Option<Vec<_>>>()?;
            let output_type = match output_mode {
                GroupAggregateOutputMode::PartialState => {
                    state_types.first().copied().unwrap_or_else(|| measure.ty())
                }
                GroupAggregateOutputMode::FinalValues => {
                    aggregate_output_type(&aggregate.schema(), key_count, index)?
                }
            };
            Some(GroupAggregate::new_with_output_and_states(
                func,
                measure,
                output_type,
                state_types,
                expr.name().to_string(),
            ))
        })
        .collect()
}

fn aggregate_output_type(
    schema: &ArrowSchemaRef,
    key_count: usize,
    aggregate_index: usize,
) -> Option<quill_plan::JitType> {
    schema
        .fields()
        .get(key_count + aggregate_index)
        .and_then(|field| crate::expr::jit_type(field.data_type()).ok())
}

fn aggregate_func(name: &str) -> Option<AggregateFunc> {
    if name.eq_ignore_ascii_case("sum") {
        Some(AggregateFunc::Sum)
    } else if name.eq_ignore_ascii_case("count") {
        Some(AggregateFunc::Count)
    } else if name.eq_ignore_ascii_case("avg") {
        Some(AggregateFunc::Avg)
    } else if name.eq_ignore_ascii_case("min") {
        Some(AggregateFunc::Min)
    } else if name.eq_ignore_ascii_case("max") {
        Some(AggregateFunc::Max)
    } else {
        None
    }
}

fn lower_sum_measure(aggregate: &AggregateExec, input_schema: &ArrowSchema) -> Option<JitExpr> {
    let aggregate_expr = aggregate.aggr_expr().first()?;
    if !aggregate_expr.fun().name().eq_ignore_ascii_case("sum")
        || aggregate_expr.is_distinct()
        || !aggregate_expr.order_bys().is_empty()
    {
        return None;
    }
    let expressions = aggregate_expr.expressions();
    if expressions.len() != 1 {
        return None;
    }
    crate::expr::from_physical(&expressions[0], input_schema).ok()
}

fn remap_projection_columns(
    expr: &JitExpr,
    projection: Option<&[usize]>,
    input_schema: &ArrowSchema,
) -> Option<JitExpr> {
    match expr {
        JitExpr::Column {
            index,
            name: _,
            ty,
            nullable,
        } => {
            let source_index = match projection {
                Some(projection) => *projection.get(*index)?,
                None => *index,
            };
            let field = input_schema.field(source_index);
            Some(JitExpr::Column {
                index: source_index,
                name: field.name().to_string(),
                ty: *ty,
                nullable: *nullable,
            })
        }
        JitExpr::Literal(value) => Some(JitExpr::Literal(value.clone())),
        JitExpr::Binary {
            op,
            left,
            right,
            ty,
            nullable,
        } => Some(JitExpr::Binary {
            op: *op,
            left: Box::new(remap_projection_columns(left, projection, input_schema)?),
            right: Box::new(remap_projection_columns(right, projection, input_schema)?),
            ty: *ty,
            nullable: *nullable,
        }),
        JitExpr::Cast { expr, ty, nullable } => Some(JitExpr::Cast {
            expr: Box::new(remap_projection_columns(expr, projection, input_schema)?),
            ty: *ty,
            nullable: *nullable,
        }),
        JitExpr::IsNull(arg) => Some(JitExpr::IsNull(Box::new(remap_projection_columns(
            arg,
            projection,
            input_schema,
        )?))),
    }
}
