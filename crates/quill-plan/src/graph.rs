use crate::{JitExpr, JitProjection, JitType};
use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineSource {
    ArrowBatch,
    ArrowStream,
}

impl PipelineSource {
    pub fn name(&self) -> &'static str {
        match self {
            Self::ArrowBatch => "arrow_batch",
            Self::ArrowStream => "arrow_stream",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperatorKind {
    Source,
    Filter,
    Project,
    Limit,
    PlainAggregate,
    GroupAggregate,
    HashJoin,
    TopK,
    Sort,
    Exchange,
    RecordBatchSink,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputMode {
    RecordBatch,
    Selection,
    Scalar,
    Grouped,
    Stream,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OperatorProperties {
    pub preserves_order: bool,
    pub changes_cardinality: bool,
    pub pipeline_breaker: bool,
    pub requires_state: bool,
    pub output_mode: OutputMode,
}

impl OperatorKind {
    pub fn name(self) -> &'static str {
        match self {
            Self::Source => "source",
            Self::Filter => "filter",
            Self::Project => "project",
            Self::Limit => "limit",
            Self::PlainAggregate => "plain_aggregate",
            Self::GroupAggregate => "group_aggregate",
            Self::HashJoin => "hash_join",
            Self::TopK => "topk",
            Self::Sort => "sort",
            Self::Exchange => "exchange",
            Self::RecordBatchSink => "record_batch",
        }
    }

    pub fn properties(self) -> OperatorProperties {
        match self {
            Self::Source => OperatorProperties {
                preserves_order: true,
                changes_cardinality: false,
                pipeline_breaker: false,
                requires_state: false,
                output_mode: OutputMode::Stream,
            },
            Self::Filter => OperatorProperties {
                preserves_order: true,
                changes_cardinality: true,
                pipeline_breaker: false,
                requires_state: false,
                output_mode: OutputMode::Selection,
            },
            Self::Project => OperatorProperties {
                preserves_order: true,
                changes_cardinality: false,
                pipeline_breaker: false,
                requires_state: false,
                output_mode: OutputMode::RecordBatch,
            },
            Self::Limit => OperatorProperties {
                preserves_order: true,
                changes_cardinality: true,
                pipeline_breaker: false,
                requires_state: true,
                output_mode: OutputMode::RecordBatch,
            },
            Self::PlainAggregate => OperatorProperties {
                preserves_order: false,
                changes_cardinality: true,
                pipeline_breaker: true,
                requires_state: true,
                output_mode: OutputMode::Scalar,
            },
            Self::GroupAggregate => OperatorProperties {
                preserves_order: false,
                changes_cardinality: true,
                pipeline_breaker: true,
                requires_state: true,
                output_mode: OutputMode::Grouped,
            },
            Self::HashJoin => OperatorProperties {
                preserves_order: false,
                changes_cardinality: true,
                pipeline_breaker: true,
                requires_state: true,
                output_mode: OutputMode::RecordBatch,
            },
            Self::TopK => OperatorProperties {
                preserves_order: true,
                changes_cardinality: true,
                pipeline_breaker: true,
                requires_state: true,
                output_mode: OutputMode::RecordBatch,
            },
            Self::Sort => OperatorProperties {
                preserves_order: false,
                changes_cardinality: false,
                pipeline_breaker: true,
                requires_state: true,
                output_mode: OutputMode::RecordBatch,
            },
            Self::Exchange => OperatorProperties {
                preserves_order: false,
                changes_cardinality: false,
                pipeline_breaker: true,
                requires_state: true,
                output_mode: OutputMode::Stream,
            },
            Self::RecordBatchSink => OperatorProperties {
                preserves_order: true,
                changes_cardinality: false,
                pipeline_breaker: false,
                requires_state: false,
                output_mode: OutputMode::RecordBatch,
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum PipelineStage {
    Filter(JitExpr),
    Projection(Vec<JitProjection>),
    Limit(usize),
}

impl PipelineStage {
    pub fn operator_kind(&self) -> OperatorKind {
        match self {
            Self::Filter(_) => OperatorKind::Filter,
            Self::Projection(_) => OperatorKind::Project,
            Self::Limit(_) => OperatorKind::Limit,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum PipelineSink {
    RecordBatch,
    Sum {
        measure: JitExpr,
    },
    GroupAggregate {
        keys: Vec<JitExpr>,
        aggregates: Vec<GroupAggregate>,
    },
}

impl PipelineSink {
    pub fn operator_kind(&self) -> OperatorKind {
        match self {
            Self::RecordBatch => OperatorKind::RecordBatchSink,
            Self::Sum { .. } => OperatorKind::PlainAggregate,
            Self::GroupAggregate { .. } => OperatorKind::GroupAggregate,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum PipelineKind {
    Record,
    Aggregate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggregateFunc {
    Sum,
    Count,
    Min,
    Max,
}

impl AggregateFunc {
    pub fn name(self) -> &'static str {
        match self {
            Self::Sum => "sum",
            Self::Count => "count",
            Self::Min => "min",
            Self::Max => "max",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct GroupAggregate {
    pub func: AggregateFunc,
    pub expr: JitExpr,
    pub output_type: JitType,
    pub alias: String,
}

impl GroupAggregate {
    pub fn new(
        func: AggregateFunc,
        expr: JitExpr,
        output_type: JitType,
        alias: impl Into<String>,
    ) -> Self {
        Self {
            func,
            expr,
            output_type,
            alias: alias.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PipelineGraph {
    pub source: PipelineSource,
    pub stages: Vec<PipelineStage>,
    pub sink: PipelineSink,
}

impl PipelineGraph {
    pub fn record(stages: Vec<PipelineStage>) -> Self {
        Self {
            source: PipelineSource::ArrowBatch,
            stages,
            sink: PipelineSink::RecordBatch,
        }
    }

    pub fn filter_sum(predicate: JitExpr, measure: JitExpr) -> Self {
        Self {
            source: PipelineSource::ArrowBatch,
            stages: vec![PipelineStage::Filter(predicate)],
            sink: PipelineSink::Sum { measure },
        }
    }

    pub fn group_aggregate(
        stages: Vec<PipelineStage>,
        keys: Vec<JitExpr>,
        aggregates: Vec<GroupAggregate>,
    ) -> Self {
        Self {
            source: PipelineSource::ArrowBatch,
            stages,
            sink: PipelineSink::GroupAggregate { keys, aggregates },
        }
    }

    pub fn stage_names(&self) -> Vec<&'static str> {
        self.stages
            .iter()
            .map(|stage| stage.operator_kind().name())
            .collect()
    }

    pub fn kind(&self) -> PipelineKind {
        match self.sink {
            PipelineSink::RecordBatch => PipelineKind::Record,
            PipelineSink::Sum { .. } | PipelineSink::GroupAggregate { .. } => {
                PipelineKind::Aggregate
            }
        }
    }

    pub fn sink_name(&self) -> &'static str {
        match &self.sink {
            PipelineSink::RecordBatch => "record_batch",
            PipelineSink::Sum { .. } => "scalar_sum",
            PipelineSink::GroupAggregate { .. } => "group_aggregate",
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        AggregateFunc, GroupAggregate, JitExpr, JitProjection, JitScalar, JitType, OperatorKind,
        OutputMode, PipelineGraph,
    };

    #[test]
    fn records_filter_project_pipeline() {
        let predicate = JitExpr::Literal(JitScalar::Bool(true));
        let projection = JitProjection::new(JitExpr::Literal(JitScalar::Int64(1)), "one");
        let pipeline = PipelineGraph::record(vec![
            crate::PipelineStage::Filter(predicate),
            crate::PipelineStage::Projection(vec![projection]),
        ]);

        assert_eq!(pipeline.stage_names(), vec!["filter", "project"]);
        assert_eq!(pipeline.sink_name(), "record_batch");
    }

    #[test]
    fn records_projection_pipeline() {
        let projection =
            JitProjection::new(JitExpr::Literal(JitScalar::Null(JitType::Int64)), "value");
        let pipeline =
            PipelineGraph::record(vec![crate::PipelineStage::Projection(vec![projection])]);

        assert_eq!(pipeline.stage_names(), vec!["project"]);
        assert_eq!(pipeline.sink_name(), "record_batch");
    }

    #[test]
    fn records_filter_sum_pipeline() {
        let predicate = JitExpr::Literal(JitScalar::Bool(true));
        let measure = JitExpr::Literal(JitScalar::Float64(1.0));
        let pipeline = PipelineGraph::filter_sum(predicate, measure);

        assert_eq!(pipeline.stage_names(), vec!["filter"]);
        assert_eq!(pipeline.sink_name(), "scalar_sum");
    }

    #[test]
    fn records_group_aggregate_pipeline() {
        let key = JitExpr::Literal(JitScalar::Int64(1));
        let aggregate = GroupAggregate::new(
            AggregateFunc::Sum,
            JitExpr::Literal(JitScalar::Float64(1.0)),
            JitType::Float64,
            "sum_value",
        );
        let pipeline = PipelineGraph::group_aggregate(vec![], vec![key], vec![aggregate]);

        assert!(pipeline.stage_names().is_empty());
        assert_eq!(pipeline.sink_name(), "group_aggregate");
        assert_eq!(pipeline.kind(), crate::PipelineKind::Aggregate);
    }

    #[test]
    fn describes_operator_properties() {
        let filter = OperatorKind::Filter.properties();
        assert!(filter.preserves_order);
        assert!(filter.changes_cardinality);
        assert!(!filter.pipeline_breaker);
        assert_eq!(filter.output_mode, OutputMode::Selection);

        let aggregate = OperatorKind::PlainAggregate.properties();
        assert!(aggregate.pipeline_breaker);
        assert!(aggregate.requires_state);
        assert_eq!(aggregate.output_mode, OutputMode::Scalar);
    }
}
