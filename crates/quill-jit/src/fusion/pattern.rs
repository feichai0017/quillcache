use quill_plan::{OperatorKind, PipelineGraph, PipelineSink, PipelineSource, PipelineStage};

use crate::PipelineSpec;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FusionConstraint {
    ArrowBatchSource,
    FixedWidthRecordBatch,
    FixedWidthPlainAggregate,
    FixedWidthGroupAggregate,
}

impl FusionConstraint {
    pub fn name(self) -> &'static str {
        match self {
            Self::ArrowBatchSource => "arrow_batch_source",
            Self::FixedWidthRecordBatch => "fixed_width_record_batch",
            Self::FixedWidthPlainAggregate => "fixed_width_plain_aggregate",
            Self::FixedWidthGroupAggregate => "fixed_width_group_aggregate",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FusionLoweringKind {
    Record,
    PlainAggregate,
    GroupAggregate,
}

impl FusionLoweringKind {
    pub fn name(self) -> &'static str {
        match self {
            Self::Record => "record_filter_project",
            Self::PlainAggregate => "plain_aggregate_loop",
            Self::GroupAggregate => "group_aggregate_loop",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FusionPattern {
    pub id: &'static str,
    pub source: PipelineSource,
    pub stages: &'static [OperatorKind],
    pub sink: OperatorKind,
    pub constraints: &'static [FusionConstraint],
    pub lowering: FusionLoweringKind,
}

impl FusionPattern {
    pub(crate) fn matches_shape(self, graph: &PipelineGraph) -> bool {
        graph.source == self.source
            && graph.sink.operator_kind() == self.sink
            && graph.stages.len() == self.stages.len()
            && graph
                .stages
                .iter()
                .zip(self.stages.iter())
                .all(|(stage, expected)| stage.operator_kind() == *expected)
    }

    pub(crate) fn satisfies_constraints(self, graph: &PipelineGraph) -> bool {
        self.constraints
            .iter()
            .copied()
            .all(|constraint| constraint.is_satisfied_by(graph))
    }
}

impl FusionConstraint {
    fn is_satisfied_by(self, graph: &PipelineGraph) -> bool {
        match self {
            Self::ArrowBatchSource => graph.source == PipelineSource::ArrowBatch,
            Self::FixedWidthRecordBatch => match (&graph.stages[..], &graph.sink) {
                (
                    [PipelineStage::Filter(predicate), PipelineStage::Projection(projections)],
                    PipelineSink::RecordBatch,
                ) => PipelineSpec::record_project(predicate, projections).is_some(),
                _ => false,
            },
            Self::FixedWidthPlainAggregate => match (&graph.stages[..], &graph.sink) {
                ([PipelineStage::Filter(predicate)], PipelineSink::Sum { measure }) => {
                    PipelineSpec::filter_sum(predicate, measure).is_some()
                }
                _ => false,
            },
            Self::FixedWidthGroupAggregate => match (&graph.stages[..], &graph.sink) {
                (
                    [],
                    PipelineSink::GroupAggregate {
                        keys, aggregates, ..
                    },
                ) => PipelineSpec::group_aggregate(None, keys, aggregates).is_some(),
                (
                    [PipelineStage::Filter(predicate)],
                    PipelineSink::GroupAggregate {
                        keys, aggregates, ..
                    },
                ) => PipelineSpec::group_aggregate(Some(predicate), keys, aggregates).is_some(),
                _ => false,
            },
        }
    }
}
