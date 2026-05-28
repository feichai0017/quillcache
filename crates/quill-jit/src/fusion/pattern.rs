use quill_plan::{OperatorKind, PipelineGraph, PipelineSource};

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
    PlainSum,
    GroupAggregate,
}

impl FusionLoweringKind {
    pub fn name(self) -> &'static str {
        match self {
            Self::Record => "record_filter_project",
            Self::PlainSum => "plain_sum_loop",
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
}
