use quill_plan::{OperatorKind, PipelineGraph, PipelineSource};

use super::lowering::{extract_lowering, PipelineLowering};
use super::{FusionConstraint, FusionLoweringKind, FusionPattern};

const FILTER_PROJECT_STAGES: [OperatorKind; 2] = [OperatorKind::Filter, OperatorKind::Project];
const FILTER_STAGES: [OperatorKind; 1] = [OperatorKind::Filter];
const NO_STAGES: [OperatorKind; 0] = [];
const RECORD_CONSTRAINTS: [FusionConstraint; 2] = [
    FusionConstraint::ArrowBatchSource,
    FusionConstraint::FixedWidthRecordBatch,
];
const PLAIN_SUM_CONSTRAINTS: [FusionConstraint; 2] = [
    FusionConstraint::ArrowBatchSource,
    FusionConstraint::FixedWidthPlainAggregate,
];
const GROUP_AGGREGATE_CONSTRAINTS: [FusionConstraint; 2] = [
    FusionConstraint::ArrowBatchSource,
    FusionConstraint::FixedWidthGroupAggregate,
];
const BUILTIN_PATTERNS: [FusionPattern; 4] = [
    FusionPattern {
        id: "filter_project_record",
        source: PipelineSource::ArrowBatch,
        stages: &FILTER_PROJECT_STAGES,
        sink: OperatorKind::RecordBatchSink,
        constraints: &RECORD_CONSTRAINTS,
        lowering: FusionLoweringKind::Record,
    },
    FusionPattern {
        id: "filter_plain_sum",
        source: PipelineSource::ArrowBatch,
        stages: &FILTER_STAGES,
        sink: OperatorKind::PlainAggregate,
        constraints: &PLAIN_SUM_CONSTRAINTS,
        lowering: FusionLoweringKind::PlainSum,
    },
    FusionPattern {
        id: "filter_group_aggregate",
        source: PipelineSource::ArrowBatch,
        stages: &FILTER_STAGES,
        sink: OperatorKind::GroupAggregate,
        constraints: &GROUP_AGGREGATE_CONSTRAINTS,
        lowering: FusionLoweringKind::GroupAggregate,
    },
    FusionPattern {
        id: "group_aggregate",
        source: PipelineSource::ArrowBatch,
        stages: &NO_STAGES,
        sink: OperatorKind::GroupAggregate,
        constraints: &GROUP_AGGREGATE_CONSTRAINTS,
        lowering: FusionLoweringKind::GroupAggregate,
    },
];

#[derive(Debug, Clone, PartialEq)]
pub struct FusionMatch {
    pub pattern: FusionPattern,
    pub lowering: PipelineLowering,
}

#[derive(Debug, Clone, Copy)]
pub struct FusionRegistry {
    patterns: &'static [FusionPattern],
}

impl Default for FusionRegistry {
    fn default() -> Self {
        Self {
            patterns: &BUILTIN_PATTERNS,
        }
    }
}

impl FusionRegistry {
    pub fn builtin() -> Self {
        Self::default()
    }

    pub fn patterns(self) -> &'static [FusionPattern] {
        self.patterns
    }

    pub fn match_pipeline(self, graph: &PipelineGraph) -> Option<FusionMatch> {
        self.patterns.iter().copied().find_map(|pattern| {
            if !pattern.matches_shape(graph) {
                return None;
            }
            let lowering = extract_lowering(pattern.lowering, graph)?;
            Some(FusionMatch { pattern, lowering })
        })
    }
}
