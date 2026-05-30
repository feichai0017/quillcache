use quill_plan::{
    GroupAggregate, JitExpr, JitProjection, PipelineGraph, PipelineKind, PipelineSink,
    PipelineStage,
};

use super::registry::FusionRegistry;
use super::FusionLoweringKind;

#[derive(Debug, Clone, PartialEq)]
pub enum PipelineLowering {
    Record {
        predicate: JitExpr,
        projections: Vec<JitProjection>,
    },
    PlainSum {
        predicate: JitExpr,
        measure: JitExpr,
    },
    GroupAggregate {
        predicate: Option<JitExpr>,
        keys: Vec<JitExpr>,
        aggregates: Vec<GroupAggregate>,
    },
}

impl PipelineLowering {
    pub fn from_graph(graph: &PipelineGraph) -> Option<Self> {
        FusionRegistry::builtin()
            .match_pipeline(graph)
            .map(|matched| matched.lowering)
    }

    pub fn kind(&self) -> PipelineKind {
        match self {
            Self::Record { .. } => PipelineKind::Record,
            Self::PlainSum { .. } | Self::GroupAggregate { .. } => PipelineKind::Aggregate,
        }
    }
}

pub(crate) fn extract_lowering(
    kind: FusionLoweringKind,
    graph: &PipelineGraph,
) -> Option<PipelineLowering> {
    match kind {
        FusionLoweringKind::Record => extract_record_lowering(graph),
        FusionLoweringKind::PlainAggregate => extract_plain_sum_lowering(graph),
        FusionLoweringKind::GroupAggregate => extract_group_aggregate_lowering(graph),
    }
}

fn extract_record_lowering(graph: &PipelineGraph) -> Option<PipelineLowering> {
    match graph.stages.as_slice() {
        [PipelineStage::Filter(predicate), PipelineStage::Projection(projections)] => {
            Some(PipelineLowering::Record {
                predicate: predicate.clone(),
                projections: projections.clone(),
            })
        }
        _ => None,
    }
}

fn extract_plain_sum_lowering(graph: &PipelineGraph) -> Option<PipelineLowering> {
    let [PipelineStage::Filter(predicate)] = graph.stages.as_slice() else {
        return None;
    };
    let PipelineSink::Sum { measure } = &graph.sink else {
        return None;
    };
    Some(PipelineLowering::PlainSum {
        predicate: predicate.clone(),
        measure: measure.clone(),
    })
}

fn extract_group_aggregate_lowering(graph: &PipelineGraph) -> Option<PipelineLowering> {
    let predicate = match graph.stages.as_slice() {
        [] => None,
        [PipelineStage::Filter(predicate)] => Some(predicate.clone()),
        _ => return None,
    };
    let PipelineSink::GroupAggregate {
        keys, aggregates, ..
    } = &graph.sink
    else {
        return None;
    };
    Some(PipelineLowering::GroupAggregate {
        predicate,
        keys: keys.clone(),
        aggregates: aggregates.clone(),
    })
}
