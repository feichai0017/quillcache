mod lowering;
mod pattern;
mod registry;

pub use lowering::PipelineLowering;
pub use pattern::{FusionConstraint, FusionLoweringKind, FusionPattern};
pub use registry::{FusionMatch, FusionRegistry};

#[cfg(test)]
mod tests {
    use quill_plan::{JitExpr, JitProjection, JitScalar, PipelineGraph, PipelineStage};

    use super::{FusionLoweringKind, FusionRegistry, PipelineLowering};

    #[test]
    fn exposes_builtin_patterns() {
        let registry = FusionRegistry::builtin();
        let ids = registry
            .patterns()
            .iter()
            .map(|pattern| pattern.id)
            .collect::<Vec<_>>();

        assert_eq!(
            ids,
            vec![
                "filter_project_record",
                "filter_plain_sum",
                "filter_group_aggregate",
                "group_aggregate"
            ]
        );
    }

    #[test]
    fn matches_filter_projection_pipeline() {
        let predicate = JitExpr::Literal(JitScalar::Bool(true));
        let projection = JitProjection::new(JitExpr::Literal(JitScalar::Int64(1)), "one");
        let pipeline = PipelineGraph::record(vec![
            PipelineStage::Filter(predicate),
            PipelineStage::Projection(vec![projection]),
        ]);

        let matched = FusionRegistry::builtin()
            .match_pipeline(&pipeline)
            .expect("fusion match");
        assert_eq!(matched.pattern.lowering, FusionLoweringKind::Record);
        assert!(matches!(matched.lowering, PipelineLowering::Record { .. }));
    }

    #[test]
    fn matches_filter_sum_pipeline() {
        let predicate = JitExpr::Literal(JitScalar::Bool(true));
        let measure = JitExpr::Literal(JitScalar::Float64(1.0));
        let pipeline = PipelineGraph::filter_sum(predicate, measure);

        let matched = FusionRegistry::builtin()
            .match_pipeline(&pipeline)
            .expect("fusion match");
        assert_eq!(matched.pattern.lowering, FusionLoweringKind::PlainSum);
        assert!(matches!(
            matched.lowering,
            PipelineLowering::PlainSum { .. }
        ));
    }

    #[test]
    fn rejects_unregistered_pipeline_shape() {
        let projection = JitProjection::new(JitExpr::Literal(JitScalar::Int64(1)), "one");
        let pipeline = PipelineGraph::record(vec![PipelineStage::Projection(vec![projection])]);

        assert!(FusionRegistry::builtin()
            .match_pipeline(&pipeline)
            .is_none());
    }

    #[test]
    fn matches_filter_group_aggregate_pipeline() {
        use quill_plan::{AggregateFunc, GroupAggregate, JitType};

        let predicate = JitExpr::Literal(JitScalar::Bool(true));
        let key = JitExpr::Literal(JitScalar::Int64(1));
        let aggregate = GroupAggregate::new(
            AggregateFunc::Count,
            JitExpr::Literal(JitScalar::Int64(1)),
            JitType::Int64,
            "count",
        );
        let pipeline = PipelineGraph::group_aggregate(
            vec![PipelineStage::Filter(predicate)],
            vec![key],
            vec![aggregate],
        );

        let matched = FusionRegistry::builtin()
            .match_pipeline(&pipeline)
            .expect("fusion match");
        assert_eq!(matched.pattern.lowering, FusionLoweringKind::GroupAggregate);
        assert!(matches!(
            matched.lowering,
            PipelineLowering::GroupAggregate { .. }
        ));
    }
}
