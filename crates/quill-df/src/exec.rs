use std::any::Any;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use datafusion::arrow::array::{ArrayRef, Decimal128Array, Float64Array};
use datafusion::arrow::datatypes::{DataType as ArrowDataType, SchemaRef as ArrowSchemaRef};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::common::{DataFusionError, Result};
use datafusion::execution::context::TaskContext;
use datafusion::physical_expr::{EquivalenceProperties, Partitioning};
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties, RecordBatchStream,
    SendableRecordBatchStream,
};
use futures::{ready, Stream, StreamExt};

use quill_jit::{CompiledKernel, KernelKind};
use quill_plan::PipelineKind;
use quill_runtime::{
    FilterProjectKernel, FilterSumKernel, FilterSumValue, GroupAggregateKernel, GroupAggregateState,
};

#[derive(Debug, Clone)]
pub enum PipelineRuntime {
    RecordBatch(FilterProjectKernel),
    ScalarSum(FilterSumKernel),
    GroupAggregate(GroupAggregateKernel),
}

#[derive(Debug, Clone)]
pub struct CompiledPipelineExec {
    input: Arc<dyn ExecutionPlan>,
    runtime: PipelineRuntime,
    kernel: CompiledKernel,
    schema: ArrowSchemaRef,
    cache: Arc<PlanProperties>,
}

#[derive(Debug, Clone)]
pub struct CompiledGlobalGroupAggregateExec {
    input: Arc<dyn ExecutionPlan>,
    runtime: GroupAggregateKernel,
    kernel: CompiledKernel,
    schema: ArrowSchemaRef,
    cache: Arc<PlanProperties>,
}

impl PipelineRuntime {
    fn expected_kernel(&self) -> KernelKind {
        match self {
            Self::RecordBatch(_) => KernelKind::FilterProject,
            Self::ScalarSum(_) => KernelKind::FilterSum,
            Self::GroupAggregate(_) => KernelKind::GroupAggregate,
        }
    }

    fn kind(&self) -> PipelineKind {
        match self {
            Self::RecordBatch(_) => PipelineKind::Record,
            Self::ScalarSum(_) | Self::GroupAggregate(_) => PipelineKind::Aggregate,
        }
    }

    fn stage_names(&self) -> &'static str {
        match self {
            Self::RecordBatch(_) => "filter -> project",
            Self::ScalarSum(_) => "filter",
            Self::GroupAggregate(runtime) => runtime.stage_names(),
        }
    }

    fn sink_name(&self) -> &'static str {
        match self {
            Self::RecordBatch(_) => "record_batch",
            Self::ScalarSum(_) => "scalar_sum",
            Self::GroupAggregate(_) => "group_aggregate",
        }
    }
}

impl CompiledPipelineExec {
    pub fn try_new(
        input: Arc<dyn ExecutionPlan>,
        runtime: PipelineRuntime,
        schema: ArrowSchemaRef,
        kernel: CompiledKernel,
    ) -> Result<Self> {
        let expected = runtime.expected_kernel();
        if kernel.kind != expected {
            return Err(DataFusionError::Internal(format!(
                "expected {:?} pipeline kernel, got {:?}",
                expected, kernel.kind
            )));
        }
        if matches!(runtime, PipelineRuntime::ScalarSum(_)) && schema.fields().len() != 1 {
            return Err(DataFusionError::Internal(format!(
                "compiled scalar aggregate pipeline expected one output field, got {}",
                schema.fields().len()
            )));
        }

        let partitioning =
            Partitioning::UnknownPartitioning(input.properties().partitioning.partition_count());
        let cache = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(Arc::clone(&schema)),
            partitioning,
            input.properties().emission_type,
            input.properties().boundedness,
        ));

        Ok(Self {
            input,
            runtime,
            kernel,
            schema,
            cache,
        })
    }

    pub fn input(&self) -> &Arc<dyn ExecutionPlan> {
        &self.input
    }

    pub fn kernel(&self) -> &CompiledKernel {
        &self.kernel
    }

    pub fn runtime(&self) -> &PipelineRuntime {
        &self.runtime
    }

    pub fn pipeline_kind(&self) -> PipelineKind {
        self.runtime.kind()
    }

    pub fn stage_names(&self) -> &'static str {
        self.runtime.stage_names()
    }

    pub fn sink_name(&self) -> &'static str {
        self.runtime.sink_name()
    }

    fn execute_record_batch(
        &self,
        runtime: &FilterProjectKernel,
        batch: RecordBatch,
    ) -> Result<RecordBatch> {
        if let Some(output) = quill_jit::execute_filter_project(&self.kernel, runtime, &batch)
            .map_err(crate::map_jit_err)?
        {
            return Ok(output);
        }

        runtime.execute(&batch).map_err(crate::map_jit_err)
    }

    fn execute_scalar_sum_batch(
        &self,
        runtime: &FilterSumKernel,
        batch: &RecordBatch,
    ) -> Result<FilterSumValue> {
        if let Some(partial) = quill_jit::execute_filter_sum(&self.kernel, runtime, batch)
            .map_err(crate::map_jit_err)?
        {
            return Ok(partial);
        }

        runtime.execute(batch).map_err(crate::map_jit_err)
    }

    fn accumulate_group_aggregate_batch(
        &self,
        runtime: &GroupAggregateKernel,
        state: &mut GroupAggregateState,
        batch: &RecordBatch,
    ) -> Result<()> {
        accumulate_group_aggregate_batch(&self.kernel, runtime, state, batch)
    }
}

impl CompiledGlobalGroupAggregateExec {
    pub fn try_new(
        input: Arc<dyn ExecutionPlan>,
        runtime: GroupAggregateKernel,
        schema: ArrowSchemaRef,
        kernel: CompiledKernel,
    ) -> Result<Self> {
        if kernel.kind != KernelKind::GroupAggregate {
            return Err(DataFusionError::Internal(format!(
                "expected group aggregate kernel, got {:?}",
                kernel.kind
            )));
        }
        let cache = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(Arc::clone(&schema)),
            Partitioning::UnknownPartitioning(1),
            input.properties().emission_type,
            input.properties().boundedness,
        ));

        Ok(Self {
            input,
            runtime,
            kernel,
            schema,
            cache,
        })
    }

    pub fn input(&self) -> &Arc<dyn ExecutionPlan> {
        &self.input
    }

    pub fn kernel(&self) -> &CompiledKernel {
        &self.kernel
    }

    pub fn runtime(&self) -> &GroupAggregateKernel {
        &self.runtime
    }

    pub fn stage_names(&self) -> &'static str {
        self.runtime.stage_names()
    }

    pub fn sink_name(&self) -> &'static str {
        "group_aggregate"
    }
}

impl DisplayAs for CompiledPipelineExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match (&self.runtime, t) {
            (PipelineRuntime::RecordBatch(runtime), DisplayFormatType::Default)
            | (PipelineRuntime::RecordBatch(runtime), DisplayFormatType::Verbose) => {
                let projections = runtime
                    .projections()
                    .iter()
                    .map(|projection| projection.alias.clone())
                    .collect::<Vec<_>>()
                    .join(", ");
                write!(
                    f,
                    "CompiledPipelineExec: kind=record, stages={}, sink={}, backend={}, executable={}, predicate={:?}, expr=[{}]",
                    self.stage_names(),
                    self.sink_name(),
                    self.kernel.backend,
                    self.kernel.executable,
                    runtime.predicate(),
                    projections
                )
            }
            (PipelineRuntime::ScalarSum(runtime), DisplayFormatType::Default)
            | (PipelineRuntime::ScalarSum(runtime), DisplayFormatType::Verbose) => {
                write!(
                    f,
                    "CompiledPipelineExec: kind=aggregate, stages={}, sink={}, backend={}, executable={}, predicate={:?}, measure={:?}",
                    self.stage_names(),
                    self.sink_name(),
                    self.kernel.backend,
                    self.kernel.executable,
                    runtime.predicate(),
                    runtime.measure()
                )
            }
            (PipelineRuntime::GroupAggregate(runtime), DisplayFormatType::Default)
            | (PipelineRuntime::GroupAggregate(runtime), DisplayFormatType::Verbose) => {
                write!(
                    f,
                    "CompiledPipelineExec: kind=aggregate, stages={}, sink={}, backend={}, executable={}, keys={}, aggregates={}",
                    self.stage_names(),
                    self.sink_name(),
                    self.kernel.backend,
                    self.kernel.executable,
                    runtime.keys().len(),
                    runtime.aggregates().len()
                )
            }
            (PipelineRuntime::RecordBatch(runtime), DisplayFormatType::TreeRender) => {
                writeln!(
                    f,
                    "kind=record, stages={}, sink={}, backend={}, executable={}",
                    self.stage_names(),
                    self.sink_name(),
                    self.kernel.backend,
                    self.kernel.executable
                )?;
                writeln!(f, "predicate={:?}", runtime.predicate())?;
                for (index, projection) in runtime.projections().iter().enumerate() {
                    writeln!(f, "expr{index}={}", projection.alias)?;
                }
                Ok(())
            }
            (PipelineRuntime::ScalarSum(runtime), DisplayFormatType::TreeRender) => {
                writeln!(
                    f,
                    "kind=aggregate, stages={}, sink={}, backend={}, executable={}",
                    self.stage_names(),
                    self.sink_name(),
                    self.kernel.backend,
                    self.kernel.executable
                )?;
                writeln!(f, "predicate={:?}", runtime.predicate())?;
                writeln!(f, "measure={:?}", runtime.measure())
            }
            (PipelineRuntime::GroupAggregate(runtime), DisplayFormatType::TreeRender) => {
                writeln!(
                    f,
                    "kind=aggregate, stages={}, sink={}, backend={}, executable={}",
                    self.stage_names(),
                    self.sink_name(),
                    self.kernel.backend,
                    self.kernel.executable
                )?;
                writeln!(f, "keys={}", runtime.keys().len())?;
                writeln!(f, "aggregates={}", runtime.aggregates().len())
            }
        }
    }
}

impl ExecutionPlan for CompiledPipelineExec {
    fn name(&self) -> &str {
        "CompiledPipelineExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.cache
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        vec![matches!(self.runtime, PipelineRuntime::RecordBatch(_))]
    }

    fn benefits_from_input_partitioning(&self) -> Vec<bool> {
        vec![true]
    }

    fn with_new_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if children.len() != 1 {
            return Err(DataFusionError::Internal(format!(
                "CompiledPipelineExec expected one child, got {}",
                children.len()
            )));
        }

        Self::try_new(
            children.swap_remove(0),
            self.runtime.clone(),
            Arc::clone(&self.schema),
            self.kernel.clone(),
        )
        .map(|exec| Arc::new(exec) as Arc<dyn ExecutionPlan>)
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        match &self.runtime {
            PipelineRuntime::RecordBatch(_) => Ok(Box::pin(CompiledRecordPipelineStream {
                schema: Arc::clone(&self.schema),
                input: self.input.execute(partition, context)?,
                exec: self.clone(),
            })),
            PipelineRuntime::ScalarSum(_) => Ok(Box::pin(CompiledScalarSumStream {
                schema: Arc::clone(&self.schema),
                input: self.input.execute(partition, context)?,
                exec: self.clone(),
                sum: None,
                emitted: false,
            })),
            PipelineRuntime::GroupAggregate(runtime) => {
                Ok(Box::pin(CompiledGroupAggregateStream {
                    schema: Arc::clone(&self.schema),
                    input: self.input.execute(partition, context)?,
                    exec: self.clone(),
                    state: runtime.new_state(),
                    emitted: false,
                }))
            }
        }
    }
}

impl DisplayAs for CompiledGlobalGroupAggregateExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(
                    f,
                    "CompiledGlobalGroupAggregateExec: kind=aggregate, stages={}, sink={}, mode={}, backend={}, executable={}, keys={}, aggregates={}",
                    self.stage_names(),
                    self.sink_name(),
                    self.runtime.output_mode().name(),
                    self.kernel.backend,
                    self.kernel.executable,
                    self.runtime.keys().len(),
                    self.runtime.aggregates().len()
                )
            }
            DisplayFormatType::TreeRender => {
                writeln!(
                    f,
                    "kind=aggregate, stages={}, sink={}, mode={}, backend={}, executable={}",
                    self.stage_names(),
                    self.sink_name(),
                    self.runtime.output_mode().name(),
                    self.kernel.backend,
                    self.kernel.executable
                )?;
                writeln!(f, "keys={}", self.runtime.keys().len())?;
                writeln!(f, "aggregates={}", self.runtime.aggregates().len())
            }
        }
    }
}

impl ExecutionPlan for CompiledGlobalGroupAggregateExec {
    fn name(&self) -> &str {
        "CompiledGlobalGroupAggregateExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.cache
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        vec![false]
    }

    fn benefits_from_input_partitioning(&self) -> Vec<bool> {
        vec![true]
    }

    fn with_new_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if children.len() != 1 {
            return Err(DataFusionError::Internal(format!(
                "CompiledGlobalGroupAggregateExec expected one child, got {}",
                children.len()
            )));
        }

        Self::try_new(
            children.swap_remove(0),
            self.runtime.clone(),
            Arc::clone(&self.schema),
            self.kernel.clone(),
        )
        .map(|exec| Arc::new(exec) as Arc<dyn ExecutionPlan>)
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        if partition != 0 {
            return Err(DataFusionError::Internal(format!(
                "CompiledGlobalGroupAggregateExec has one output partition, got {partition}"
            )));
        }

        Ok(Box::pin(CompiledGlobalGroupAggregateStream {
            schema: Arc::clone(&self.schema),
            input: Arc::clone(&self.input),
            context,
            current: None,
            next_partition: 0,
            input_partitions: self.input.properties().partitioning.partition_count(),
            runtime: self.runtime.clone(),
            kernel: self.kernel.clone(),
            state: self.runtime.new_state(),
            emitted: false,
        }))
    }
}

struct CompiledRecordPipelineStream {
    schema: ArrowSchemaRef,
    input: SendableRecordBatchStream,
    exec: CompiledPipelineExec,
}

struct CompiledScalarSumStream {
    schema: ArrowSchemaRef,
    input: SendableRecordBatchStream,
    exec: CompiledPipelineExec,
    sum: Option<FilterSumValue>,
    emitted: bool,
}

struct CompiledGroupAggregateStream {
    schema: ArrowSchemaRef,
    input: SendableRecordBatchStream,
    exec: CompiledPipelineExec,
    state: GroupAggregateState,
    emitted: bool,
}

struct CompiledGlobalGroupAggregateStream {
    schema: ArrowSchemaRef,
    input: Arc<dyn ExecutionPlan>,
    context: Arc<TaskContext>,
    current: Option<SendableRecordBatchStream>,
    next_partition: usize,
    input_partitions: usize,
    runtime: GroupAggregateKernel,
    kernel: CompiledKernel,
    state: GroupAggregateState,
    emitted: bool,
}

impl Stream for CompiledRecordPipelineStream {
    type Item = Result<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let runtime = match &self.exec.runtime {
            PipelineRuntime::RecordBatch(runtime) => runtime.clone(),
            PipelineRuntime::ScalarSum(_) | PipelineRuntime::GroupAggregate(_) => {
                return Poll::Ready(Some(Err(DataFusionError::Internal(
                    "record pipeline stream cannot execute aggregate runtime".to_string(),
                ))));
            }
        };

        match ready!(self.input.poll_next_unpin(cx)) {
            Some(Ok(batch)) => Poll::Ready(Some(self.exec.execute_record_batch(&runtime, batch))),
            Some(Err(err)) => Poll::Ready(Some(Err(err))),
            None => Poll::Ready(None),
        }
    }
}

impl Stream for CompiledScalarSumStream {
    type Item = Result<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.emitted {
            return Poll::Ready(None);
        }

        let runtime = match &self.exec.runtime {
            PipelineRuntime::ScalarSum(runtime) => runtime.clone(),
            PipelineRuntime::RecordBatch(_) | PipelineRuntime::GroupAggregate(_) => {
                return Poll::Ready(Some(Err(DataFusionError::Internal(
                    "scalar sum stream cannot execute record runtime".to_string(),
                ))));
            }
        };

        loop {
            match ready!(self.input.poll_next_unpin(cx)) {
                Some(Ok(batch)) => match self.exec.execute_scalar_sum_batch(&runtime, &batch) {
                    Ok(partial) => {
                        if let Some(sum) = &mut self.sum {
                            if let Err(err) = sum.merge(partial) {
                                return Poll::Ready(Some(Err(crate::map_jit_err(err))));
                            }
                        } else {
                            self.sum = Some(partial);
                        }
                    }
                    Err(err) => return Poll::Ready(Some(Err(err))),
                },
                Some(Err(err)) => return Poll::Ready(Some(Err(err))),
                None => {
                    self.emitted = true;
                    return Poll::Ready(Some(finish_scalar_sum_batch(
                        Arc::clone(&self.schema),
                        self.sum,
                    )));
                }
            }
        }
    }
}

impl Stream for CompiledGroupAggregateStream {
    type Item = Result<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.emitted {
            return Poll::Ready(None);
        }

        let runtime = match &self.exec.runtime {
            PipelineRuntime::GroupAggregate(runtime) => runtime.clone(),
            PipelineRuntime::RecordBatch(_) | PipelineRuntime::ScalarSum(_) => {
                return Poll::Ready(Some(Err(DataFusionError::Internal(
                    "group aggregate stream cannot execute non-group runtime".to_string(),
                ))));
            }
        };

        loop {
            match ready!(self.input.poll_next_unpin(cx)) {
                Some(Ok(batch)) => {
                    let mut state = std::mem::replace(&mut self.state, runtime.new_state());
                    let result = self
                        .exec
                        .accumulate_group_aggregate_batch(&runtime, &mut state, &batch);
                    self.state = state;
                    if let Err(err) = result {
                        return Poll::Ready(Some(Err(err)));
                    }
                }
                Some(Err(err)) => return Poll::Ready(Some(Err(err))),
                None => {
                    self.emitted = true;
                    let state = std::mem::replace(&mut self.state, runtime.new_state());
                    return Poll::Ready(Some(runtime.finish(state).map_err(crate::map_jit_err)));
                }
            }
        }
    }
}

impl Stream for CompiledGlobalGroupAggregateStream {
    type Item = Result<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.emitted {
            return Poll::Ready(None);
        }

        loop {
            if let Some(input) = &mut self.current {
                match ready!(input.poll_next_unpin(cx)) {
                    Some(Ok(batch)) => {
                        let runtime = self.runtime.clone();
                        let kernel = self.kernel.clone();
                        let mut state = std::mem::replace(&mut self.state, runtime.new_state());
                        let result =
                            accumulate_group_aggregate_batch(&kernel, &runtime, &mut state, &batch);
                        self.state = state;
                        if let Err(err) = result {
                            return Poll::Ready(Some(Err(err)));
                        }
                    }
                    Some(Err(err)) => return Poll::Ready(Some(Err(err))),
                    None => {
                        self.current = None;
                    }
                }
                continue;
            }

            if self.next_partition < self.input_partitions {
                match self
                    .input
                    .execute(self.next_partition, Arc::clone(&self.context))
                {
                    Ok(stream) => {
                        self.current = Some(stream);
                        self.next_partition += 1;
                    }
                    Err(err) => return Poll::Ready(Some(Err(err))),
                }
                continue;
            }

            self.emitted = true;
            let runtime = self.runtime.clone();
            let state = std::mem::replace(&mut self.state, runtime.new_state());
            return Poll::Ready(Some(
                runtime.finish_final(state).map_err(crate::map_jit_err),
            ));
        }
    }
}

impl RecordBatchStream for CompiledRecordPipelineStream {
    fn schema(&self) -> ArrowSchemaRef {
        Arc::clone(&self.schema)
    }
}

impl RecordBatchStream for CompiledScalarSumStream {
    fn schema(&self) -> ArrowSchemaRef {
        Arc::clone(&self.schema)
    }
}

impl RecordBatchStream for CompiledGroupAggregateStream {
    fn schema(&self) -> ArrowSchemaRef {
        Arc::clone(&self.schema)
    }
}

impl RecordBatchStream for CompiledGlobalGroupAggregateStream {
    fn schema(&self) -> ArrowSchemaRef {
        Arc::clone(&self.schema)
    }
}

fn accumulate_group_aggregate_batch(
    kernel: &CompiledKernel,
    runtime: &GroupAggregateKernel,
    state: &mut GroupAggregateState,
    batch: &RecordBatch,
) -> Result<()> {
    if quill_jit::execute_group_aggregate_update(kernel, runtime, state, batch)
        .map_err(crate::map_jit_err)?
        .is_some()
    {
        return Ok(());
    }

    let binding = runtime
        .bind_batch(state, batch)
        .map_err(crate::map_jit_err)?;
    runtime
        .flush_dense_state(state)
        .map_err(crate::map_jit_err)?;
    runtime
        .accumulate_bound(state, batch, &binding)
        .map_err(crate::map_jit_err)
}

fn finish_scalar_sum_batch(
    schema: ArrowSchemaRef,
    sum: Option<FilterSumValue>,
) -> Result<RecordBatch> {
    let field = schema.field(0);
    let values = match field.data_type() {
        ArrowDataType::Float64 => {
            let value = match sum {
                Some(FilterSumValue::Float64(value)) => value,
                None => None,
                Some(other) => {
                    return Err(DataFusionError::Execution(format!(
                        "expected f64 sum, got {:?}",
                        other.ty()
                    )));
                }
            };
            Arc::new(Float64Array::from(vec![value])) as ArrayRef
        }
        ArrowDataType::Decimal128(precision, scale) => {
            let value = match sum {
                Some(FilterSumValue::Decimal128 {
                    value,
                    scale: value_scale,
                }) => {
                    if value_scale != *scale {
                        return Err(DataFusionError::Execution(format!(
                            "expected decimal scale {}, got {}",
                            scale, value_scale
                        )));
                    }
                    value
                }
                None => None,
                Some(other) => {
                    return Err(DataFusionError::Execution(format!(
                        "expected decimal sum, got {:?}",
                        other.ty()
                    )));
                }
            };
            Arc::new(
                Decimal128Array::from(vec![value])
                    .with_precision_and_scale(*precision, *scale)
                    .map_err(|err| DataFusionError::Execution(err.to_string()))?,
            ) as ArrayRef
        }
        other => {
            return Err(DataFusionError::Execution(format!(
                "unsupported scalar sum output type {other:?}"
            )));
        }
    };

    RecordBatch::try_new(schema, vec![values])
        .map_err(|err| DataFusionError::Execution(err.to_string()))
}
