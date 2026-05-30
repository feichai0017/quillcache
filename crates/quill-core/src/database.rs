use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use datafusion::arrow::{datatypes::SchemaRef as ArrowSchemaRef, record_batch::RecordBatch};
use datafusion::common::ScalarValue;
use datafusion::datasource::file_format::options::ParquetReadOptions;
use datafusion::datasource::MemTable;
use datafusion::execution::context::{SessionConfig, SessionContext};
use datafusion::execution::session_state::SessionStateBuilder;
use datafusion::logical_expr::LogicalPlan;
use datafusion::physical_plan::{collect, displayable};
use serde::Serialize;
use tempfile::TempDir;

use crate::error::{QuillSQLError, QuillSQLResult};
use quill_df::{JitCandidate, MlirJitRule, PipelineCandidate};
use quill_jit::JitOptions;

#[derive(Debug, Clone)]
pub struct DatabaseOptions {
    pub data_dir: Option<PathBuf>,
    pub debug_trace: bool,
    pub jit: JitOptions,
}

impl Default for DatabaseOptions {
    fn default() -> Self {
        Self {
            data_dir: None,
            debug_trace: true,
            jit: JitOptions::default(),
        }
    }
}

pub struct Database {
    ctx: SessionContext,
    debug_trace: Arc<Mutex<Option<DebugTrace>>>,
    debug_trace_enabled: bool,
    _temp_dir: Option<TempDir>,
    _data_dir: PathBuf,
}

#[derive(Clone)]
pub struct PreparedQuery {
    ctx: SessionContext,
    logical_plan: LogicalPlan,
    physical_plan: String,
}

#[derive(Debug, Clone)]
pub struct QueryOutput {
    pub batches: Vec<RecordBatch>,
}

impl QueryOutput {
    pub fn new(batches: Vec<RecordBatch>) -> Self {
        Self { batches }
    }

    pub fn is_empty(&self) -> bool {
        self.batches.iter().all(|batch| batch.num_rows() == 0)
    }

    pub fn rows_as_strings(&self) -> Vec<Vec<String>> {
        record_batches_to_string_rows(&self.batches)
    }

    pub fn pretty_table(&self) -> QuillSQLResult<comfy_table::Table> {
        pretty_format_batches(&self.batches)
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct DebugTrace {
    pub logical_plan: String,
    pub physical_plan: String,
    pub jit_candidates: Vec<JitCandidate>,
    pub pipeline_candidates: Vec<PipelineCandidate>,
    pub rows: usize,
    pub duration_ms: u128,
    pub logical_tree: DebugPlanNode,
    pub physical_tree: DebugPlanNode,
}

#[derive(Debug, Clone, Serialize)]
pub struct DebugPlanNode {
    pub op: String,
    pub children: Vec<DebugPlanNode>,
}

#[derive(Debug, Clone, Serialize)]
pub struct DebugPlanSnapshot {
    pub logical: DebugPlanNode,
    pub physical: DebugPlanNode,
}

impl DebugPlanNode {
    fn from_logical(plan: &LogicalPlan) -> Self {
        Self {
            op: plan.display().to_string(),
            children: plan
                .inputs()
                .iter()
                .map(|child| Self::from_logical(child))
                .collect(),
        }
    }

    fn leaf(op: impl Into<String>) -> Self {
        Self {
            op: op.into(),
            children: Vec::new(),
        }
    }
}

impl Database {
    pub fn new(options: DatabaseOptions) -> QuillSQLResult<Self> {
        let DatabaseOptions {
            data_dir,
            debug_trace,
            jit,
        } = options;
        match data_dir {
            Some(data_dir) => Self::open(data_dir, None, debug_trace, jit),
            None => {
                let temp_dir = TempDir::new()?;
                let data_dir = temp_dir.path().join("data");
                Self::open(data_dir, Some(temp_dir), debug_trace, jit)
            }
        }
    }

    pub fn new_with_data_dir(data_dir: impl Into<PathBuf>) -> QuillSQLResult<Self> {
        Self::open(data_dir.into(), None, true, JitOptions::default())
    }

    pub fn new_temp() -> QuillSQLResult<Self> {
        let temp_dir = TempDir::new()?;
        let data_dir = temp_dir.path().join("data");
        Self::open(data_dir, Some(temp_dir), true, JitOptions::default())
    }

    fn open(
        data_dir: PathBuf,
        temp_dir: Option<TempDir>,
        debug_trace_enabled: bool,
        jit_options: JitOptions,
    ) -> QuillSQLResult<Self> {
        std::fs::create_dir_all(&data_dir)?;

        let config = SessionConfig::new()
            .with_information_schema(true)
            .with_default_catalog_and_schema("datafusion", "public");
        let state = SessionStateBuilder::new()
            .with_config(config)
            .with_default_features()
            .with_physical_optimizer_rule(Arc::new(MlirJitRule::with_options(jit_options)))
            .build();
        let ctx = SessionContext::new_with_state(state);

        Ok(Self {
            ctx,
            debug_trace: Arc::new(Mutex::new(None)),
            debug_trace_enabled,
            _temp_dir: temp_dir,
            _data_dir: data_dir,
        })
    }

    pub async fn run(&self, sql: &str) -> QuillSQLResult<QueryOutput> {
        let start = Instant::now();
        let logical_plan = self
            .ctx
            .state()
            .create_logical_plan(sql)
            .await
            .map_err(map_datafusion_err)?;
        let trace_input = if self.debug_trace_enabled {
            let logical_tree = DebugPlanNode::from_logical(&logical_plan);
            let logical_plan_str = logical_plan.display_indent().to_string();
            let physical_plan = self
                .ctx
                .state()
                .create_physical_plan(&logical_plan)
                .await
                .ok();
            let physical_plan_str = physical_plan
                .as_ref()
                .map(|plan| displayable(plan.as_ref()).indent(false).to_string())
                .unwrap_or_else(|| {
                    "DataFusion physical plan unavailable for this statement".into()
                });
            let jit_rule = MlirJitRule::new();
            let jit_candidates = physical_plan
                .as_ref()
                .map(|plan| jit_rule.inspect_plan(Arc::clone(plan)))
                .unwrap_or_default();
            let pipeline_candidates = physical_plan
                .as_ref()
                .map(|plan| jit_rule.inspect_pipelines(Arc::clone(plan)))
                .unwrap_or_default();
            Some((
                logical_tree,
                logical_plan_str,
                physical_plan_str,
                jit_candidates,
                pipeline_candidates,
            ))
        } else {
            None
        };

        let df = self
            .ctx
            .execute_logical_plan(logical_plan)
            .await
            .map_err(map_datafusion_err)?;
        let batches = df.collect().await.map_err(map_datafusion_err)?;
        let rows = batches.iter().map(|batch| batch.num_rows()).sum();
        if let Some((
            logical_tree,
            logical_plan,
            physical_plan,
            jit_candidates,
            pipeline_candidates,
        )) = trace_input
        {
            self.record_trace(DebugTrace {
                logical_plan,
                physical_plan: physical_plan.clone(),
                jit_candidates,
                pipeline_candidates,
                rows,
                duration_ms: start.elapsed().as_millis(),
                logical_tree,
                physical_tree: DebugPlanNode::leaf(physical_plan),
            });
        } else {
            self.clear_trace();
        }
        Ok(QueryOutput::new(batches))
    }

    pub async fn prepare(&self, sql: &str) -> QuillSQLResult<PreparedQuery> {
        let logical_plan = self
            .ctx
            .state()
            .create_logical_plan(sql)
            .await
            .map_err(map_datafusion_err)?;
        let plan = self
            .ctx
            .state()
            .create_physical_plan(&logical_plan)
            .await
            .map_err(map_datafusion_err)?;
        let physical_plan = displayable(plan.as_ref()).indent(false).to_string();
        Ok(PreparedQuery {
            ctx: self.ctx.clone(),
            logical_plan,
            physical_plan,
        })
    }

    pub async fn register_parquet(&self, table: &str, path: &str) -> QuillSQLResult<()> {
        self.ctx
            .register_parquet(table, path, ParquetReadOptions::default())
            .await
            .map_err(map_datafusion_err)
    }

    pub fn register_batches(
        &self,
        table: &str,
        schema: ArrowSchemaRef,
        batches: Vec<RecordBatch>,
    ) -> QuillSQLResult<()> {
        self.register_partitions(table, schema, vec![batches])
    }

    pub fn register_partitions(
        &self,
        table: &str,
        schema: ArrowSchemaRef,
        partitions: Vec<Vec<RecordBatch>>,
    ) -> QuillSQLResult<()> {
        let table_provider = MemTable::try_new(schema, partitions).map_err(map_datafusion_err)?;
        self.ctx
            .register_table(table, Arc::new(table_provider))
            .map(|_| ())
            .map_err(map_datafusion_err)
    }

    pub fn flush(&self) -> QuillSQLResult<()> {
        Ok(())
    }

    pub fn debug_last_trace(&self) -> Option<DebugTrace> {
        self.debug_trace.lock().expect("debug trace lock").clone()
    }

    pub fn debug_last_plan(&self) -> Option<DebugPlanSnapshot> {
        self.debug_trace
            .lock()
            .expect("debug trace lock")
            .clone()
            .map(|trace| DebugPlanSnapshot {
                logical: trace.logical_tree,
                physical: trace.physical_tree,
            })
    }

    fn record_trace(&self, trace: DebugTrace) {
        *self.debug_trace.lock().expect("debug trace lock") = Some(trace);
    }

    fn clear_trace(&self) {
        *self.debug_trace.lock().expect("debug trace lock") = None;
    }
}

impl PreparedQuery {
    pub async fn run(&self) -> QuillSQLResult<QueryOutput> {
        let plan = self
            .ctx
            .state()
            .create_physical_plan(&self.logical_plan)
            .await
            .map_err(map_datafusion_err)?;
        let batches = collect(plan, self.ctx.task_ctx())
            .await
            .map_err(map_datafusion_err)?;
        Ok(QueryOutput::new(batches))
    }

    pub fn physical_plan(&self) -> &str {
        &self.physical_plan
    }
}

fn record_batches_to_string_rows(batches: &[RecordBatch]) -> Vec<Vec<String>> {
    let mut rows = Vec::new();
    for batch in batches {
        for row_idx in 0..batch.num_rows() {
            rows.push(
                batch
                    .columns()
                    .iter()
                    .map(
                        |array| match ScalarValue::try_from_array(array.as_ref(), row_idx) {
                            Ok(value) => value.to_string(),
                            Err(err) => format!("<error: {err}>"),
                        },
                    )
                    .collect(),
            );
        }
    }
    rows
}

fn pretty_format_batches(batches: &[RecordBatch]) -> QuillSQLResult<comfy_table::Table> {
    let mut table = comfy_table::Table::new();
    table.load_preset("||--+-++|    ++++++");
    let Some(first) = batches.first() else {
        return Ok(table);
    };
    table.set_header(
        first
            .schema()
            .fields()
            .iter()
            .map(|field| comfy_table::Cell::new(field.name()))
            .collect::<Vec<_>>(),
    );
    for row in record_batches_to_string_rows(batches) {
        table.add_row(row);
    }
    Ok(table)
}

fn map_datafusion_err(err: datafusion::error::DataFusionError) -> QuillSQLError {
    QuillSQLError::Execution(format!("DataFusion error: {err}"))
}
