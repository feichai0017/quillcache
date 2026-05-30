use std::sync::Arc;

use datafusion::arrow::array::Date32Array;
use datafusion::arrow::array::{Array, Decimal128Array, Float64Array, Int64Array, StringArray};
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::dataframe::DataFrameWriteOptions;
use datafusion::execution::context::SessionContext;
use quill_core::database::{Database, DatabaseOptions, QueryOutput};
use quill_jit::JitOptions;
use quill_plan::PipelineKind;
use tempfile::TempDir;

fn rows(result: QueryOutput) -> Vec<Vec<String>> {
    result.rows_as_strings()
}

#[tokio::test]
async fn datafusion_memory_table_executes_sql() {
    let db = Database::new_temp().expect("database");
    db.run("create table t as select 1::bigint as id, 10::bigint as v")
        .await
        .expect("create table");
    db.run("insert into t values (2, 20), (3, 30)")
        .await
        .expect("insert rows");

    assert_eq!(
        rows(
            db.run("select id, v from t where id >= 2 order by id")
                .await
                .unwrap()
        ),
        vec![
            vec!["2".to_string(), "20".to_string()],
            vec!["3".to_string(), "30".to_string()]
        ]
    );
}

#[tokio::test]
async fn parquet_table_is_the_persistent_storage_path() {
    let dir = TempDir::new().expect("temp dir");
    let parquet_path = dir.path().join("people.parquet");
    write_people_parquet(parquet_path.to_str().unwrap()).await;

    let db = Database::new_temp().expect("database");
    db.register_parquet("people", parquet_path.to_str().unwrap())
        .await
        .expect("register parquet");

    assert_eq!(
        rows(
            db.run("select name from people where id >= 2 order by id")
                .await
                .unwrap()
        ),
        vec![vec!["bob".to_string()], vec!["cara".to_string()]]
    );
}

#[tokio::test]
async fn explain_uses_datafusion_physical_plan() {
    let db = Database::new_temp().expect("database");
    let text = rows(db.run("explain select 1").await.unwrap())
        .into_iter()
        .flatten()
        .collect::<Vec<_>>()
        .join("\n");
    assert!(text.contains("ProjectionExec"), "{text}");
}

#[tokio::test]
async fn debug_trace_can_be_disabled_for_benchmarks() {
    let db = Database::new(DatabaseOptions {
        debug_trace: false,
        ..Default::default()
    })
    .expect("database");

    assert_eq!(
        rows(db.run("select 1::bigint").await.unwrap()),
        vec![vec!["1".to_string()]]
    );
    assert!(db.debug_last_trace().is_none());
}

#[tokio::test]
async fn prepared_query_reuses_logical_plan() {
    let db = Database::new(DatabaseOptions {
        debug_trace: false,
        ..Default::default()
    })
    .expect("database");
    db.run("create table t as select 1::bigint as id union all select 2::bigint")
        .await
        .expect("create table");

    let query = db
        .prepare("select id + 10 as value from t where id > 1")
        .await
        .expect("prepare");

    assert!(
        query.physical_plan().contains("CompiledPipelineExec"),
        "{}",
        query.physical_plan()
    );
    assert_eq!(
        rows(query.run().await.expect("first run")),
        vec![vec!["12".to_string()]]
    );
    assert_eq!(
        rows(query.run().await.expect("second run")),
        vec![vec!["12".to_string()]]
    );
}

#[tokio::test]
async fn debug_trace_reports_jit_candidates() {
    let db = Database::new_temp().expect("database");
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("v", DataType::Int64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from(vec![1, 2, 3])),
            Arc::new(Int64Array::from(vec![10, 20, 30])),
        ],
    )
    .expect("batch");
    db.register_batches("t", schema, vec![batch])
        .expect("table");

    let mut output_rows = rows(
        db.run("select id + 1 as next_id from t where v > 10")
            .await
            .expect("query"),
    );
    output_rows.sort();
    assert_eq!(
        output_rows,
        vec![vec!["3".to_string()], vec!["4".to_string()]]
    );

    let trace = db.debug_last_trace().expect("trace");

    assert!(
        trace.physical_plan.contains("CompiledPipelineExec"),
        "{}",
        trace.physical_plan
    );
    assert!(
        trace
            .jit_candidates
            .iter()
            .any(|candidate| matches!(candidate.kernel.name(), "filter_project" | "filter")),
        "{:?}",
        trace.jit_candidates
    );
    assert!(
        trace
            .pipeline_candidates
            .iter()
            .any(|candidate| candidate.node == "CompiledPipelineExec"
                && candidate.kind == PipelineKind::Record
                && candidate.compiled
                && candidate.stages == vec!["filter", "project"]
                && candidate.sink == "record_batch"
                && candidate.backend.as_deref() == Some("mlir")
                && candidate.reason == "compiled"),
        "{:?}",
        trace.pipeline_candidates
    );
}

#[tokio::test]
async fn disabled_jit_keeps_datafusion_physical_plan() {
    let db = Database::new(DatabaseOptions {
        jit: JitOptions::disabled(),
        ..Default::default()
    })
    .expect("database");
    db.run("create table t as select 1::bigint as id union all select 2::bigint")
        .await
        .expect("create table");

    assert_eq!(
        rows(
            db.run("select id + 1 as next_id from t where id > 1")
                .await
                .expect("query")
        ),
        vec![vec!["3".to_string()]]
    );

    let trace = db.debug_last_trace().expect("trace");
    assert!(
        !trace.physical_plan.contains("CompiledPipelineExec"),
        "{}",
        trace.physical_plan
    );
    assert!(
        trace.jit_candidates.is_empty(),
        "{:?}",
        trace.jit_candidates
    );
}

#[tokio::test]
async fn filter_project_mlir_execution_returns_expected_rows() {
    let db = database_with_mlir_execution();
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("v", DataType::Int64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from(vec![1, 2, 3])),
            Arc::new(Int64Array::from(vec![10, 20, 30])),
        ],
    )
    .expect("batch");
    db.register_batches("t", schema, vec![batch])
        .expect("table");

    assert_eq!(
        rows(
            db.run("select id + 1 as next_id from t where v > 10")
                .await
                .expect("query")
        ),
        vec![vec!["3".to_string()], vec!["4".to_string()]]
    );

    let trace = db.debug_last_trace().expect("trace");
    assert!(
        trace
            .jit_candidates
            .iter()
            .any(|candidate| candidate.kernel.name() == "filter_project"
                && candidate.backend == "mlir"
                && candidate.executable),
        "{:?}",
        trace.jit_candidates
    );
}

#[tokio::test]
async fn filter_project_mlir_execution_materializes_fixed_width_columns() {
    let db = database_with_mlir_execution();
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("price", DataType::Float64, false),
        Field::new("shipdate", DataType::Date32, false),
        Field::new("amount", DataType::Decimal128(15, 2), false),
    ]));
    let amount = Decimal128Array::from(vec![1000_i128, 2500, 3000])
        .with_precision_and_scale(15, 2)
        .expect("decimal scale");
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from(vec![1, 2, 3])),
            Arc::new(Float64Array::from(vec![10.0, 20.5, 30.25])),
            Arc::new(Date32Array::from(vec![19_724, 19_725, 19_726])),
            Arc::new(amount),
        ],
    )
    .expect("batch");
    db.register_batches("t", schema, vec![batch])
        .expect("table");

    let output = db
        .run(
            "select id + 1 as next_id, price, shipdate, amount \
             from t where id > 1 order by next_id",
        )
        .await
        .expect("query");

    let batch = &output.batches[0];
    let next_id = batch
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("next_id");
    let price = batch
        .column(1)
        .as_any()
        .downcast_ref::<Float64Array>()
        .expect("price");
    let shipdate = batch
        .column(2)
        .as_any()
        .downcast_ref::<Date32Array>()
        .expect("shipdate");
    let amount = batch
        .column(3)
        .as_any()
        .downcast_ref::<Decimal128Array>()
        .expect("amount");
    assert_eq!(next_id.values().as_ref(), &[3, 4]);
    assert_eq!(price.values().as_ref(), &[20.5, 30.25]);
    assert_eq!(shipdate.values().as_ref(), &[19_725, 19_726]);
    assert_eq!(amount.values().as_ref(), &[2_500, 3_000]);
    assert_eq!(amount.precision(), 15);
    assert_eq!(amount.scale(), 2);

    let trace = db.debug_last_trace().expect("trace");
    assert!(
        trace
            .jit_candidates
            .iter()
            .any(|candidate| candidate.kernel.name() == "filter_project"
                && candidate.backend == "mlir"
                && candidate.executable),
        "{:?}",
        trace.jit_candidates
    );
}

#[tokio::test]
async fn debug_trace_reports_plain_sum_candidate() {
    let db = Database::new_temp().expect("database");
    let schema = Arc::new(Schema::new(vec![
        Field::new("v", DataType::Int64, false),
        Field::new("price", DataType::Float64, false),
        Field::new("discount", DataType::Float64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from(vec![9, 11, 12])),
            Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0])),
            Arc::new(Float64Array::from(vec![0.1, 0.2, 0.3])),
        ],
    )
    .expect("batch");
    db.register_batches("t", schema, vec![batch])
        .expect("table");

    let output = db
        .run("select sum(price * discount) from t where v > 10")
        .await
        .expect("query");
    let values = output.batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Float64Array>()
        .expect("f64 sum");
    assert_eq!(values.len(), 1);
    assert!((values.value(0) - 13.0).abs() < 0.000_001);

    let trace = db.debug_last_trace().expect("trace");
    assert!(
        trace.physical_plan.contains("CompiledPipelineExec"),
        "{}",
        trace.physical_plan
    );
    assert!(
        trace
            .jit_candidates
            .iter()
            .any(|candidate| candidate.kernel.name() == "filter_sum"),
        "{:?}",
        trace.jit_candidates
    );
    assert!(
        trace
            .pipeline_candidates
            .iter()
            .any(|candidate| candidate.kind == PipelineKind::Aggregate
                && candidate.node == "CompiledPipelineExec"
                && candidate.compiled
                && candidate.stages == vec!["filter"]
                && candidate.sink == "scalar_sum"
                && candidate.backend.as_deref() == Some("mlir")
                && candidate.reason == "compiled"),
        "{:?}",
        trace.pipeline_candidates
    );
}

#[tokio::test]
async fn debug_trace_reports_group_aggregate_candidate() {
    let db = Database::new(DatabaseOptions {
        jit: JitOptions::disabled(),
        ..Default::default()
    })
    .expect("database");
    let schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, false),
        Field::new("v", DataType::Int64, false),
        Field::new("shipdate", DataType::Date32, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from(vec![1, 1, 2, 2])),
            Arc::new(Int64Array::from(vec![10, 20, 30, 40])),
            Arc::new(Date32Array::from(vec![19_724, 19_725, 19_724, 19_726])),
        ],
    )
    .expect("batch");
    db.register_batches("t", schema, vec![batch])
        .expect("table");

    assert_eq!(
        rows(
            db.run(
                "select k, sum(v) as sum_v, count(*) as count_v \
                 from t \
                 where shipdate <= date '2024-01-04' \
                 group by k \
                 order by k",
            )
            .await
            .expect("query")
        ),
        vec![
            vec!["1".to_string(), "30".to_string(), "2".to_string()],
            vec!["2".to_string(), "70".to_string(), "2".to_string()],
        ]
    );

    let trace = db.debug_last_trace().expect("trace");
    assert!(
        !trace.physical_plan.contains("CompiledPipelineExec"),
        "{}",
        trace.physical_plan
    );
    assert!(
        trace
            .pipeline_candidates
            .iter()
            .any(|candidate| candidate.node == "AggregateExec"
                && candidate.kind == PipelineKind::Aggregate
                && !candidate.compiled
                && candidate.source == "arrow_batch"
                && candidate.stages == vec!["filter"]
                && candidate.sink == "group_aggregate"
                && candidate.backend.is_none()
                && candidate.reason == "candidate"),
        "{:?}",
        trace.pipeline_candidates
    );
}

#[tokio::test]
async fn group_aggregate_mlir_executes_dense_update_pipeline() {
    let db = Database::new_temp().expect("database");
    let schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, false),
        Field::new("v", DataType::Int64, false),
        Field::new("shipdate", DataType::Date32, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from(vec![1, 1, 2, 2])),
            Arc::new(Int64Array::from(vec![10, 20, 30, 40])),
            Arc::new(Date32Array::from(vec![19_724, 19_725, 19_724, 19_726])),
        ],
    )
    .expect("batch");
    db.register_batches("t", schema, vec![batch])
        .expect("table");

    assert_eq!(
        rows(
            db.run(
                "select k, sum(v) as sum_v, count(*) as count_v, min(v) as min_v, max(v) as max_v \
                 from t \
                 where shipdate <= date '2024-01-04' \
                 group by k \
                 order by k",
            )
            .await
            .expect("query")
        ),
        vec![
            vec![
                "1".to_string(),
                "30".to_string(),
                "2".to_string(),
                "10".to_string(),
                "20".to_string()
            ],
            vec![
                "2".to_string(),
                "70".to_string(),
                "2".to_string(),
                "30".to_string(),
                "40".to_string()
            ],
        ]
    );

    let trace = db.debug_last_trace().expect("trace");
    assert!(
        trace
            .physical_plan
            .contains("CompiledGlobalGroupAggregateExec"),
        "{}",
        trace.physical_plan
    );
    assert!(
        trace
            .pipeline_candidates
            .iter()
            .any(
                |candidate| candidate.node == "CompiledGlobalGroupAggregateExec"
                    && candidate.kind == PipelineKind::Aggregate
                    && candidate.compiled
                    && candidate.source == "arrow_batch"
                    && candidate.stages == vec!["filter"]
                    && candidate.sink == "group_aggregate"
                    && candidate.output_mode == Some("final_values")
                    && candidate.backend.as_deref() == Some("mlir")
                    && candidate.reason == "compiled"
            ),
        "{:?}",
        trace.pipeline_candidates
    );
}

#[tokio::test]
async fn string_group_keys_use_mlir_dense_update_boundary() {
    let db = Database::new_temp().expect("database");
    let schema = Arc::new(Schema::new(vec![
        Field::new("flag", DataType::Utf8, false),
        Field::new("v", DataType::Int64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from(vec!["A", "A", "B"])),
            Arc::new(Int64Array::from(vec![10, 20, 30])),
        ],
    )
    .expect("batch");
    db.register_batches("t", schema, vec![batch])
        .expect("table");

    assert_eq!(
        rows(
            db.run("select flag, sum(v) from t group by flag order by flag")
                .await
                .expect("query")
        ),
        vec![
            vec!["A".to_string(), "30".to_string()],
            vec!["B".to_string(), "30".to_string()],
        ]
    );

    let trace = db.debug_last_trace().expect("trace");
    assert!(
        trace
            .physical_plan
            .contains("CompiledGlobalGroupAggregateExec"),
        "{}",
        trace.physical_plan
    );
    assert!(
        trace
            .pipeline_candidates
            .iter()
            .any(
                |candidate| candidate.node == "CompiledGlobalGroupAggregateExec"
                    && candidate.kind == PipelineKind::Aggregate
                    && candidate.compiled
                    && candidate.source == "arrow_batch"
                    && candidate.sink == "group_aggregate"
                    && candidate.output_mode == Some("final_values")
                    && candidate.backend.as_deref() == Some("mlir")
                    && candidate.reason == "compiled"
            ),
        "{:?}",
        trace.pipeline_candidates
    );
}

#[tokio::test]
async fn global_group_aggregate_merges_input_partitions() {
    let db = Database::new_temp().expect("database");
    let schema = Arc::new(Schema::new(vec![
        Field::new("flag", DataType::Utf8, false),
        Field::new("v", DataType::Int64, false),
    ]));
    let left = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from(vec!["A", "B"])),
            Arc::new(Int64Array::from(vec![10, 30])),
        ],
    )
    .expect("left batch");
    let right = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from(vec!["A", "A"])),
            Arc::new(Int64Array::from(vec![20, 5])),
        ],
    )
    .expect("right batch");
    db.register_partitions("t", schema, vec![vec![left], vec![right]])
        .expect("table");

    assert_eq!(
        rows(
            db.run("select flag, sum(v) as sum_v, count(*) as count_v from t group by flag order by flag")
                .await
                .expect("query")
        ),
        vec![
            vec!["A".to_string(), "35".to_string(), "3".to_string()],
            vec!["B".to_string(), "30".to_string(), "1".to_string()],
        ]
    );

    let trace = db.debug_last_trace().expect("trace");
    assert!(
        trace
            .physical_plan
            .contains("CompiledGlobalGroupAggregateExec"),
        "{}",
        trace.physical_plan
    );
    assert!(
        trace.physical_plan.contains("DataSourceExec: partitions=2"),
        "{}",
        trace.physical_plan
    );
    assert!(
        trace
            .pipeline_candidates
            .iter()
            .any(
                |candidate| candidate.node == "CompiledGlobalGroupAggregateExec"
                    && candidate.kind == PipelineKind::Aggregate
                    && candidate.compiled
                    && candidate.sink == "group_aggregate"
                    && candidate.output_mode == Some("final_values")
                    && candidate.backend.as_deref() == Some("mlir")
                    && candidate.reason == "compiled"
            ),
        "{:?}",
        trace.pipeline_candidates
    );
}

#[tokio::test]
async fn avg_group_aggregate_uses_global_final_pipeline() {
    let db = Database::new_temp().expect("database");
    let schema = Arc::new(Schema::new(vec![
        Field::new("flag", DataType::Utf8, false),
        Field::new("v", DataType::Int64, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from(vec!["A", "A", "B", "B"])),
            Arc::new(Int64Array::from(vec![Some(10), Some(20), Some(30), None])),
        ],
    )
    .expect("batch");
    db.register_batches("t", schema, vec![batch])
        .expect("table");

    assert_eq!(
        rows(
            db.run(
                "select flag, avg(v) as avg_v, sum(v) as sum_v, count(v) as count_v \
                 from t \
                 group by flag \
                 order by flag",
            )
            .await
            .expect("query")
        ),
        vec![
            vec![
                "A".to_string(),
                "15".to_string(),
                "30".to_string(),
                "2".to_string()
            ],
            vec![
                "B".to_string(),
                "30".to_string(),
                "30".to_string(),
                "1".to_string()
            ],
        ]
    );

    let trace = db.debug_last_trace().expect("trace");
    assert!(
        trace
            .physical_plan
            .contains("CompiledGlobalGroupAggregateExec"),
        "{}",
        trace.physical_plan
    );
    assert!(
        trace
            .pipeline_candidates
            .iter()
            .any(
                |candidate| candidate.node == "CompiledGlobalGroupAggregateExec"
                    && candidate.kind == PipelineKind::Aggregate
                    && candidate.compiled
                    && candidate.source == "arrow_batch"
                    && candidate.sink == "group_aggregate"
                    && candidate.output_mode == Some("final_values")
                    && candidate.backend.as_deref() == Some("mlir")
                    && candidate.reason == "compiled"
            ),
        "{:?}",
        trace.pipeline_candidates
    );
}

#[tokio::test]
async fn q1_shaped_composite_group_aggregate_uses_global_final_pipeline() {
    let db = Database::new_temp().expect("database");
    let schema = Arc::new(Schema::new(vec![
        Field::new("returnflag", DataType::Utf8, false),
        Field::new("linestatus", DataType::Utf8, false),
        Field::new("quantity", DataType::Int64, false),
        Field::new("shipdate", DataType::Date32, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from(vec!["A", "A", "A", "R"])),
            Arc::new(StringArray::from(vec!["F", "F", "O", "F"])),
            Arc::new(Int64Array::from(vec![10, 20, 30, 40])),
            Arc::new(Date32Array::from(vec![10, 11, 12, 13])),
        ],
    )
    .expect("batch");
    db.register_batches("lineitem", schema, vec![batch])
        .expect("table");

    assert_eq!(
        rows(
            db.run(
                "select returnflag, linestatus, \
                        sum(quantity) as sum_qty, \
                        avg(quantity) as avg_qty, \
                        count(*) as count_order \
                 from lineitem \
                 where shipdate <= date '1970-01-13' \
                 group by returnflag, linestatus \
                 order by returnflag, linestatus",
            )
            .await
            .expect("query")
        ),
        vec![
            vec![
                "A".to_string(),
                "F".to_string(),
                "30".to_string(),
                "15".to_string(),
                "2".to_string()
            ],
            vec![
                "A".to_string(),
                "O".to_string(),
                "30".to_string(),
                "30".to_string(),
                "1".to_string()
            ],
        ]
    );

    let trace = db.debug_last_trace().expect("trace");
    assert!(
        trace
            .physical_plan
            .contains("CompiledGlobalGroupAggregateExec"),
        "{}",
        trace.physical_plan
    );
    assert!(
        trace
            .pipeline_candidates
            .iter()
            .any(
                |candidate| candidate.node == "CompiledGlobalGroupAggregateExec"
                    && candidate.kind == PipelineKind::Aggregate
                    && candidate.compiled
                    && candidate.source == "arrow_batch"
                    && candidate.stages == vec!["filter"]
                    && candidate.sink == "group_aggregate"
                    && candidate.output_mode == Some("final_values")
                    && candidate.backend.as_deref() == Some("mlir")
                    && candidate.reason == "compiled"
            ),
        "{:?}",
        trace.pipeline_candidates
    );
}

#[tokio::test]
async fn q1_decimal_group_aggregate_shape_uses_global_final_pipeline() {
    let db = Database::new_temp().expect("database");
    let money_type = DataType::Decimal128(12, 2);
    let rate_type = DataType::Decimal128(4, 2);
    let schema = Arc::new(Schema::new(vec![
        Field::new("returnflag", DataType::Utf8, false),
        Field::new("linestatus", DataType::Utf8, false),
        Field::new("quantity", DataType::Int64, false),
        Field::new("extendedprice", money_type.clone(), false),
        Field::new("discount", rate_type.clone(), false),
        Field::new("shipdate", DataType::Date32, false),
    ]));
    let price = Decimal128Array::from(vec![1000_i128, 2000, 3000, 4000])
        .with_precision_and_scale(12, 2)
        .expect("price decimal");
    let discount = Decimal128Array::from(vec![10_i128, 20, 10, 0])
        .with_precision_and_scale(4, 2)
        .expect("discount decimal");
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from(vec!["A", "A", "A", "R"])),
            Arc::new(StringArray::from(vec!["F", "F", "O", "F"])),
            Arc::new(Int64Array::from(vec![10, 20, 30, 40])),
            Arc::new(price),
            Arc::new(discount),
            Arc::new(Date32Array::from(vec![10, 11, 12, 13])),
        ],
    )
    .expect("batch");
    db.register_batches("lineitem", schema, vec![batch])
        .expect("table");

    assert_eq!(
        rows(
            db.run(
                "select returnflag, linestatus, \
                        sum(quantity) as sum_qty, \
                        sum(extendedprice) as sum_base_price, \
                        sum(extendedprice * discount) as sum_disc_price, \
                        avg(extendedprice) as avg_price, \
                        count(*) as count_order \
                 from lineitem \
                 where shipdate <= date '1970-01-13' \
                 group by returnflag, linestatus \
                 order by returnflag, linestatus",
            )
            .await
            .expect("query")
        ),
        vec![
            vec![
                "A".to_string(),
                "F".to_string(),
                "30".to_string(),
                "Some(3000),22,2".to_string(),
                "Some(50000),27,4".to_string(),
                "Some(15000000),16,6".to_string(),
                "2".to_string()
            ],
            vec![
                "A".to_string(),
                "O".to_string(),
                "30".to_string(),
                "Some(3000),22,2".to_string(),
                "Some(30000),27,4".to_string(),
                "Some(30000000),16,6".to_string(),
                "1".to_string()
            ],
        ]
    );

    let trace = db.debug_last_trace().expect("trace");
    assert!(
        trace
            .physical_plan
            .contains("CompiledGlobalGroupAggregateExec"),
        "{}",
        trace.physical_plan
    );
    assert!(
        trace
            .pipeline_candidates
            .iter()
            .any(
                |candidate| candidate.node == "CompiledGlobalGroupAggregateExec"
                    && candidate.kind == PipelineKind::Aggregate
                    && candidate.compiled
                    && candidate.source == "arrow_batch"
                    && candidate.stages == vec!["filter"]
                    && candidate.sink == "group_aggregate"
                    && candidate.output_mode == Some("final_values")
                    && candidate.backend.as_deref() == Some("mlir")
                    && candidate.reason == "compiled"
            ),
        "{:?}",
        trace.pipeline_candidates
    );
}

#[tokio::test]
async fn f64_plain_sum_mlir_execution_preserves_empty_sum_null() {
    let db = database_with_mlir_execution();
    let schema = Arc::new(Schema::new(vec![
        Field::new("v", DataType::Int64, false),
        Field::new("price", DataType::Float64, false),
        Field::new("discount", DataType::Float64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from(vec![9, 11, 12])),
            Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0])),
            Arc::new(Float64Array::from(vec![0.1, 0.2, 0.3])),
        ],
    )
    .expect("batch");
    db.register_batches("t", schema, vec![batch])
        .expect("table");

    let output = db
        .run("select sum(price * discount) from t where v > 100")
        .await
        .expect("query");
    let values = output.batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Float64Array>()
        .expect("f64 sum");
    assert_eq!(values.len(), 1);
    assert!(values.is_null(0));

    let trace = db.debug_last_trace().expect("trace");
    assert!(
        trace
            .jit_candidates
            .iter()
            .any(|candidate| candidate.kernel.name() == "filter_sum"
                && candidate.backend == "mlir"
                && candidate.executable),
        "{:?}",
        trace.jit_candidates
    );
}

#[tokio::test]
async fn parquet_q6_shape_uses_decimal_plain_sum_candidate() {
    let dir = TempDir::new().expect("temp dir");
    let parquet_path = dir.path().join("lineitem.parquet");
    write_q6_lineitem_parquet(parquet_path.to_str().unwrap()).await;

    let db = database_with_mlir_execution();
    db.register_parquet("lineitem", parquet_path.to_str().unwrap())
        .await
        .expect("register parquet");

    let output = db
        .run(
            "select sum(l_extendedprice * l_discount) as revenue \
             from lineitem \
             where l_shipdate >= date '1994-01-01' \
               and l_shipdate < date '1995-01-01' \
               and l_discount between cast(0.05 as decimal(15,2)) \
                                  and cast(0.07 as decimal(15,2)) \
               and l_quantity < cast(24.00 as decimal(15,2))",
        )
        .await
        .expect("query");
    let values = output.batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Decimal128Array>()
        .expect("decimal sum");
    assert_eq!(values.len(), 1);
    assert_eq!(values.value(0), 210_000);
    assert_eq!(values.scale(), 4);

    let trace = db.debug_last_trace().expect("trace");
    assert!(
        trace.physical_plan.contains("CompiledPipelineExec"),
        "{}",
        trace.physical_plan
    );
    assert!(
        trace
            .pipeline_candidates
            .iter()
            .any(|candidate| candidate.node == "CompiledPipelineExec"
                && candidate.kind == PipelineKind::Aggregate
                && candidate.compiled
                && candidate.source == "arrow_batch"
                && candidate.stages == vec!["filter"]
                && candidate.sink == "scalar_sum"
                && candidate.backend.as_deref() == Some("mlir")
                && candidate.reason == "compiled"),
        "{:?}",
        trace.pipeline_candidates
    );
    assert!(
        trace
            .jit_candidates
            .iter()
            .any(|candidate| candidate.kernel.name() == "filter_sum"
                && candidate.backend == "mlir"
                && candidate.executable),
        "{:?}",
        trace.jit_candidates
    );
}

#[tokio::test]
async fn parquet_q6_mlir_execution_returns_null_for_empty_sum() {
    let dir = TempDir::new().expect("temp dir");
    let parquet_path = dir.path().join("lineitem.parquet");
    write_q6_lineitem_parquet(parquet_path.to_str().unwrap()).await;

    let db = database_with_mlir_execution();
    db.register_parquet("lineitem", parquet_path.to_str().unwrap())
        .await
        .expect("register parquet");

    let output = db
        .run(
            "select sum(l_extendedprice * l_discount) as revenue \
             from lineitem \
             where l_shipdate < date '1994-01-01' \
               and l_discount between cast(0.05 as decimal(15,2)) \
                                  and cast(0.07 as decimal(15,2)) \
               and l_quantity < cast(24.00 as decimal(15,2))",
        )
        .await
        .expect("query");
    let values = output.batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Decimal128Array>()
        .expect("decimal sum");
    assert_eq!(values.len(), 1);
    assert!(values.is_null(0));

    let trace = db.debug_last_trace().expect("trace");
    assert!(
        trace
            .jit_candidates
            .iter()
            .any(|candidate| candidate.kernel.name() == "filter_sum"
                && candidate.backend == "mlir"
                && candidate.executable),
        "{:?}",
        trace.jit_candidates
    );
}

fn database_with_mlir_execution() -> Database {
    Database::new(DatabaseOptions {
        jit: JitOptions::mlir_execution(),
        ..Default::default()
    })
    .expect("database")
}

async fn write_people_parquet(path: &str) {
    let ctx = SessionContext::new();
    ctx.sql(
        "select 1::bigint as id, 'ada' as name \
         union all select 2::bigint, 'bob' \
         union all select 3::bigint, 'cara'",
    )
    .await
    .expect("build dataframe")
    .write_parquet(path, DataFrameWriteOptions::new(), None)
    .await
    .expect("write parquet");
}

async fn write_q6_lineitem_parquet(path: &str) {
    let ctx = SessionContext::new();
    ctx.sql(
        "select date '1994-02-01' as l_shipdate, \
                cast(300.00 as decimal(15,2)) as l_extendedprice, \
                cast(0.07 as decimal(15,2)) as l_discount, \
                cast(20.00 as decimal(15,2)) as l_quantity \
         union all select date '1994-03-01', \
                cast(100.00 as decimal(15,2)), \
                cast(0.06 as decimal(15,2)), \
                cast(25.00 as decimal(15,2)) \
         union all select date '1994-04-01', \
                cast(200.00 as decimal(15,2)), \
                cast(0.04 as decimal(15,2)), \
                cast(10.00 as decimal(15,2)) \
         union all select date '1995-01-01', \
                cast(400.00 as decimal(15,2)), \
                cast(0.05 as decimal(15,2)), \
                cast(10.00 as decimal(15,2))",
    )
    .await
    .expect("build q6 dataframe")
    .write_parquet(path, DataFrameWriteOptions::new(), None)
    .await
    .expect("write q6 parquet");
}
