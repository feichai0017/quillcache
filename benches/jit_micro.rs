use std::sync::Arc;

use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion};
use datafusion::arrow::array::{Float64Array, Int64Array};
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::arrow::record_batch::RecordBatch;
use quill_core::database::{Database, DatabaseOptions};
use quill_jit::{FixedColumnInput, RecordPipelineOutput};
use quill_jit::{JitOptions, MlirBackend, PipelineLowering};
use quill_plan::{
    AggregateFunc, GroupAggregate, JitBinaryOp, JitExpr, JitProjection, JitScalar, JitType,
    PipelineGraph, PipelineStage,
};
use quill_runtime::{GroupAggregateKernel, GroupAggregateState, GroupAggregateStateField};

fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("v", DataType::Int64, false),
    ]))
}

fn sum_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("v", DataType::Int64, false),
        Field::new("price", DataType::Float64, false),
        Field::new("discount", DataType::Float64, false),
    ]))
}

fn predicate() -> JitExpr {
    JitExpr::Binary {
        op: JitBinaryOp::Gt,
        left: Box::new(JitExpr::Column {
            index: 1,
            name: "v".to_string(),
            ty: JitType::Int64,
            nullable: false,
        }),
        right: Box::new(JitExpr::Literal(JitScalar::Int64(500))),
        ty: JitType::Bool,
        nullable: false,
    }
}

fn projections() -> Vec<JitProjection> {
    vec![JitProjection::new(
        JitExpr::Binary {
            op: JitBinaryOp::Add,
            left: Box::new(JitExpr::Column {
                index: 0,
                name: "id".to_string(),
                ty: JitType::Int64,
                nullable: false,
            }),
            right: Box::new(JitExpr::Literal(JitScalar::Int64(1))),
            ty: JitType::Int64,
            nullable: false,
        },
        "next_id",
    )]
}

fn benchmark_database() -> Database {
    Database::new(DatabaseOptions {
        debug_trace: false,
        jit: JitOptions::from_env(),
        ..Default::default()
    })
    .expect("database")
}

fn sum_predicate() -> JitExpr {
    JitExpr::Binary {
        op: JitBinaryOp::Gt,
        left: Box::new(JitExpr::Column {
            index: 0,
            name: "v".to_string(),
            ty: JitType::Int64,
            nullable: false,
        }),
        right: Box::new(JitExpr::Literal(JitScalar::Int64(500))),
        ty: JitType::Bool,
        nullable: false,
    }
}

fn measure() -> JitExpr {
    JitExpr::Binary {
        op: JitBinaryOp::Mul,
        left: Box::new(JitExpr::Column {
            index: 1,
            name: "price".to_string(),
            ty: JitType::Float64,
            nullable: false,
        }),
        right: Box::new(JitExpr::Column {
            index: 2,
            name: "discount".to_string(),
            ty: JitType::Float64,
            nullable: false,
        }),
        ty: JitType::Float64,
        nullable: false,
    }
}

fn q6_decimal_predicate() -> JitExpr {
    and(
        and(
            compare(JitBinaryOp::GtEq, date_col(0, "shipdate"), date_lit(10)),
            compare(JitBinaryOp::Lt, date_col(0, "shipdate"), date_lit(20)),
        ),
        and(
            and(
                compare(
                    JitBinaryOp::GtEq,
                    decimal_col(2, "discount", 2),
                    decimal_lit(5, 15, 2),
                ),
                compare(
                    JitBinaryOp::LtEq,
                    decimal_col(2, "discount", 2),
                    decimal_lit(7, 15, 2),
                ),
            ),
            compare(
                JitBinaryOp::Lt,
                decimal_col(3, "quantity", 2),
                decimal_lit(2_400, 15, 2),
            ),
        ),
    )
}

fn q6_decimal_measure() -> JitExpr {
    JitExpr::Binary {
        op: JitBinaryOp::Mul,
        left: Box::new(decimal_col(1, "extendedprice", 2)),
        right: Box::new(decimal_col(2, "discount", 2)),
        ty: JitType::Decimal128 {
            precision: 38,
            scale: 4,
        },
        nullable: false,
    }
}

fn and(left: JitExpr, right: JitExpr) -> JitExpr {
    JitExpr::Binary {
        op: JitBinaryOp::And,
        left: Box::new(left),
        right: Box::new(right),
        ty: JitType::Bool,
        nullable: false,
    }
}

fn compare(op: JitBinaryOp, left: JitExpr, right: JitExpr) -> JitExpr {
    JitExpr::Binary {
        op,
        left: Box::new(left),
        right: Box::new(right),
        ty: JitType::Bool,
        nullable: false,
    }
}

fn date_col(index: usize, name: &str) -> JitExpr {
    JitExpr::Column {
        index,
        name: name.to_string(),
        ty: JitType::Date32,
        nullable: false,
    }
}

fn date_lit(value: i32) -> JitExpr {
    JitExpr::Literal(JitScalar::Date32(value))
}

fn decimal_col(index: usize, name: &str, scale: i8) -> JitExpr {
    JitExpr::Column {
        index,
        name: name.to_string(),
        ty: JitType::Decimal128 {
            precision: 15,
            scale,
        },
        nullable: false,
    }
}

fn decimal_lit(value: i128, precision: u8, scale: i8) -> JitExpr {
    JitExpr::Literal(JitScalar::Decimal128 {
        value,
        precision,
        scale,
    })
}

fn bench_pipeline_graph_and_mlir(c: &mut Criterion) {
    let predicate = predicate();
    let projections = projections();
    let backend = MlirBackend::new();

    c.bench_function("lowering/filter_project_graph", |b| {
        b.iter(|| {
            let pipeline = PipelineGraph::record(vec![
                PipelineStage::Filter(black_box(predicate.clone())),
                PipelineStage::Projection(black_box(projections.clone())),
            ]);
            black_box(PipelineLowering::from_graph(&pipeline))
        });
    });

    c.bench_function("lowering/mlir_record_dialect", |b| {
        b.iter(|| {
            let pipeline = PipelineGraph::record(vec![
                PipelineStage::Filter(black_box(predicate.clone())),
                PipelineStage::Projection(black_box(projections.clone())),
            ]);
            black_box(
                backend
                    .lower_graph_to_quill_mlir("bench_record_pipeline", &pipeline)
                    .expect("lower graph to dialect"),
            )
        });
    });

    c.bench_function("compile/mlir_record_pipeline", |b| {
        b.iter(|| {
            black_box(
                backend
                    .compile_record_pipeline(black_box(&predicate), black_box(&projections))
                    .expect("compile record pipeline"),
            )
        });
    });
    c.bench_function("compile/mlir_f64_plain_sum", |b| {
        let measure = measure();
        let predicate = sum_predicate();
        b.iter(|| {
            black_box(
                backend
                    .compile_plain_sum(black_box(&predicate), black_box(&measure))
                    .expect("compile f64 plain sum"),
            )
        });
    });
    c.bench_function("compile/mlir_decimal_plain_sum", |b| {
        let predicate = q6_decimal_predicate();
        let measure = q6_decimal_measure();
        b.iter(|| {
            black_box(
                backend
                    .compile_plain_sum(black_box(&predicate), black_box(&measure))
                    .expect("compile decimal plain sum"),
            )
        });
    });
    c.bench_function("compile/mlir_group_aggregate_dense_update", |b| {
        let key = group_key();
        let aggregates = group_aggregates();
        b.iter(|| {
            black_box(
                backend
                    .compile_group_aggregate_update(
                        black_box(std::slice::from_ref(&key)),
                        black_box(&aggregates),
                    )
                    .expect("compile group aggregate update"),
            )
        });
    });
}

fn bench_datafusion_filter_project(c: &mut Criterion) {
    let runtime = tokio::runtime::Runtime::new().expect("tokio runtime");
    let db = benchmark_database();
    let input_schema = schema();
    let row_count = 65_536_i64;
    let ids = (0..row_count).collect::<Vec<_>>();
    let values = (0..row_count)
        .map(|value| value % 1_000)
        .collect::<Vec<_>>();
    let batch = RecordBatch::try_new(
        Arc::clone(&input_schema),
        vec![
            Arc::new(Int64Array::from(ids)),
            Arc::new(Int64Array::from(values)),
        ],
    )
    .expect("record batch");
    db.register_batches("t", input_schema, vec![batch])
        .expect("register table");

    runtime
        .block_on(db.run("select id + 1 as next_id from t where v > 500"))
        .expect("warmup");

    c.bench_function("sql/df/filter_project_64k", |b| {
        b.iter(|| {
            black_box(
                runtime
                    .block_on(db.run(black_box("select id + 1 as next_id from t where v > 500")))
                    .expect("query"),
            )
        });
    });
}

fn bench_datafusion_filter_sum(c: &mut Criterion) {
    let runtime = tokio::runtime::Runtime::new().expect("tokio runtime");
    let db = benchmark_database();
    let input_schema = sum_schema();
    let row_count = 65_536_i64;
    let values = (0..row_count)
        .map(|value| value % 1_000)
        .collect::<Vec<_>>();
    let prices = (0..row_count)
        .map(|value| 100.0 + (value % 10) as f64)
        .collect::<Vec<_>>();
    let discounts = (0..row_count)
        .map(|value| 0.01 * ((value % 7) as f64))
        .collect::<Vec<_>>();
    let batch = RecordBatch::try_new(
        Arc::clone(&input_schema),
        vec![
            Arc::new(Int64Array::from(values)),
            Arc::new(Float64Array::from(prices)),
            Arc::new(Float64Array::from(discounts)),
        ],
    )
    .expect("record batch");
    db.register_batches("t", input_schema, vec![batch])
        .expect("register table");

    runtime
        .block_on(db.run("select sum(price * discount) from t where v > 500"))
        .expect("warmup");
    let prepared = runtime
        .block_on(db.prepare("select sum(price * discount) from t where v > 500"))
        .expect("prepare");
    runtime.block_on(prepared.run()).expect("prepared warmup");

    c.bench_function("sql/df/filter_sum_64k", |b| {
        b.iter(|| {
            black_box(
                runtime
                    .block_on(db.run(black_box(
                        "select sum(price * discount) from t where v > 500",
                    )))
                    .expect("query"),
            )
        });
    });
    c.bench_function("sql/df/prepared_filter_sum_64k", |b| {
        b.iter(|| black_box(runtime.block_on(prepared.run()).expect("query")));
    });
}

fn bench_compiled_record_pipeline_kernel(c: &mut Criterion) {
    let row_count = 65_536_i64;
    let ids = (0..row_count).collect::<Vec<_>>();
    let values = (0..row_count)
        .map(|value| value % 1_000)
        .collect::<Vec<_>>();
    let mut output = vec![0_i64; values.len()];
    let kernel = MlirBackend::new()
        .compile_record_pipeline(&predicate(), &projections())
        .expect("compiled record pipeline");

    c.bench_function("kernel/record_pipeline_64k", |b| {
        b.iter(|| {
            let output_len = {
                let mut outputs = [RecordPipelineOutput::Int64 {
                    values: output.as_mut_slice(),
                }];
                kernel
                    .invoke(
                        black_box(&[
                            FixedColumnInput::Int64 {
                                index: 0,
                                values: ids.as_slice(),
                            },
                            FixedColumnInput::Int64 {
                                index: 1,
                                values: values.as_slice(),
                            },
                        ]),
                        black_box(&mut outputs),
                    )
                    .expect("execute compiled record pipeline")
            };
            black_box(output_len);
            black_box(&output[..output_len]);
        });
    });
}

fn bench_compiled_f64_plain_sum_kernel(c: &mut Criterion) {
    let row_count = 65_536_i64;
    let predicate_values = (0..row_count)
        .map(|value| value % 1_000)
        .collect::<Vec<_>>();
    let prices = (0..row_count)
        .map(|value| 100.0 + (value % 10) as f64)
        .collect::<Vec<_>>();
    let discounts = (0..row_count)
        .map(|value| 0.01 * ((value % 7) as f64))
        .collect::<Vec<_>>();
    let kernel = MlirBackend::new()
        .compile_plain_sum(&sum_predicate(), &measure())
        .expect("compiled f64 plain sum");

    c.bench_function("kernel/f64_plain_sum_64k", |b| {
        b.iter(|| {
            black_box(
                kernel
                    .invoke(black_box(&[
                        FixedColumnInput::Int64 {
                            index: 0,
                            values: predicate_values.as_slice(),
                        },
                        FixedColumnInput::Float64 {
                            index: 1,
                            values: prices.as_slice(),
                        },
                        FixedColumnInput::Float64 {
                            index: 2,
                            values: discounts.as_slice(),
                        },
                    ]))
                    .expect("execute compiled plain sum"),
            );
        });
    });
}

fn bench_compiled_decimal_plain_sum_kernel(c: &mut Criterion) {
    let row_count = 65_536_i32;
    let shipdates = (0..row_count)
        .map(|value| 10 + (value % 12))
        .collect::<Vec<_>>();
    let prices = (0..row_count)
        .map(|value| 10_000_i128 + i128::from(value % 1_000))
        .collect::<Vec<_>>();
    let discounts = (0..row_count)
        .map(|value| 4_i128 + i128::from(value % 5))
        .collect::<Vec<_>>();
    let quantities = (0..row_count)
        .map(|value| 2_000_i128 + i128::from(value % 600))
        .collect::<Vec<_>>();
    let kernel = MlirBackend::new()
        .compile_plain_sum(&q6_decimal_predicate(), &q6_decimal_measure())
        .expect("compiled decimal plain sum");

    c.bench_function("kernel/decimal_plain_sum_64k", |b| {
        b.iter(|| {
            black_box(
                kernel
                    .invoke(&[
                        FixedColumnInput::Date32 {
                            index: 0,
                            values: black_box(shipdates.as_slice()),
                        },
                        FixedColumnInput::Decimal128 {
                            index: 1,
                            values: black_box(prices.as_slice()),
                        },
                        FixedColumnInput::Decimal128 {
                            index: 2,
                            values: black_box(discounts.as_slice()),
                        },
                        FixedColumnInput::Decimal128 {
                            index: 3,
                            values: black_box(quantities.as_slice()),
                        },
                    ])
                    .expect("execute compiled decimal plain sum"),
            );
        });
    });
}

fn bench_compiled_group_aggregate_dense_update_kernel(c: &mut Criterion) {
    let row_count = 65_536_i64;
    let group_count = 1_024_usize;
    let group_ids = (0..row_count)
        .map(|value| value % group_count as i64)
        .collect::<Vec<_>>();
    let values = (0..row_count)
        .map(|value| value % 1_000)
        .collect::<Vec<_>>();
    let key = group_key();
    let aggregates = group_aggregates();
    let kernel = MlirBackend::new()
        .compile_group_aggregate_update(&[key], &aggregates)
        .expect("compiled group aggregate update");
    let mut state_fields = vec![
        int64_state(group_count),
        int64_state(group_count),
        uint64_state(group_count),
        int64_state(group_count),
    ];

    c.bench_function("kernel/group_aggregate_dense_update_64k", |b| {
        b.iter(|| {
            reset_states(&mut state_fields);
            kernel
                .invoke(
                    black_box(group_ids.as_slice()),
                    black_box(&[FixedColumnInput::Int64 {
                        index: 1,
                        values: values.as_slice(),
                    }]),
                    black_box(&mut state_fields),
                )
                .expect("execute compiled group aggregate update");
            black_box(&state_fields);
        });
    });
}

fn bench_group_aggregate_runtime_boundaries(c: &mut Criterion) {
    let row_count = 65_536_i64;
    let group_count = 1_024_usize;
    let batch = group_batch(row_count, group_count);
    let kernel = group_kernel();

    c.bench_function("runtime/group_aggregate_bind_build_64k", |b| {
        b.iter(|| {
            let mut state = kernel.new_state();
            black_box(
                kernel
                    .bind_batch(&mut state, black_box(&batch))
                    .expect("bind group aggregate batch"),
            );
            black_box(state);
        });
    });

    let mut probe_state = kernel.new_state();
    kernel
        .bind_batch(&mut probe_state, &batch)
        .expect("seed group aggregate state");
    c.bench_function("runtime/group_aggregate_bind_probe_64k", |b| {
        b.iter(|| {
            black_box(
                kernel
                    .bind_batch(black_box(&mut probe_state), black_box(&batch))
                    .expect("probe group aggregate batch"),
            );
        });
    });

    c.bench_function("runtime/group_aggregate_finish_dense_1k_groups", |b| {
        b.iter_batched(
            || seeded_dense_group_state(&kernel, &batch),
            |state| black_box(kernel.finish(state).expect("finish dense group aggregate")),
            BatchSize::SmallInput,
        );
    });
}

fn group_batch(row_count: i64, group_count: usize) -> RecordBatch {
    let keys = (0..row_count)
        .map(|value| value % group_count as i64)
        .collect::<Vec<_>>();
    let values = (0..row_count)
        .map(|value| value % 1_000)
        .collect::<Vec<_>>();
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int64, false),
            Field::new("v", DataType::Int64, false),
        ])),
        vec![
            Arc::new(Int64Array::from(keys)),
            Arc::new(Int64Array::from(values)),
        ],
    )
    .expect("group aggregate batch")
}

fn group_kernel() -> GroupAggregateKernel {
    GroupAggregateKernel::try_new(
        &[],
        vec![group_key()],
        group_aggregates(),
        Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int64, false),
            Field::new("sum_v", DataType::Int64, true),
            Field::new("count_star", DataType::Int64, true),
            Field::new("avg_count", DataType::UInt64, true),
            Field::new("avg_v", DataType::Int64, true),
        ])),
    )
    .expect("group aggregate kernel")
}

fn seeded_dense_group_state(
    kernel: &GroupAggregateKernel,
    batch: &RecordBatch,
) -> GroupAggregateState {
    let mut state = kernel.new_state();
    kernel
        .bind_batch(&mut state, batch)
        .expect("bind group aggregate batch");
    let dense = kernel
        .dense_state_mut(&mut state)
        .expect("dense group aggregate state");
    for field in dense.fields_mut() {
        match field {
            GroupAggregateStateField::Int64 { values, valid } => {
                values.fill(1);
                valid.fill(1);
            }
            GroupAggregateStateField::UInt64 { values, valid } => {
                values.fill(1);
                valid.fill(1);
            }
            GroupAggregateStateField::Float64 { values, valid } => {
                values.fill(1.0);
                valid.fill(1);
            }
            GroupAggregateStateField::Decimal128 { values, valid, .. } => {
                values.fill(1);
                valid.fill(1);
            }
        }
    }
    state
}

fn group_key() -> JitExpr {
    JitExpr::Column {
        index: 0,
        name: "k".to_string(),
        ty: JitType::Int64,
        nullable: false,
    }
}

fn group_aggregates() -> Vec<GroupAggregate> {
    let value = JitExpr::Column {
        index: 1,
        name: "v".to_string(),
        ty: JitType::Int64,
        nullable: false,
    };
    vec![
        GroupAggregate::new(AggregateFunc::Sum, value.clone(), JitType::Int64, "sum_v"),
        GroupAggregate::new(
            AggregateFunc::Count,
            JitExpr::Literal(JitScalar::Int64(1)),
            JitType::Int64,
            "count_star",
        ),
        GroupAggregate::new_with_states(
            AggregateFunc::Avg,
            value,
            vec![JitType::UInt64, JitType::Int64],
            "avg_v",
        ),
    ]
}

fn int64_state(len: usize) -> GroupAggregateStateField {
    GroupAggregateStateField::Int64 {
        values: vec![0; len],
        valid: vec![0; len],
    }
}

fn uint64_state(len: usize) -> GroupAggregateStateField {
    GroupAggregateStateField::UInt64 {
        values: vec![0; len],
        valid: vec![0; len],
    }
}

fn reset_states(fields: &mut [GroupAggregateStateField]) {
    for field in fields {
        match field {
            GroupAggregateStateField::Int64 { values, valid } => {
                values.fill(0);
                valid.fill(0);
            }
            GroupAggregateStateField::UInt64 { values, valid } => {
                values.fill(0);
                valid.fill(0);
            }
            GroupAggregateStateField::Float64 { values, valid } => {
                values.fill(0.0);
                valid.fill(0);
            }
            GroupAggregateStateField::Decimal128 { values, valid, .. } => {
                values.fill(0);
                valid.fill(0);
            }
        }
    }
}

criterion_group!(
    benches,
    bench_pipeline_graph_and_mlir,
    bench_compiled_record_pipeline_kernel,
    bench_compiled_f64_plain_sum_kernel,
    bench_compiled_decimal_plain_sum_kernel,
    bench_compiled_group_aggregate_dense_update_kernel,
    bench_group_aggregate_runtime_boundaries,
    bench_datafusion_filter_project,
    bench_datafusion_filter_sum
);
criterion_main!(benches);
