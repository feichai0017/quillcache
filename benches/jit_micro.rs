use std::sync::Arc;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use datafusion::arrow::array::{Float64Array, Int64Array};
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::arrow::record_batch::RecordBatch;
use quill_core::database::{Database, DatabaseOptions};
use quill_jit::{FixedColumnInput, RecordPipelineOutput};
use quill_jit::{JitOptions, MlirBackend, PipelineLowering};
use quill_plan::{
    JitBinaryOp, JitExpr, JitProjection, JitScalar, JitType, PipelineGraph, PipelineStage,
};
use quill_runtime::KernelBackend;

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
    let input_schema = schema();
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

    c.bench_function("compile/mlir_filter_text", |b| {
        b.iter(|| {
            black_box(
                backend
                    .compile_filter(black_box(Arc::clone(&input_schema)), black_box(&predicate))
                    .expect("compile filter"),
            )
        });
    });

    c.bench_function("compile/mlir_filter_project_text", |b| {
        b.iter(|| {
            black_box(
                backend
                    .compile_filter_project(
                        black_box(Arc::clone(&input_schema)),
                        black_box(&predicate),
                        black_box(&projections),
                    )
                    .expect("compile filter project"),
            )
        });
    });
    c.bench_function("compile/mlir_i64_filter", |b| {
        b.iter(|| {
            black_box(
                backend
                    .compile_i64_filter(black_box(&predicate))
                    .expect("compile i64 filter"),
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

fn bench_compiled_i64_filter_kernel(c: &mut Criterion) {
    let row_count = 65_536_i64;
    let values = (0..row_count)
        .map(|value| value % 1_000)
        .collect::<Vec<_>>();
    let mut output = vec![0_u8; values.len()];
    let kernel = MlirBackend::new()
        .compile_i64_filter(&predicate())
        .expect("compiled i64 filter");

    c.bench_function("kernel/i64_filter_64k", |b| {
        b.iter(|| {
            kernel
                .invoke(black_box(&values), black_box(&mut output))
                .expect("execute compiled filter");
            black_box(&output);
        });
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

criterion_group!(
    benches,
    bench_pipeline_graph_and_mlir,
    bench_compiled_i64_filter_kernel,
    bench_compiled_record_pipeline_kernel,
    bench_compiled_f64_plain_sum_kernel,
    bench_compiled_decimal_plain_sum_kernel,
    bench_datafusion_filter_project,
    bench_datafusion_filter_sum
);
criterion_main!(benches);
