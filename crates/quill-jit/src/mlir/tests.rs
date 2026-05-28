use crate::{AggregateFunc, FilterSumValue, FixedColumnInput, GroupAggregate};
use crate::{
    JitBinaryOp, JitExpr, JitProjection, JitScalar, JitType, MlirBackend, PipelineGraph,
    PipelineKind, PipelineStage,
};

#[test]
fn emits_textual_filter_module() {
    let expr = i64_gt_ten(true);

    let module = MlirBackend::new().lower_filter(&expr).unwrap();
    assert!(module.text.contains("func.func @quill_filter_"));
    assert!(module.text.contains("arith.cmpi sgt"));
    assert!(module.text.contains("qjit.kind = filter"));
    assert!(module
        .text
        .contains("col(0, a, i64, nullable=true) > 10:i64"));
}

#[test]
fn backend_verifies_generated_module() {
    let expr = i64_gt_ten(true);

    let module = MlirBackend::new().lower_filter(&expr).unwrap();
    MlirBackend::new().verify_module(&module).unwrap();
}

#[test]
fn emits_filter_project_module() {
    let predicate = i64_gt_ten(true);
    let projection = JitProjection::new(
        JitExpr::Binary {
            op: JitBinaryOp::Add,
            left: Box::new(JitExpr::Column {
                index: 0,
                name: "a".to_string(),
                ty: JitType::Int64,
                nullable: true,
            }),
            right: Box::new(JitExpr::Literal(JitScalar::Int64(1))),
            ty: JitType::Int64,
            nullable: true,
        },
        "a_plus_one",
    );

    let module = MlirBackend::new()
        .lower_filter_project(&predicate, &[projection])
        .unwrap();
    assert!(module.text.contains("qjit.kind = filter_project"));
    assert!(module.text.contains("arith.cmpi sgt"));
    assert!(module.text.contains("arith.addi"));
    MlirBackend::new().verify_module(&module).unwrap();
}

#[test]
fn emits_quill_dialect_pipeline_skeleton() {
    let predicate = i64_gt_ten(false);
    let projections = vec![i64_plus_one_projection(0)];
    let pipeline = PipelineGraph::record(vec![
        PipelineStage::Filter(predicate),
        PipelineStage::Projection(projections),
    ]);

    let module = MlirBackend::new().emit_quill_dialect("record_pipeline", &pipeline);
    let text = module.to_mlir_text().unwrap();

    assert_eq!(module.kind, PipelineKind::Record);
    assert_eq!(
        module.pipeline_spec().map(|spec| spec.name()),
        Some("record_project")
    );
    assert!(text.contains("func.func @record_pipeline"));
    assert!(text.contains("// qjit.pipeline = record_project"));
    assert!(text.contains("quill.exec.filter"));
    assert!(text.contains("quill.exec.project"));
    assert!(!text.contains("predicate ="));
}

#[test]
fn emits_q6_quill_dialect_pipeline_spec() {
    let pipeline = PipelineGraph::filter_sum(q6_decimal_predicate(), q6_decimal_measure());
    let module = MlirBackend::new().emit_quill_dialect("q6_pipeline", &pipeline);
    let text = module.to_mlir_text().unwrap();

    assert_eq!(
        module.pipeline_spec().map(|spec| spec.name()),
        Some("plain_sum")
    );
    assert!(text.contains("// qjit.pipeline = plain_sum"));
    assert!(text.contains("quill.sink.plain_sum"));
    assert!(!text.contains("measure ="));
}

#[test]
fn verifies_formal_quill_dialect_pipeline() {
    let pipeline = PipelineGraph::filter_sum(q6_decimal_predicate(), q6_decimal_measure());
    let module = MlirBackend::new()
        .lower_graph_to_quill_mlir("q6_quill_region", &pipeline)
        .unwrap();

    assert!(module.text.contains("quill.column"));
    assert!(module.text.contains("quill.yield"));
    MlirBackend::new().verify_module(&module).unwrap();
}

#[test]
fn verifies_formal_group_aggregate_dialect_pipeline() {
    let key = JitExpr::Column {
        index: 0,
        name: "group_key".to_string(),
        ty: JitType::Int64,
        nullable: false,
    };
    let aggregate = GroupAggregate::new(
        AggregateFunc::Sum,
        JitExpr::Column {
            index: 1,
            name: "measure".to_string(),
            ty: JitType::Float64,
            nullable: false,
        },
        JitType::Float64,
        "sum_measure",
    );
    let pipeline = PipelineGraph::group_aggregate(vec![], vec![key], vec![aggregate]);
    let module = MlirBackend::new()
        .lower_graph_to_quill_mlir("group_quill_region", &pipeline)
        .unwrap();

    assert!(module.text.contains("quill.sink.group_aggregate"));
    MlirBackend::new().verify_module(&module).unwrap();
}

#[test]
fn rejects_invalid_quill_filter_region_result() {
    let text = r#"
module {
  func.func @bad_filter() {
    %batch = quill.source.arrow_batch : !quill.batch
    %sel = quill.exec.filter %batch {
    ^bb0(%row: !quill.row):
      %v = quill.column %row { index = 0 : i64 } : !quill.row -> i64
      quill.yield %v : i64
    } : !quill.batch -> !quill.selection
    quill.sink.record_batch %batch, %sel : !quill.batch, !quill.selection
    return
  }
}
"#;

    let module = super::MlirModule {
        symbol: "bad_filter".to_string(),
        text: text.to_string(),
    };

    assert!(MlirBackend::new().verify_module(&module).is_err());
}

#[test]
fn rejects_invalid_quill_group_aggregate_region() {
    let text = r#"
module {
  func.func @bad_group() {
    %batch = quill.source.arrow_batch : !quill.batch
    %sel = quill.exec.filter %batch {
    ^bb0(%row: !quill.row):
      %ok = arith.constant true
      quill.yield %ok : i1
    } : !quill.batch -> !quill.selection
    %out = quill.sink.group_aggregate %batch, %sel {
    ^bb0(%row: !quill.row):
      %k = quill.column %row { index = 0 : i64 } : !quill.row -> i64
      quill.yield %k : i64
    } : !quill.batch, !quill.selection -> !quill.batch
    return
  }
}
"#;

    let module = super::MlirModule {
        symbol: "bad_group".to_string(),
        text: text.to_string(),
    };

    assert!(MlirBackend::new().verify_module(&module).is_err());
}

#[test]
fn lowers_q6_quill_dialect_to_mlir() {
    let pipeline = PipelineGraph::filter_sum(q6_decimal_predicate(), q6_decimal_measure());
    let dialect = MlirBackend::new().emit_quill_dialect("q6_decimal_pipeline", &pipeline);

    let module = MlirBackend::new().lower_quill_dialect(&dialect).unwrap();

    assert_eq!(module.symbol, "q6_decimal_pipeline");
    assert!(module.text.contains("func.func @q6_decimal_pipeline"));
    assert!(module.text.contains("i128"));
    assert!(module.text.contains("llvm.emit_c_interface"));
    assert!(module.text.contains("scf.for"));
    assert!(!module.text.contains("quill."));
    MlirBackend::new().verify_module(&module).unwrap();
}

#[test]
fn emits_i64_predicate_module() {
    let predicate = i64_gt_ten(false);

    let module = MlirBackend::new().lower_i64_predicate(&predicate).unwrap();
    assert!(module.text.contains("llvm.emit_c_interface"));
    assert!(module.text.contains("arith.select"));
    MlirBackend::new().verify_module(&module).unwrap();
}

#[test]
fn emits_i64_filter_module() {
    let predicate = i64_gt_ten(false);

    let module = MlirBackend::new().lower_i64_filter(&predicate).unwrap();
    assert!(module.text.contains("func.func @quill_i64_filter_"));
    assert!(module.text.contains("scf.for"));
    assert!(module.text.contains("llvm.load"));
    assert!(module.text.contains("llvm.store"));
    MlirBackend::new().verify_module(&module).unwrap();
}

#[test]
fn emits_record_pipeline_module() {
    let predicate = i64_gt_ten(false);
    let projections = vec![i64_plus_one_projection(0)];

    let module = MlirBackend::new()
        .lower_record_pipeline(&predicate, &projections)
        .unwrap();
    assert!(module.text.contains("func.func @quill_record_pipeline_"));
    assert!(module.text.contains("llvm.emit_c_interface"));
    assert!(module.text.contains("scf.for"));
    assert!(!module.text.contains("quill."));
    assert!(module.text.contains("scf.if"));
    assert!(module.text.contains("llvm.load"));
    assert!(module.text.contains("llvm.store"));
    MlirBackend::new().verify_module(&module).unwrap();
}

#[test]
fn emits_f64_plain_sum_module() {
    let predicate = i64_gt_ten(false);
    let measure = f64_product_measure();

    let module = MlirBackend::new()
        .lower_plain_sum(&predicate, &measure)
        .unwrap();
    assert!(module.text.contains("func.func @quill_plain_sum_"));
    assert!(module.text.contains("llvm.emit_c_interface"));
    assert!(!module.text.contains("quill."));
    assert!(module.text.contains("scf.for"));
    assert!(module.text.contains("scf.if"));
    assert!(module.text.contains("arith.mulf"));
    assert!(module.text.contains("arith.addf"));
    assert!(module.text.contains("llvm.store"));
    MlirBackend::new().verify_module(&module).unwrap();
}

#[test]
fn emits_decimal_plain_sum_module() {
    let predicate = q6_decimal_predicate();
    let measure = q6_decimal_measure();

    let module = MlirBackend::new()
        .lower_plain_sum(&predicate, &measure)
        .unwrap();

    assert!(module.text.contains("func.func @quill_plain_sum_"));
    assert!(module.text.contains("llvm.emit_c_interface"));
    assert!(module.text.contains("scf.for"));
    assert!(!module.text.contains("quill."));
    assert!(module.text.contains("scf.if"));
    assert!(module.text.contains("i128"));
    assert!(module.text.contains("arith.muli"));
    assert!(module.text.contains("arith.addi"));
    assert!(module.text.contains("llvm.store"));
    MlirBackend::new().verify_module(&module).unwrap();
}

#[test]
fn invokes_i64_predicate_with_execution_engine() {
    let predicate = i64_gt_ten(false);

    let backend = MlirBackend::new();
    assert!(!backend.invoke_i64_predicate(&predicate, 10).unwrap());
    assert!(backend.invoke_i64_predicate(&predicate, 11).unwrap());
}

#[test]
fn reuses_compiled_i64_predicate_artifact() {
    let predicate = i64_gt_ten(false);
    let module = MlirBackend::new().lower_i64_predicate(&predicate).unwrap();
    let compiled = super::compiled::compile_i64_predicate(&module).unwrap();

    assert!(!compiled.invoke(9).unwrap());
    assert!(!compiled.invoke(10).unwrap());
    assert!(compiled.invoke(11).unwrap());
}

#[test]
fn invokes_compiled_i64_filter_kernel() {
    let predicate = i64_gt_ten(false);
    let compiled = MlirBackend::new().compile_i64_filter(&predicate).unwrap();
    let input = [9_i64, 10, 11, 42];
    let mut output = [255_u8; 4];

    compiled.invoke(&input, &mut output).unwrap();

    assert_eq!(output, [0, 0, 1, 1]);
}

#[test]
fn invokes_compiled_i64_filter_with_nonzero_column_index() {
    let predicate = JitExpr::Binary {
        op: JitBinaryOp::Gt,
        left: Box::new(JitExpr::Column {
            index: 1,
            name: "v".to_string(),
            ty: JitType::Int64,
            nullable: false,
        }),
        right: Box::new(JitExpr::Literal(JitScalar::Int64(10))),
        ty: JitType::Bool,
        nullable: false,
    };
    let compiled = MlirBackend::new().compile_i64_filter(&predicate).unwrap();
    let input = [10_i64, 11, 12];
    let mut output = [0_u8; 3];

    compiled.invoke(&input, &mut output).unwrap();

    assert_eq!(output, [0, 1, 1]);
}

#[test]
fn invokes_compiled_record_pipeline_kernel() {
    let predicate = JitExpr::Binary {
        op: JitBinaryOp::Gt,
        left: Box::new(JitExpr::Column {
            index: 1,
            name: "v".to_string(),
            ty: JitType::Int64,
            nullable: false,
        }),
        right: Box::new(JitExpr::Literal(JitScalar::Int64(10))),
        ty: JitType::Bool,
        nullable: false,
    };
    let projections = vec![i64_plus_one_projection(0)];
    let compiled = MlirBackend::new()
        .compile_record_pipeline(&predicate, &projections)
        .unwrap();
    let predicate_values = [9_i64, 11, 12];
    let projection_values = [100_i64, 200, 300];
    let mut output = [0_i64; 3];
    let output_len = {
        let mut outputs = [crate::RecordPipelineOutput::Int64 {
            values: &mut output,
        }];
        compiled
            .invoke(
                &[
                    crate::FixedColumnInput::Int64 {
                        index: 0,
                        values: &projection_values,
                    },
                    crate::FixedColumnInput::Int64 {
                        index: 1,
                        values: &predicate_values,
                    },
                ],
                &mut outputs,
            )
            .unwrap()
    };

    assert_eq!(output_len, 2);
    assert_eq!(&output[..output_len], [201, 301]);
}

#[test]
fn invokes_compiled_f64_plain_sum_kernel() {
    let predicate = i64_gt_ten(false);
    let measure = f64_product_measure();
    let compiled = MlirBackend::new()
        .compile_plain_sum(&predicate, &measure)
        .unwrap();
    let predicate_values = [9_i64, 11, 12];
    let left_values = [10.0_f64, 20.0, 30.0];
    let right_values = [0.1_f64, 0.2, 0.3];

    let output = compiled
        .invoke(&[
            crate::FixedColumnInput::Int64 {
                index: 0,
                values: &predicate_values,
            },
            crate::FixedColumnInput::Float64 {
                index: 1,
                values: &left_values,
            },
            crate::FixedColumnInput::Float64 {
                index: 2,
                values: &right_values,
            },
        ])
        .unwrap();

    assert!(matches!(
        output,
        FilterSumValue::Float64(Some(value)) if (value - 13.0).abs() < 0.000_001
    ));
}

#[test]
fn invokes_compiled_f64_plain_sum_with_or_predicate() {
    let predicate = or(
        compare(
            JitBinaryOp::Lt,
            JitExpr::Column {
                index: 0,
                name: "a".to_string(),
                ty: JitType::Int64,
                nullable: false,
            },
            JitExpr::Literal(JitScalar::Int64(10)),
        ),
        compare(
            JitBinaryOp::Gt,
            JitExpr::Column {
                index: 0,
                name: "a".to_string(),
                ty: JitType::Int64,
                nullable: false,
            },
            JitExpr::Literal(JitScalar::Int64(11)),
        ),
    );
    let measure = f64_product_measure();
    let compiled = MlirBackend::new()
        .compile_plain_sum(&predicate, &measure)
        .unwrap();
    let predicate_values = [9_i64, 10, 12];
    let left_values = [10.0_f64, 20.0, 30.0];
    let right_values = [0.1_f64, 0.2, 0.3];

    let output = compiled
        .invoke(&[
            crate::FixedColumnInput::Int64 {
                index: 0,
                values: &predicate_values,
            },
            crate::FixedColumnInput::Float64 {
                index: 1,
                values: &left_values,
            },
            crate::FixedColumnInput::Float64 {
                index: 2,
                values: &right_values,
            },
        ])
        .unwrap();

    assert!(matches!(
        output,
        FilterSumValue::Float64(Some(value)) if (value - 10.0).abs() < 0.000_001
    ));
}

#[test]
fn invokes_compiled_decimal_plain_sum_kernel() {
    let predicate = q6_decimal_predicate();
    let measure = q6_decimal_measure();
    let compiled = MlirBackend::new()
        .compile_plain_sum(&predicate, &measure)
        .unwrap();
    let shipdates = [9_i32, 10, 12, 20];
    let prices = [10_000_i128, 20_000, 30_000, 40_000];
    let discounts = [4_i128, 5, 7, 6];
    let quantities = [1_000_i128, 2_500, 2_000, 2_000];

    let output = compiled
        .invoke(&[
            FixedColumnInput::Date32 {
                index: 0,
                values: &shipdates,
            },
            FixedColumnInput::Decimal128 {
                index: 1,
                values: &prices,
            },
            FixedColumnInput::Decimal128 {
                index: 2,
                values: &discounts,
            },
            FixedColumnInput::Decimal128 {
                index: 3,
                values: &quantities,
            },
        ])
        .unwrap();

    assert_eq!(
        output,
        FilterSumValue::Decimal128 {
            value: Some(210_000),
            scale: 4
        }
    );
}

fn i64_gt_ten(nullable: bool) -> JitExpr {
    JitExpr::Binary {
        op: JitBinaryOp::Gt,
        left: Box::new(JitExpr::Column {
            index: 0,
            name: "a".to_string(),
            ty: JitType::Int64,
            nullable,
        }),
        right: Box::new(JitExpr::Literal(JitScalar::Int64(10))),
        ty: JitType::Bool,
        nullable,
    }
}

fn i64_plus_one_projection(index: usize) -> JitProjection {
    JitProjection::new(
        JitExpr::Binary {
            op: JitBinaryOp::Add,
            left: Box::new(JitExpr::Column {
                index,
                name: format!("c{index}"),
                ty: JitType::Int64,
                nullable: false,
            }),
            right: Box::new(JitExpr::Literal(JitScalar::Int64(1))),
            ty: JitType::Int64,
            nullable: false,
        },
        "plus_one",
    )
}

fn f64_product_measure() -> JitExpr {
    JitExpr::Binary {
        op: JitBinaryOp::Mul,
        left: Box::new(JitExpr::Column {
            index: 1,
            name: "left".to_string(),
            ty: JitType::Float64,
            nullable: false,
        }),
        right: Box::new(JitExpr::Column {
            index: 2,
            name: "right".to_string(),
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

fn or(left: JitExpr, right: JitExpr) -> JitExpr {
    JitExpr::Binary {
        op: JitBinaryOp::Or,
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
