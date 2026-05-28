use std::sync::Arc;

use arrow::array::{Array, BooleanArray, Date32Array, Decimal128Array, Float64Array, Int64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;

use quill_plan::{JitBinaryOp, JitExpr, JitProjection, JitScalar, JitType};

use super::{FilterProjectKernel, FilterSumKernel, FilterSumValue, FixedColumn, PipelineSpec};

#[test]
fn executes_filter_project_with_nulls() {
    let input_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, true),
        Field::new("v", DataType::Int64, true),
    ]));
    let output_schema = Arc::new(Schema::new(vec![Field::new(
        "next_id",
        DataType::Int64,
        true,
    )]));
    let batch = RecordBatch::try_new(
        input_schema,
        vec![
            Arc::new(Int64Array::from(vec![Some(1), Some(2), Some(3), None])),
            Arc::new(Int64Array::from(vec![Some(10), Some(20), None, Some(40)])),
        ],
    )
    .unwrap();
    let kernel = FilterProjectKernel::try_new(
        JitExpr::Binary {
            op: JitBinaryOp::Gt,
            left: Box::new(JitExpr::Column {
                index: 1,
                name: "v".to_string(),
                ty: JitType::Int64,
                nullable: true,
            }),
            right: Box::new(JitExpr::Literal(JitScalar::Int64(10))),
            ty: JitType::Bool,
            nullable: true,
        },
        vec![JitProjection::new(
            JitExpr::Binary {
                op: JitBinaryOp::Add,
                left: Box::new(JitExpr::Column {
                    index: 0,
                    name: "id".to_string(),
                    ty: JitType::Int64,
                    nullable: true,
                }),
                right: Box::new(JitExpr::Literal(JitScalar::Int64(1))),
                ty: JitType::Int64,
                nullable: true,
            },
            "next_id",
        )],
        output_schema,
    )
    .unwrap();

    assert_eq!(
        kernel.spec(),
        Some(&PipelineSpec::RecordProject {
            columns: vec![
                FixedColumn {
                    index: 0,
                    ty: JitType::Int64
                },
                FixedColumn {
                    index: 1,
                    ty: JitType::Int64
                }
            ],
            output_types: vec![JitType::Int64],
        })
    );

    let output = kernel.execute(&batch).unwrap();
    let values = output
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    assert_eq!(values.len(), 2);
    assert_eq!(values.value(0), 3);
    assert!(values.is_null(1));
}

#[test]
fn implements_sql_three_valued_boolean_logic() {
    let input_schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Boolean, true),
        Field::new("b", DataType::Boolean, true),
    ]));
    let output_schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Boolean, true)]));
    let batch = RecordBatch::try_new(
        input_schema,
        vec![
            Arc::new(BooleanArray::from(vec![
                Some(true),
                Some(true),
                None,
                Some(false),
            ])),
            Arc::new(BooleanArray::from(vec![Some(true), None, Some(true), None])),
        ],
    )
    .unwrap();
    let kernel = FilterProjectKernel::try_new(
        JitExpr::Binary {
            op: JitBinaryOp::Or,
            left: Box::new(JitExpr::Column {
                index: 0,
                name: "a".to_string(),
                ty: JitType::Bool,
                nullable: true,
            }),
            right: Box::new(JitExpr::Column {
                index: 1,
                name: "b".to_string(),
                ty: JitType::Bool,
                nullable: true,
            }),
            ty: JitType::Bool,
            nullable: true,
        },
        vec![JitProjection::new(
            JitExpr::Column {
                index: 0,
                name: "a".to_string(),
                ty: JitType::Bool,
                nullable: true,
            },
            "a",
        )],
        output_schema,
    )
    .unwrap();

    let output = kernel.execute(&batch).unwrap();
    let values = output
        .column(0)
        .as_any()
        .downcast_ref::<BooleanArray>()
        .unwrap();
    assert_eq!(values.len(), 3);
    assert!(values.value(0));
    assert!(values.value(1));
    assert!(values.is_null(2));
}

#[test]
fn executes_plain_sum_with_nulls() {
    let input_schema = Arc::new(Schema::new(vec![
        Field::new("v", DataType::Int64, true),
        Field::new("price", DataType::Float64, true),
        Field::new("discount", DataType::Float64, true),
    ]));
    let batch = RecordBatch::try_new(
        input_schema,
        vec![
            Arc::new(Int64Array::from(vec![Some(9), Some(11), None, Some(12)])),
            Arc::new(Float64Array::from(vec![
                Some(10.0),
                Some(20.0),
                Some(30.0),
                None,
            ])),
            Arc::new(Float64Array::from(vec![
                Some(0.1),
                Some(0.2),
                Some(0.3),
                Some(0.4),
            ])),
        ],
    )
    .unwrap();
    let kernel = FilterSumKernel::try_new(
        JitExpr::Binary {
            op: JitBinaryOp::Gt,
            left: Box::new(JitExpr::Column {
                index: 0,
                name: "v".to_string(),
                ty: JitType::Int64,
                nullable: true,
            }),
            right: Box::new(JitExpr::Literal(JitScalar::Int64(10))),
            ty: JitType::Bool,
            nullable: true,
        },
        JitExpr::Binary {
            op: JitBinaryOp::Mul,
            left: Box::new(JitExpr::Column {
                index: 1,
                name: "price".to_string(),
                ty: JitType::Float64,
                nullable: true,
            }),
            right: Box::new(JitExpr::Column {
                index: 2,
                name: "discount".to_string(),
                ty: JitType::Float64,
                nullable: true,
            }),
            ty: JitType::Float64,
            nullable: true,
        },
    )
    .unwrap();

    assert_eq!(
        kernel.spec(),
        Some(&PipelineSpec::PlainSum {
            columns: vec![
                FixedColumn {
                    index: 0,
                    ty: JitType::Int64
                },
                FixedColumn {
                    index: 1,
                    ty: JitType::Float64
                },
                FixedColumn {
                    index: 2,
                    ty: JitType::Float64
                }
            ],
            output_type: JitType::Float64
        })
    );

    let output = kernel.execute(&batch).unwrap();
    assert_eq!(output, FilterSumValue::Float64(Some(4.0)));
}

#[test]
fn executes_decimal_plain_sum_with_date_predicate() {
    let input_schema = Arc::new(Schema::new(vec![
        Field::new("shipdate", DataType::Date32, true),
        Field::new("price", DataType::Decimal128(15, 2), true),
        Field::new("discount", DataType::Decimal128(15, 2), true),
        Field::new("quantity", DataType::Decimal128(15, 2), true),
    ]));
    let batch = RecordBatch::try_new(
        input_schema,
        vec![
            Arc::new(Date32Array::from(vec![Some(9), Some(10), Some(12), None])),
            Arc::new(
                Decimal128Array::from(vec![
                    Some(10_000_i128),
                    Some(20_000),
                    Some(30_000),
                    Some(40_000),
                ])
                .with_precision_and_scale(15, 2)
                .unwrap(),
            ),
            Arc::new(
                Decimal128Array::from(vec![Some(4_i128), Some(5), Some(7), Some(6)])
                    .with_precision_and_scale(15, 2)
                    .unwrap(),
            ),
            Arc::new(
                Decimal128Array::from(vec![
                    Some(1_000_i128),
                    Some(2_500),
                    Some(2_000),
                    Some(2_000),
                ])
                .with_precision_and_scale(15, 2)
                .unwrap(),
            ),
        ],
    )
    .unwrap();
    let predicate = and(
        date_cmp(JitBinaryOp::GtEq, 0, 10),
        and(
            decimal_cmp(JitBinaryOp::GtEq, 2, 5),
            and(
                decimal_cmp(JitBinaryOp::LtEq, 2, 7),
                decimal_cmp(JitBinaryOp::Lt, 3, 2_400),
            ),
        ),
    );
    let measure = JitExpr::Binary {
        op: JitBinaryOp::Mul,
        left: Box::new(decimal_col(1, "price", 15, 2)),
        right: Box::new(decimal_col(2, "discount", 15, 2)),
        ty: JitType::Decimal128 {
            precision: 30,
            scale: 4,
        },
        nullable: true,
    };
    let kernel = FilterSumKernel::try_new(predicate, measure).unwrap();

    assert_eq!(
        kernel.spec(),
        Some(&PipelineSpec::PlainSum {
            columns: vec![
                FixedColumn {
                    index: 0,
                    ty: JitType::Date32
                },
                FixedColumn {
                    index: 1,
                    ty: JitType::Decimal128 {
                        precision: 15,
                        scale: 2
                    }
                },
                FixedColumn {
                    index: 2,
                    ty: JitType::Decimal128 {
                        precision: 15,
                        scale: 2
                    }
                },
                FixedColumn {
                    index: 3,
                    ty: JitType::Decimal128 {
                        precision: 15,
                        scale: 2
                    }
                },
            ],
            output_type: JitType::Decimal128 {
                precision: 30,
                scale: 4
            }
        })
    );

    let output = kernel.execute(&batch).unwrap();
    assert_eq!(
        output,
        FilterSumValue::Decimal128 {
            value: Some(210_000),
            scale: 4
        }
    );
}

fn and(left: JitExpr, right: JitExpr) -> JitExpr {
    JitExpr::Binary {
        op: JitBinaryOp::And,
        left: Box::new(left),
        right: Box::new(right),
        ty: JitType::Bool,
        nullable: true,
    }
}

fn date_cmp(op: JitBinaryOp, index: usize, value: i32) -> JitExpr {
    JitExpr::Binary {
        op,
        left: Box::new(JitExpr::Column {
            index,
            name: "shipdate".to_string(),
            ty: JitType::Date32,
            nullable: true,
        }),
        right: Box::new(JitExpr::Literal(JitScalar::Date32(value))),
        ty: JitType::Bool,
        nullable: true,
    }
}

fn decimal_cmp(op: JitBinaryOp, index: usize, value: i128) -> JitExpr {
    JitExpr::Binary {
        op,
        left: Box::new(decimal_col(index, "decimal", 15, 2)),
        right: Box::new(JitExpr::Literal(JitScalar::Decimal128 {
            value,
            precision: 15,
            scale: 2,
        })),
        ty: JitType::Bool,
        nullable: true,
    }
}

fn decimal_col(index: usize, name: &str, precision: u8, scale: i8) -> JitExpr {
    JitExpr::Column {
        index,
        name: name.to_string(),
        ty: JitType::Decimal128 { precision, scale },
        nullable: true,
    }
}
