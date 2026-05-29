use std::sync::Arc;

use arrow::array::{
    Array, BooleanArray, Date32Array, Decimal128Array, Float64Array, Int64Array, StringArray,
    UInt64Array,
};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;

use quill_plan::{
    AggregateFunc, GroupAggregate, JitBinaryOp, JitExpr, JitProjection, JitScalar, JitType,
    PipelineStage,
};

use super::{
    FilterProjectKernel, FilterSumKernel, FilterSumValue, GroupAggregateKernel,
    GroupAggregateStateField,
};

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

    let output = kernel.execute(&batch).unwrap();
    assert_eq!(
        output,
        FilterSumValue::Decimal128 {
            value: Some(210_000),
            scale: 4
        }
    );
}

#[test]
fn executes_group_aggregate_with_filter() {
    let input_schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, true),
        Field::new("v", DataType::Int64, true),
        Field::new("shipdate", DataType::Date32, true),
    ]));
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, true),
        Field::new("sum_v", DataType::Int64, true),
        Field::new("count_v", DataType::Int64, false),
        Field::new("min_v", DataType::Int64, true),
        Field::new("max_v", DataType::Int64, true),
    ]));
    let batch = RecordBatch::try_new(
        input_schema,
        vec![
            Arc::new(Int64Array::from(vec![Some(1), Some(1), Some(2), Some(2)])),
            Arc::new(Int64Array::from(vec![Some(10), Some(20), Some(30), None])),
            Arc::new(Date32Array::from(vec![
                Some(10),
                Some(11),
                Some(10),
                Some(12),
            ])),
        ],
    )
    .unwrap();
    let predicate = date_cmp(JitBinaryOp::LtEq, 2, 12);
    let key = JitExpr::Column {
        index: 0,
        name: "k".to_string(),
        ty: JitType::Int64,
        nullable: true,
    };
    let value = JitExpr::Column {
        index: 1,
        name: "v".to_string(),
        ty: JitType::Int64,
        nullable: true,
    };
    let aggregates = vec![
        GroupAggregate::new(AggregateFunc::Sum, value.clone(), JitType::Int64, "sum_v"),
        GroupAggregate::new(
            AggregateFunc::Count,
            value.clone(),
            JitType::Int64,
            "count_v",
        ),
        GroupAggregate::new(AggregateFunc::Min, value.clone(), JitType::Int64, "min_v"),
        GroupAggregate::new(AggregateFunc::Max, value, JitType::Int64, "max_v"),
    ];
    let kernel = GroupAggregateKernel::try_new(
        &[PipelineStage::Filter(predicate)],
        vec![key],
        aggregates,
        output_schema,
    )
    .unwrap();
    let mut state = kernel.new_state();
    kernel.accumulate(&mut state, &batch).unwrap();
    let output = kernel.finish(state).unwrap();

    let keys = output
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    let sums = output
        .column(1)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    let counts = output
        .column(2)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    let mins = output
        .column(3)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    let maxes = output
        .column(4)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();

    assert_eq!(keys.values().as_ref(), &[1, 2]);
    assert_eq!(sums.values().as_ref(), &[30, 30]);
    assert_eq!(counts.values().as_ref(), &[2, 1]);
    assert_eq!(mins.values().as_ref(), &[10, 30]);
    assert_eq!(maxes.values().as_ref(), &[20, 30]);
}

#[test]
fn executes_group_aggregate_with_utf8_key() {
    let input_schema = Arc::new(Schema::new(vec![
        Field::new("flag", DataType::Utf8, true),
        Field::new("v", DataType::Int64, true),
    ]));
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("flag", DataType::Utf8, true),
        Field::new("sum_v", DataType::Int64, true),
        Field::new("min_flag", DataType::Utf8, true),
    ]));
    let batch = RecordBatch::try_new(
        input_schema,
        vec![
            Arc::new(StringArray::from(vec![
                Some("B"),
                Some("A"),
                Some("A"),
                None,
            ])),
            Arc::new(Int64Array::from(vec![
                Some(30),
                Some(10),
                Some(20),
                Some(5),
            ])),
        ],
    )
    .unwrap();
    let key = JitExpr::Column {
        index: 0,
        name: "flag".to_string(),
        ty: JitType::Utf8,
        nullable: true,
    };
    let value = JitExpr::Column {
        index: 1,
        name: "v".to_string(),
        ty: JitType::Int64,
        nullable: true,
    };
    let aggregates = vec![
        GroupAggregate::new(AggregateFunc::Sum, value, JitType::Int64, "sum_v"),
        GroupAggregate::new(AggregateFunc::Min, key.clone(), JitType::Utf8, "min_flag"),
    ];
    let kernel = GroupAggregateKernel::try_new(&[], vec![key], aggregates, output_schema).unwrap();
    let mut state = kernel.new_state();
    kernel.accumulate(&mut state, &batch).unwrap();
    let output = kernel.finish(state).unwrap();

    let keys = output
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let sums = output
        .column(1)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    let mins = output
        .column(2)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();

    assert!(keys.is_null(0));
    assert_eq!(keys.value(1), "A");
    assert_eq!(keys.value(2), "B");
    assert_eq!(sums.values().as_ref(), &[5, 30, 30]);
    assert!(mins.is_null(0));
    assert_eq!(mins.value(1), "A");
    assert_eq!(mins.value(2), "B");
}

#[test]
fn executes_group_avg_as_partial_state() {
    let input_schema = Arc::new(Schema::new(vec![
        Field::new("flag", DataType::Utf8, false),
        Field::new("v", DataType::Int64, true),
    ]));
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("flag", DataType::Utf8, false),
        Field::new("avg_v[count]", DataType::UInt64, false),
        Field::new("avg_v[sum]", DataType::Int64, true),
    ]));
    let batch = RecordBatch::try_new(
        input_schema,
        vec![
            Arc::new(StringArray::from(vec!["A", "A", "B", "B"])),
            Arc::new(Int64Array::from(vec![Some(10), Some(20), Some(30), None])),
        ],
    )
    .unwrap();
    let key = JitExpr::Column {
        index: 0,
        name: "flag".to_string(),
        ty: JitType::Utf8,
        nullable: false,
    };
    let value = JitExpr::Column {
        index: 1,
        name: "v".to_string(),
        ty: JitType::Int64,
        nullable: true,
    };
    let aggregate = GroupAggregate::new_with_states(
        AggregateFunc::Avg,
        value,
        vec![JitType::UInt64, JitType::Int64],
        "avg_v",
    );
    let kernel =
        GroupAggregateKernel::try_new(&[], vec![key], vec![aggregate], output_schema).unwrap();
    let mut state = kernel.new_state();
    kernel.accumulate(&mut state, &batch).unwrap();
    let output = kernel.finish(state).unwrap();

    let keys = output
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let counts = output
        .column(1)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap();
    let sums = output
        .column(2)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();

    assert_eq!(keys.value(0), "A");
    assert_eq!(keys.value(1), "B");
    assert_eq!(counts.values().as_ref(), &[2, 1]);
    assert_eq!(sums.values().as_ref(), &[30, 30]);
}

#[test]
fn binds_group_ids_for_selected_rows() {
    let input_schema = Arc::new(Schema::new(vec![
        Field::new("flag", DataType::Utf8, false),
        Field::new("v", DataType::Int64, false),
    ]));
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("flag", DataType::Utf8, false),
        Field::new("sum_v", DataType::Int64, true),
    ]));
    let batch = RecordBatch::try_new(
        input_schema,
        vec![
            Arc::new(StringArray::from(vec!["A", "B", "A", "C"])),
            Arc::new(Int64Array::from(vec![10, 20, 30, 40])),
        ],
    )
    .unwrap();
    let key = JitExpr::Column {
        index: 0,
        name: "flag".to_string(),
        ty: JitType::Utf8,
        nullable: false,
    };
    let value = JitExpr::Column {
        index: 1,
        name: "v".to_string(),
        ty: JitType::Int64,
        nullable: false,
    };
    let predicate = JitExpr::Binary {
        op: JitBinaryOp::Lt,
        left: Box::new(value.clone()),
        right: Box::new(JitExpr::Literal(JitScalar::Int64(40))),
        ty: JitType::Bool,
        nullable: false,
    };
    let aggregate = GroupAggregate::new(AggregateFunc::Sum, value, JitType::Int64, "sum_v");
    let kernel = GroupAggregateKernel::try_new(
        &[PipelineStage::Filter(predicate)],
        vec![key],
        vec![aggregate],
        output_schema,
    )
    .unwrap();
    let mut state = kernel.new_state();
    let binding = kernel.bind_batch(&mut state, &batch).unwrap();

    assert_eq!(binding.group_ids(), &[0, 1, 0, -1]);
    assert_eq!(binding.selected_rows(), 3);
}

#[test]
fn flushes_dense_group_state_before_runtime_fallback() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, false),
        Field::new("v", DataType::Int64, false),
    ]));
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, false),
        Field::new("sum_v", DataType::Int64, true),
    ]));
    let key = JitExpr::Column {
        index: 0,
        name: "k".to_string(),
        ty: JitType::Int64,
        nullable: false,
    };
    let value = JitExpr::Column {
        index: 1,
        name: "v".to_string(),
        ty: JitType::Int64,
        nullable: false,
    };
    let aggregate = GroupAggregate::new(AggregateFunc::Sum, value, JitType::Int64, "sum_v");
    let kernel =
        GroupAggregateKernel::try_new(&[], vec![key], vec![aggregate], output_schema).unwrap();
    let mut state = kernel.new_state();
    let first = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from(vec![1])),
            Arc::new(Int64Array::from(vec![10])),
        ],
    )
    .unwrap();
    let first_binding = kernel.bind_batch(&mut state, &first).unwrap();
    assert_eq!(first_binding.group_ids(), &[0]);

    {
        let dense = kernel.dense_state_mut(&mut state).unwrap();
        let [GroupAggregateStateField::Int64 { values, valid }] = dense.fields_mut() else {
            panic!("expected one int64 dense state field");
        };
        values[0] = 10;
        valid[0] = 1;
    }

    let second = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(vec![1, 2])),
            Arc::new(Int64Array::from(vec![5, 7])),
        ],
    )
    .unwrap();
    let second_binding = kernel.bind_batch(&mut state, &second).unwrap();
    assert_eq!(second_binding.group_ids(), &[0, 1]);
    kernel.flush_dense_state(&mut state).unwrap();
    kernel
        .accumulate_bound(&mut state, &second, &second_binding)
        .unwrap();

    let output = kernel.finish(state).unwrap();
    let keys = output
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    let sums = output
        .column(1)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    assert_eq!(keys.values().as_ref(), &[1, 2]);
    assert_eq!(sums.values().as_ref(), &[15, 7]);
}

#[test]
fn finishes_group_aggregate_directly_from_dense_state() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, false),
        Field::new("v", DataType::Int64, false),
    ]));
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, false),
        Field::new("sum_v", DataType::Int64, true),
        Field::new("count_v", DataType::Int64, true),
    ]));
    let key = JitExpr::Column {
        index: 0,
        name: "k".to_string(),
        ty: JitType::Int64,
        nullable: false,
    };
    let value = JitExpr::Column {
        index: 1,
        name: "v".to_string(),
        ty: JitType::Int64,
        nullable: false,
    };
    let aggregates = vec![
        GroupAggregate::new(AggregateFunc::Sum, value, JitType::Int64, "sum_v"),
        GroupAggregate::new(
            AggregateFunc::Count,
            JitExpr::Literal(JitScalar::Int64(1)),
            JitType::Int64,
            "count_v",
        ),
    ];
    let kernel = GroupAggregateKernel::try_new(&[], vec![key], aggregates, output_schema).unwrap();
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(vec![1, 2])),
            Arc::new(Int64Array::from(vec![10, 20])),
        ],
    )
    .unwrap();
    let mut state = kernel.new_state();
    let binding = kernel.bind_batch(&mut state, &batch).unwrap();
    assert_eq!(binding.group_ids(), &[0, 1]);

    {
        let dense = kernel.dense_state_mut(&mut state).unwrap();
        let [GroupAggregateStateField::Int64 {
            values: sums,
            valid: sum_valid,
        }, GroupAggregateStateField::Int64 {
            values: counts,
            valid: count_valid,
        }] = dense.fields_mut()
        else {
            panic!("expected sum and count dense state fields");
        };
        sums.copy_from_slice(&[30, 20]);
        sum_valid.copy_from_slice(&[1, 1]);
        counts.copy_from_slice(&[2, 1]);
        count_valid.copy_from_slice(&[1, 1]);
    }

    let output = kernel.finish(state).unwrap();
    let keys = output
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    let sums = output
        .column(1)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    let counts = output
        .column(2)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    assert_eq!(keys.values().as_ref(), &[1, 2]);
    assert_eq!(sums.values().as_ref(), &[30, 20]);
    assert_eq!(counts.values().as_ref(), &[2, 1]);
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
