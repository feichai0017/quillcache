use std::sync::Arc;

use datafusion::arrow::datatypes::{DataType as ArrowDataType, Schema as ArrowSchema};
use datafusion::common::ScalarValue;
use datafusion::logical_expr::Operator;
use datafusion::physical_expr::expressions::{BinaryExpr, Column, IsNullExpr, Literal};
use datafusion::physical_expr::PhysicalExpr;
use quill_plan::{JitBinaryOp, JitError, JitExpr, JitResult, JitScalar, JitType};

pub(crate) fn from_physical(
    expr: &Arc<dyn PhysicalExpr>,
    input_schema: &ArrowSchema,
) -> JitResult<JitExpr> {
    if let Some(column) = expr.as_any().downcast_ref::<Column>() {
        let field = input_schema.field(column.index());
        return Ok(JitExpr::Column {
            index: column.index(),
            name: column.name().to_string(),
            ty: jit_type(field.data_type())?,
            nullable: field.is_nullable(),
        });
    }

    if let Some(literal) = expr.as_any().downcast_ref::<Literal>() {
        return Ok(JitExpr::Literal(jit_scalar(literal.value())?));
    }

    if let Some(binary) = expr.as_any().downcast_ref::<BinaryExpr>() {
        let left = from_physical(binary.left(), input_schema)?;
        let right = from_physical(binary.right(), input_schema)?;
        let output_type = expr.data_type(input_schema).map_err(|err| {
            JitError::UnsupportedExpr(format!("cannot infer binary expression type: {err}"))
        })?;
        let nullable = expr.nullable(input_schema).map_err(|err| {
            JitError::UnsupportedExpr(format!("cannot infer binary expression nullability: {err}"))
        })?;
        return Ok(JitExpr::Binary {
            op: jit_binary_op(binary.op())?,
            left: Box::new(left),
            right: Box::new(right),
            ty: jit_type(&output_type)?,
            nullable,
        });
    }

    if let Some(is_null) = expr.as_any().downcast_ref::<IsNullExpr>() {
        let arg = from_physical(is_null.arg(), input_schema)?;
        return Ok(JitExpr::IsNull(Box::new(arg)));
    }

    Err(JitError::UnsupportedExpr(expr.to_string()))
}

pub(crate) fn jit_type(data_type: &ArrowDataType) -> JitResult<JitType> {
    match data_type {
        ArrowDataType::Boolean => Ok(JitType::Bool),
        ArrowDataType::Date32 => Ok(JitType::Date32),
        ArrowDataType::Int32 => Ok(JitType::Int32),
        ArrowDataType::Int64 => Ok(JitType::Int64),
        ArrowDataType::Float64 => Ok(JitType::Float64),
        ArrowDataType::Decimal128(precision, scale) => Ok(JitType::Decimal128 {
            precision: *precision,
            scale: *scale,
        }),
        other => Err(JitError::UnsupportedType(format!("{other:?}"))),
    }
}

fn jit_scalar(value: &ScalarValue) -> JitResult<JitScalar> {
    match value {
        ScalarValue::Null => Ok(JitScalar::Null(JitType::Bool)),
        ScalarValue::Boolean(Some(value)) => Ok(JitScalar::Bool(*value)),
        ScalarValue::Boolean(None) => Ok(JitScalar::Null(JitType::Bool)),
        ScalarValue::Date32(Some(value)) => Ok(JitScalar::Date32(*value)),
        ScalarValue::Date32(None) => Ok(JitScalar::Null(JitType::Date32)),
        ScalarValue::Int32(Some(value)) => Ok(JitScalar::Int32(*value)),
        ScalarValue::Int32(None) => Ok(JitScalar::Null(JitType::Int32)),
        ScalarValue::Int64(Some(value)) => Ok(JitScalar::Int64(*value)),
        ScalarValue::Int64(None) => Ok(JitScalar::Null(JitType::Int64)),
        ScalarValue::Float64(Some(value)) => Ok(JitScalar::Float64(*value)),
        ScalarValue::Float64(None) => Ok(JitScalar::Null(JitType::Float64)),
        ScalarValue::Decimal128(Some(value), precision, scale) => Ok(JitScalar::Decimal128 {
            value: *value,
            precision: *precision,
            scale: *scale,
        }),
        ScalarValue::Decimal128(None, precision, scale) => {
            Ok(JitScalar::Null(JitType::Decimal128 {
                precision: *precision,
                scale: *scale,
            }))
        }
        other => Err(JitError::UnsupportedType(format!("{other:?}"))),
    }
}

fn jit_binary_op(op: &Operator) -> JitResult<JitBinaryOp> {
    match op {
        Operator::Plus => Ok(JitBinaryOp::Add),
        Operator::Minus => Ok(JitBinaryOp::Sub),
        Operator::Multiply => Ok(JitBinaryOp::Mul),
        Operator::Divide => Ok(JitBinaryOp::Div),
        Operator::Eq => Ok(JitBinaryOp::Eq),
        Operator::NotEq => Ok(JitBinaryOp::NotEq),
        Operator::Lt => Ok(JitBinaryOp::Lt),
        Operator::LtEq => Ok(JitBinaryOp::LtEq),
        Operator::Gt => Ok(JitBinaryOp::Gt),
        Operator::GtEq => Ok(JitBinaryOp::GtEq),
        Operator::And => Ok(JitBinaryOp::And),
        Operator::Or => Ok(JitBinaryOp::Or),
        other => Err(JitError::UnsupportedExpr(format!(
            "operator {other:?} is not supported"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use datafusion::arrow::datatypes::{DataType as ArrowDataType, Field, Schema as ArrowSchema};
    use datafusion::common::ScalarValue;
    use datafusion::logical_expr::Operator;
    use datafusion::physical_expr::expressions::{binary, col, lit};
    use quill_plan::{JitBinaryOp, JitExpr, JitScalar, JitType};

    #[test]
    fn lowers_simple_filter_expr() {
        let schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "a",
            ArrowDataType::Int64,
            true,
        )]));
        let expr = binary(
            col("a", &schema).unwrap(),
            Operator::Gt,
            lit(ScalarValue::Int64(Some(10))),
            &schema,
        )
        .unwrap();

        let lowered = super::from_physical(&expr, &schema).unwrap();
        let JitExpr::Binary {
            op,
            left,
            right,
            ty,
            nullable,
        } = lowered
        else {
            panic!("expected binary expression");
        };

        assert_eq!(op, JitBinaryOp::Gt);
        assert_eq!(ty, JitType::Bool);
        assert!(nullable);
        assert!(matches!(
            *left,
            JitExpr::Column {
                index: 0,
                ty: JitType::Int64,
                nullable: true,
                ..
            }
        ));
        assert_eq!(*right, JitExpr::Literal(JitScalar::Int64(10)));
    }
}
