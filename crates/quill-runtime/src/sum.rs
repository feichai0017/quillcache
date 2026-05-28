use arrow::record_batch::RecordBatch;

use quill_plan::{JitError, JitExpr, JitResult, JitType};

use super::eval::{ensure_supported_expr, eval_expr};
use super::value::Scalar;
use super::{BatchView, PipelineSpec};

#[derive(Debug, Clone)]
pub struct FilterSumKernel {
    predicate: JitExpr,
    measure: JitExpr,
    spec: Option<PipelineSpec>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FilterSumValue {
    Float64(Option<f64>),
    Decimal128 { value: Option<i128>, scale: i8 },
}

impl FilterSumKernel {
    pub fn try_new(predicate: JitExpr, measure: JitExpr) -> JitResult<Self> {
        if predicate.ty() != JitType::Bool {
            return Err(JitError::UnsupportedExpr(format!(
                "filter predicate must be bool, got {:?}",
                predicate.ty()
            )));
        }
        if !matches!(measure.ty(), JitType::Float64 | JitType::Decimal128 { .. }) {
            return Err(JitError::UnsupportedExpr(format!(
                "sum measure must be f64 or decimal128, got {:?}",
                measure.ty()
            )));
        }
        ensure_supported_expr(&predicate)?;
        ensure_supported_expr(&measure)?;
        let spec = PipelineSpec::filter_sum(&predicate, &measure);
        Ok(Self {
            predicate,
            measure,
            spec,
        })
    }

    pub fn predicate(&self) -> &JitExpr {
        &self.predicate
    }

    pub fn measure(&self) -> &JitExpr {
        &self.measure
    }

    pub fn spec(&self) -> Option<&PipelineSpec> {
        self.spec.as_ref()
    }

    pub fn execute(&self, batch: &RecordBatch) -> JitResult<FilterSumValue> {
        let view = BatchView::try_new(batch)?;
        let mut sum = FilterSumValue::null(self.measure.ty())?;

        for row in 0..batch.num_rows() {
            if !eval_expr(&self.predicate, &view, row)?.is_filter_true()? {
                continue;
            }
            sum.add_scalar(eval_expr(&self.measure, &view, row)?)?;
        }

        Ok(sum)
    }
}

impl FilterSumValue {
    pub fn null(ty: JitType) -> JitResult<Self> {
        match ty {
            JitType::Float64 => Ok(Self::Float64(None)),
            JitType::Decimal128 { scale, .. } => Ok(Self::Decimal128 { value: None, scale }),
            other => Err(JitError::UnsupportedExpr(format!(
                "sum value must be f64 or decimal128, got {other:?}"
            ))),
        }
    }

    pub fn merge(&mut self, other: Self) -> JitResult<()> {
        match (self, other) {
            (Self::Float64(lhs), Self::Float64(rhs)) => {
                if let Some(rhs) = rhs {
                    *lhs = Some(lhs.unwrap_or(0.0) + rhs);
                }
                Ok(())
            }
            (
                Self::Decimal128 {
                    value: lhs,
                    scale: lhs_scale,
                },
                Self::Decimal128 {
                    value: rhs,
                    scale: rhs_scale,
                },
            ) => {
                if *lhs_scale != rhs_scale {
                    return Err(JitError::UnsupportedExpr(format!(
                        "cannot merge decimal sums with scales {} and {}",
                        *lhs_scale, rhs_scale
                    )));
                }
                if let Some(rhs) = rhs {
                    *lhs = Some(lhs.unwrap_or(0) + rhs);
                }
                Ok(())
            }
            (_, rhs) => Err(JitError::UnsupportedExpr(format!(
                "cannot merge incompatible sum value {:?}",
                rhs.ty()
            ))),
        }
    }

    pub fn ty(self) -> JitType {
        match self {
            Self::Float64(_) => JitType::Float64,
            Self::Decimal128 { scale, .. } => JitType::Decimal128 {
                precision: 38,
                scale,
            },
        }
    }

    fn add_scalar(&mut self, value: Scalar) -> JitResult<()> {
        match (self, value) {
            (Self::Float64(sum), Scalar::Float64(value)) => {
                if let Some(value) = value {
                    *sum = Some(sum.unwrap_or(0.0) + value);
                }
                Ok(())
            }
            (
                Self::Decimal128 {
                    value: sum,
                    scale: sum_scale,
                },
                Scalar::Decimal128 {
                    value,
                    scale,
                    precision: _,
                },
            ) => {
                if *sum_scale != scale {
                    return Err(JitError::UnsupportedExpr(format!(
                        "decimal sum requires scale {}, got {}",
                        *sum_scale, scale
                    )));
                }
                if let Some(value) = value {
                    *sum = Some(sum.unwrap_or(0) + value);
                }
                Ok(())
            }
            (_, other) => Err(JitError::Backend(format!(
                "sum measure produced unsupported value {:?}",
                other.ty()
            ))),
        }
    }
}
