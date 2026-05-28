use std::cmp::Ordering;

use quill_plan::{JitError, JitType};

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) enum Scalar {
    Bool(Option<bool>),
    Date32(Option<i32>),
    Int32(Option<i32>),
    Int64(Option<i64>),
    Float64(Option<f64>),
    Decimal128 {
        value: Option<i128>,
        precision: u8,
        scale: i8,
    },
}

impl Scalar {
    pub(super) fn is_filter_true(self) -> Result<bool, JitError> {
        match self {
            Self::Bool(value) => Ok(value.unwrap_or(false)),
            other => Err(JitError::Backend(format!(
                "predicate produced non-bool value {:?}",
                other.ty()
            ))),
        }
    }

    pub(super) fn ty(self) -> JitType {
        match self {
            Self::Bool(_) => JitType::Bool,
            Self::Date32(_) => JitType::Date32,
            Self::Int32(_) => JitType::Int32,
            Self::Int64(_) => JitType::Int64,
            Self::Float64(_) => JitType::Float64,
            Self::Decimal128 {
                precision, scale, ..
            } => JitType::Decimal128 { precision, scale },
        }
    }

    pub(super) fn is_null(self) -> bool {
        match self {
            Self::Bool(value) => value.is_none(),
            Self::Date32(value) => value.is_none(),
            Self::Int32(value) => value.is_none(),
            Self::Int64(value) => value.is_none(),
            Self::Float64(value) => value.is_none(),
            Self::Decimal128 { value, .. } => value.is_none(),
        }
    }

    pub(super) fn checked_add(self, rhs: Self) -> Result<Self, JitError> {
        match (self, rhs) {
            (Self::Int32(lhs), Self::Int32(rhs)) => {
                Ok(Self::Int32(add_options(lhs, rhs, i32::wrapping_add)))
            }
            (Self::Int64(lhs), Self::Int64(rhs)) => {
                Ok(Self::Int64(add_options(lhs, rhs, i64::wrapping_add)))
            }
            (Self::Float64(lhs), Self::Float64(rhs)) => {
                Ok(Self::Float64(add_options(lhs, rhs, |lhs, rhs| lhs + rhs)))
            }
            (
                Self::Decimal128 {
                    value: lhs,
                    precision: lhs_precision,
                    scale: lhs_scale,
                },
                Self::Decimal128 {
                    value: rhs,
                    precision: rhs_precision,
                    scale: rhs_scale,
                },
            ) => {
                if lhs_scale != rhs_scale {
                    return Err(JitError::UnsupportedExpr(format!(
                        "decimal sum requires matching scale, got {lhs_scale} and {rhs_scale}"
                    )));
                }
                Ok(Self::Decimal128 {
                    value: add_options(lhs, rhs, |lhs, rhs| lhs + rhs),
                    precision: lhs_precision.max(rhs_precision).saturating_add(1).min(38),
                    scale: lhs_scale,
                })
            }
            _ => Err(type_mismatch(self, rhs)),
        }
    }

    pub(super) fn partial_cmp_value(self, rhs: Self) -> Result<Option<Ordering>, JitError> {
        match (self, rhs) {
            (Self::Bool(lhs), Self::Bool(rhs)) => {
                Ok(option_zip(lhs, rhs).map(|(lhs, rhs)| lhs.cmp(&rhs)))
            }
            (Self::Date32(lhs), Self::Date32(rhs)) => {
                Ok(option_zip(lhs, rhs).map(|(lhs, rhs)| lhs.cmp(&rhs)))
            }
            (Self::Int32(lhs), Self::Int32(rhs)) => {
                Ok(option_zip(lhs, rhs).map(|(lhs, rhs)| lhs.cmp(&rhs)))
            }
            (Self::Int64(lhs), Self::Int64(rhs)) => {
                Ok(option_zip(lhs, rhs).map(|(lhs, rhs)| lhs.cmp(&rhs)))
            }
            (Self::Float64(lhs), Self::Float64(rhs)) => {
                Ok(option_zip(lhs, rhs).and_then(|(lhs, rhs)| lhs.partial_cmp(&rhs)))
            }
            (
                Self::Decimal128 {
                    value: lhs,
                    scale: lhs_scale,
                    ..
                },
                Self::Decimal128 {
                    value: rhs,
                    scale: rhs_scale,
                    ..
                },
            ) => {
                if lhs_scale != rhs_scale {
                    return Err(JitError::UnsupportedExpr(format!(
                        "decimal comparison requires matching scale, got {lhs_scale} and {rhs_scale}"
                    )));
                }
                Ok(option_zip(lhs, rhs).map(|(lhs, rhs)| lhs.cmp(&rhs)))
            }
            _ => Err(type_mismatch(self, rhs)),
        }
    }
}

fn add_options<T>(lhs: Option<T>, rhs: Option<T>, add: impl FnOnce(T, T) -> T) -> Option<T> {
    match (lhs, rhs) {
        (Some(lhs), Some(rhs)) => Some(add(lhs, rhs)),
        (Some(lhs), None) => Some(lhs),
        (None, Some(rhs)) => Some(rhs),
        (None, None) => None,
    }
}

pub(super) fn option_zip<T, U>(lhs: Option<T>, rhs: Option<U>) -> Option<(T, U)> {
    match (lhs, rhs) {
        (Some(lhs), Some(rhs)) => Some((lhs, rhs)),
        _ => None,
    }
}

pub(super) fn type_mismatch(lhs: Scalar, rhs: Scalar) -> JitError {
    JitError::UnsupportedExpr(format!("type mismatch: {:?} vs {:?}", lhs.ty(), rhs.ty()))
}
