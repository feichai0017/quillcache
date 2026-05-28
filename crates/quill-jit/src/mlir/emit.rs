use std::collections::BTreeMap;
use std::fmt::Write;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::{JitBinaryOp, JitError, JitExpr, JitProjection, JitResult, JitScalar, JitType};

use super::MlirModule;

static NEXT_KERNEL_ID: AtomicU64 = AtomicU64::new(1);

pub(super) fn lower_filter(predicate: &JitExpr) -> JitResult<MlirModule> {
    let symbol = next_symbol("quill_filter");
    let mut text = start_module();
    text.push_str(&scalar_function(&format!("{symbol}_expr"), predicate)?);
    append_batch_function_header(&mut text, &symbol);
    let _ = writeln!(text, "    // qjit.kind = filter");
    let _ = writeln!(text, "    // qjit.predicate = {}", format_expr(predicate));
    text.push_str("    %ok = arith.constant 0 : i32\n");
    text.push_str("    return %ok : i32\n");
    text.push_str("  }\n}\n");
    Ok(MlirModule { symbol, text })
}

pub(super) fn lower_projection(projections: &[JitProjection]) -> JitResult<MlirModule> {
    let symbol = next_symbol("quill_project");
    let mut text = start_module();
    append_projection_scalar_functions(&mut text, &symbol, projections)?;
    append_batch_function_header(&mut text, &symbol);
    let _ = writeln!(text, "    // qjit.kind = projection");
    for projection in projections {
        let _ = writeln!(
            text,
            "    // qjit.project {} = {}",
            projection.alias,
            format_expr(&projection.expr)
        );
    }
    text.push_str("    %ok = arith.constant 0 : i32\n");
    text.push_str("    return %ok : i32\n");
    text.push_str("  }\n}\n");
    Ok(MlirModule { symbol, text })
}

pub(super) fn lower_filter_project(
    predicate: &JitExpr,
    projections: &[JitProjection],
) -> JitResult<MlirModule> {
    let symbol = next_symbol("quill_filter_project");
    let mut text = start_module();
    text.push_str(&scalar_function(&format!("{symbol}_predicate"), predicate)?);
    append_projection_scalar_functions(&mut text, &symbol, projections)?;
    append_batch_function_header(&mut text, &symbol);
    let _ = writeln!(text, "    // qjit.kind = filter_project");
    let _ = writeln!(text, "    // qjit.predicate = {}", format_expr(predicate));
    for projection in projections {
        let _ = writeln!(
            text,
            "    // qjit.project {} = {}",
            projection.alias,
            format_expr(&projection.expr)
        );
    }
    text.push_str("    %ok = arith.constant 0 : i32\n");
    text.push_str("    return %ok : i32\n");
    text.push_str("  }\n}\n");
    Ok(MlirModule { symbol, text })
}

pub(super) fn lower_i64_predicate(predicate: &JitExpr) -> JitResult<MlirModule> {
    ensure_single_i64_predicate(predicate, "compiled predicate wrapper")?;
    let symbol = next_symbol("quill_i64_predicate");
    let expr_symbol = format!("{symbol}_expr");
    let mut text = start_module();
    text.push_str(&scalar_function(&expr_symbol, predicate)?);
    let _ = writeln!(
        text,
        "  func.func @{symbol}(%c0: i64) -> i32 attributes {{ llvm.emit_c_interface }} {{"
    );
    let _ = writeln!(
        text,
        "    %pred = func.call @{expr_symbol}(%c0) : (i64) -> i1"
    );
    text.push_str("    %one = arith.constant 1 : i32\n");
    text.push_str("    %zero = arith.constant 0 : i32\n");
    text.push_str("    %out = arith.select %pred, %one, %zero : i32\n");
    text.push_str("    return %out : i32\n");
    text.push_str("  }\n}\n");
    Ok(MlirModule { symbol, text })
}

pub(super) fn lower_i64_filter(predicate: &JitExpr) -> JitResult<MlirModule> {
    ensure_single_i64_predicate(predicate, "compiled filter kernel")?;
    let symbol = next_symbol("quill_i64_filter");
    let expr_symbol = format!("{symbol}_expr");
    let mut text = start_module();
    text.push_str(&scalar_function(&expr_symbol, predicate)?);
    let _ = writeln!(
        text,
        "  func.func @{symbol}(%len: i64, %values: !llvm.ptr, %out: !llvm.ptr) -> i32 attributes {{ llvm.emit_c_interface }} {{"
    );
    text.push_str("    %c0_i64 = arith.constant 0 : i64\n");
    text.push_str("    %c1_i64 = arith.constant 1 : i64\n");
    text.push_str("    %false = arith.constant 0 : i8\n");
    text.push_str("    %true = arith.constant 1 : i8\n");
    text.push_str("    scf.for unsigned %i = %c0_i64 to %len step %c1_i64 : i64 {\n");
    text.push_str(
        "      %value_ptr = llvm.getelementptr %values[%i] : (!llvm.ptr, i64) -> !llvm.ptr, i64\n",
    );
    text.push_str("      %value = llvm.load %value_ptr : !llvm.ptr -> i64\n");
    let _ = writeln!(
        text,
        "      %pred = func.call @{expr_symbol}(%value) : (i64) -> i1"
    );
    text.push_str("      %mask = arith.select %pred, %true, %false : i8\n");
    text.push_str(
        "      %out_ptr = llvm.getelementptr %out[%i] : (!llvm.ptr, i64) -> !llvm.ptr, i8\n",
    );
    text.push_str("      llvm.store %mask, %out_ptr : i8, !llvm.ptr\n");
    text.push_str("    }\n");
    text.push_str("    %ok = arith.constant 0 : i32\n");
    text.push_str("    return %ok : i32\n");
    text.push_str("  }\n}\n");
    Ok(MlirModule { symbol, text })
}

pub(super) fn next_symbol(prefix: &str) -> String {
    let id = NEXT_KERNEL_ID.fetch_add(1, Ordering::Relaxed);
    format!("{prefix}_{id}")
}

fn start_module() -> String {
    "module {\n".to_string()
}

fn append_batch_function_header(text: &mut String, symbol: &str) {
    let _ = writeln!(
        text,
        "  func.func @{symbol}(%len: index, %input: !llvm.ptr, %output: !llvm.ptr) -> i32 {{"
    );
}

fn append_projection_scalar_functions(
    text: &mut String,
    symbol: &str,
    projections: &[JitProjection],
) -> JitResult<()> {
    for (index, projection) in projections.iter().enumerate() {
        text.push_str(&scalar_function(
            &format!("{symbol}_expr_{index}"),
            &projection.expr,
        )?);
    }
    Ok(())
}

fn scalar_function(symbol: &str, expr: &JitExpr) -> JitResult<String> {
    let mut columns = BTreeMap::new();
    collect_columns(expr, &mut columns);
    let args = columns
        .iter()
        .map(|(index, ty)| format!("%c{index}: {}", mlir_type(*ty)))
        .collect::<Vec<_>>()
        .join(", ");

    let mut emitter = ScalarEmitter::default();
    let value = emitter.emit_expr(expr)?;
    let mut text = String::new();
    let _ = writeln!(
        text,
        "  func.func private @{symbol}({args}) -> {} {{",
        mlir_type(value.ty)
    );
    for line in emitter.lines {
        let _ = writeln!(text, "{line}");
    }
    let _ = writeln!(text, "    return {} : {}", value.name, mlir_type(value.ty));
    text.push_str("  }\n");
    Ok(text)
}

fn collect_columns(expr: &JitExpr, columns: &mut BTreeMap<usize, JitType>) {
    match expr {
        JitExpr::Column { index, ty, .. } => {
            columns.insert(*index, *ty);
        }
        JitExpr::Literal(_) => {}
        JitExpr::Binary { left, right, .. } => {
            collect_columns(left, columns);
            collect_columns(right, columns);
        }
        JitExpr::IsNull(arg) => collect_columns(arg, columns),
    }
}

fn ensure_single_i64_predicate(predicate: &JitExpr, context: &str) -> JitResult<()> {
    if predicate.ty() != JitType::Bool {
        return Err(JitError::UnsupportedExpr(format!(
            "{context} requires bool output, got {}",
            mlir_type(predicate.ty())
        )));
    }

    ensure_single_i64_input(predicate, context)?;
    Ok(())
}

fn ensure_single_i64_input(expr: &JitExpr, context: &str) -> JitResult<()> {
    let mut columns = BTreeMap::new();
    collect_columns(expr, &mut columns);
    if columns.len() != 1 || !columns.values().all(|ty| *ty == JitType::Int64) {
        return Err(JitError::UnsupportedExpr(format!(
            "{context} currently supports exactly one i64 input column"
        )));
    }
    Ok(())
}

#[derive(Debug, Clone)]
struct ScalarValueRef {
    name: String,
    ty: JitType,
}

#[derive(Debug, Default)]
struct ScalarEmitter {
    next_id: usize,
    lines: Vec<String>,
}

impl ScalarEmitter {
    fn emit_expr(&mut self, expr: &JitExpr) -> JitResult<ScalarValueRef> {
        match expr {
            JitExpr::Column { index, ty, .. } => Ok(ScalarValueRef {
                name: format!("%c{index}"),
                ty: *ty,
            }),
            JitExpr::Literal(value) => self.emit_literal(value),
            JitExpr::Binary {
                op, left, right, ..
            } => self.emit_binary(*op, left, right),
            JitExpr::IsNull(_) => Err(JitError::UnsupportedExpr(
                "MLIR lowering does not yet model Arrow validity bitmaps".to_string(),
            )),
        }
    }

    fn emit_literal(&mut self, value: &JitScalar) -> JitResult<ScalarValueRef> {
        let ty = value.ty();
        let name = self.next_value("lit");
        match value {
            JitScalar::Null(_) => {
                return Err(JitError::UnsupportedExpr(
                    "MLIR lowering does not yet model null literals".to_string(),
                ));
            }
            JitScalar::Bool(value) => {
                self.lines
                    .push(format!("    {name} = arith.constant {value}"));
            }
            JitScalar::Date32(value) => {
                self.lines.push(format!(
                    "    {name} = arith.constant {value} : {}",
                    mlir_type(ty)
                ));
            }
            JitScalar::Int32(value) => {
                self.lines.push(format!(
                    "    {name} = arith.constant {value} : {}",
                    mlir_type(ty)
                ));
            }
            JitScalar::Int64(value) => {
                self.lines.push(format!(
                    "    {name} = arith.constant {value} : {}",
                    mlir_type(ty)
                ));
            }
            JitScalar::Float64(value) => {
                self.lines.push(format!(
                    "    {name} = arith.constant {} : {}",
                    format_float(*value),
                    mlir_type(ty)
                ));
            }
            JitScalar::Decimal128 { value, .. } => {
                self.lines.push(format!(
                    "    {name} = arith.constant {value} : {}",
                    mlir_type(ty)
                ));
            }
        }
        Ok(ScalarValueRef { name, ty })
    }

    fn emit_binary(
        &mut self,
        op: JitBinaryOp,
        left: &JitExpr,
        right: &JitExpr,
    ) -> JitResult<ScalarValueRef> {
        let lhs = self.emit_expr(left)?;
        let rhs = self.emit_expr(right)?;
        match op {
            JitBinaryOp::Add | JitBinaryOp::Sub | JitBinaryOp::Mul | JitBinaryOp::Div => {
                self.emit_arithmetic(op, lhs, rhs)
            }
            JitBinaryOp::Eq
            | JitBinaryOp::NotEq
            | JitBinaryOp::Lt
            | JitBinaryOp::LtEq
            | JitBinaryOp::Gt
            | JitBinaryOp::GtEq => self.emit_comparison(op, lhs, rhs),
            JitBinaryOp::And | JitBinaryOp::Or => self.emit_boolean(op, lhs, rhs),
        }
    }

    fn emit_arithmetic(
        &mut self,
        op: JitBinaryOp,
        lhs: ScalarValueRef,
        rhs: ScalarValueRef,
    ) -> JitResult<ScalarValueRef> {
        ensure_same_type(&lhs, &rhs)?;
        let opcode = match (op, lhs.ty) {
            (JitBinaryOp::Add, JitType::Int32 | JitType::Int64 | JitType::Decimal128 { .. }) => {
                "addi"
            }
            (JitBinaryOp::Sub, JitType::Int32 | JitType::Int64 | JitType::Decimal128 { .. }) => {
                "subi"
            }
            (JitBinaryOp::Mul, JitType::Int32 | JitType::Int64 | JitType::Decimal128 { .. }) => {
                "muli"
            }
            (JitBinaryOp::Div, JitType::Int32 | JitType::Int64) => "divsi",
            (JitBinaryOp::Add, JitType::Float64) => "addf",
            (JitBinaryOp::Sub, JitType::Float64) => "subf",
            (JitBinaryOp::Mul, JitType::Float64) => "mulf",
            (JitBinaryOp::Div, JitType::Float64) => "divf",
            _ => {
                return Err(JitError::UnsupportedExpr(format!(
                    "operator {} is not supported for {}",
                    format_op(op),
                    mlir_type(lhs.ty)
                )));
            }
        };
        let result = self.next_value("arith");
        self.lines.push(format!(
            "    {result} = arith.{opcode} {}, {} : {}",
            lhs.name,
            rhs.name,
            mlir_type(lhs.ty)
        ));
        Ok(ScalarValueRef {
            name: result,
            ty: lhs.ty,
        })
    }

    fn emit_comparison(
        &mut self,
        op: JitBinaryOp,
        lhs: ScalarValueRef,
        rhs: ScalarValueRef,
    ) -> JitResult<ScalarValueRef> {
        ensure_same_type(&lhs, &rhs)?;
        let result = self.next_value("cmp");
        match lhs.ty {
            JitType::Float64 => {
                let predicate = match op {
                    JitBinaryOp::Eq => "oeq",
                    JitBinaryOp::NotEq => "one",
                    JitBinaryOp::Lt => "olt",
                    JitBinaryOp::LtEq => "ole",
                    JitBinaryOp::Gt => "ogt",
                    JitBinaryOp::GtEq => "oge",
                    _ => unreachable!(),
                };
                self.lines.push(format!(
                    "    {result} = arith.cmpf {predicate}, {}, {} : {}",
                    lhs.name,
                    rhs.name,
                    mlir_type(lhs.ty)
                ));
            }
            JitType::Bool if matches!(op, JitBinaryOp::Eq | JitBinaryOp::NotEq) => {
                let predicate = if matches!(op, JitBinaryOp::Eq) {
                    "eq"
                } else {
                    "ne"
                };
                self.lines.push(format!(
                    "    {result} = arith.cmpi {predicate}, {}, {} : {}",
                    lhs.name,
                    rhs.name,
                    mlir_type(lhs.ty)
                ));
            }
            JitType::Bool => {
                return Err(JitError::UnsupportedExpr(format!(
                    "ordered comparison {} is not supported for bool",
                    format_op(op)
                )));
            }
            JitType::Date32 | JitType::Int32 | JitType::Int64 | JitType::Decimal128 { .. } => {
                let predicate = match op {
                    JitBinaryOp::Eq => "eq",
                    JitBinaryOp::NotEq => "ne",
                    JitBinaryOp::Lt => "slt",
                    JitBinaryOp::LtEq => "sle",
                    JitBinaryOp::Gt => "sgt",
                    JitBinaryOp::GtEq => "sge",
                    _ => unreachable!(),
                };
                self.lines.push(format!(
                    "    {result} = arith.cmpi {predicate}, {}, {} : {}",
                    lhs.name,
                    rhs.name,
                    mlir_type(lhs.ty)
                ));
            }
        }
        Ok(ScalarValueRef {
            name: result,
            ty: JitType::Bool,
        })
    }

    fn emit_boolean(
        &mut self,
        op: JitBinaryOp,
        lhs: ScalarValueRef,
        rhs: ScalarValueRef,
    ) -> JitResult<ScalarValueRef> {
        ensure_same_type(&lhs, &rhs)?;
        if lhs.ty != JitType::Bool {
            return Err(JitError::UnsupportedExpr(format!(
                "boolean operator {} requires i1 inputs",
                format_op(op)
            )));
        }
        let opcode = match op {
            JitBinaryOp::And => "andi",
            JitBinaryOp::Or => "ori",
            _ => unreachable!(),
        };
        let result = self.next_value("bool");
        self.lines.push(format!(
            "    {result} = arith.{opcode} {}, {} : i1",
            lhs.name, rhs.name
        ));
        Ok(ScalarValueRef {
            name: result,
            ty: JitType::Bool,
        })
    }

    fn next_value(&mut self, prefix: &str) -> String {
        let id = self.next_id;
        self.next_id += 1;
        format!("%{prefix}{id}")
    }
}

fn ensure_same_type(lhs: &ScalarValueRef, rhs: &ScalarValueRef) -> JitResult<()> {
    if lhs.ty == rhs.ty {
        Ok(())
    } else {
        Err(JitError::UnsupportedExpr(format!(
            "type mismatch: {} vs {}",
            mlir_type(lhs.ty),
            mlir_type(rhs.ty)
        )))
    }
}

fn mlir_type(ty: JitType) -> &'static str {
    match ty {
        JitType::Bool => "i1",
        JitType::Date32 => "i32",
        JitType::Int32 => "i32",
        JitType::Int64 => "i64",
        JitType::Float64 => "f64",
        JitType::Decimal128 { .. } => "i128",
    }
}

fn format_float(value: f64) -> String {
    if value.is_finite() {
        format!("{value:e}")
    } else if value.is_nan() {
        "0.0".to_string()
    } else if value.is_sign_positive() {
        "1.7976931348623157e308".to_string()
    } else {
        "-1.7976931348623157e308".to_string()
    }
}

fn format_expr(expr: &JitExpr) -> String {
    match expr {
        JitExpr::Column {
            index,
            name,
            ty,
            nullable,
        } => format!(
            "col({index}, {name}, {}, nullable={nullable})",
            format_type(*ty)
        ),
        JitExpr::Literal(value) => format_scalar(value),
        JitExpr::Binary {
            op, left, right, ..
        } => format!(
            "({} {} {})",
            format_expr(left),
            format_op(*op),
            format_expr(right)
        ),
        JitExpr::IsNull(arg) => format!("is_null({})", format_expr(arg)),
    }
}

fn format_scalar(value: &JitScalar) -> String {
    match value {
        JitScalar::Null(ty) => format!("null:{}", format_type(*ty)),
        JitScalar::Bool(value) => value.to_string(),
        JitScalar::Date32(value) => format!("{value}:date32"),
        JitScalar::Int32(value) => format!("{value}:i32"),
        JitScalar::Int64(value) => format!("{value}:i64"),
        JitScalar::Float64(value) => format!("{value}:f64"),
        JitScalar::Decimal128 {
            value,
            precision,
            scale,
        } => {
            format!("{value}:decimal128({precision},{scale})")
        }
    }
}

fn format_type(ty: JitType) -> &'static str {
    match ty {
        JitType::Bool => "i1",
        JitType::Date32 => "date32",
        JitType::Int32 => "i32",
        JitType::Int64 => "i64",
        JitType::Float64 => "f64",
        JitType::Decimal128 { .. } => "decimal128",
    }
}

fn format_op(op: JitBinaryOp) -> &'static str {
    match op {
        JitBinaryOp::Add => "+",
        JitBinaryOp::Sub => "-",
        JitBinaryOp::Mul => "*",
        JitBinaryOp::Div => "/",
        JitBinaryOp::Eq => "==",
        JitBinaryOp::NotEq => "!=",
        JitBinaryOp::Lt => "<",
        JitBinaryOp::LtEq => "<=",
        JitBinaryOp::Gt => ">",
        JitBinaryOp::GtEq => ">=",
        JitBinaryOp::And => "and",
        JitBinaryOp::Or => "or",
    }
}
