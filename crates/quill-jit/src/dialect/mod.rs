use std::fmt::Write;

use crate::{
    GroupAggregate, JitBinaryOp, JitError, JitExpr, JitProjection, JitResult, JitScalar, JitType,
    PipelineGraph, PipelineKind, PipelineSink, PipelineSource, PipelineSpec, PipelineStage,
};

#[derive(Debug, Clone, PartialEq)]
pub struct QuillDialectModule {
    pub symbol: String,
    pub kind: PipelineKind,
    pub source: QuillDialectSource,
    pub ops: Vec<QuillDialectOp>,
    pub sink: QuillDialectSink,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuillDialectSource {
    ArrowBatch,
    ArrowStream,
}

#[derive(Debug, Clone, PartialEq)]
pub enum QuillDialectOp {
    Filter { predicate: JitExpr },
    Project { projections: Vec<JitProjection> },
    Limit { fetch: usize },
}

#[derive(Debug, Clone, PartialEq)]
pub enum QuillDialectSink {
    RecordBatch,
    PlainSum {
        measure: JitExpr,
    },
    GroupAggregate {
        keys: Vec<JitExpr>,
        aggregates: Vec<GroupAggregate>,
    },
}

impl QuillDialectModule {
    pub fn from_graph(symbol: impl Into<String>, graph: &PipelineGraph) -> Self {
        let source = match &graph.source {
            PipelineSource::ArrowBatch => QuillDialectSource::ArrowBatch,
            PipelineSource::ArrowStream => QuillDialectSource::ArrowStream,
        };
        let ops = graph
            .stages
            .iter()
            .map(|stage| match stage {
                PipelineStage::Filter(predicate) => QuillDialectOp::Filter {
                    predicate: predicate.clone(),
                },
                PipelineStage::Projection(projections) => QuillDialectOp::Project {
                    projections: projections.clone(),
                },
                PipelineStage::Limit(fetch) => QuillDialectOp::Limit { fetch: *fetch },
            })
            .collect();
        let sink = match &graph.sink {
            PipelineSink::RecordBatch => QuillDialectSink::RecordBatch,
            PipelineSink::Sum { measure } => QuillDialectSink::PlainSum {
                measure: measure.clone(),
            },
            PipelineSink::GroupAggregate {
                keys, aggregates, ..
            } => QuillDialectSink::GroupAggregate {
                keys: keys.clone(),
                aggregates: aggregates.clone(),
            },
        };

        Self {
            symbol: symbol.into(),
            kind: sink.kind(),
            source,
            ops,
            sink,
        }
    }

    pub fn to_mlir_text(&self) -> JitResult<String> {
        let mut text = String::new();
        let _ = writeln!(text, "module {{");
        let _ = writeln!(text, "  func.func @{}() {{", self.symbol);
        let _ = writeln!(text, "    %batch0 = {} : !quill.batch", self.source.name());
        if let Some(spec) = self.pipeline_spec() {
            let _ = writeln!(text, "    // qjit.pipeline = {}", spec.name());
        }

        let mut batch = "%batch0".to_string();
        let mut selection = None::<String>;
        let mut next_batch = 1_usize;
        let mut next_selection = 0_usize;

        for op in &self.ops {
            match op {
                QuillDialectOp::Filter { predicate } => {
                    let out = format!("%sel{next_selection}");
                    next_selection += 1;
                    append_filter(&mut text, &out, &batch, predicate)?;
                    selection = Some(out);
                }
                QuillDialectOp::Project { projections } => {
                    let sel =
                        ensure_selection(&mut text, &batch, &mut selection, &mut next_selection)?;
                    let out = format!("%batch{next_batch}");
                    next_batch += 1;
                    append_project(&mut text, &out, &batch, &sel, projections)?;
                    batch = out;
                }
                QuillDialectOp::Limit { fetch } => {
                    let sel =
                        ensure_selection(&mut text, &batch, &mut selection, &mut next_selection)?;
                    let out = format!("%sel{next_selection}");
                    next_selection += 1;
                    let _ = writeln!(
                        text,
                        "    {out} = quill.exec.limit {sel} {{ fetch = {fetch} : i64 }} : !quill.selection -> !quill.selection"
                    );
                    selection = Some(out);
                }
            }
        }

        self.append_sink(&mut text, &batch, selection.as_deref(), &mut next_selection)?;
        let _ = writeln!(text, "    return");
        let _ = writeln!(text, "  }}");
        let _ = writeln!(text, "}}");
        Ok(text)
    }

    fn append_sink(
        &self,
        text: &mut String,
        batch: &str,
        selection: Option<&str>,
        next_selection: &mut usize,
    ) -> JitResult<()> {
        match &self.sink {
            QuillDialectSink::RecordBatch => {
                let selection = match selection {
                    Some(selection) => selection.to_string(),
                    None => {
                        let mut selection = None;
                        ensure_selection(text, batch, &mut selection, next_selection)?
                    }
                };
                let _ = writeln!(
                    text,
                    "    quill.sink.record_batch {batch}, {selection} : !quill.batch, !quill.selection"
                );
            }
            QuillDialectSink::PlainSum { measure } => {
                let selection = match selection {
                    Some(selection) => selection.to_string(),
                    None => {
                        let mut selection = None;
                        ensure_selection(text, batch, &mut selection, next_selection)?
                    }
                };
                append_plain_sum(text, "%sum0", batch, &selection, measure)?;
            }
            QuillDialectSink::GroupAggregate { keys, aggregates } => {
                let selection = match selection {
                    Some(selection) => selection.to_string(),
                    None => {
                        let mut selection = None;
                        ensure_selection(text, batch, &mut selection, next_selection)?
                    }
                };
                append_group_aggregate(
                    text, "%groups0", "%group0", batch, &selection, keys, aggregates,
                )?;
            }
        }
        Ok(())
    }

    pub fn pipeline_spec(&self) -> Option<PipelineSpec> {
        match (&self.source, self.ops.as_slice(), &self.sink) {
            (
                QuillDialectSource::ArrowBatch,
                [QuillDialectOp::Filter { predicate }, QuillDialectOp::Project { projections }],
                QuillDialectSink::RecordBatch,
            ) => PipelineSpec::record_project(predicate, projections),
            (
                QuillDialectSource::ArrowBatch,
                [QuillDialectOp::Filter { predicate }],
                QuillDialectSink::PlainSum { measure },
            ) => PipelineSpec::filter_sum(predicate, measure),
            (
                QuillDialectSource::ArrowBatch,
                [],
                QuillDialectSink::GroupAggregate { keys, aggregates },
            ) => PipelineSpec::group_aggregate(None, keys, aggregates),
            (
                QuillDialectSource::ArrowBatch,
                [QuillDialectOp::Filter { predicate }],
                QuillDialectSink::GroupAggregate { keys, aggregates },
            ) => PipelineSpec::group_aggregate(Some(predicate), keys, aggregates),
            _ => None,
        }
    }
}

impl QuillDialectSource {
    fn name(self) -> &'static str {
        match self {
            Self::ArrowBatch => "quill.source.arrow_batch",
            Self::ArrowStream => "quill.source.arrow_stream",
        }
    }
}

impl QuillDialectSink {
    fn kind(&self) -> PipelineKind {
        match self {
            Self::RecordBatch => PipelineKind::Record,
            Self::PlainSum { .. } | Self::GroupAggregate { .. } => PipelineKind::Aggregate,
        }
    }
}

fn ensure_selection(
    text: &mut String,
    batch: &str,
    selection: &mut Option<String>,
    next_selection: &mut usize,
) -> JitResult<String> {
    if let Some(selection) = selection {
        return Ok(selection.clone());
    }

    let out = format!("%sel{next_selection}");
    *next_selection += 1;
    append_filter(text, &out, batch, &JitExpr::Literal(JitScalar::Bool(true)))?;
    *selection = Some(out.clone());
    Ok(out)
}

fn append_filter(text: &mut String, out: &str, batch: &str, predicate: &JitExpr) -> JitResult<()> {
    let mut emitter = RegionEmitter::new("row");
    let value = emitter.emit_expr(predicate)?;
    if value.ty != JitType::Bool {
        return Err(JitError::UnsupportedExpr(
            "filter region must yield i1".to_string(),
        ));
    }
    let _ = writeln!(text, "    {out} = quill.exec.filter {batch} {{");
    append_region_body(text, &emitter, &value)?;
    let _ = writeln!(text, "    }} : !quill.batch -> !quill.selection");
    Ok(())
}

fn append_project(
    text: &mut String,
    out: &str,
    batch: &str,
    selection: &str,
    projections: &[JitProjection],
) -> JitResult<()> {
    let mut emitter = RegionEmitter::new("row");
    let values = projections
        .iter()
        .map(|projection| emitter.emit_expr(&projection.expr))
        .collect::<JitResult<Vec<_>>>()?;
    if values.is_empty() {
        return Err(JitError::UnsupportedExpr(
            "project region must yield at least one value".to_string(),
        ));
    }
    let _ = writeln!(
        text,
        "    {out} = quill.exec.project {batch}, {selection} {{"
    );
    append_region_values(text, &emitter, &values)?;
    let _ = writeln!(
        text,
        "    }} : !quill.batch, !quill.selection -> !quill.batch"
    );
    Ok(())
}

fn append_plain_sum(
    text: &mut String,
    out: &str,
    batch: &str,
    selection: &str,
    measure: &JitExpr,
) -> JitResult<()> {
    let mut emitter = RegionEmitter::new("row");
    let value = emitter.emit_expr(measure)?;
    if !is_numeric_type(value.ty) {
        return Err(JitError::UnsupportedExpr(
            "plain_sum region must yield a numeric scalar".to_string(),
        ));
    }
    let _ = writeln!(
        text,
        "    {out} = quill.sink.plain_sum {batch}, {selection} {{"
    );
    append_region_body(text, &emitter, &value)?;
    let _ = writeln!(
        text,
        "    }} : !quill.batch, !quill.selection -> !quill.scalar"
    );
    Ok(())
}

fn append_group_aggregate(
    text: &mut String,
    group_ids: &str,
    out: &str,
    batch: &str,
    selection: &str,
    keys: &[JitExpr],
    aggregates: &[GroupAggregate],
) -> JitResult<()> {
    if keys.is_empty() || aggregates.is_empty() {
        return Err(JitError::UnsupportedExpr(
            "group_aggregate requires at least one key and one aggregate".to_string(),
        ));
    }

    append_group_ids(text, group_ids, batch, selection, keys)?;
    append_group_update(text, out, batch, selection, group_ids, aggregates)
}

fn append_group_ids(
    text: &mut String,
    out: &str,
    batch: &str,
    selection: &str,
    keys: &[JitExpr],
) -> JitResult<()> {
    let mut emitter = RegionEmitter::new("row");
    let values = keys
        .iter()
        .map(|key| emitter.emit_expr(key))
        .collect::<JitResult<Vec<_>>>()?;
    let _ = writeln!(
        text,
        "    {out} = quill.exec.group_ids {batch}, {selection} {{"
    );
    append_region_values(text, &emitter, &values)?;
    let _ = writeln!(
        text,
        "    }} : !quill.batch, !quill.selection -> !quill.group_ids"
    );
    Ok(())
}

fn append_group_update(
    text: &mut String,
    out: &str,
    batch: &str,
    selection: &str,
    group_ids: &str,
    aggregates: &[GroupAggregate],
) -> JitResult<()> {
    let mut emitter = RegionEmitter::new("row");
    let values = aggregates
        .iter()
        .map(|aggregate| emitter.emit_expr(&aggregate.expr))
        .collect::<JitResult<Vec<_>>>()?;

    let funcs = aggregates
        .iter()
        .map(|aggregate| aggregate.func.name())
        .collect::<Vec<_>>();
    let state_types = aggregates
        .iter()
        .flat_map(|aggregate| aggregate.state_types.iter().copied())
        .map(state_type_name)
        .collect::<Vec<_>>();
    let funcs_attr = mlir_string_array(&funcs);
    let state_types_attr = mlir_string_array(&state_types);
    let _ = writeln!(
        text,
        "    {out} = quill.sink.group_update {batch}, {selection}, {group_ids} {{"
    );
    append_region_values(text, &emitter, &values)?;
    let _ = writeln!(
        text,
        "    }} {{ aggregate_funcs = {funcs_attr}, state_types = {state_types_attr} }} : !quill.batch, !quill.selection, !quill.group_ids -> !quill.batch"
    );
    Ok(())
}

fn mlir_string_array(values: &[&str]) -> String {
    let values = values
        .iter()
        .map(|value| format!("\"{value}\""))
        .collect::<Vec<_>>()
        .join(", ");
    format!("[{values}]")
}

fn append_region_body(
    text: &mut String,
    emitter: &RegionEmitter,
    value: &ScalarValueRef,
) -> JitResult<()> {
    append_region_values(text, emitter, std::slice::from_ref(value))
}

fn append_region_values(
    text: &mut String,
    emitter: &RegionEmitter,
    values: &[ScalarValueRef],
) -> JitResult<()> {
    let _ = writeln!(text, "    ^bb0(%{}: !quill.row):", emitter.row_name);
    for line in &emitter.lines {
        let _ = writeln!(text, "      {line}");
    }
    let value_names = values
        .iter()
        .map(|value| value.name.as_str())
        .collect::<Vec<_>>()
        .join(", ");
    let value_types = values
        .iter()
        .map(|value| mlir_type(value.ty))
        .collect::<Vec<_>>()
        .join(", ");
    let _ = writeln!(text, "      quill.yield {value_names} : {value_types}");
    Ok(())
}

#[derive(Debug, Clone)]
struct ScalarValueRef {
    name: String,
    ty: JitType,
}

#[derive(Debug)]
struct RegionEmitter {
    row_name: String,
    next_id: usize,
    lines: Vec<String>,
}

impl RegionEmitter {
    fn new(row_name: impl Into<String>) -> Self {
        Self {
            row_name: row_name.into(),
            next_id: 0,
            lines: Vec::new(),
        }
    }

    fn emit_expr(&mut self, expr: &JitExpr) -> JitResult<ScalarValueRef> {
        match expr {
            JitExpr::Column { index, ty, .. } => {
                let name = self.next_value("col");
                self.lines.push(format!(
                    "{name} = quill.column %{} {{ index = {index} : i64 }} : !quill.row -> {}",
                    self.row_name,
                    mlir_type(*ty)
                ));
                Ok(ScalarValueRef { name, ty: *ty })
            }
            JitExpr::Literal(value) => self.emit_literal(value),
            JitExpr::Binary {
                op, left, right, ..
            } => self.emit_binary(*op, left, right),
            JitExpr::Cast { expr, ty, .. } => {
                let value = self.emit_expr(expr)?;
                self.emit_cast(value, *ty)
            }
            JitExpr::IsNull(_) => Err(JitError::UnsupportedExpr(
                "Quill dialect regions do not yet model Arrow validity bitmaps".to_string(),
            )),
        }
    }

    fn emit_literal(&mut self, value: &JitScalar) -> JitResult<ScalarValueRef> {
        let ty = value.ty();
        let name = self.next_value("lit");
        let literal = match value {
            JitScalar::Null(_) => {
                return Err(JitError::UnsupportedExpr(
                    "Quill dialect regions do not yet model null literals".to_string(),
                ));
            }
            JitScalar::Bool(value) => {
                self.lines.push(format!("{name} = arith.constant {value}"));
                return Ok(ScalarValueRef { name, ty });
            }
            JitScalar::Date32(value) => value.to_string(),
            JitScalar::Int32(value) => value.to_string(),
            JitScalar::Int64(value) => value.to_string(),
            JitScalar::UInt64(value) => value.to_string(),
            JitScalar::Float64(value) => format_float(*value),
            JitScalar::Utf8(_) => {
                return Err(JitError::UnsupportedExpr(
                    "Quill dialect regions do not yet lower Utf8 literals".to_string(),
                ));
            }
            JitScalar::Decimal128 { value, .. } => value.to_string(),
        };
        self.lines.push(format!(
            "{name} = arith.constant {literal} : {}",
            mlir_type(ty)
        ));
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

    fn emit_cast(&mut self, value: ScalarValueRef, ty: JitType) -> JitResult<ScalarValueRef> {
        if value.ty == ty {
            return Ok(value);
        }

        let opcode = match (value.ty, ty) {
            (JitType::Int32, JitType::Int64) => "extsi",
            (JitType::Int32 | JitType::Int64, JitType::Float64) => "sitofp",
            (JitType::UInt64, JitType::Float64) => "uitofp",
            (JitType::Float64, JitType::Int64) => "fptosi",
            _ => {
                return Err(JitError::UnsupportedExpr(format!(
                    "cast from {} to {} is not supported",
                    mlir_type(value.ty),
                    mlir_type(ty)
                )));
            }
        };
        let result = self.next_value("cast");
        self.lines.push(format!(
            "{result} = arith.{opcode} {} : {} to {}",
            value.name,
            mlir_type(value.ty),
            mlir_type(ty)
        ));
        Ok(ScalarValueRef { name: result, ty })
    }

    fn emit_arithmetic(
        &mut self,
        op: JitBinaryOp,
        lhs: ScalarValueRef,
        rhs: ScalarValueRef,
    ) -> JitResult<ScalarValueRef> {
        ensure_same_type(&lhs, &rhs)?;
        let opcode = match (op, lhs.ty) {
            (
                JitBinaryOp::Add,
                JitType::Int32 | JitType::Int64 | JitType::UInt64 | JitType::Decimal128 { .. },
            ) => "addi",
            (
                JitBinaryOp::Sub,
                JitType::Int32 | JitType::Int64 | JitType::UInt64 | JitType::Decimal128 { .. },
            ) => "subi",
            (
                JitBinaryOp::Mul,
                JitType::Int32 | JitType::Int64 | JitType::UInt64 | JitType::Decimal128 { .. },
            ) => "muli",
            (JitBinaryOp::Div, JitType::Int32 | JitType::Int64 | JitType::Decimal128 { .. }) => {
                "divsi"
            }
            (JitBinaryOp::Add, JitType::Float64) => "addf",
            (JitBinaryOp::Sub, JitType::Float64) => "subf",
            (JitBinaryOp::Mul, JitType::Float64) => "mulf",
            (JitBinaryOp::Div, JitType::Float64) => "divf",
            _ => {
                return Err(JitError::UnsupportedExpr(format!(
                    "operator {op} is not supported for {}",
                    mlir_type(lhs.ty)
                )));
            }
        };
        let result = self.next_value("arith");
        self.lines.push(format!(
            "{result} = arith.{opcode} {}, {} : {}",
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
                    "{result} = arith.cmpf {predicate}, {}, {} : {}",
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
                    "{result} = arith.cmpi {predicate}, {}, {} : i1",
                    lhs.name, rhs.name
                ));
            }
            JitType::Bool => {
                return Err(JitError::UnsupportedExpr(format!(
                    "ordered comparison {op} is not supported for bool"
                )));
            }
            JitType::Utf8 => {
                return Err(JitError::UnsupportedExpr(
                    "Utf8 comparisons are not supported by MLIR lowering".to_string(),
                ));
            }
            JitType::UInt64 => {
                let predicate = match op {
                    JitBinaryOp::Eq => "eq",
                    JitBinaryOp::NotEq => "ne",
                    JitBinaryOp::Lt => "ult",
                    JitBinaryOp::LtEq => "ule",
                    JitBinaryOp::Gt => "ugt",
                    JitBinaryOp::GtEq => "uge",
                    _ => unreachable!(),
                };
                self.lines.push(format!(
                    "{result} = arith.cmpi {predicate}, {}, {} : {}",
                    lhs.name,
                    rhs.name,
                    mlir_type(lhs.ty)
                ));
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
                    "{result} = arith.cmpi {predicate}, {}, {} : {}",
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
                "boolean operator {op} requires i1 inputs"
            )));
        }
        let opcode = match op {
            JitBinaryOp::And => "andi",
            JitBinaryOp::Or => "ori",
            _ => unreachable!(),
        };
        let result = self.next_value("bool");
        self.lines.push(format!(
            "{result} = arith.{opcode} {}, {} : i1",
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

fn is_numeric_type(ty: JitType) -> bool {
    matches!(
        ty,
        JitType::Date32
            | JitType::Int32
            | JitType::Int64
            | JitType::UInt64
            | JitType::Float64
            | JitType::Decimal128 { .. }
    )
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
        JitType::UInt64 => "i64",
        JitType::Float64 => "f64",
        JitType::Utf8 => "!quill.scalar",
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

fn state_type_name(ty: JitType) -> &'static str {
    match ty {
        JitType::Int64 => "i64",
        JitType::UInt64 => "u64",
        JitType::Float64 => "f64",
        JitType::Decimal128 { .. } => "i128",
        JitType::Bool => "i1",
        JitType::Date32 | JitType::Int32 => "i32",
        JitType::Utf8 => "scalar",
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        AggregateFunc, GroupAggregate, JitBinaryOp, JitExpr, JitProjection, JitScalar, JitType,
        PipelineGraph, PipelineKind, PipelineStage, QuillDialectModule,
    };

    #[test]
    fn emits_record_pipeline_dialect_skeleton() {
        let predicate = i64_gt_ten();
        let projection = JitProjection::new(
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
        );
        let pipeline = PipelineGraph::record(vec![
            PipelineStage::Filter(predicate),
            PipelineStage::Projection(vec![projection]),
        ]);

        let module = QuillDialectModule::from_graph("record0", &pipeline);
        let text = module.to_mlir_text().unwrap();

        assert_eq!(module.kind, PipelineKind::Record);
        assert!(text.contains("func.func @record0"));
        assert!(text.contains("quill.source.arrow_batch"));
        assert!(text.contains("quill.exec.filter"));
        assert!(text.contains("quill.exec.project"));
        assert!(text.contains("quill.sink.record_batch"));
        assert!(!text.contains("predicate ="));
        assert!(text.contains("quill.yield"));
    }

    #[test]
    fn emits_plain_sum_pipeline_dialect_skeleton() {
        let predicate = i64_gt_ten();
        let measure = JitExpr::Column {
            index: 1,
            name: "price".to_string(),
            ty: JitType::Float64,
            nullable: false,
        };
        let pipeline = PipelineGraph::filter_sum(predicate, measure);

        let module = QuillDialectModule::from_graph("sum0", &pipeline);
        let text = module.to_mlir_text().unwrap();

        assert_eq!(module.kind, PipelineKind::Aggregate);
        assert!(text.contains("quill.exec.filter"));
        assert!(text.contains("quill.sink.plain_sum"));
        assert!(!text.contains("measure ="));
        assert!(text.contains("quill.column %row { index = 1 : i64 } : !quill.row -> f64"));
    }

    #[test]
    fn emits_group_aggregate_pipeline_dialect_skeleton() {
        let key = JitExpr::Column {
            index: 0,
            name: "returnflag".to_string(),
            ty: JitType::Int64,
            nullable: false,
        };
        let aggregate = GroupAggregate::new(
            AggregateFunc::Sum,
            JitExpr::Column {
                index: 1,
                name: "quantity".to_string(),
                ty: JitType::Float64,
                nullable: false,
            },
            JitType::Float64,
            "sum_qty",
        );
        let pipeline = PipelineGraph::group_aggregate(vec![], vec![key], vec![aggregate]);

        let module = QuillDialectModule::from_graph("group0", &pipeline);
        let text = module.to_mlir_text().unwrap();

        assert_eq!(module.kind, PipelineKind::Aggregate);
        assert_eq!(
            module.pipeline_spec().map(|spec| spec.name()),
            Some("group_aggregate")
        );
        assert!(text.contains("quill.exec.group_ids"));
        assert!(text.contains("quill.sink.group_update"));
        assert!(text.contains("aggregate_funcs = [\"sum\"]"));
        assert!(text.contains("// qjit.pipeline = group_aggregate"));
        assert!(text.contains("!quill.group_ids"));
        assert!(text.contains("quill.yield"));
    }

    fn i64_gt_ten() -> JitExpr {
        JitExpr::Binary {
            op: JitBinaryOp::Gt,
            left: Box::new(JitExpr::Column {
                index: 0,
                name: "v".to_string(),
                ty: JitType::Int64,
                nullable: false,
            }),
            right: Box::new(JitExpr::Literal(JitScalar::Int64(10))),
            ty: JitType::Bool,
            nullable: false,
        }
    }
}
