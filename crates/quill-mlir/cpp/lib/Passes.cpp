#include "Quill/IR/Dialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/ADT/SmallVector.h"

#include <map>

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif

namespace {

struct ColumnInfo {
  int64_t index;
  mlir::Type type;
  mlir::Value pointer;
};

struct StateFieldInfo {
  llvm::StringRef kind;
  mlir::Type type;
  mlir::Value values;
  mlir::Value valid;
};

static mlir::LogicalResult collectColumns(mlir::Region &region,
                                          std::map<int64_t, mlir::Type> &columns,
                                          mlir::Operation *owner,
                                          bool requireColumn = true) {
  auto walkResult = region.walk([&](mlir::quill::ColumnOp column) {
    int64_t index = static_cast<int64_t>(column.getIndex());
    mlir::Type type = column.getResult().getType();
    auto [it, inserted] = columns.try_emplace(index, type);
    if (!inserted && it->second != type) {
      column.emitOpError("has inconsistent type for repeated column index ")
          << index;
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });
  if (walkResult.wasInterrupted())
    return mlir::failure();
  if (requireColumn && columns.empty())
    return owner->emitOpError("lowering requires at least one column access");
  return mlir::success();
}

static mlir::Value loadColumn(mlir::OpBuilder &builder, mlir::Location loc,
                              mlir::Value index, const ColumnInfo &column) {
  auto ptrType = mlir::LLVM::LLVMPointerType::get(builder.getContext());
  llvm::SmallVector<mlir::LLVM::GEPArg> indices;
  indices.push_back(index);
  auto elementPtr = builder.create<mlir::LLVM::GEPOp>(
      loc, ptrType, column.type, column.pointer, indices);
  return builder.create<mlir::LLVM::LoadOp>(loc, column.type, elementPtr);
}

static bool isSupportedFixedType(mlir::Type type) {
  return type.isInteger(32) || type.isInteger(64) || type.isInteger(128) ||
         type.isF64();
}

static mlir::LogicalResult cloneRegionValues(
    mlir::OpBuilder &builder, mlir::Region &region,
    const llvm::DenseMap<int64_t, mlir::Value> &loadedColumns,
    mlir::Operation *owner, llvm::SmallVectorImpl<mlir::Value> &results) {
  mlir::Block &block = region.front();
  mlir::IRMapping mapping;

  for (mlir::Operation &op : block.without_terminator()) {
    if (auto column = llvm::dyn_cast<mlir::quill::ColumnOp>(op)) {
      auto loaded = loadedColumns.find(static_cast<int64_t>(column.getIndex()));
      if (loaded == loadedColumns.end())
        return column.emitOpError("has no loaded value for column index ")
               << column.getIndex();
      if (loaded->second.getType() != column.getResult().getType())
        return column.emitOpError("loaded column type does not match result");
      mapping.map(column.getResult(), loaded->second);
      continue;
    }

    if (op.getDialect()->getNamespace() == "quill")
      return op.emitOpError("cannot be cloned into loop body");

    mlir::Operation *cloned = builder.clone(op, mapping);
    mapping.map(op.getResults(), cloned->getResults());
  }

  auto yield = llvm::dyn_cast<mlir::quill::YieldOp>(block.getTerminator());
  if (!yield || yield.getValues().empty())
    return owner->emitOpError("region must yield at least one value");

  for (mlir::Value value : yield.getValues())
    results.push_back(mapping.lookupOrDefault(value));
  return mlir::success();
}

static mlir::LogicalResult cloneRegionValue(
    mlir::OpBuilder &builder, mlir::Region &region,
    const llvm::DenseMap<int64_t, mlir::Value> &loadedColumns,
    mlir::Operation *owner, mlir::Value &result) {
  llvm::SmallVector<mlir::Value> values;
  if (mlir::failed(cloneRegionValues(builder, region, loadedColumns, owner,
                                     values)))
    return mlir::failure();
  if (values.size() != 1)
    return owner->emitOpError("region must yield one value");
  result = values.front();
  return mlir::success();
}

static bool isLiteralTrueFilter(mlir::quill::FilterOp filter) {
  if (!filter || !llvm::hasSingleElement(filter.getPredicate()))
    return false;
  mlir::Block &block = filter.getPredicate().front();
  auto yield = llvm::dyn_cast<mlir::quill::YieldOp>(block.getTerminator());
  if (!yield || yield.getValues().size() != 1)
    return false;
  auto constant =
      yield.getValues().front().getDefiningOp<mlir::arith::ConstantOp>();
  if (!constant)
    return false;
  auto value = llvm::dyn_cast<mlir::BoolAttr>(constant.getValue());
  return value && value.getValue();
}

static mlir::LogicalResult lowerFilterProject(mlir::func::FuncOp func) {
  if (func.isExternal() || !llvm::hasSingleElement(func.getBody()))
    return mlir::failure();

  mlir::quill::FilterOp filter;
  mlir::quill::ProjectOp project;
  mlir::quill::RecordBatchSinkOp sink;
  func.walk([&](mlir::quill::FilterOp op) {
    if (!filter)
      filter = op;
  });
  func.walk([&](mlir::quill::ProjectOp op) {
    if (!project)
      project = op;
  });
  func.walk([&](mlir::quill::RecordBatchSinkOp op) {
    if (!sink)
      sink = op;
  });

  if (!filter || !project || !sink)
    return mlir::failure();

  std::map<int64_t, mlir::Type> columnTypes;
  if (mlir::failed(collectColumns(filter.getPredicate(), columnTypes, filter)))
    return mlir::failure();
  if (mlir::failed(collectColumns(project.getProjector(), columnTypes, project)))
    return mlir::failure();

  mlir::Block &projectBlock = project.getProjector().front();
  auto projectYield =
      llvm::dyn_cast<mlir::quill::YieldOp>(projectBlock.getTerminator());
  if (!projectYield || projectYield.getValues().empty())
    return project.emitOpError("projector region must yield at least one value");

  llvm::SmallVector<mlir::Type> outputTypes;
  for (mlir::Value value : projectYield.getValues()) {
    mlir::Type type = value.getType();
    if (!isSupportedFixedType(type))
      return project.emitOpError("lowering requires fixed-width projection results");
    outputTypes.push_back(type);
  }

  mlir::Region predicateRegion;
  mlir::IRMapping predicateMapping;
  filter.getPredicate().cloneInto(&predicateRegion, predicateMapping);
  mlir::Region projectorRegion;
  mlir::IRMapping projectorMapping;
  project.getProjector().cloneInto(&projectorRegion, projectorMapping);

  mlir::MLIRContext *context = func.getContext();
  mlir::OpBuilder builder(context);
  mlir::Location loc = func.getLoc();
  auto i64Type = builder.getI64Type();
  auto i32Type = builder.getI32Type();
  auto ptrType = mlir::LLVM::LLVMPointerType::get(context);

  llvm::SmallVector<mlir::Type> inputTypes;
  inputTypes.push_back(i64Type);
  for (size_t i = 0; i < columnTypes.size(); ++i)
    inputTypes.push_back(ptrType);
  for (size_t i = 0; i < outputTypes.size(); ++i)
    inputTypes.push_back(ptrType);
  inputTypes.push_back(ptrType);

  func.eraseBody();
  func.setFunctionType(mlir::FunctionType::get(context, inputTypes, i32Type));
  func->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
  mlir::Block *entry = func.addEntryBlock();
  builder.setInsertionPointToStart(entry);

  mlir::Value len = entry->getArgument(0);
  llvm::SmallVector<ColumnInfo> columns;
  size_t argIndex = 1;
  for (const auto &[index, type] : columnTypes)
    columns.push_back(ColumnInfo{index, type, entry->getArgument(argIndex++)});

  llvm::SmallVector<mlir::Value> outputPtrs;
  for (size_t i = 0; i < outputTypes.size(); ++i)
    outputPtrs.push_back(entry->getArgument(argIndex++));
  mlir::Value outputLen = entry->getArgument(argIndex++);

  mlir::Value zeroI64 = builder.create<mlir::arith::ConstantIntOp>(loc, 0, 64);
  mlir::Value oneI64 = builder.create<mlir::arith::ConstantIntOp>(loc, 1, 64);

  bool cloneFailed = false;
  auto loop = builder.create<mlir::scf::ForOp>(
      loc, zeroI64, len, oneI64, mlir::ValueRange{zeroI64},
      [&](mlir::OpBuilder &bodyBuilder, mlir::Location bodyLoc,
          mlir::Value iv, mlir::ValueRange iterArgs) {
        llvm::DenseMap<int64_t, mlir::Value> loadedColumns;
        for (const ColumnInfo &column : columns)
          loadedColumns.try_emplace(column.index,
                                    loadColumn(bodyBuilder, bodyLoc, iv, column));

        mlir::Value predicate;
        if (mlir::failed(cloneRegionValue(bodyBuilder, predicateRegion,
                                          loadedColumns, func.getOperation(),
                                          predicate))) {
          cloneFailed = true;
          bodyBuilder.create<mlir::scf::YieldOp>(bodyLoc, iterArgs);
          return;
        }

        auto branch = bodyBuilder.create<mlir::scf::IfOp>(
            bodyLoc, mlir::TypeRange{i64Type}, predicate, true);
        {
          mlir::OpBuilder thenBuilder = branch.getThenBodyBuilder();
          llvm::SmallVector<mlir::Value> projected;
          if (mlir::failed(cloneRegionValues(thenBuilder, projectorRegion,
                                             loadedColumns, func.getOperation(),
                                             projected))) {
            cloneFailed = true;
            thenBuilder.create<mlir::scf::YieldOp>(bodyLoc, iterArgs);
            return;
          }

          for (size_t index = 0; index < projected.size(); ++index) {
            mlir::Value value = projected[index];
            auto outPtr = thenBuilder.create<mlir::LLVM::GEPOp>(
                bodyLoc, ptrType, value.getType(), outputPtrs[index],
                llvm::SmallVector<mlir::LLVM::GEPArg>{iterArgs[0]});
            thenBuilder.create<mlir::LLVM::StoreOp>(bodyLoc, value, outPtr);
          }
          mlir::Value nextCount =
              thenBuilder.create<mlir::arith::AddIOp>(bodyLoc, iterArgs[0],
                                                      oneI64);
          thenBuilder.create<mlir::scf::YieldOp>(
              bodyLoc, mlir::ValueRange{nextCount});
        }
        {
          mlir::OpBuilder elseBuilder = branch.getElseBodyBuilder();
          elseBuilder.create<mlir::scf::YieldOp>(bodyLoc, iterArgs);
        }
        bodyBuilder.create<mlir::scf::YieldOp>(bodyLoc, branch.getResults());
      });
  if (cloneFailed)
    return mlir::failure();

  builder.create<mlir::LLVM::StoreOp>(loc, loop.getResult(0), outputLen);
  mlir::Value ok = builder.create<mlir::arith::ConstantIntOp>(loc, 0, 32);
  builder.create<mlir::func::ReturnOp>(loc, ok);
  return mlir::success();
}

static mlir::Value zeroForType(mlir::OpBuilder &builder, mlir::Location loc,
                               mlir::Type type) {
  if (auto integerType = llvm::dyn_cast<mlir::IntegerType>(type)) {
    return builder.create<mlir::arith::ConstantIntOp>(
        loc, 0, integerType.getWidth());
  }
  if (auto floatType = llvm::dyn_cast<mlir::FloatType>(type)) {
    return builder.create<mlir::arith::ConstantFloatOp>(
        loc, floatType, llvm::APFloat(0.0));
  }
  return {};
}

static mlir::Value addValues(mlir::OpBuilder &builder, mlir::Location loc,
                             mlir::Value lhs, mlir::Value rhs) {
  mlir::Type type = lhs.getType();
  if (llvm::isa<mlir::FloatType>(type))
    return builder.create<mlir::arith::AddFOp>(loc, lhs, rhs);
  return builder.create<mlir::arith::AddIOp>(loc, lhs, rhs);
}

static mlir::Value compareForMinMax(mlir::OpBuilder &builder,
                                    mlir::Location loc, llvm::StringRef kind,
                                    mlir::Value candidate,
                                    mlir::Value current, bool isMin) {
  mlir::Type type = candidate.getType();
  if (llvm::isa<mlir::FloatType>(type)) {
    auto predicate = isMin ? mlir::arith::CmpFPredicate::OLT
                           : mlir::arith::CmpFPredicate::OGT;
    return builder.create<mlir::arith::CmpFOp>(loc, predicate, candidate,
                                               current);
  }

  mlir::arith::CmpIPredicate predicate;
  if (kind == "u64")
    predicate = isMin ? mlir::arith::CmpIPredicate::ult
                      : mlir::arith::CmpIPredicate::ugt;
  else
    predicate = isMin ? mlir::arith::CmpIPredicate::slt
                      : mlir::arith::CmpIPredicate::sgt;
  return builder.create<mlir::arith::CmpIOp>(loc, predicate, candidate,
                                             current);
}

static mlir::Type stateTypeFromName(mlir::OpBuilder &builder,
                                    llvm::StringRef name) {
  if (name == "i64" || name == "u64")
    return builder.getI64Type();
  if (name == "f64")
    return builder.getF64Type();
  if (name == "i128")
    return builder.getIntegerType(128);
  return {};
}

static mlir::Value stateValuePtr(mlir::OpBuilder &builder, mlir::Location loc,
                                 const StateFieldInfo &field,
                                 mlir::Value groupId) {
  auto ptrType = mlir::LLVM::LLVMPointerType::get(builder.getContext());
  return builder.create<mlir::LLVM::GEPOp>(
      loc, ptrType, field.type, field.values,
      llvm::SmallVector<mlir::LLVM::GEPArg>{groupId});
}

static mlir::Value stateValidPtr(mlir::OpBuilder &builder, mlir::Location loc,
                                 const StateFieldInfo &field,
                                 mlir::Value groupId) {
  auto ptrType = mlir::LLVM::LLVMPointerType::get(builder.getContext());
  return builder.create<mlir::LLVM::GEPOp>(
      loc, ptrType, builder.getI8Type(), field.valid,
      llvm::SmallVector<mlir::LLVM::GEPArg>{groupId});
}

static mlir::Value loadStateValue(mlir::OpBuilder &builder, mlir::Location loc,
                                  const StateFieldInfo &field,
                                  mlir::Value groupId) {
  return builder.create<mlir::LLVM::LoadOp>(
      loc, field.type, stateValuePtr(builder, loc, field, groupId));
}

static mlir::Value loadStateValid(mlir::OpBuilder &builder, mlir::Location loc,
                                  const StateFieldInfo &field,
                                  mlir::Value groupId) {
  return builder.create<mlir::LLVM::LoadOp>(
      loc, builder.getI8Type(), stateValidPtr(builder, loc, field, groupId));
}

static void storeStateValue(mlir::OpBuilder &builder, mlir::Location loc,
                            const StateFieldInfo &field, mlir::Value groupId,
                            mlir::Value value) {
  builder.create<mlir::LLVM::StoreOp>(
      loc, value, stateValuePtr(builder, loc, field, groupId));
}

static void storeStateValid(mlir::OpBuilder &builder, mlir::Location loc,
                            const StateFieldInfo &field, mlir::Value groupId) {
  mlir::Value one = builder.create<mlir::arith::ConstantIntOp>(loc, 1, 8);
  builder.create<mlir::LLVM::StoreOp>(
      loc, one, stateValidPtr(builder, loc, field, groupId));
}

static void updateNullableValue(mlir::OpBuilder &builder, mlir::Location loc,
                                const StateFieldInfo &field,
                                mlir::Value groupId, mlir::Value value,
                                bool addToCurrent) {
  mlir::Value valid = loadStateValid(builder, loc, field, groupId);
  mlir::Value zeroI8 = builder.create<mlir::arith::ConstantIntOp>(loc, 0, 8);
  mlir::Value isValid = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::ne, valid, zeroI8);
  auto branch = builder.create<mlir::scf::IfOp>(
      loc, mlir::TypeRange{field.type}, isValid, true);
  {
    mlir::OpBuilder thenBuilder = branch.getThenBodyBuilder();
    mlir::Value current = loadStateValue(thenBuilder, loc, field, groupId);
    mlir::Value next = addToCurrent ? addValues(thenBuilder, loc, current, value)
                                    : value;
    thenBuilder.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{next});
  }
  {
    mlir::OpBuilder elseBuilder = branch.getElseBodyBuilder();
    elseBuilder.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{value});
  }
  storeStateValue(builder, loc, field, groupId, branch.getResult(0));
  storeStateValid(builder, loc, field, groupId);
}

static void updateMinMaxValue(mlir::OpBuilder &builder, mlir::Location loc,
                              const StateFieldInfo &field, mlir::Value groupId,
                              mlir::Value value, bool isMin) {
  mlir::Value current = loadStateValue(builder, loc, field, groupId);
  mlir::Value valid = loadStateValid(builder, loc, field, groupId);
  mlir::Value zeroI8 = builder.create<mlir::arith::ConstantIntOp>(loc, 0, 8);
  mlir::Value invalid = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::eq, valid, zeroI8);
  mlir::Value better =
      compareForMinMax(builder, loc, field.kind, value, current, isMin);
  mlir::Value replace =
      builder.create<mlir::arith::OrIOp>(loc, invalid, better);
  auto branch = builder.create<mlir::scf::IfOp>(
      loc, mlir::TypeRange{builder.getI32Type()}, replace, true);
  mlir::OpBuilder thenBuilder = branch.getThenBodyBuilder();
  storeStateValue(thenBuilder, loc, field, groupId, value);
  storeStateValid(thenBuilder, loc, field, groupId);
  mlir::Value zeroI32 = thenBuilder.create<mlir::arith::ConstantIntOp>(loc, 0, 32);
  thenBuilder.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{zeroI32});
  mlir::OpBuilder elseBuilder = branch.getElseBodyBuilder();
  mlir::Value elseZeroI32 =
      elseBuilder.create<mlir::arith::ConstantIntOp>(loc, 0, 32);
  elseBuilder.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{elseZeroI32});
}

static mlir::LogicalResult updateAggregateState(
    mlir::OpBuilder &builder, mlir::Location loc, llvm::StringRef func,
    mlir::Value measure, llvm::SmallVectorImpl<StateFieldInfo> &states,
    size_t &stateIndex, mlir::Value groupId) {
  if (func == "sum") {
    if (stateIndex >= states.size())
      return mlir::failure();
    updateNullableValue(builder, loc, states[stateIndex++], groupId, measure,
                        true);
    return mlir::success();
  }
  if (func == "count") {
    if (stateIndex >= states.size())
      return mlir::failure();
    const StateFieldInfo &field = states[stateIndex++];
    mlir::Value current = loadStateValue(builder, loc, field, groupId);
    auto integerType = llvm::dyn_cast<mlir::IntegerType>(field.type);
    if (!integerType)
      return mlir::failure();
    mlir::Value one = builder.create<mlir::arith::ConstantIntOp>(
        loc, 1, integerType.getWidth());
    mlir::Value next = builder.create<mlir::arith::AddIOp>(loc, current, one);
    storeStateValue(builder, loc, field, groupId, next);
    storeStateValid(builder, loc, field, groupId);
    return mlir::success();
  }
  if (func == "avg") {
    if (stateIndex + 1 >= states.size())
      return mlir::failure();
    const StateFieldInfo &countField = states[stateIndex++];
    mlir::Value current = loadStateValue(builder, loc, countField, groupId);
    auto integerType = llvm::dyn_cast<mlir::IntegerType>(countField.type);
    if (!integerType)
      return mlir::failure();
    mlir::Value one = builder.create<mlir::arith::ConstantIntOp>(
        loc, 1, integerType.getWidth());
    mlir::Value next = builder.create<mlir::arith::AddIOp>(loc, current, one);
    storeStateValue(builder, loc, countField, groupId, next);
    storeStateValid(builder, loc, countField, groupId);
    updateNullableValue(builder, loc, states[stateIndex++], groupId, measure,
                        true);
    return mlir::success();
  }
  if (func == "min" || func == "max") {
    if (stateIndex >= states.size())
      return mlir::failure();
    updateMinMaxValue(builder, loc, states[stateIndex++], groupId, measure,
                      func == "min");
    return mlir::success();
  }
  return mlir::failure();
}

static mlir::LogicalResult lowerFilterPlainSum(mlir::func::FuncOp func) {
  if (func.isExternal() || !llvm::hasSingleElement(func.getBody()))
    return mlir::failure();

  mlir::quill::FilterOp filter;
  mlir::quill::PlainSumSinkOp plainSum;
  func.walk([&](mlir::quill::FilterOp op) {
    if (!filter)
      filter = op;
  });
  func.walk([&](mlir::quill::PlainSumSinkOp op) {
    if (!plainSum)
      plainSum = op;
  });

  if (!filter || !plainSum)
    return mlir::failure();

  std::map<int64_t, mlir::Type> columnTypes;
  if (mlir::failed(collectColumns(filter.getPredicate(), columnTypes, filter)))
    return mlir::failure();
  if (mlir::failed(collectColumns(plainSum.getMeasure(), columnTypes, plainSum)))
    return mlir::failure();

  mlir::Block &measureBlock = plainSum.getMeasure().front();
  auto measureYield =
      llvm::dyn_cast<mlir::quill::YieldOp>(measureBlock.getTerminator());
  if (!measureYield || measureYield.getValues().size() != 1)
    return plainSum.emitOpError("measure region must yield one value");
  mlir::Type sumType = measureYield.getValues().front().getType();
  if (!llvm::isa<mlir::IntegerType, mlir::FloatType>(sumType))
    return plainSum.emitOpError("measure region must yield an integer or float");

  mlir::Region predicateRegion;
  mlir::IRMapping predicateMapping;
  filter.getPredicate().cloneInto(&predicateRegion, predicateMapping);
  mlir::Region measureRegion;
  mlir::IRMapping measureMapping;
  plainSum.getMeasure().cloneInto(&measureRegion, measureMapping);

  mlir::MLIRContext *context = func.getContext();
  mlir::OpBuilder builder(context);
  mlir::Location loc = func.getLoc();
  auto i64Type = builder.getI64Type();
  auto i32Type = builder.getI32Type();
  auto ptrType = mlir::LLVM::LLVMPointerType::get(context);

  llvm::SmallVector<mlir::Type> inputTypes;
  inputTypes.push_back(i64Type);
  for (const auto &[_, type] : columnTypes)
    inputTypes.push_back(ptrType);
  inputTypes.push_back(ptrType);
  inputTypes.push_back(ptrType);

  func.eraseBody();
  func.setFunctionType(mlir::FunctionType::get(context, inputTypes, i32Type));
  func->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
  mlir::Block *entry = func.addEntryBlock();
  builder.setInsertionPointToStart(entry);

  mlir::Value len = entry->getArgument(0);
  llvm::SmallVector<ColumnInfo> columns;
  size_t argIndex = 1;
  for (const auto &[index, type] : columnTypes)
    columns.push_back(ColumnInfo{index, type, entry->getArgument(argIndex++)});
  mlir::Value outSum = entry->getArgument(argIndex++);
  mlir::Value outCount = entry->getArgument(argIndex++);

  mlir::Value zeroI64 = builder.create<mlir::arith::ConstantIntOp>(loc, 0, 64);
  mlir::Value oneI64 = builder.create<mlir::arith::ConstantIntOp>(loc, 1, 64);
  mlir::Value zeroSum = zeroForType(builder, loc, sumType);
  if (!zeroSum)
    return plainSum.emitOpError("unsupported SUM state type");

  bool cloneFailed = false;
  auto loop = builder.create<mlir::scf::ForOp>(
      loc, zeroI64, len, oneI64, mlir::ValueRange{zeroSum, zeroI64},
      [&](mlir::OpBuilder &bodyBuilder, mlir::Location bodyLoc,
          mlir::Value iv, mlir::ValueRange iterArgs) {
        llvm::DenseMap<int64_t, mlir::Value> loadedColumns;
        for (const ColumnInfo &column : columns)
          loadedColumns.try_emplace(column.index,
                                    loadColumn(bodyBuilder, bodyLoc, iv, column));

        mlir::Value predicate;
        if (mlir::failed(cloneRegionValue(bodyBuilder, predicateRegion,
                                          loadedColumns, func.getOperation(),
                                          predicate))) {
          cloneFailed = true;
          bodyBuilder.create<mlir::scf::YieldOp>(
              bodyLoc, mlir::ValueRange{iterArgs[0], iterArgs[1]});
          return;
        }

        auto branch = bodyBuilder.create<mlir::scf::IfOp>(
            bodyLoc, mlir::TypeRange{sumType, i64Type}, predicate, true);
        {
          mlir::OpBuilder thenBuilder = branch.getThenBodyBuilder();
          mlir::Value measure;
          if (mlir::failed(cloneRegionValue(thenBuilder, measureRegion,
                                            loadedColumns, func.getOperation(),
                                            measure))) {
            cloneFailed = true;
            thenBuilder.create<mlir::scf::YieldOp>(
                bodyLoc, mlir::ValueRange{iterArgs[0], iterArgs[1]});
            return;
          }
          mlir::Value nextSum =
              addValues(thenBuilder, bodyLoc, iterArgs[0], measure);
          mlir::Value nextCount =
              thenBuilder.create<mlir::arith::AddIOp>(bodyLoc, iterArgs[1],
                                                      oneI64);
          thenBuilder.create<mlir::scf::YieldOp>(
              bodyLoc, mlir::ValueRange{nextSum, nextCount});
        }
        {
          mlir::OpBuilder elseBuilder = branch.getElseBodyBuilder();
          elseBuilder.create<mlir::scf::YieldOp>(
              bodyLoc, mlir::ValueRange{iterArgs[0], iterArgs[1]});
        }
        bodyBuilder.create<mlir::scf::YieldOp>(bodyLoc, branch.getResults());
      });
  if (cloneFailed)
    return mlir::failure();

  builder.create<mlir::LLVM::StoreOp>(loc, loop.getResult(0), outSum);
  builder.create<mlir::LLVM::StoreOp>(loc, loop.getResult(1), outCount);
  mlir::Value ok = builder.create<mlir::arith::ConstantIntOp>(loc, 0, 32);
  builder.create<mlir::func::ReturnOp>(loc, ok);
  return mlir::success();
}

static mlir::LogicalResult lowerGroupAggregateUpdate(mlir::func::FuncOp func) {
  if (func.isExternal() || !llvm::hasSingleElement(func.getBody()))
    return mlir::failure();

  mlir::quill::GroupUpdateSinkOp sink;
  func.walk([&](mlir::quill::GroupUpdateSinkOp op) {
    if (!sink)
      sink = op;
  });
  if (!sink)
    return mlir::failure();
  auto filter =
      sink.getSelection().getDefiningOp<mlir::quill::FilterOp>();

  std::map<int64_t, mlir::Type> columnTypes;
  if (filter && !isLiteralTrueFilter(filter) &&
      mlir::failed(collectColumns(filter.getPredicate(), columnTypes, filter,
                                  false)))
    return mlir::failure();
  if (mlir::failed(
          collectColumns(sink.getState(), columnTypes, sink, false)))
    return mlir::failure();

  mlir::Block &stateBlock = sink.getState().front();
  auto stateYield =
      llvm::dyn_cast<mlir::quill::YieldOp>(stateBlock.getTerminator());
  if (!stateYield || stateYield.getValues().empty())
    return sink.emitOpError("state region must yield at least one aggregate value");

  llvm::SmallVector<llvm::StringRef> funcs;
  for (mlir::Attribute attr : sink.getAggregateFuncs())
    funcs.push_back(llvm::cast<mlir::StringAttr>(attr).getValue());
  if (funcs.size() != stateYield.getValues().size())
    return sink.emitOpError("state region must yield one value per aggregate function");

  llvm::SmallVector<llvm::StringRef> stateKinds;
  for (mlir::Attribute attr : sink.getStateTypes())
    stateKinds.push_back(llvm::cast<mlir::StringAttr>(attr).getValue());
  if (stateKinds.empty())
    return sink.emitOpError("state_types must not be empty");

  mlir::Region stateRegion;
  mlir::IRMapping stateMapping;
  sink.getState().cloneInto(&stateRegion, stateMapping);
  mlir::Region predicateRegion;
  bool hasPredicate = false;
  if (filter && !isLiteralTrueFilter(filter)) {
    mlir::IRMapping predicateMapping;
    filter.getPredicate().cloneInto(&predicateRegion, predicateMapping);
    hasPredicate = true;
  }

  mlir::MLIRContext *context = func.getContext();
  mlir::OpBuilder builder(context);
  mlir::Location loc = func.getLoc();
  auto i64Type = builder.getI64Type();
  auto i8Type = builder.getI8Type();
  auto i32Type = builder.getI32Type();
  auto ptrType = mlir::LLVM::LLVMPointerType::get(context);

  llvm::SmallVector<mlir::Type> stateTypes;
  for (llvm::StringRef kind : stateKinds) {
    mlir::Type type = stateTypeFromName(builder, kind);
    if (!type)
      return sink.emitOpError("unsupported state type ") << kind;
    stateTypes.push_back(type);
  }

  llvm::SmallVector<mlir::Type> inputTypes;
  inputTypes.push_back(i64Type);
  inputTypes.push_back(ptrType);
  inputTypes.push_back(ptrType);
  for (size_t i = 0; i < columnTypes.size(); ++i)
    inputTypes.push_back(ptrType);
  for (size_t i = 0; i < stateTypes.size(); ++i)
    inputTypes.push_back(ptrType);
  for (size_t i = 0; i < stateTypes.size(); ++i)
    inputTypes.push_back(ptrType);

  func.eraseBody();
  func.setFunctionType(mlir::FunctionType::get(context, inputTypes, i32Type));
  func->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
  mlir::Block *entry = func.addEntryBlock();
  builder.setInsertionPointToStart(entry);

  mlir::Value len = entry->getArgument(0);
  mlir::Value groupIds = entry->getArgument(1);
  mlir::Value touched = entry->getArgument(2);
  llvm::SmallVector<ColumnInfo> columns;
  size_t argIndex = 3;
  for (const auto &[index, type] : columnTypes)
    columns.push_back(ColumnInfo{index, type, entry->getArgument(argIndex++)});

  llvm::SmallVector<StateFieldInfo> states;
  states.reserve(stateTypes.size());
  for (size_t i = 0; i < stateTypes.size(); ++i)
    states.push_back(
        StateFieldInfo{stateKinds[i], stateTypes[i], entry->getArgument(argIndex++), {}});
  for (size_t i = 0; i < stateTypes.size(); ++i)
    states[i].valid = entry->getArgument(argIndex++);

  mlir::Value zeroI64 = builder.create<mlir::arith::ConstantIntOp>(loc, 0, 64);
  mlir::Value oneI64 = builder.create<mlir::arith::ConstantIntOp>(loc, 1, 64);

  bool cloneFailed = false;
  bool updateFailed = false;
  builder.create<mlir::scf::ForOp>(
      loc, zeroI64, len, oneI64, mlir::ValueRange{},
      [&](mlir::OpBuilder &bodyBuilder, mlir::Location bodyLoc,
          mlir::Value iv, mlir::ValueRange) {
        auto groupIdPtr = bodyBuilder.create<mlir::LLVM::GEPOp>(
            bodyLoc, ptrType, i64Type, groupIds,
            llvm::SmallVector<mlir::LLVM::GEPArg>{iv});
        mlir::Value groupId =
            bodyBuilder.create<mlir::LLVM::LoadOp>(bodyLoc, i64Type, groupIdPtr);
        mlir::Value selected = bodyBuilder.create<mlir::arith::CmpIOp>(
            bodyLoc, mlir::arith::CmpIPredicate::sge, groupId, zeroI64);

        llvm::DenseMap<int64_t, mlir::Value> loadedColumns;
        for (const ColumnInfo &column : columns)
          loadedColumns.try_emplace(column.index,
                                    loadColumn(bodyBuilder, bodyLoc, iv, column));
        if (hasPredicate) {
          mlir::Value predicate;
          if (mlir::failed(cloneRegionValue(bodyBuilder, predicateRegion,
                                            loadedColumns, func.getOperation(),
                                            predicate))) {
            cloneFailed = true;
            bodyBuilder.create<mlir::scf::YieldOp>(bodyLoc);
            return;
          }
          selected = bodyBuilder.create<mlir::arith::AndIOp>(bodyLoc, selected,
                                                             predicate);
        }

        auto branch = bodyBuilder.create<mlir::scf::IfOp>(
            bodyLoc, mlir::TypeRange{i32Type}, selected, true);
        mlir::OpBuilder thenBuilder = branch.getThenBodyBuilder();

        llvm::SmallVector<mlir::Value> measures;
        if (mlir::failed(cloneRegionValues(thenBuilder, stateRegion,
                                           loadedColumns, func.getOperation(),
                                           measures))) {
          cloneFailed = true;
          mlir::Value zeroI32 =
              thenBuilder.create<mlir::arith::ConstantIntOp>(bodyLoc, 0, 32);
          thenBuilder.create<mlir::scf::YieldOp>(bodyLoc,
                                                 mlir::ValueRange{zeroI32});
          bodyBuilder.create<mlir::scf::YieldOp>(bodyLoc);
          return;
        }

        size_t stateIndex = 0;
        for (size_t i = 0; i < funcs.size(); ++i) {
          if (mlir::failed(updateAggregateState(thenBuilder, bodyLoc, funcs[i],
                                                measures[i], states,
                                                stateIndex, groupId))) {
            updateFailed = true;
            break;
          }
        }
        if (stateIndex != states.size())
          updateFailed = true;

        auto touchedPtr = thenBuilder.create<mlir::LLVM::GEPOp>(
            bodyLoc, ptrType, i8Type, touched,
            llvm::SmallVector<mlir::LLVM::GEPArg>{groupId});
        mlir::Value touchedValue =
            thenBuilder.create<mlir::arith::ConstantIntOp>(bodyLoc, 1, 8);
        thenBuilder.create<mlir::LLVM::StoreOp>(bodyLoc, touchedValue,
                                                touchedPtr);

        mlir::Value zeroI32 =
            thenBuilder.create<mlir::arith::ConstantIntOp>(bodyLoc, 0, 32);
        thenBuilder.create<mlir::scf::YieldOp>(bodyLoc,
                                               mlir::ValueRange{zeroI32});
        mlir::OpBuilder elseBuilder = branch.getElseBodyBuilder();
        mlir::Value elseZeroI32 =
            elseBuilder.create<mlir::arith::ConstantIntOp>(bodyLoc, 0, 32);
        elseBuilder.create<mlir::scf::YieldOp>(bodyLoc,
                                               mlir::ValueRange{elseZeroI32});
        bodyBuilder.create<mlir::scf::YieldOp>(bodyLoc);
      });
  if (cloneFailed || updateFailed)
    return mlir::failure();

  mlir::Value ok = builder.create<mlir::arith::ConstantIntOp>(loc, 0, 32);
  builder.create<mlir::func::ReturnOp>(loc, ok);
  return mlir::success();
}

struct QuillCanonicalizePipelinePass
    : public mlir::PassWrapper<QuillCanonicalizePipelinePass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(QuillCanonicalizePipelinePass)

  llvm::StringRef getArgument() const final {
    return "quill-canonicalize-pipeline";
  }

  llvm::StringRef getDescription() const final {
    return "canonicalize Quill pipeline graph before loop lowering";
  }

  void runOnOperation() final {}
};

struct ConvertQuillToLoopsPass
    : public mlir::PassWrapper<ConvertQuillToLoopsPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertQuillToLoopsPass)

  llvm::StringRef getArgument() const final { return "convert-quill-to-loops"; }

  llvm::StringRef getDescription() const final {
    return "lower Quill pipeline operations to loop-level MLIR";
  }

  void runOnOperation() final {
    for (auto func : getOperation().getOps<mlir::func::FuncOp>()) {
      bool hasPlainSum = false;
      func.walk([&](mlir::quill::PlainSumSinkOp) { hasPlainSum = true; });
      bool hasRecordSink = false;
      func.walk([&](mlir::quill::RecordBatchSinkOp) { hasRecordSink = true; });
      bool hasGroupUpdate = false;
      func.walk([&](mlir::quill::GroupUpdateSinkOp) {
        hasGroupUpdate = true;
      });

      mlir::LogicalResult result = mlir::success();
      if (hasGroupUpdate)
        result = lowerGroupAggregateUpdate(func);
      else if (hasPlainSum)
        result = lowerFilterPlainSum(func);
      else if (hasRecordSink)
        result = lowerFilterProject(func);

      if (mlir::failed(result)) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace

extern "C" void quillMlirRegisterPasses() {
  static const bool registered = [] {
    mlir::PassRegistration<QuillCanonicalizePipelinePass>();
    mlir::PassRegistration<ConvertQuillToLoopsPass>();
    return true;
  }();
  (void)registered;
}

#ifdef __clang__
#pragma clang diagnostic pop
#endif
