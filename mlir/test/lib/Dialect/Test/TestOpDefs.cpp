//===- TestOpDefs.cpp - MLIR Test Dialect Operation Hooks -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Verifier.h"

using namespace mlir;
using namespace test;

//===----------------------------------------------------------------------===//
// FormatInferType2Op
//===----------------------------------------------------------------------===//

LogicalResult FormatInferType2Op::inferReturnTypes(
    MLIRContext *context, Optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  inferredReturnTypes.assign({IntegerType::get(context, 16)});
  return success();
}

//===----------------------------------------------------------------------===//
// TestBranchOp
//===----------------------------------------------------------------------===//

SuccessorOperands TestBranchOp::getSuccessorOperands(unsigned index) {
  assert(index == 0 && "invalid successor index");
  return SuccessorOperands(getTargetOperandsMutable());
}

//===----------------------------------------------------------------------===//
// TestProducingBranchOp
//===----------------------------------------------------------------------===//

SuccessorOperands TestProducingBranchOp::getSuccessorOperands(unsigned index) {
  assert(index <= 1 && "invalid successor index");
  if (index == 1)
    return SuccessorOperands(getFirstOperandsMutable());
  return SuccessorOperands(getSecondOperandsMutable());
}

//===----------------------------------------------------------------------===//
// TestInternalBranchOp
//===----------------------------------------------------------------------===//

SuccessorOperands TestInternalBranchOp::getSuccessorOperands(unsigned index) {
  assert(index <= 1 && "invalid successor index");
  if (index == 0)
    return SuccessorOperands(0, getSuccessOperandsMutable());
  return SuccessorOperands(1, getErrorOperandsMutable());
}

//===----------------------------------------------------------------------===//
// TestCallOp
//===----------------------------------------------------------------------===//

LogicalResult TestCallOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // Check that the callee attribute was specified.
  auto fnAttr = (*this)->getAttrOfType<FlatSymbolRefAttr>("callee");
  if (!fnAttr)
    return emitOpError("requires a 'callee' symbol reference attribute");
  if (!symbolTable.lookupNearestSymbolFrom<FunctionOpInterface>(*this, fnAttr))
    return emitOpError() << "'" << fnAttr.getValue()
                         << "' does not reference a valid function";
  return success();
}

//===----------------------------------------------------------------------===//
// FoldToCallOp
//===----------------------------------------------------------------------===//

namespace {
struct FoldToCallOpPattern : public OpRewritePattern<FoldToCallOp> {
  using OpRewritePattern<FoldToCallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(FoldToCallOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<func::CallOp>(op, TypeRange(),
                                              op.getCalleeAttr(), ValueRange());
    return success();
  }
};
} // namespace

void FoldToCallOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                               MLIRContext *context) {
  results.add<FoldToCallOpPattern>(context);
}
//===----------------------------------------------------------------------===//
// IsolatedRegionOp - test parsing passthrough operands
//===----------------------------------------------------------------------===//

ParseResult IsolatedRegionOp::parse(OpAsmParser &parser,
                                    OperationState &result) {
  // Parse the input operand.
  OpAsmParser::Argument argInfo;
  argInfo.type = parser.getBuilder().getIndexType();
  if (parser.parseOperand(argInfo.ssaName) ||
      parser.resolveOperand(argInfo.ssaName, argInfo.type, result.operands))
    return failure();

  // Parse the body region, and reuse the operand info as the argument info.
  Region *body = result.addRegion();
  return parser.parseRegion(*body, argInfo, /*enableNameShadowing=*/true);
}

void IsolatedRegionOp::print(OpAsmPrinter &p) {
  p << "test.isolated_region ";
  p.printOperand(getOperand());
  p.shadowRegionArgs(getRegion(), getOperand());
  p << ' ';
  p.printRegion(getRegion(), /*printEntryBlockArgs=*/false);
}

//===----------------------------------------------------------------------===//
// SSACFGRegionOp
//===----------------------------------------------------------------------===//

RegionKind SSACFGRegionOp::getRegionKind(unsigned index) {
  return RegionKind::SSACFG;
}

//===----------------------------------------------------------------------===//
// GraphRegionOp
//===----------------------------------------------------------------------===//

RegionKind GraphRegionOp::getRegionKind(unsigned index) {
  return RegionKind::Graph;
}

//===----------------------------------------------------------------------===//
// AffineScopeOp
//===----------------------------------------------------------------------===//

ParseResult AffineScopeOp::parse(OpAsmParser &parser, OperationState &result) {
  // Parse the body region, and reuse the operand info as the argument info.
  Region *body = result.addRegion();
  return parser.parseRegion(*body, /*arguments=*/{}, /*argTypes=*/{});
}

void AffineScopeOp::print(OpAsmPrinter &p) {
  p << "test.affine_scope ";
  p.printRegion(getRegion(), /*printEntryBlockArgs=*/false);
}

//===----------------------------------------------------------------------===//
// ParseIntegerLiteralOp
//===----------------------------------------------------------------------===//

ParseResult ParseIntegerLiteralOp::parse(OpAsmParser &parser,
                                         OperationState &result) {
  if (parser.parseOptionalColon())
    return success();
  uint64_t numResults;
  if (parser.parseInteger(numResults))
    return failure();

  IndexType type = parser.getBuilder().getIndexType();
  for (unsigned i = 0; i < numResults; ++i)
    result.addTypes(type);
  return success();
}

void ParseIntegerLiteralOp::print(OpAsmPrinter &p) {
  if (unsigned numResults = getNumResults())
    p << " : " << numResults;
}

//===----------------------------------------------------------------------===//
// ParseWrappedKeywordOp
//===----------------------------------------------------------------------===//

ParseResult ParseWrappedKeywordOp::parse(OpAsmParser &parser,
                                         OperationState &result) {
  StringRef keyword;
  if (parser.parseKeyword(&keyword))
    return failure();
  result.addAttribute("keyword", parser.getBuilder().getStringAttr(keyword));
  return success();
}

void ParseWrappedKeywordOp::print(OpAsmPrinter &p) { p << " " << getKeyword(); }

//===----------------------------------------------------------------------===//
// WrappingRegionOp - wrapping op exercising `parseGenericOperation()`.
//===----------------------------------------------------------------------===//

ParseResult WrappingRegionOp::parse(OpAsmParser &parser,
                                    OperationState &result) {
  if (parser.parseKeyword("wraps"))
    return failure();

  // Parse the wrapped op in a region
  Region &body = *result.addRegion();
  body.push_back(new Block);
  Block &block = body.back();
  Operation *wrappedOp = parser.parseGenericOperation(&block, block.begin());
  if (!wrappedOp)
    return failure();

  // Create a return terminator in the inner region, pass as operand to the
  // terminator the returned values from the wrapped operation.
  SmallVector<Value, 8> returnOperands(wrappedOp->getResults());
  OpBuilder builder(parser.getContext());
  builder.setInsertionPointToEnd(&block);
  builder.create<TestReturnOp>(wrappedOp->getLoc(), returnOperands);

  // Get the results type for the wrapping op from the terminator operands.
  Operation &returnOp = body.back().back();
  result.types.append(returnOp.operand_type_begin(),
                      returnOp.operand_type_end());

  // Use the location of the wrapped op for the "test.wrapping_region" op.
  result.location = wrappedOp->getLoc();

  return success();
}

void WrappingRegionOp::print(OpAsmPrinter &p) {
  p << " wraps ";
  p.printGenericOp(&getRegion().front().front());
}

//===----------------------------------------------------------------------===//
// PrettyPrintedRegionOp -  exercising the following parser APIs
//   parseGenericOperationAfterOpName
//   parseCustomOperationName
//===----------------------------------------------------------------------===//

ParseResult PrettyPrintedRegionOp::parse(OpAsmParser &parser,
                                         OperationState &result) {
  SMLoc loc = parser.getCurrentLocation();
  Location currLocation = parser.getEncodedSourceLoc(loc);

  // Parse the operands.
  SmallVector<OpAsmParser::UnresolvedOperand, 2> operands;
  if (parser.parseOperandList(operands))
    return failure();

  // Check if we are parsing the pretty-printed version
  //  test.pretty_printed_region start <inner-op> end : <functional-type>
  // Else fallback to parsing the "non pretty-printed" version.
  if (!succeeded(parser.parseOptionalKeyword("start")))
    return parser.parseGenericOperationAfterOpName(
        result, llvm::makeArrayRef(operands));

  FailureOr<OperationName> parseOpNameInfo = parser.parseCustomOperationName();
  if (failed(parseOpNameInfo))
    return failure();

  StringAttr innerOpName = parseOpNameInfo->getIdentifier();

  FunctionType opFntype;
  Optional<Location> explicitLoc;
  if (parser.parseKeyword("end") || parser.parseColon() ||
      parser.parseType(opFntype) ||
      parser.parseOptionalLocationSpecifier(explicitLoc))
    return failure();

  // If location of the op is explicitly provided, then use it; Else use
  // the parser's current location.
  Location opLoc = explicitLoc.getValueOr(currLocation);

  // Derive the SSA-values for op's operands.
  if (parser.resolveOperands(operands, opFntype.getInputs(), loc,
                             result.operands))
    return failure();

  // Add a region for op.
  Region &region = *result.addRegion();

  // Create a basic-block inside op's region.
  Block &block = region.emplaceBlock();

  // Create and insert an "inner-op" operation in the block.
  // Just for testing purposes, we can assume that inner op is a binary op with
  // result and operand types all same as the test-op's first operand.
  Type innerOpType = opFntype.getInput(0);
  Value lhs = block.addArgument(innerOpType, opLoc);
  Value rhs = block.addArgument(innerOpType, opLoc);

  OpBuilder builder(parser.getBuilder().getContext());
  builder.setInsertionPointToStart(&block);

  Operation *innerOp =
      builder.create(opLoc, innerOpName, /*operands=*/{lhs, rhs}, innerOpType);

  // Insert a return statement in the block returning the inner-op's result.
  builder.create<TestReturnOp>(innerOp->getLoc(), innerOp->getResults());

  // Populate the op operation-state with result-type and location.
  result.addTypes(opFntype.getResults());
  result.location = innerOp->getLoc();

  return success();
}

void PrettyPrintedRegionOp::print(OpAsmPrinter &p) {
  p << ' ';
  p.printOperands(getOperands());

  Operation &innerOp = getRegion().front().front();
  // Assuming that region has a single non-terminator inner-op, if the inner-op
  // meets some criteria (which in this case is a simple one  based on the name
  // of inner-op), then we can print the entire region in a succinct way.
  // Here we assume that the prototype of "special.op" can be trivially derived
  // while parsing it back.
  if (innerOp.getName().getStringRef().equals("special.op")) {
    p << " start special.op end";
  } else {
    p << " (";
    p.printRegion(getRegion());
    p << ")";
  }

  p << " : ";
  p.printFunctionalType(*this);
}

//===----------------------------------------------------------------------===//
// PolyForOp - parse list of region arguments.
//===----------------------------------------------------------------------===//

ParseResult PolyForOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::Argument, 4> ivsInfo;
  // Parse list of region arguments without a delimiter.
  if (parser.parseArgumentList(ivsInfo, OpAsmParser::Delimiter::None))
    return failure();

  // Parse the body region.
  Region *body = result.addRegion();
  for (auto &iv : ivsInfo)
    iv.type = parser.getBuilder().getIndexType();
  return parser.parseRegion(*body, ivsInfo);
}

void PolyForOp::print(OpAsmPrinter &p) { p.printGenericOp(*this); }

void PolyForOp::getAsmBlockArgumentNames(Region &region,
                                         OpAsmSetValueNameFn setNameFn) {
  auto arrayAttr = getOperation()->getAttrOfType<ArrayAttr>("arg_names");
  if (!arrayAttr)
    return;
  auto args = getRegion().front().getArguments();
  auto e = std::min(arrayAttr.size(), args.size());
  for (unsigned i = 0; i < e; ++i) {
    if (auto strAttr = arrayAttr[i].dyn_cast<StringAttr>())
      setNameFn(args[i], strAttr.getValue());
  }
}

//===----------------------------------------------------------------------===//
// TestRemoveOpWithInnerOps
//===----------------------------------------------------------------------===//

namespace {
struct TestRemoveOpWithInnerOps
    : public OpRewritePattern<TestOpWithRegionPattern> {
  using OpRewritePattern<TestOpWithRegionPattern>::OpRewritePattern;

  void initialize() { setDebugName("TestRemoveOpWithInnerOps"); }

  LogicalResult matchAndRewrite(TestOpWithRegionPattern op,
                                PatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// TestOpWithRegionPattern
//===----------------------------------------------------------------------===//

void TestOpWithRegionPattern::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.add<TestRemoveOpWithInnerOps>(context);
}

//===----------------------------------------------------------------------===//
// TestOpWithRegionFold
//===----------------------------------------------------------------------===//

OpFoldResult TestOpWithRegionFold::fold(ArrayRef<Attribute> operands) {
  return getOperand();
}

//===----------------------------------------------------------------------===//
// TestOpConstant
//===----------------------------------------------------------------------===//

OpFoldResult TestOpConstant::fold(ArrayRef<Attribute> operands) {
  return getValue();
}

//===----------------------------------------------------------------------===//
// TestOpWithVariadicResultsAndFolder
//===----------------------------------------------------------------------===//

LogicalResult TestOpWithVariadicResultsAndFolder::fold(
    ArrayRef<Attribute> operands, SmallVectorImpl<OpFoldResult> &results) {
  for (Value input : this->getOperands()) {
    results.push_back(input);
  }
  return success();
}

//===----------------------------------------------------------------------===//
// TestOpInPlaceFold
//===----------------------------------------------------------------------===//

OpFoldResult TestOpInPlaceFold::fold(ArrayRef<Attribute> operands) {
  assert(operands.size() == 1);
  if (operands.front()) {
    (*this)->setAttr("attr", operands.front());
    return getResult();
  }
  return {};
}

//===----------------------------------------------------------------------===//
// TestPassthroughFold
//===----------------------------------------------------------------------===//

OpFoldResult TestPassthroughFold::fold(ArrayRef<Attribute> operands) {
  return getOperand();
}

//===----------------------------------------------------------------------===//
// OpWithInferTypeInterfaceOp
//===----------------------------------------------------------------------===//

LogicalResult OpWithInferTypeInterfaceOp::inferReturnTypes(
    MLIRContext *, Optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  if (operands[0].getType() != operands[1].getType()) {
    return emitOptionalError(location, "operand type mismatch ",
                             operands[0].getType(), " vs ",
                             operands[1].getType());
  }
  inferredReturnTypes.assign({operands[0].getType()});
  return success();
}

//===----------------------------------------------------------------------===//
// OpWithShapedTypeInferTypeInterfaceOp
//===----------------------------------------------------------------------===//

LogicalResult OpWithShapedTypeInferTypeInterfaceOp::inferReturnTypeComponents(
    MLIRContext *context, Optional<Location> location, ValueShapeRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  // Create return type consisting of the last element of the first operand.
  auto operandType = operands.front().getType();
  auto sval = operandType.dyn_cast<ShapedType>();
  if (!sval) {
    return emitOptionalError(location, "only shaped type operands allowed");
  }
  int64_t dim =
      sval.hasRank() ? sval.getShape().front() : ShapedType::kDynamicSize;
  auto type = IntegerType::get(context, 17);
  inferredReturnShapes.push_back(ShapedTypeComponents({dim}, type));
  return success();
}

LogicalResult OpWithShapedTypeInferTypeInterfaceOp::reifyReturnTypeShapes(
    OpBuilder &builder, ValueRange operands,
    llvm::SmallVectorImpl<Value> &shapes) {
  shapes = SmallVector<Value, 1>{
      builder.createOrFold<tensor::DimOp>(getLoc(), operands.front(), 0)};
  return success();
}

//===----------------------------------------------------------------------===//
// OpWithResultShapeInterfaceOp
//===----------------------------------------------------------------------===//

LogicalResult OpWithResultShapeInterfaceOp::reifyReturnTypeShapes(
    OpBuilder &builder, ValueRange operands,
    llvm::SmallVectorImpl<Value> &shapes) {
  Location loc = getLoc();
  shapes.reserve(operands.size());
  for (Value operand : llvm::reverse(operands)) {
    auto rank = operand.getType().cast<RankedTensorType>().getRank();
    auto currShape = llvm::to_vector<4>(
        llvm::map_range(llvm::seq<int64_t>(0, rank), [&](int64_t dim) -> Value {
          return builder.createOrFold<tensor::DimOp>(loc, operand, dim);
        }));
    shapes.push_back(builder.create<tensor::FromElementsOp>(
        getLoc(), RankedTensorType::get({rank}, builder.getIndexType()),
        currShape));
  }
  return success();
}

//===----------------------------------------------------------------------===//
// OpWithResultShapePerDimInterfaceOp
//===----------------------------------------------------------------------===//

LogicalResult OpWithResultShapePerDimInterfaceOp::reifyResultShapes(
    OpBuilder &builder, ReifiedRankedShapedTypeDims &shapes) {
  Location loc = getLoc();
  shapes.reserve(getNumOperands());
  for (Value operand : llvm::reverse(getOperands())) {
    auto currShape = llvm::to_vector<4>(llvm::map_range(
        llvm::seq<int64_t>(
            0, operand.getType().cast<RankedTensorType>().getRank()),
        [&](int64_t dim) -> Value {
          return builder.createOrFold<tensor::DimOp>(loc, operand, dim);
        }));
    shapes.emplace_back(std::move(currShape));
  }
  return success();
}

//===----------------------------------------------------------------------===//
// SideEffectOp
//===----------------------------------------------------------------------===//

void SideEffectOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  // Check for an effects attribute on the op instance.
  ArrayAttr effectsAttr = (*this)->getAttrOfType<ArrayAttr>("effects");
  if (!effectsAttr)
    return;

  // If there is one, it is an array of dictionary attributes that hold
  // information on the effects of this operation.
  for (Attribute element : effectsAttr) {
    DictionaryAttr effectElement = element.cast<DictionaryAttr>();

    // Get the specific memory effect.
    MemoryEffects::Effect *effect =
        StringSwitch<MemoryEffects::Effect *>(
            effectElement.get("effect").cast<StringAttr>().getValue())
            .Case("allocate", MemoryEffects::Allocate::get())
            .Case("free", MemoryEffects::Free::get())
            .Case("read", MemoryEffects::Read::get())
            .Case("write", MemoryEffects::Write::get());

    // Check for a non-default resource to use.
    SideEffects::Resource *resource = SideEffects::DefaultResource::get();
    if (effectElement.get("test_resource"))
      resource = TestResource::get();

    // Check for a result to affect.
    if (effectElement.get("on_result"))
      effects.emplace_back(effect, getResult(), resource);
    else if (Attribute ref = effectElement.get("on_reference"))
      effects.emplace_back(effect, ref.cast<SymbolRefAttr>(), resource);
    else
      effects.emplace_back(effect, resource);
  }
}

void SideEffectOp::getEffects(
    SmallVectorImpl<TestEffects::EffectInstance> &effects) {
  auto effectsAttr = (*this)->getAttrOfType<AffineMapAttr>("effect_parameter");
  if (!effectsAttr)
    return;

  effects.emplace_back(TestEffects::Concrete::get(), effectsAttr);
}

//===----------------------------------------------------------------------===//
// StringAttrPrettyNameOp
//===----------------------------------------------------------------------===//

// This op has fancy handling of its SSA result name.
ParseResult StringAttrPrettyNameOp::parse(OpAsmParser &parser,
                                          OperationState &result) {
  // Add the result types.
  for (size_t i = 0, e = parser.getNumResults(); i != e; ++i)
    result.addTypes(parser.getBuilder().getIntegerType(32));

  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  // If the attribute dictionary contains no 'names' attribute, infer it from
  // the SSA name (if specified).
  bool hadNames = llvm::any_of(result.attributes, [](NamedAttribute attr) {
    return attr.getName() == "names";
  });

  // If there was no name specified, check to see if there was a useful name
  // specified in the asm file.
  if (hadNames || parser.getNumResults() == 0)
    return success();

  SmallVector<StringRef, 4> names;
  auto *context = result.getContext();

  for (size_t i = 0, e = parser.getNumResults(); i != e; ++i) {
    auto resultName = parser.getResultName(i);
    StringRef nameStr;
    if (!resultName.first.empty() && !isdigit(resultName.first[0]))
      nameStr = resultName.first;

    names.push_back(nameStr);
  }

  auto namesAttr = parser.getBuilder().getStrArrayAttr(names);
  result.attributes.push_back({StringAttr::get(context, "names"), namesAttr});
  return success();
}

void StringAttrPrettyNameOp::print(OpAsmPrinter &p) {
  // Note that we only need to print the "name" attribute if the asmprinter
  // result name disagrees with it.  This can happen in strange cases, e.g.
  // when there are conflicts.
  bool namesDisagree = getNames().size() != getNumResults();

  SmallString<32> resultNameStr;
  for (size_t i = 0, e = getNumResults(); i != e && !namesDisagree; ++i) {
    resultNameStr.clear();
    llvm::raw_svector_ostream tmpStream(resultNameStr);
    p.printOperand(getResult(i), tmpStream);

    auto expectedName = getNames()[i].dyn_cast<StringAttr>();
    if (!expectedName ||
        tmpStream.str().drop_front() != expectedName.getValue()) {
      namesDisagree = true;
    }
  }

  if (namesDisagree)
    p.printOptionalAttrDictWithKeyword((*this)->getAttrs());
  else
    p.printOptionalAttrDictWithKeyword((*this)->getAttrs(), {"names"});
}

// We set the SSA name in the asm syntax to the contents of the name
// attribute.
void StringAttrPrettyNameOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  auto value = getNames();
  for (size_t i = 0, e = value.size(); i != e; ++i)
    if (auto str = value[i].dyn_cast<StringAttr>())
      if (!str.getValue().empty())
        setNameFn(getResult(i), str.getValue());
}

//===----------------------------------------------------------------------===//
// CustomResultsNameOp
//===----------------------------------------------------------------------===//

void CustomResultsNameOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  ArrayAttr value = getNames();
  for (size_t i = 0, e = value.size(); i != e; ++i)
    if (auto str = value[i].dyn_cast<StringAttr>())
      if (!str.getValue().empty())
        setNameFn(getResult(i), str.getValue());
}

//===----------------------------------------------------------------------===//
// ResultTypeWithTraitOp
//===----------------------------------------------------------------------===//

LogicalResult ResultTypeWithTraitOp::verify() {
  if ((*this)->getResultTypes()[0].hasTrait<TypeTrait::TestTypeTrait>())
    return success();
  return emitError("result type should have trait 'TestTypeTrait'");
}

//===----------------------------------------------------------------------===//
// AttrWithTraitOp
//===----------------------------------------------------------------------===//

LogicalResult AttrWithTraitOp::verify() {
  if (getAttr().hasTrait<AttributeTrait::TestAttrTrait>())
    return success();
  return emitError("'attr' attribute should have trait 'TestAttrTrait'");
}

//===----------------------------------------------------------------------===//
// RegionIfOp
//===----------------------------------------------------------------------===//

void RegionIfOp::print(OpAsmPrinter &p) {
  p << " ";
  p.printOperands(getOperands());
  p << ": " << getOperandTypes();
  p.printArrowTypeList(getResultTypes());
  p << " then ";
  p.printRegion(getThenRegion(),
                /*printEntryBlockArgs=*/true,
                /*printBlockTerminators=*/true);
  p << " else ";
  p.printRegion(getElseRegion(),
                /*printEntryBlockArgs=*/true,
                /*printBlockTerminators=*/true);
  p << " join ";
  p.printRegion(getJoinRegion(),
                /*printEntryBlockArgs=*/true,
                /*printBlockTerminators=*/true);
}

ParseResult RegionIfOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand, 2> operandInfos;
  SmallVector<Type, 2> operandTypes;

  result.regions.reserve(3);
  Region *thenRegion = result.addRegion();
  Region *elseRegion = result.addRegion();
  Region *joinRegion = result.addRegion();

  // Parse operand, type and arrow type lists.
  if (parser.parseOperandList(operandInfos) ||
      parser.parseColonTypeList(operandTypes) ||
      parser.parseArrowTypeList(result.types))
    return failure();

  // Parse all attached regions.
  if (parser.parseKeyword("then") || parser.parseRegion(*thenRegion, {}, {}) ||
      parser.parseKeyword("else") || parser.parseRegion(*elseRegion, {}, {}) ||
      parser.parseKeyword("join") || parser.parseRegion(*joinRegion, {}, {}))
    return failure();

  return parser.resolveOperands(operandInfos, operandTypes,
                                parser.getCurrentLocation(), result.operands);
}

OperandRange RegionIfOp::getSuccessorEntryOperands(unsigned index) {
  assert(index < 2 && "invalid region index");
  return getOperands();
}

void RegionIfOp::getSuccessorRegions(
    Optional<unsigned> index, ArrayRef<Attribute> operands,
    SmallVectorImpl<RegionSuccessor> &regions) {
  // We always branch to the join region.
  if (index.hasValue()) {
    if (index.getValue() < 2)
      regions.push_back(RegionSuccessor(&getJoinRegion(), getJoinArgs()));
    else
      regions.push_back(RegionSuccessor(getResults()));
    return;
  }

  // The then and else regions are the entry regions of this op.
  regions.push_back(RegionSuccessor(&getThenRegion(), getThenArgs()));
  regions.push_back(RegionSuccessor(&getElseRegion(), getElseArgs()));
}

void RegionIfOp::getRegionInvocationBounds(
    ArrayRef<Attribute> operands,
    SmallVectorImpl<InvocationBounds> &invocationBounds) {
  // Each region is invoked at most once.
  invocationBounds.assign(/*NumElts=*/3, /*Elt=*/{0, 1});
}

//===----------------------------------------------------------------------===//
// AnyCondOp
//===----------------------------------------------------------------------===//

void AnyCondOp::getSuccessorRegions(Optional<unsigned> index,
                                    ArrayRef<Attribute> operands,
                                    SmallVectorImpl<RegionSuccessor> &regions) {
  // The parent op branches into the only region, and the region branches back
  // to the parent op.
  if (index)
    regions.emplace_back(&getRegion());
  else
    regions.emplace_back(getResults());
}

void AnyCondOp::getRegionInvocationBounds(
    ArrayRef<Attribute> operands,
    SmallVectorImpl<InvocationBounds> &invocationBounds) {
  invocationBounds.emplace_back(1, 1);
}

//===----------------------------------------------------------------------===//
// SingleBlockImplicitTerminatorO
//===----------------------------------------------------------------------===//

/// Testing the correctness of some traits.
static_assert(
    llvm::is_detected<OpTrait::has_implicit_terminator_t,
                      SingleBlockImplicitTerminatorOp>::value,
    "has_implicit_terminator_t does not match SingleBlockImplicitTerminatorOp");
static_assert(OpTrait::hasSingleBlockImplicitTerminator<
                  SingleBlockImplicitTerminatorOp>::value,
              "hasSingleBlockImplicitTerminator does not match "
              "SingleBlockImplicitTerminatorOp");

//===----------------------------------------------------------------------===//
// SingleNoTerminatorCustomAsmOp
//===----------------------------------------------------------------------===//

ParseResult SingleNoTerminatorCustomAsmOp::parse(OpAsmParser &parser,
                                                 OperationState &result) {
  Region *body = result.addRegion();
  if (parser.parseRegion(*body, /*arguments=*/{}, /*argTypes=*/{}))
    return failure();
  return success();
}

void SingleNoTerminatorCustomAsmOp::print(OpAsmPrinter &printer) {
  printer.printRegion(
      getRegion(), /*printEntryBlockArgs=*/false,
      // This op has a single block without terminators. But explicitly mark
      // as not printing block terminators for testing.
      /*printBlockTerminators=*/false);
}

//===----------------------------------------------------------------------===//
// TestVerifiersOp
//===----------------------------------------------------------------------===//

LogicalResult TestVerifiersOp::verify() {
  if (!getRegion().hasOneBlock())
    return emitOpError("`hasOneBlock` trait hasn't been verified");

  Operation *definingOp = getInput().getDefiningOp();
  if (definingOp && failed(mlir::verify(definingOp)))
    return emitOpError("operand hasn't been verified");

  emitRemark("success run of verifier");

  return success();
}

LogicalResult TestVerifiersOp::verifyRegions() {
  if (!getRegion().hasOneBlock())
    return emitOpError("`hasOneBlock` trait hasn't been verified");

  for (Block &block : getRegion())
    for (Operation &op : block)
      if (failed(mlir::verify(&op)))
        return emitOpError("nested op hasn't been verified");

  emitRemark("success run of region verifier");

  return success();
}

//===----------------------------------------------------------------------===//
// Test InferIntRangeInterface
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// TestWithBoundsOp

void TestWithBoundsOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                         SetIntRangeFn setResultRanges) {
  setResultRanges(getResult(), {getUmin(), getUmax(), getSmin(), getSmax()});
}

//===----------------------------------------------------------------------===//
// TestWithBoundsRegionOp

ParseResult TestWithBoundsRegionOp::parse(OpAsmParser &parser,
                                          OperationState &result) {
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  // Parse the input argument
  OpAsmParser::Argument argInfo;
  argInfo.type = parser.getBuilder().getIndexType();
  if (failed(parser.parseArgument(argInfo)))
    return failure();

  // Parse the body region, and reuse the operand info as the argument info.
  Region *body = result.addRegion();
  return parser.parseRegion(*body, argInfo, /*enableNameShadowing=*/false);
}

void TestWithBoundsRegionOp::print(OpAsmPrinter &p) {
  p.printOptionalAttrDict((*this)->getAttrs());
  p << ' ';
  p.printRegionArgument(getRegion().getArgument(0), /*argAttrs=*/{},
                        /*omitType=*/true);
  p << ' ';
  p.printRegion(getRegion(), /*printEntryBlockArgs=*/false);
}

void TestWithBoundsRegionOp::inferResultRanges(
    ArrayRef<ConstantIntRanges> argRanges, SetIntRangeFn setResultRanges) {
  Value arg = getRegion().getArgument(0);
  setResultRanges(arg, {getUmin(), getUmax(), getSmin(), getSmax()});
}

//===----------------------------------------------------------------------===//
// TestIncrementOp

void TestIncrementOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                        SetIntRangeFn setResultRanges) {
  const ConstantIntRanges &range = argRanges[0];
  APInt one(range.umin().getBitWidth(), 1);
  setResultRanges(getResult(),
                  {range.umin().uadd_sat(one), range.umax().uadd_sat(one),
                   range.smin().sadd_sat(one), range.smax().sadd_sat(one)});
}

//===----------------------------------------------------------------------===//
// TestReflectBoundsOp

void TestReflectBoundsOp::inferResultRanges(
    ArrayRef<ConstantIntRanges> argRanges, SetIntRangeFn setResultRanges) {
  const ConstantIntRanges &range = argRanges[0];
  MLIRContext *ctx = getContext();
  Builder b(ctx);
  setUminAttr(b.getIndexAttr(range.umin().getZExtValue()));
  setUmaxAttr(b.getIndexAttr(range.umax().getZExtValue()));
  setSminAttr(b.getIndexAttr(range.smin().getSExtValue()));
  setSmaxAttr(b.getIndexAttr(range.smax().getSExtValue()));
  setResultRanges(getResult(), range);
}
