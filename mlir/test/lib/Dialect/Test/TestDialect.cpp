//===- TestDialect.cpp - MLIR Dialect for Testing -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestDialect.h"
#include "TestOps.h"
#include "TestTypes.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Interfaces/FoldInterfaces.h"
#include "mlir/Reducer/ReductionPatternInterface.h"
#include "mlir/Transforms/InliningUtils.h"

// Include this before the using namespace lines below to test that we don't
// have namespace dependencies.
#include "TestOpsDialect.cpp.inc"

using namespace mlir;
using namespace test;

void test::registerTestDialect(DialectRegistry &registry) {
  registry.insert<TestDialect>();
}

//===----------------------------------------------------------------------===//
// TestDialect Interfaces
//===----------------------------------------------------------------------===//

namespace {

//===----------------------------------------------------------------------===//
// OpAsmDialectInterface

// Test support for interacting with the AsmPrinter.
struct TestOpAsmInterface : public OpAsmDialectInterface {
  using OpAsmDialectInterface::OpAsmDialectInterface;

  AliasResult getAlias(Attribute attr, raw_ostream &os) const final {
    StringAttr strAttr = attr.dyn_cast<StringAttr>();
    if (!strAttr)
      return AliasResult::NoAlias;

    // Check the contents of the string attribute to see what the test alias
    // should be named.
    Optional<StringRef> aliasName =
        StringSwitch<Optional<StringRef>>(strAttr.getValue())
            .Case("alias_test:dot_in_name", StringRef("test.alias"))
            .Case("alias_test:trailing_digit", StringRef("test_alias0"))
            .Case("alias_test:prefixed_digit", StringRef("0_test_alias"))
            .Case("alias_test:sanitize_conflict_a",
                  StringRef("test_alias_conflict0"))
            .Case("alias_test:sanitize_conflict_b",
                  StringRef("test_alias_conflict0_"))
            .Case("alias_test:tensor_encoding", StringRef("test_encoding"))
            .Default(llvm::None);
    if (!aliasName)
      return AliasResult::NoAlias;

    os << *aliasName;
    return AliasResult::FinalAlias;
  }

  AliasResult getAlias(Type type, raw_ostream &os) const final {
    if (auto tupleType = type.dyn_cast<TupleType>()) {
      if (tupleType.size() > 0 &&
          llvm::all_of(tupleType.getTypes(), [](Type elemType) {
            return elemType.isa<SimpleAType>();
          })) {
        os << "test_tuple";
        return AliasResult::FinalAlias;
      }
    }
    if (auto intType = type.dyn_cast<TestIntegerType>()) {
      if (intType.getSignedness() ==
              TestIntegerType::SignednessSemantics::Unsigned &&
          intType.getWidth() == 8) {
        os << "test_ui8";
        return AliasResult::FinalAlias;
      }
    }
    return AliasResult::NoAlias;
  }
};

//===----------------------------------------------------------------------===//
// DialectFoldInterface

struct TestDialectFoldInterface : public DialectFoldInterface {
  using DialectFoldInterface::DialectFoldInterface;

  /// Registered hook to check if the given region, which is attached to an
  /// operation that is *not* isolated from above, should be used when
  /// materializing constants.
  bool shouldMaterializeInto(Region *region) const final {
    // If this is a one region operation, then insert into it.
    return isa<OneRegionOp>(region->getParentOp());
  }
};

//===----------------------------------------------------------------------===//
// DialectInlinerInterface

/// This class defines the interface for handling inlining with standard
/// operations.
struct TestInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  //===--------------------------------------------------------------------===//
  // Analysis Hooks

  bool isLegalToInline(Operation *call, Operation *callable,
                       bool wouldBeCloned) const final {
    // Don't allow inlining calls that are marked `noinline`.
    return !call->hasAttr("noinline");
  }
  bool isLegalToInline(Region *, Region *, bool,
                       BlockAndValueMapping &) const final {
    // Inlining into test dialect regions is legal.
    return true;
  }
  bool isLegalToInline(Operation *, Region *, bool,
                       BlockAndValueMapping &) const final {
    return true;
  }

  bool shouldAnalyzeRecursively(Operation *op) const final {
    // Analyze recursively if this is not a functional region operation, it
    // froms a separate functional scope.
    return !isa<FunctionalRegionOp>(op);
  }

  //===--------------------------------------------------------------------===//
  // Transformation Hooks

  /// Handle the given inlined terminator by replacing it with a new operation
  /// as necessary.
  void handleTerminator(Operation *op,
                        ArrayRef<Value> valuesToRepl) const final {
    // Only handle "test.return" here.
    auto returnOp = dyn_cast<TestReturnOp>(op);
    if (!returnOp)
      return;

    // Replace the values directly with the return operands.
    assert(returnOp.getNumOperands() == valuesToRepl.size());
    for (const auto &it : llvm::enumerate(returnOp.getOperands()))
      valuesToRepl[it.index()].replaceAllUsesWith(it.value());
  }

  /// Attempt to materialize a conversion for a type mismatch between a call
  /// from this dialect, and a callable region. This method should generate an
  /// operation that takes 'input' as the only operand, and produces a single
  /// result of 'resultType'. If a conversion can not be generated, nullptr
  /// should be returned.
  Operation *materializeCallConversion(OpBuilder &builder, Value input,
                                       Type resultType,
                                       Location conversionLoc) const final {
    // Only allow conversion for i16/i32 types.
    if (!(resultType.isSignlessInteger(16) ||
          resultType.isSignlessInteger(32)) ||
        !(input.getType().isSignlessInteger(16) ||
          input.getType().isSignlessInteger(32)))
      return nullptr;
    return builder.create<TestCastOp>(conversionLoc, resultType, input);
  }

  void processInlinedCallBlocks(
      Operation *call,
      iterator_range<Region::iterator> inlinedBlocks) const final {
    if (!isa<ConversionCallOp>(call))
      return;

    // Set attributed on all ops in the inlined blocks.
    for (Block &block : inlinedBlocks) {
      block.walk([&](Operation *op) {
        op->setAttr("inlined_conversion", UnitAttr::get(call->getContext()));
      });
    }
  }
};

//===----------------------------------------------------------------------===//
// DialectReductionPatternInterface

struct TestReductionPatternInterface : public DialectReductionPatternInterface {
  TestReductionPatternInterface(Dialect *dialect)
      : DialectReductionPatternInterface(dialect) {}

  void populateReductionPatterns(RewritePatternSet &patterns) const final {
    populateTestReductionPatterns(patterns);
  }
};

//===----------------------------------------------------------------------===//
// TestEffectOpInterface

// This is the implementation of a dialect fallback for `TestEffectOpInterface`.
struct TestOpEffectInterfaceFallback
    : public TestEffectOpInterface::FallbackModel<
          TestOpEffectInterfaceFallback> {
  static bool classof(Operation *op) {
    bool isSupportedOp =
        op->getName().getStringRef() == "test.unregistered_side_effect_op";
    assert(isSupportedOp && "Unexpected dispatch");
    return isSupportedOp;
  }

  void
  getEffects(Operation *op,
             SmallVectorImpl<SideEffects::EffectInstance<TestEffects::Effect>>
                 &effects) const {
    auto effectsAttr = op->getAttrOfType<AffineMapAttr>("effect_parameter");
    if (!effectsAttr)
      return;

    effects.emplace_back(TestEffects::Concrete::get(), effectsAttr);
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Dynamic operations
//===----------------------------------------------------------------------===//

std::unique_ptr<DynamicOpDefinition> getDynamicGenericOp(TestDialect *dialect) {
  return DynamicOpDefinition::get(
      "dynamic_generic", dialect, [](Operation *op) { return success(); },
      [](Operation *op) { return success(); });
}

std::unique_ptr<DynamicOpDefinition>
getDynamicOneOperandTwoResultsOp(TestDialect *dialect) {
  return DynamicOpDefinition::get(
      "dynamic_one_operand_two_results", dialect,
      [](Operation *op) {
        if (op->getNumOperands() != 1) {
          op->emitOpError()
              << "expected 1 operand, but had " << op->getNumOperands();
          return failure();
        }
        if (op->getNumResults() != 2) {
          op->emitOpError()
              << "expected 2 results, but had " << op->getNumResults();
          return failure();
        }
        return success();
      },
      [](Operation *op) { return success(); });
}

std::unique_ptr<DynamicOpDefinition>
getDynamicCustomParserPrinterOp(TestDialect *dialect) {
  auto verifier = [](Operation *op) {
    if (op->getNumOperands() == 0 && op->getNumResults() == 0)
      return success();
    op->emitError() << "operation should have no operands and no results";
    return failure();
  };
  auto regionVerifier = [](Operation *op) { return success(); };

  auto parser = [](OpAsmParser &parser, OperationState &state) {
    return parser.parseKeyword("custom_keyword");
  };

  auto printer = [](Operation *op, OpAsmPrinter &printer, llvm::StringRef) {
    printer << op->getName() << " custom_keyword";
  };

  return DynamicOpDefinition::get("dynamic_custom_parser_printer", dialect,
                                  verifier, regionVerifier, parser, printer);
}

//===----------------------------------------------------------------------===//
// TestDialect
//===----------------------------------------------------------------------===//

void TestDialect::initialize() {
  registerAttributes();
  registerTypes();
  registerTestDialectOperations(this);
  registerDynamicOp(getDynamicGenericOp(this));
  registerDynamicOp(getDynamicOneOperandTwoResultsOp(this));
  registerDynamicOp(getDynamicCustomParserPrinterOp(this));

  addInterfaces<TestOpAsmInterface, TestDialectFoldInterface,
                TestInlinerInterface, TestReductionPatternInterface>();
  allowUnknownOperations();

  // Instantiate our fallback op interface that we'll use on specific
  // unregistered op.
  fallbackEffectOpInterfaces = new TestOpEffectInterfaceFallback;
}
TestDialect::~TestDialect() {
  delete static_cast<TestOpEffectInterfaceFallback *>(
      fallbackEffectOpInterfaces);
}

//===----------------------------------------------------------------------===//
// TestDialect Hooks

Operation *TestDialect::materializeConstant(OpBuilder &builder, Attribute value,
                                            Type type, Location loc) {
  return builder.create<TestOpConstant>(loc, type, value);
}

void *TestDialect::getRegisteredInterfaceForOp(TypeID interfaceID,
                                               OperationName opName) {
  if (opName.getIdentifier() == "test.unregistered_side_effect_op" &&
      interfaceID == TypeID::get<TestEffectOpInterface>())
    return fallbackEffectOpInterfaces;
  return nullptr;
}

LogicalResult TestDialect::verifyOperationAttribute(Operation *op,
                                                    NamedAttribute attribute) {
  if (attribute.getName() == "test.invalid_attr")
    return op->emitError() << "invalid to use 'test.invalid_attr'";
  return success();
}

LogicalResult TestDialect::verifyRegionArgAttribute(Operation *op,
                                                    unsigned regionIndex,
                                                    unsigned argIndex,
                                                    NamedAttribute attribute) {
  if (attribute.getName() == "test.invalid_attr")
    return op->emitError() << "invalid to use 'test.invalid_attr'";
  return success();
}

LogicalResult
TestDialect::verifyRegionResultAttribute(Operation *op, unsigned regionIndex,
                                         unsigned resultIndex,
                                         NamedAttribute attribute) {
  if (attribute.getName() == "test.invalid_attr")
    return op->emitError() << "invalid to use 'test.invalid_attr'";
  return success();
}

Optional<Dialect::ParseOpHook>
TestDialect::getParseOperationHook(StringRef opName) const {
  if (opName == "test.dialect_custom_printer") {
    return ParseOpHook{[](OpAsmParser &parser, OperationState &state) {
      return parser.parseKeyword("custom_format");
    }};
  }
  if (opName == "test.dialect_custom_format_fallback") {
    return ParseOpHook{[](OpAsmParser &parser, OperationState &state) {
      return parser.parseKeyword("custom_format_fallback");
    }};
  }
  if (opName == "test.dialect_custom_printer.with.dot") {
    return ParseOpHook{[](OpAsmParser &parser, OperationState &state) {
      return ParseResult::success();
    }};
  }
  return None;
}

llvm::unique_function<void(Operation *, OpAsmPrinter &)>
TestDialect::getOperationPrinter(Operation *op) const {
  StringRef opName = op->getName().getStringRef();
  if (opName == "test.dialect_custom_printer") {
    return [](Operation *op, OpAsmPrinter &printer) {
      printer.getStream() << " custom_format";
    };
  }
  if (opName == "test.dialect_custom_format_fallback") {
    return [](Operation *op, OpAsmPrinter &printer) {
      printer.getStream() << " custom_format_fallback";
    };
  }
  return {};
}

//===----------------------------------------------------------------------===//
// TestDialectCanonicalizerOp
//===----------------------------------------------------------------------===//

static LogicalResult
dialectCanonicalizationPattern(TestDialectCanonicalizerOp op,
                               PatternRewriter &rewriter) {
  rewriter.replaceOpWithNewOp<arith::ConstantOp>(
      op, rewriter.getI32IntegerAttr(42));
  return success();
}

void TestDialect::getCanonicalizationPatterns(
    RewritePatternSet &results) const {
  results.add(&dialectCanonicalizationPattern);
}
