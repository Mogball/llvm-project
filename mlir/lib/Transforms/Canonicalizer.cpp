//===- Canonicalizer.cpp - Canonicalize MLIR operations -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This transformation pass converts operations into their canonical forms by
// folding constants, applying operation identity transformations etc.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

/// Initialize the patterns for a canonicalization pass. Collect
/// canonicalization patterns from all currently loaded dialects and registered
/// operations.
static FrozenRewritePatternSet
initializeCanonicalizer(MLIRContext *context,
                        ArrayRef<std::string> disabledPatterns,
                        ArrayRef<std::string> enabledPatterns) {
  RewritePatternSet owningPatterns(context);
  for (auto *dialect : context->getLoadedDialects())
    dialect->getCanonicalizationPatterns(owningPatterns);
  for (RegisteredOperationName op : context->getRegisteredOperations())
    op.getCanonicalizationPatterns(owningPatterns, context);

  return {std::move(owningPatterns), disabledPatterns, enabledPatterns};
}

namespace {
/// Canonicalize operations in nested regions.
struct Canonicalizer : public CanonicalizerBase<Canonicalizer> {
  Canonicalizer(const GreedyRewriteConfig &config,
                ArrayRef<std::string> disabledPatterns,
                ArrayRef<std::string> enabledPatterns)
      : config(config) {
    this->disabledPatterns = disabledPatterns;
    this->enabledPatterns = enabledPatterns;
  }

  Canonicalizer() {
    // Default constructed Canonicalizer takes its settings from command line
    // options.
    config.useTopDownTraversal = topDownProcessingEnabled;
    config.enableRegionSimplification = enableRegionSimplification;
    config.maxIterations = maxIterations;
  }

  /// Initialize the canonicalizer by building the set of patterns used during
  /// execution.
  LogicalResult initialize(MLIRContext *context) override {
    patterns =
        initializeCanonicalizer(context, disabledPatterns, enabledPatterns);
    return success();
  }
  void runOnOperation() override {
    (void)applyPatternsAndFoldGreedily(getOperation()->getRegions(), patterns,
                                       config);
  }

  /// The greedy rewrite config to use when applying patterns.
  GreedyRewriteConfig config;
  /// The canonicalization patterns.
  FrozenRewritePatternSet patterns;
};
} // namespace

/// Create a Canonicalizer pass.
std::unique_ptr<Pass> mlir::createCanonicalizerPass() {
  return std::make_unique<Canonicalizer>();
}

/// Creates an instance of the Canonicalizer pass with the specified config.
std::unique_ptr<Pass>
mlir::createCanonicalizerPass(const GreedyRewriteConfig &config,
                              ArrayRef<std::string> disabledPatterns,
                              ArrayRef<std::string> enabledPatterns) {
  return std::make_unique<Canonicalizer>(config, disabledPatterns,
                                         enabledPatterns);
}

/// Run canonicalization on the provided operation.
LogicalResult
mlir::canonicalizeOperations(Operation *op, const GreedyRewriteConfig &config,
                             ArrayRef<std::string> disabledPatterns,
                             ArrayRef<std::string> enabledPatterns,
                             RewriteListener *listener) {
  FrozenRewritePatternSet patterns = initializeCanonicalizer(
      op->getContext(), disabledPatterns, enabledPatterns);
  return applyPatternsAndFoldGreedily(op, patterns, config, listener);
}
