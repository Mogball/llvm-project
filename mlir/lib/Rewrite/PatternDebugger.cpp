//===- PatternDebugger.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Rewrite/PatternDebugger.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/Debug.h"
#include <iostream>

#define DEBUG_TYPE "pattern-debugger"

using namespace mlir;

namespace {
struct SimplePatternRewriter : public PatternRewriter {
  SimplePatternRewriter(MLIRContext *context) : PatternRewriter(context) {}
};
} // end anonymous namespace

static Operation *getNearestScope(Operation *op) {
  // Find the nearest parent op that is isolated from above. Pattern sets are
  // required to be applied on ops that are isolated from above, so there it is
  // not possible to climb up high enough that we exit the op scoped to a
  // thread.
  Operation *scope = op;
  do {
    scope = scope->getParentOp();
  } while (scope && !scope->hasTrait<OpTrait::IsIsolatedFromAbove>());
  assert(scope && "failed to find a transform scope");
  return scope;
}

static std::tuple<Operation *, Operation *>
cloneAndRemapHandle(Operation *scope, Operation *op) {
  unsigned opIndex = 0;
  WalkResult result = scope->walk([&](Operation *foundOp) {
    WalkResult result =
        foundOp == op ? WalkResult::interrupt() : WalkResult::advance();
    ++opIndex;
    return result;
  });
  assert(result.wasInterrupted() && "scope does not contain target op");

  Operation *scopeClone = scope->clone();
  Operation *opClone = nullptr;
  unsigned walkIndex = 0;
  scopeClone->walk([&](Operation *newOp) {
    if (++walkIndex == opIndex) {
      opClone = newOp;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  assert(opClone && "didn't find remapped op");
  return std::make_tuple(scopeClone, opClone);
}

static bool getUserYesNo(std::istream &in) {
  char result = 0;
  do {
    in >> result;
    if (in.fail()) {
      result = 0;
      in.clear();
    } else {
      result = std::tolower(result);
    }
    in.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  } while (result != 'n' && result != 'y');
  return result == 'y';
}

FailureOr<bool>
PatternDebuggerHandler::shouldExecute(const RewritePattern *pattern,
                                      Operation *op) {
  Operation *scope = getNearestScope(op);
  assert(scope->hasTrait<OpTrait::IsIsolatedFromAbove>() &&
         "expected transform scope to be isolated from above");
  SimplePatternRewriter rewriter(op->getContext());
  Operation *scopeClone, *opClone;
  std::tie(scopeClone, opClone) = cloneAndRemapHandle(scope, op);
  auto cleanup = llvm::make_scope_exit([&] { scopeClone->destroy(); });

  // If -debug is specified, the pattern applicator will print context messages.
  bool debugPrints = true;
  LLVM_DEBUG(debugPrints = false);

  rewriter.setInsertionPoint(opClone);
  if (succeeded(pattern->matchAndRewrite(opClone, rewriter))) {
    if (debugPrints) {
      llvm::errs() << "\n" << *op << "\n";
      llvm::errs() << "Pattern matched \"" << pattern->getDebugName() << "\"\n";
    }
    for (StringRef label : pattern->getDebugLabels())
      llvm::errs() << "  " << label << "\n";
    if (debugPrints) {
      llvm::errs() << "-> Apply? [y/n]: ";
    } else {
      llvm::errs() << "-> Pattern matched. Apply? [y/n]: ";
    }
    return getUserYesNo(std::cin);
  }
  return false;
}
