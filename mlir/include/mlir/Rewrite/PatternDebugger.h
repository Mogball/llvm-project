//===- PatternDebugger.h --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_REWRITE_PATTERN_DEBUGGER_H_
#define MLIR_REWRITE_PATTERN_DEBUGGER_H_

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/DebugAction.h"

namespace mlir {
struct PatternDebuggerAction : DebugAction<const RewritePattern *, Operation *> {
  static StringRef getTag() { return "pattern-debugger"; }
  static StringRef getDescription() { return "Enable the pattern debugger."; }
};

struct PatternDebuggerHandler : public PatternDebuggerAction::Handler {
  FailureOr<bool> shouldExecute(const RewritePattern *pattern,
                                Operation *op) override;
};
} // end namespace mlir

#endif // MLIR_REWRITE_PATTERN_DEBUGGER_H_
