//===- TestDialect.h - MLIR Dialect for testing -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a fake 'test' dialect that can be used for testing things
// that do not have a respective counterpart in the main source directories.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TESTDIALECT_H
#define MLIR_TESTDIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/ExtensibleDialect.h"
#include "llvm/ADT/SetVector.h"

namespace mlir {
class RewritePatternSet;
} // end namespace mlir

#include "TestOpsDialect.h.inc"

namespace test {
void registerTestDialect(mlir::DialectRegistry &registry);
void populateTestReductionPatterns(mlir::RewritePatternSet &patterns);
} // end namespace test

#endif // MLIR_TESTDIALECT_H
