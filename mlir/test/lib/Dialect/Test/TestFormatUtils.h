//===- TestFormatUtils.h - MLIR Test Dialect Assembly Format Utilities ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TESTFORMATUTILS_H
#define MLIR_TESTFORMATUTILS_H

#include "mlir/IR/OpImplementation.h"

namespace test {

//===----------------------------------------------------------------------===//
// CustomDirectiveOperands
//===----------------------------------------------------------------------===//

mlir::ParseResult parseCustomDirectiveOperands(
    mlir::OpAsmParser &parser, mlir::OpAsmParser::UnresolvedOperand &operand,
    llvm::Optional<mlir::OpAsmParser::UnresolvedOperand> &optOperand,
    llvm::SmallVectorImpl<mlir::OpAsmParser::UnresolvedOperand> &varOperands);

void printCustomDirectiveOperands(mlir::OpAsmPrinter &printer,
                                  mlir::Operation *, mlir::Value operand,
                                  mlir::Value optOperand,
                                  mlir::OperandRange varOperands);

//===----------------------------------------------------------------------===//
// CustomDirectiveResults
//===----------------------------------------------------------------------===//

mlir::ParseResult
parseCustomDirectiveResults(mlir::OpAsmParser &parser, mlir::Type &operandType,
                            mlir::Type &optOperandType,
                            llvm::SmallVectorImpl<mlir::Type> &varOperandTypes);

void printCustomDirectiveResults(mlir::OpAsmPrinter &printer, mlir::Operation *,
                                 mlir::Type operandType,
                                 mlir::Type optOperandType,
                                 mlir::TypeRange varOperandTypes);

//===----------------------------------------------------------------------===//
// CustomDirectiveWithTypeRefs
//===----------------------------------------------------------------------===//

mlir::ParseResult parseCustomDirectiveWithTypeRefs(
    mlir::OpAsmParser &parser, mlir::Type operandType,
    mlir::Type optOperandType,
    const llvm::SmallVectorImpl<mlir::Type> &varOperandTypes);

void printCustomDirectiveWithTypeRefs(mlir::OpAsmPrinter &printer,
                                      mlir::Operation *op,
                                      mlir::Type operandType,
                                      mlir::Type optOperandType,
                                      mlir::TypeRange varOperandTypes);

//===----------------------------------------------------------------------===//
// CustomDirectiveOperandsAndTypes
//===----------------------------------------------------------------------===//

mlir::ParseResult parseCustomDirectiveOperandsAndTypes(
    mlir::OpAsmParser &parser, mlir::OpAsmParser::UnresolvedOperand &operand,
    llvm::Optional<mlir::OpAsmParser::UnresolvedOperand> &optOperand,
    llvm::SmallVectorImpl<mlir::OpAsmParser::UnresolvedOperand> &varOperands,
    mlir::Type &operandType, mlir::Type &optOperandType,
    llvm::SmallVectorImpl<mlir::Type> &varOperandTypes);

void printCustomDirectiveOperandsAndTypes(
    mlir::OpAsmPrinter &printer, mlir::Operation *op, mlir::Value operand,
    mlir::Value optOperand, mlir::OperandRange varOperands,
    mlir::Type operandType, mlir::Type optOperandType,
    mlir::TypeRange varOperandTypes);

//===----------------------------------------------------------------------===//
// CustomDirectiveRegions
//===----------------------------------------------------------------------===//

mlir::ParseResult parseCustomDirectiveRegions(
    mlir::OpAsmParser &parser, mlir::Region &region,
    llvm::SmallVectorImpl<std::unique_ptr<mlir::Region>> &varRegions);

void printCustomDirectiveRegions(
    mlir::OpAsmPrinter &printer, mlir::Operation *, mlir::Region &region,
    llvm::MutableArrayRef<mlir::Region> varRegions);

//===----------------------------------------------------------------------===//
// CustomDirectiveSuccessors
//===----------------------------------------------------------------------===//

mlir::ParseResult parseCustomDirectiveSuccessors(
    mlir::OpAsmParser &parser, mlir::Block *&successor,
    llvm::SmallVectorImpl<mlir::Block *> &varSuccessors);

void printCustomDirectiveSuccessors(mlir::OpAsmPrinter &printer,
                                    mlir::Operation *, mlir::Block *successor,
                                    mlir::SuccessorRange varSuccessors);

//===----------------------------------------------------------------------===//
// CustomDirectiveAttributes
//===----------------------------------------------------------------------===//

mlir::ParseResult parseCustomDirectiveAttributes(mlir::OpAsmParser &parser,
                                                 mlir::IntegerAttr &attr,
                                                 mlir::IntegerAttr &optAttr);

void printCustomDirectiveAttributes(mlir::OpAsmPrinter &printer,
                                    mlir::Operation *,
                                    mlir::Attribute attribute,
                                    mlir::Attribute optAttribute);

//===----------------------------------------------------------------------===//
// CustomDirectiveAttrDict
//===----------------------------------------------------------------------===//

mlir::ParseResult parseCustomDirectiveAttrDict(mlir::OpAsmParser &parser,
                                               mlir::NamedAttrList &attrs);

void printCustomDirectiveAttrDict(mlir::OpAsmPrinter &printer,
                                  mlir::Operation *op,
                                  mlir::DictionaryAttr attrs);

//===----------------------------------------------------------------------===//
// CustomDirectiveOptionalOperandRef
//===----------------------------------------------------------------------===//

mlir::ParseResult parseCustomDirectiveOptionalOperandRef(
    mlir::OpAsmParser &parser,
    llvm::Optional<mlir::OpAsmParser::UnresolvedOperand> &optOperand);

void printCustomDirectiveOptionalOperandRef(mlir::OpAsmPrinter &printer,
                                            mlir::Operation *op,
                                            mlir::Value optOperand);

//===----------------------------------------------------------------------===//
// CustomDirectiveOptionalOperand
//===----------------------------------------------------------------------===//

mlir::ParseResult parseCustomOptionalOperand(
    mlir::OpAsmParser &parser,
    llvm::Optional<mlir::OpAsmParser::UnresolvedOperand> &optOperand);

void printCustomOptionalOperand(mlir::OpAsmPrinter &printer, mlir::Operation *,
                                mlir::Value optOperand);

} // end namespace test

#endif // MLIR_TESTFORMATUTILS_H
