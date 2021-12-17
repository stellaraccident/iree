// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_NUMOPT_IR_NUMOPTDIALECT_H_
#define IREE_COMPILER_DIALECT_NUMOPT_IR_NUMOPTDIALECT_H_

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Numopt {

class NumoptDialect : public Dialect {
 public:
  explicit NumoptDialect(MLIRContext* context);
  static StringRef getDialectNamespace() { return "numopt"; }

  // Attribute parseAttribute(DialectAsmParser& parser, Type type) const
  // override; void printAttribute(Attribute attr, DialectAsmPrinter& p) const
  // override;

  // Type parseType(DialectAsmParser& parser) const override;
  // void printType(Type type, DialectAsmPrinter& os) const override;

  // Operation* materializeConstant(OpBuilder& builder, Attribute value, Type
  // type,
  //                                Location loc) override;

  // void getCanonicalizationPatterns(RewritePatternSet& results) const
  // override;

 private:
  // void registerAttributes();
  // void registerTypes();
};

}  // namespace Numopt
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_NUMOPT_IR_NUMOPTDIALECT_H_
