// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Numopt/IR/NumoptDialect.h"

#include "iree/compiler/Dialect/Numopt/IR/NumoptOps.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Parser.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Numopt {

NumoptDialect::NumoptDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context, TypeID::get<NumoptDialect>()) {
#define GET_OP_LIST
  addOperations<
#include "iree/compiler/Dialect/Numopt/IR/NumoptOps.cpp.inc"
      >();
}

}  // namespace Numopt
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
