// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_NUMOPT_IR_NUMOPTOPS_H_
#define IREE_COMPILER_DIALECT_NUMOPT_IR_NUMOPTOPS_H_

//#include "iree/compiler/Dialect/Numopt/IR/NumoptTypes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"

#define GET_OP_CLASSES
#include "iree/compiler/Dialect/Numopt/IR/NumoptOps.h.inc"  // IWYU pragma: export

namespace mlir {
namespace iree_compiler {

//

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_NUMOPT_IR_NUMOPTOPS_H_
