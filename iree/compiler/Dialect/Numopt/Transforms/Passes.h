// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_NUMOPT_TRANSFORMS_PASSES_H_
#define IREE_COMPILER_DIALECT_NUMOPT_TRANSFORMS_PASSES_H_

#include "iree/compiler/Dialect/Numopt/IR/NumoptOps.h"
#include "llvm/ADT/StringMap.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Numopt {

void registerNumoptPasses();

}  // namespace Numopt
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_NUMOPT_TRANSFORMS_PASSES_H_
