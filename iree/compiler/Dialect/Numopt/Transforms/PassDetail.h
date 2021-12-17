// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_NUMOPT_TRANSFORMS_PASS_DETAIL_H_
#define IREE_COMPILER_DIALECT_NUMOPT_TRANSFORMS_PASS_DETAIL_H_

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Numopt {

#define GEN_PASS_CLASSES
#include "iree/compiler/Dialect/Numopt/Transforms/Passes.h.inc"  // IWYU pragma: keep

}  // namespace Numopt
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_NUMOPT_TRANSFORMS_PASS_DETAIL_H_
