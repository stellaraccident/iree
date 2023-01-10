// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_CPU_COMMON_H_
#define IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_CPU_COMMON_H_

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir {
namespace iree_compiler {
namespace cpu {

/// Take care of the last common steps in a CPU strategy (i.e. vectorize,
/// bufferize, maps to blocks/workgroups and lower vectors).
/// Return the handles to the updated variant and the func::FuncOp ops under
/// the variant op.
// TODO: pass control to LowerVectorsOp once the builder allows it.
std::pair<Value, Value> buildCommonTrailingStrategy(ImplicitLocOpBuilder &b,
                                                    Value variantH);

/// Return success if the IR matches what the GPU reduction strategy can
/// handle. If it is success it will append the transform dialect after the
/// entry point module.
LogicalResult matchAndSetReductionStrategy(func::FuncOp entryPoint,
                                           linalg::LinalgOp op);
}  // namespace cpu
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_CPU_COMMON_H_