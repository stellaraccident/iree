// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_MODULES_VMVX_TRANSFORMS_PASSES_H_
#define IREE_COMPILER_DIALECT_MODULES_VMVX_TRANSFORMS_PASSES_H_

#include "iree/compiler/Dialect/VMVX/IR/VMVXOps.h"
#include "llvm/ADT/StringMap.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace VMVX {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

// Adds a set of passes to the given pass manager that run the required VMVX
// transforms in the canonical order.
//
// Most translation code should prefer to use this instead of manually adding
// the passes themselves to ensure that expected pass ordering is observed.
//
// The expected usage is:
//   <run conversion from TF/HLO/etc to flow>
//   buildVMVXTransformPassPipeline & run
//   <serialize VM module>
void buildVMVXTransformPassPipeline(OpPassManager &passManager);

//===----------------------------------------------------------------------===//
// Dialect conversion
//===----------------------------------------------------------------------===//

// Converts from various dialects (HAL, standard, etc) to the VMVX dialect.
std::unique_ptr<OperationPass<mlir::ModuleOp>> createConversionPass();

// Resolves any outstanding get_buffer_descriptor ops.
std::unique_ptr<OperationPass<>> createResolveBufferDescriptorsPass();

//===----------------------------------------------------------------------===//
// Register all Passes
//===----------------------------------------------------------------------===//

void registerVMVXPasses();

}  // namespace VMVX
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_MODULES_VMVX_TRANSFORMS_PASSES_H_
