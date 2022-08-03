// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_MODULES_VMVX_PASSES
#define IREE_MODULES_VMVX_PASSES

include "mlir/Pass/PassBase.td"

def ConvertHalEntrypoints: Pass<"iree-vmvx-convert-hal-entrypoints", "mlir::ModuleOp"> {
  let summary = "Converts function signature and HAL objects to util/memref";
  let description = [{
    Converts hal entry-points of the form:

    ```
    func.func @_tensor_float_dispatch_0() {
      %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(64) : memref<?xf32>{%c6}
    ```

    These are converted to the VMVX runtime ABI where all bindings and HAL
    ops can be resolved from the function signature operating on memrefs.
  }];
  let constructor = "mlir::iree_compiler::IREE::VMVX::createConvertHalEntrypointsPass()";
}

def Conversion : Pass<"iree-vmvx-conversion", "mlir::ModuleOp"> {
  let summary = "Converts from various dialects to the VMVX dialect";
  let constructor = "mlir::iree_compiler::IREE::VMVX::createConversionPass()";
}

def ResolveBufferDescriptors : Pass<"iree-vmvx-resolve-buffer-descriptors", ""> {
  let summary = "Resolves any existing vmvx.get_buffer_descriptor ops";
  let constructor = "mlir::iree_compiler::IREE::VMVX::createResolveBufferDescriptorsPass()";
}

#endif  // IREE_MODULES_VMVX_PASSES
