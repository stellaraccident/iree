// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Numopt/Transforms/Passes.h"

#include <memory>

#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Numopt {

namespace {
#define GEN_PASS_REGISTRATION
#include "iree/compiler/Dialect/Numopt/Transforms/Passes.h.inc"  // IWYU pragma: export
}  // namespace

void registerNumoptPasses() {
  // Generated.
  registerPasses();
}

}  // namespace Numopt
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
