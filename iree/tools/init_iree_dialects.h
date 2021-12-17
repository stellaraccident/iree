// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// This files defines a helper to trigger the registration of dialects to
// the system.
//
// Based on MLIR's InitAllDialects but for IREE dialects.

#ifndef IREE_TOOLS_INIT_IREE_DIALECTS_H_
#define IREE_TOOLS_INIT_IREE_DIALECTS_H_

#include "iree-dialects/Dialect/Input/InputDialect.h"
#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree-dialects/Dialect/LinalgExt/IR/TiledOpInterface.h"
#include "iree-dialects/Dialect/PyDM/IR/PyDMDialect.h"
#include "iree/compiler/Codegen/Dialect/IREECodegenDialect.h"
#include "iree/compiler/Codegen/Interfaces/Interfaces.h"
#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/Modules/VMVX/IR/VMVXDialect.h"
#include "iree/compiler/Dialect/Numopt/IR/NumoptDialect.h"
#include "iree/compiler/Dialect/Stream/IR/StreamDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/VM/IR/VMDialect.h"
#include "iree/compiler/Dialect/Vulkan/IR/VulkanDialect.h"
#include "mlir/IR/Dialect.h"

namespace mlir {
namespace iree_compiler {

// Add all the IREE dialects to the provided registry.
inline void registerIreeDialects(DialectRegistry &registry) {
  // clang-format off
  registry.insert<IREE::Codegen::IREECodegenDialect,
                  IREE::Flow::FlowDialect,
                  IREE::HAL::HALDialect,
                  IREE::LinalgExt::IREELinalgExtDialect,
                  IREE::Numopt::NumoptDialect,
                  IREE::Stream::StreamDialect,
                  IREE::Util::UtilDialect,
                  IREE::VM::VMDialect,
                  IREE::VMVX::VMVXDialect,
                  IREE::Vulkan::VulkanDialect,
                  IREE::Input::IREEInputDialect,
                  IREE::PYDM::IREEPyDMDialect>();
  // clang-format on

  IREE::LinalgExt::registerTiledOpInterfaceExternalModels(registry);
  registerCodegenInterfaces(registry);
}

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_TOOLS_INIT_IREE_DIALECTS_H_
