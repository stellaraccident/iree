// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "iree/compiler/Translation/IREEVM.h"

#include "iree/compiler/Bindings/Native/Transforms/Passes.h"
#include "iree/compiler/Bindings/SIP/Transforms/Passes.h"
#include "iree/compiler/Bindings/TFLite/Transforms/Passes.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h"
#include "iree/compiler/Dialect/IREE/Transforms/Passes.h"
#include "iree/compiler/Dialect/VM/Target/Bytecode/TranslationFlags.h"
#include "iree/compiler/Dialect/VM/Transforms/Passes.h"
#include "iree/compiler/Utils/TracingUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Translation.h"

#ifdef IREE_HAVE_EMITC_DIALECT
#include "iree/compiler/Dialect/VM/Target/C/CModuleTarget.h"
#include "iree/compiler/Dialect/VM/Target/C/TranslationFlags.h"
#endif  // IREE_HAVE_EMITC_DIALECT

namespace mlir {
namespace iree_compiler {

// TODO(#3817): move all of this code to the iree-compile driver/API.
// Breaking this up such that for development iree-opt runs all passes/pipelines
// and iree-translate strictly does the VM dialect to bytecode/emitc files will
// match upstream better, and then our own iree-compile C API/binary will do the
// whole end-to-end with options for bindings/targets/etc.
struct BindingOptions {
  // Whether to include runtime support functions for the IREE native ABI.
  bool native = false;
  // Whether to include runtime support functions and metadata required for
  // SIP-compatible bindings (like bindings/python/iree).
  bool sip = true;
  // Whether to include runtime support functions required for the IREE TFLite
  // API compatibility bindings.
  bool tflite = false;
};

static BindingOptions getBindingOptionsFromFlags() {
  static llvm::cl::OptionCategory bindingOptionsCategory(
      "IREE translation binding support options");

  static llvm::cl::opt<bool> *bindingsNativeFlag = new llvm::cl::opt<bool>{
      "iree-native-bindings-support",
      llvm::cl::desc(
          "Include runtime support for native IREE ABI-compatible bindings"),
      llvm::cl::init(true), llvm::cl::cat(bindingOptionsCategory)};

  static llvm::cl::opt<bool> *bindingsSIPFlag = new llvm::cl::opt<bool>{
      "iree-sip-bindings-support",
      llvm::cl::desc("Include runtime support for SIP-compatible bindings"),
      llvm::cl::init(false), llvm::cl::cat(bindingOptionsCategory)};

  static llvm::cl::opt<bool> *bindingsTFLiteFlag = new llvm::cl::opt<bool>{
      "iree-tflite-bindings-support",
      llvm::cl::desc(
          "Include runtime support for the IREE TFLite compatibility bindings"),
      llvm::cl::init(false), llvm::cl::cat(bindingOptionsCategory)};

  BindingOptions bindingOptions;
  bindingOptions.native = *bindingsNativeFlag;
  bindingOptions.sip = *bindingsSIPFlag;
  bindingOptions.tflite = *bindingsTFLiteFlag;
  return bindingOptions;
}

// Performs initial dialect conversion to get the canonical input lowered into
// the IREE execution/dataflow dialect.
//
// This will fail if we cannot support the input yet. The hope is that any
// error that happens after this point is either backend-specific (like
// unsupported SPIR-V lowering) or a bug.
static LogicalResult convertToFlowModule(ModuleOp moduleOp) {
  PassManager passManager(moduleOp.getContext());
  mlir::applyPassManagerCLOptions(passManager);
  passManager.addInstrumentation(std::make_unique<PassTracing>());
  IREE::Flow::buildInputTransformPassPipeline(passManager);
  IREE::Flow::buildFlowTransformPassPipeline(passManager);
  if (failed(passManager.run(moduleOp))) {
    return moduleOp.emitError()
           << "failed to run flow transformation pass pipeline";
  }
  return success();
}

// Runs the flow->HAL transform pipeline to lower a flow module and compile
// executables for the specified target backends.
static LogicalResult convertToHALModule(
    ModuleOp moduleOp, IREE::HAL::TargetOptions executableOptions) {
  PassManager passManager(moduleOp.getContext());
  mlir::applyPassManagerCLOptions(passManager);
  passManager.addInstrumentation(std::make_unique<PassTracing>());
  IREE::HAL::buildHALTransformPassPipeline(passManager, executableOptions);
  if (failed(passManager.run(moduleOp))) {
    return moduleOp.emitError()
           << "failed to run HAL transformation pass pipeline";
  }
  return success();
}

// Converts the lowered module to a canonical vm.module containing only vm ops.
// This uses patterns to convert from standard ops and other dialects to their
// vm ABI form.
static LogicalResult convertToVMModule(ModuleOp moduleOp,
                                       IREE::VM::TargetOptions targetOptions) {
  PassManager passManager(moduleOp.getContext());
  mlir::applyPassManagerCLOptions(passManager);
  passManager.addInstrumentation(std::make_unique<PassTracing>());
  IREE::VM::buildVMTransformPassPipeline(passManager, targetOptions);
  if (failed(passManager.run(moduleOp))) {
    return moduleOp.emitError()
           << "failed to run VM transformation pass pipeline";
  }
  return success();
}

static void buildIREEVMTransformPassPipeline(
    BindingOptions bindingOptions, IREE::HAL::TargetOptions executableOptions,
    IREE::VM::TargetOptions targetOptions, OpPassManager &passManager) {
  if (bindingOptions.native) {
    IREE::ABI::buildTransformPassPipeline(passManager);
  }
  if (bindingOptions.sip) {
    IREE::SIP::buildTransformPassPipeline(passManager);
  }
  if (bindingOptions.tflite) {
    IREE::TFLite::buildTransformPassPipeline(passManager);
  }
  IREE::Flow::buildInputTransformPassPipeline(passManager);
  // VMLA must go with legacy path.
  bool dispatchLinalgOnTensors = true;
  if (std::find(executableOptions.targets.begin(),
                executableOptions.targets.end(),
                "vmla") != executableOptions.targets.end()) {
    dispatchLinalgOnTensors = false;
  }
  IREE::Flow::buildFlowTransformPassPipeline(passManager,
                                             dispatchLinalgOnTensors);
  IREE::HAL::buildHALTransformPassPipeline(passManager, executableOptions);
  IREE::VM::buildVMTransformPassPipeline(passManager, targetOptions);
  passManager.addPass(mlir::iree_compiler::IREE::createDropCompilerHintsPass());
}

void buildDefaultIREEVMTransformPassPipeline(OpPassManager &passManager) {
  buildIREEVMTransformPassPipeline(
      getBindingOptionsFromFlags(), IREE::HAL::getTargetOptionsFromFlags(),
      IREE::VM::getTargetOptionsFromFlags(), passManager);
}

void registerIREEVMTransformPassPipeline() {
  PassPipelineRegistration<> transformPassPipeline(
      "iree-transformation-pipeline",
      "Runs the full IREE input to VM transformation pipeline",
      [](OpPassManager &passManager) {
        buildDefaultIREEVMTransformPassPipeline(passManager);
      });
}

// Converts from our source to a vm.module in canonical form.
// After this completes we have a non-bytecode-specific vm.module that we
// could lower to other forms (LLVM IR, C, etc).
static LogicalResult translateFromMLIRToVM(
    ModuleOp moduleOp, BindingOptions bindingOptions,
    IREE::HAL::TargetOptions executableOptions,
    IREE::VM::TargetOptions targetOptions) {
  PassManager passManager(moduleOp.getContext());
  mlir::applyPassManagerCLOptions(passManager);
  passManager.addInstrumentation(std::make_unique<PassTracing>());
  buildIREEVMTransformPassPipeline(bindingOptions, executableOptions,
                                   targetOptions, passManager);
  if (failed(passManager.run(moduleOp))) {
    return moduleOp.emitError() << "conversion from source -> vm failed";
  }
  return success();
}

// Translates an MLIR module containing a set of supported IREE input dialects
// to an IREE VM bytecode module for loading at runtime.
//
// See iree/schemas/bytecode_module_def.fbs for the description of the
// serialized module format.
//
// Exposed via the --iree-mlir-to-vm-bytecode-module translation.
static LogicalResult translateFromMLIRToVMBytecodeModuleWithFlags(
    ModuleOp moduleOp, llvm::raw_ostream &output) {
  mlir::registerPassManagerCLOptions();
  auto bindingOptions = getBindingOptionsFromFlags();
  auto halTargetOptions = IREE::HAL::getTargetOptionsFromFlags();
  auto vmTargetOptions = IREE::VM::getTargetOptionsFromFlags();
  auto bytecodeTargetOptions = IREE::VM::getBytecodeTargetOptionsFromFlags();
  auto result = translateFromMLIRToVM(moduleOp, bindingOptions,
                                      halTargetOptions, vmTargetOptions);
  if (failed(result)) {
    return result;
  }
  return translateModuleToBytecode(moduleOp, bytecodeTargetOptions, output);
}

#ifdef IREE_HAVE_EMITC_DIALECT
// Translates an MLIR module containing a set of supported IREE input dialects
// to an IREE VM C module.
//
// Exposed via the --iree-mlir-to-vm-c-module translation.
static LogicalResult translateFromMLIRToVMCModule(
    ModuleOp moduleOp, BindingOptions bindingOptions,
    IREE::HAL::TargetOptions executableOptions,
    IREE::VM::TargetOptions targetOptions,
    IREE::VM::CTargetOptions cTargetOptions, llvm::raw_ostream &output) {
  auto result = translateFromMLIRToVM(moduleOp, bindingOptions,
                                      executableOptions, targetOptions);
  if (failed(result)) {
    return result;
  }

  // Serialize to c code.
  return mlir::iree_compiler::IREE::VM::translateModuleToC(
      moduleOp, cTargetOptions, output);
}

static LogicalResult translateFromMLIRToVMCModuleWithFlags(
    ModuleOp moduleOp, llvm::raw_ostream &output) {
  mlir::registerPassManagerCLOptions();
  auto bindingOptions = getBindingOptionsFromFlags();
  auto halTargetOptions = IREE::HAL::getTargetOptionsFromFlags();
  auto vmTargetOptions = IREE::VM::getTargetOptionsFromFlags();
  auto cTargetOptions = IREE::VM::getCTargetOptionsFromFlags();
  return translateFromMLIRToVMCModule(moduleOp, bindingOptions,
                                      halTargetOptions, vmTargetOptions,
                                      cTargetOptions, output);
}
#endif  // IREE_HAVE_EMITC_DIALECT

void registerIREEVMTranslation() {
  getBindingOptionsFromFlags();

  TranslateFromMLIRRegistration toVMBytecodeModuleWithFlags(
      "iree-mlir-to-vm-bytecode-module",
      translateFromMLIRToVMBytecodeModuleWithFlags);

#ifdef IREE_HAVE_EMITC_DIALECT
  TranslateFromMLIRRegistration toVMCModuleWithFlags(
      "iree-mlir-to-vm-c-module", translateFromMLIRToVMCModuleWithFlags);
#endif  // IREE_HAVE_EMITC_DIALECT
}

}  // namespace iree_compiler
}  // namespace mlir
