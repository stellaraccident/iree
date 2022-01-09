// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree_tf_compiler/TFL/Passes.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "tensorflow/compiler/mlir/lite/flatbuffer_import.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/dialect_registration.h"

using namespace llvm;
using namespace mlir;

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);

  static cl::opt<std::string> inputPath(
      cl::Positional, cl::desc("<TFLite FlatBuffer>"), cl::Required);
  static cl::opt<std::string> outputFilename("o", cl::desc("Output filename"),
                                             cl::value_desc("filename"),
                                             cl::init("-"));

  static cl::opt<std::string> saveTempTflInput(
      "save-temp-tfl-input",
      cl::desc("Save the TFL pipeline input to this file"), cl::init(""));
  static cl::opt<std::string> saveTempIreeImport(
      "save-temp-iree-input",
      cl::desc("Save the resultant IR to this file (useful for saving an "
               "intermediate in a pipeline)"),
      cl::init(""));

  static cl::list<std::string> inputArrayFlag(
      "input-array",
      cl::desc("Input tensor, if different from the default inputs"),
      cl::ZeroOrMore);
  static cl::list<std::string> outputArrayFlag(
      "output-array",
      cl::desc("Output tensor, if different from the default outputs"),
      cl::ZeroOrMore);

  // Register any command line options.
  registerAsmPrinterCLOptions();
  registerMLIRContextCLOptions();
  registerPassManagerCLOptions();
  registerDefaultTimingManagerCLOptions();
  cl::ParseCommandLineOptions(argc, argv);

  // Initialize dialects.
  DialectRegistry registry;
  registry.insert<mlir::TFL::TensorFlowLiteDialect>();
  registry.insert<mlir::tosa::TosaDialect>();
  registry.insert<quant::QuantizationDialect>();
  registry.insert<StandardOpsDialect, mlir::arith::ArithmeticDialect>();

  RegisterAllTensorFlowDialects(registry);

  // Convert the Module proto into MLIR.
  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  llvm::SourceMgr sourceMgr;
  mlir::SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);

  // Load input buffer.
  std::string errorMessage;
  auto inputFile = openInputFile(inputPath, &errorMessage);
  if (!inputFile) {
    llvm::errs() << errorMessage << "\n";
    return 1;
  }

  // Convert.
  std::vector<std::string> inputArrays(inputArrayFlag.begin(),
                                       inputArrayFlag.end());
  std::vector<std::string> outputArrays(outputArrayFlag.begin(),
                                        outputArrayFlag.end());
  auto loc = mlir::FileLineColLoc::get(&context,
                                       inputFile->getBufferIdentifier(), 0, 0);
  OwningModuleRef module = tflite::FlatBufferToMlir(
      absl::string_view(inputFile->getBufferStart(),
                        inputFile->getBufferSize()),
      &context, loc,
      /*use_external_constant=*/false, inputArrays, outputArrays);
  if (!module) {
    // Error should have emitted.
    llvm::errs() << "Unable to import TFLite flatbuffer to MLIR Module\n";
    return 2;
  }

  // Save.
  auto saveToFile = [&](llvm::StringRef savePath) -> LogicalResult {
    auto outputFile = openOutputFile(savePath);
    if (!outputFile) {
      llvm::errs() << "Could not open output file: " << savePath << "\n";
      return failure();
    }
    OpPrintingFlags printFlags;
    module->print(outputFile->os(), printFlags);
    outputFile->os() << "\n";
    outputFile->keep();
    return success();
  };

  // Save temp input.
  if (!saveTempTflInput.empty()) {
    if (failed(saveToFile(saveTempTflInput))) return 10;
  }

  // Run transformations.
  PassManager pm(&context, PassManager::Nesting::Implicit);
  applyPassManagerCLOptions(pm);
  applyDefaultTimingPassManagerCLOptions(pm);
  mlir::iree_integrations::TFL::buildTFLImportPassPipeline(pm);
  if (failed(pm.run(*module))) {
    llvm::errs() << "Running iree-import-tflite pass pipeline failed (see "
                    "diagnostics)\n";
    return 3;
  }

  // Save temp output.
  if (!saveTempIreeImport.empty()) {
    if (failed(saveToFile(saveTempIreeImport))) return 10;
  }

  // Save output.
  if (failed(saveToFile(outputFilename))) return 3;
  return 0;
}
