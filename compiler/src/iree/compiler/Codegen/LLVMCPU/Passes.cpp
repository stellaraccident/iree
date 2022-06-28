// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Passes.h"

#include "iree-dialects/Dialect/LinalgExt/IR/TiledOpInterface.h"
#include "iree-dialects/Dialect/LinalgExt/Passes/Passes.h"
#include "iree-dialects/Dialect/LinalgTransform/Passes.h"
#include "iree/compiler/Codegen/Interfaces/PartitionableLoopsInterface.h"
#include "iree/compiler/Codegen/LLVMCPU/KernelDispatch.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Sandbox/Passes.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arithmetic/Transforms/Passes.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace iree_compiler {

/// Command line options used purely for development purposes. Not to be relied
/// on in any way.
static llvm::cl::opt<bool> clCheckIRBeforeLLVMConversion(
    "iree-codegen-check-ir-before-llvm-conversion",
    llvm::cl::desc("Runs the pass to check the IR generated from LLVMCPU "
                   "before conversion to LLVM IR"),
    llvm::cl::init(true));

static llvm::cl::opt<bool> clCheckLinalgVectorization(
    "iree-llvmcpu-check-linalg-vectorization",
    llvm::cl::desc(
        "Runs the pass to check if all the Linalg ops are vectorized"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> clEnableHoistPadding(
    "iree-llvmcpu-enable-hoist-padding",
    llvm::cl::desc("Flag to enable hoist padding"), llvm::cl::init(false));

// MLIR file containing a top-level module that specifies the transformations to
// apply to form dispatch regions.
// Defined externally in KernelDispatch.cpp to control the codegen pass
// pipeline.
extern llvm::cl::opt<std::string> clCPUCodegenTransformDialectFileName;

//===---------------------------------------------------------------------===//
// Default allocation functions for CPU backend
//===---------------------------------------------------------------------===//

// Allocation callbacks to use with upstream comprehensive bufferization
static FailureOr<Value> cpuAllocationFn(OpBuilder &builder, Location loc,
                                        MemRefType memRefType,
                                        ValueRange dynamicSizes,
                                        unsigned alignment) {
  return builder
      .create<memref::AllocaOp>(loc, memRefType, dynamicSizes,
                                builder.getI64IntegerAttr(alignment))
      .getResult();
}

static LogicalResult cpuDeallocationFn(OpBuilder &builder, Location loc,
                                       Value allocation) {
  return success();
}

static LogicalResult cpuCopyFn(OpBuilder &builder, Location loc, Value from,
                               Value to) {
  createLinalgCopyOp(builder, loc, from, to);
  return success();
}

static void addBufferizePasses(OpPassManager &passManager) {
  BufferizationOptions::AllocationFn allocationFn = cpuAllocationFn;
  BufferizationOptions::DeallocationFn deallocationFn = cpuDeallocationFn;
  BufferizationOptions::MemCpyFn memcpyFn = cpuCopyFn;
  addIREEComprehensiveBufferizePasses(passManager, allocationFn, deallocationFn,
                                      memcpyFn);
}

static void addTileAndDistributePasses(OpPassManager &pm) {
  pm.addPass(createTileAndDistributeToWorkgroupsPass());
  auto &nestedModulePM = pm.nest<ModuleOp>();
  nestedModulePM.addNestedPass<func::FuncOp>(
      createConvertToDestinationPassingStylePass());
  nestedModulePM.addNestedPass<func::FuncOp>(
      createFoldAffineMinInDistributedLoopsPass());
  nestedModulePM.addPass(createCanonicalizerPass());
  nestedModulePM.addPass(createCSEPass());
}

//===---------------------------------------------------------------------===//
// Codegen configuration verifications.
//===---------------------------------------------------------------------===//

static bool isValidInterchange(ArrayRef<int64_t> interchange, int numLoops) {
  if (interchange.empty()) return true;
  llvm::SmallDenseSet<int64_t> s;
  s.insert(interchange.begin(), interchange.end());
  for (int i = 0; i < numLoops; ++i) {
    if (!s.contains(i)) return false;
  }
  return true;
}

LogicalResult verifyDoubleTilingExpertPassPipelineConfig(
    Operation *op, IREE::Codegen::LoweringConfigAttr loweringConfig,
    IREE::Codegen::TranslationInfoAttr translationInfo,
    ArrayRef<int64_t> workgroupSize) {
  if (!workgroupSize.empty()) {
    return op->emitOpError(
        "expected workgroup size to be empty for CPU pipelines");
  }

  // Verify that the translation info is using the right pipeline.
  if (translationInfo.getDispatchLoweringPassPipeline() !=
          IREE::Codegen::DispatchLoweringPassPipeline::CPUDoubleTilingExpert &&
      translationInfo.getDispatchLoweringPassPipeline() !=
          IREE::Codegen::DispatchLoweringPassPipeline::
              CPUDoubleTilingPadExpert) {
    return op->emitOpError("expected pipeline in translation_info to be ")
           << stringifyEnum(IREE::Codegen::DispatchLoweringPassPipeline::
                                CPUDoubleTilingExpert)
           << " or "
           << stringifyEnum(IREE::Codegen::DispatchLoweringPassPipeline::
                                CPUDoubleTilingPadExpert);
  }

  // Verify that the workload per workgroup is not set.
  // TODO(ravishankarm): Remove workload_per_wg eventually.
  SmallVector<int64_t> workloadPerWorkgroup =
      translationInfo.getWorkloadPerWorkgroupVals();
  if (!workloadPerWorkgroup.empty()) {
    return op->emitOpError(
               "workload_per_wg expected to be empty since its internal "
               "compiler implementation detail")
           << kNumMaxParallelDims;
  }

  if (loweringConfig.getTileSizes().size() !=
      static_cast<unsigned>(StrategyTilingLevel::NumStrategyTileLevels)) {
    return op->emitOpError("expected three tiling sizes, got ")
           << loweringConfig.getTileSizes().size();
  }

  auto interfaceOp = dyn_cast_or_null<PartitionableLoopsInterface>(op);
  if (interfaceOp) {
    llvm::SmallDenseSet<unsigned> pLoopsSet;
    for (auto iteratorType : llvm::enumerate(interfaceOp.getIteratorTypes())) {
      if (iteratorType.value() == getParallelIteratorTypeName()) {
        pLoopsSet.insert(iteratorType.index());
      }
    }

    SmallVector<int64_t> secondLevelTileSizes = loweringConfig.getTileSizeVals(
        static_cast<unsigned>(StrategyTilingLevel::ParallelTiles));
    for (auto en : llvm::enumerate(secondLevelTileSizes)) {
      if (en.value() != 0 && !pLoopsSet.contains(en.index())) {
        return op->emitOpError(
                   "expected only parallel dims to be set in the "
                   "second tiling sizes, got ")
               << en.index() << "-th tile size set";
      }
    }

    SmallVector<int64_t> thirdLevelTileSizes = loweringConfig.getTileSizeVals(
        static_cast<unsigned>(StrategyTilingLevel::ReductionTiles));
    for (auto en : llvm::enumerate(thirdLevelTileSizes)) {
      if (en.value() != 0 && pLoopsSet.contains(en.index())) {
        return op->emitOpError(
                   "expected only reduction dims to be set in the third "
                   "tiling sizes, got ")
               << en.index() << "-th tile size set";
      }
    }
  }

  // Verify interchange
  if (!loweringConfig.getTileInterchange().empty()) {
    for (auto level : llvm::seq<unsigned>(
             0, static_cast<unsigned>(
                    loweringConfig.getTileInterchange().size()))) {
      auto tileSizes = loweringConfig.getTileSizeVals(level);
      auto interchange = loweringConfig.getTileInterchangeVals(level);
      if (!isValidInterchange(interchange, tileSizes.size())) {
        return op->emitOpError("expected [0, ")
               << tileSizes.size()
               << ") to be set exactly once in interchange #" << level;
      }
    }
  }

  // Verify that native vector size is empty.
  SmallVector<int64_t> nativeVectorSize =
      loweringConfig.getNativeVectorSizeVals();
  if (!nativeVectorSize.empty()) {
    return op->emitOpError("native_vector_size must be empty");
  }
  return success();
}

//===---------------------------------------------------------------------===//
// Codegen pipelines.
//===---------------------------------------------------------------------===//

void addCPUBufferOpsTileAndVectorizePipeline(OpPassManager &passManager) {
  addTileAndDistributePasses(passManager);

  OpPassManager &nestedModulePM = passManager.nest<ModuleOp>();
  {
    // Skip tiling reduction loops because this is expected to apply on copy ops
    // only.
    LinalgSingleTilingExpertPassOptions options;
    options.tilingLevel =
        static_cast<int64_t>(StrategyTilingLevel::ParallelTiles);
    options.vectorize = true;
    nestedModulePM.addNestedPass<func::FuncOp>(
        createLinalgSingleTilingExpertPass(options));
    nestedModulePM.addNestedPass<func::FuncOp>(createCanonicalizerPass());
    nestedModulePM.addNestedPass<func::FuncOp>(createCSEPass());
  }

  // Run IREE specific passes before vector lowering expert.
  nestedModulePM.addNestedPass<func::FuncOp>(
      createRemoveSingleIterationLoopPass());

  // Add the vector lowering expert.
  {
    OpPassManager &nestedFuncPassManager = nestedModulePM.nest<func::FuncOp>();
    LinalgVectorLoweringPassOptions options;
    options.splitVectorTransfersTo = "linalg-copy";
    addLowerToVectorTransforms(nestedFuncPassManager, options);
  }
}

void addDoubleTilingPadExpertPassPipeline(OpPassManager &passManager) {
  addTileAndDistributePasses(passManager);

  OpPassManager &nestedModulePM = passManager.nest<ModuleOp>();
  {
    LinalgFusePassOptions options;
    options.tilingLevel =
        static_cast<int64_t>(StrategyTilingLevel::ParallelTiles);
    nestedModulePM.addNestedPass<func::FuncOp>(createLinalgFusePass(options));
    nestedModulePM.addNestedPass<func::FuncOp>(createCanonicalizerPass());
    nestedModulePM.addNestedPass<func::FuncOp>(createCSEPass());
  }

  auto pad = [&](std::string anchorOpName, bool setAnchorOpToRootOp = false,
                 ArrayRef<int64_t> packPaddings = {}) {
    LinalgFusePassOptions options;
    options.padParallelDims = true;
    if (setAnchorOpToRootOp) {
      options.setAnchorOpToRootOp = true;
    } else {
      options.anchorOpName = anchorOpName;
    }
    options.packPaddings.assign(packPaddings.begin(), packPaddings.end());
    nestedModulePM.addNestedPass<func::FuncOp>(createLinalgFusePass(options));
  };

  pad("linalg.fill");
  pad("", /*setAnchorOpToRootOp=*/true);
  // TODO(hanchung): pack and hoist padding for linalg.generic op.
  pad("linalg.generic");

  {
    LinalgSingleTilingExpertPassOptions options;
    options.tilingLevel =
        static_cast<int64_t>(StrategyTilingLevel::ReductionTiles);
    nestedModulePM.addNestedPass<func::FuncOp>(
        createLinalgSingleTilingExpertPass(options));
    nestedModulePM.addNestedPass<func::FuncOp>(createCanonicalizerPass());
    nestedModulePM.addNestedPass<func::FuncOp>(createCSEPass());
  }

  if (!clEnableHoistPadding) {
    LinalgFusePassOptions options;
    options.padReductionDims = true;
    options.setAnchorOpToRootOp = true;
    nestedModulePM.addNestedPass<func::FuncOp>(createLinalgFusePass(options));
  } else {
    {
      LinalgFusePassOptions options;
      options.padReductionDims = true;
      options.setAnchorOpToRootOp = true;
      options.packPaddings = {1, 1, 0};
      nestedModulePM.addNestedPass<func::FuncOp>(createLinalgFusePass(options));
    }

    LinalgFusePassOptions options;
    options.pad = true;
    options.setAnchorOpToRootOp = true;
    options.hoistPaddings = SmallVector<int64_t>{2, 3, 0};
    nestedModulePM.addNestedPass<func::FuncOp>(createLinalgFusePass(options));
    nestedModulePM.addNestedPass<func::FuncOp>(createCanonicalizerPass());
    nestedModulePM.addNestedPass<func::FuncOp>(createCSEPass());
  }

  // Fold dim(pad) away before vectorization.
  nestedModulePM.addPass(memref::createResolveShapedTypeResultDimsPass());

  {
    LinalgSingleTilingExpertPassOptions options;
    options.vectorize = true;
    options.vectorizePadding = true;
    nestedModulePM.addNestedPass<func::FuncOp>(
        createLinalgSingleTilingExpertPass(options));
    nestedModulePM.addNestedPass<func::FuncOp>(createCanonicalizerPass());
    nestedModulePM.addNestedPass<func::FuncOp>(createCSEPass());
  }

  addBufferizePasses(nestedModulePM);

  // Run IREE specific passes before vector lowering expert.
  nestedModulePM.addNestedPass<func::FuncOp>(
      createRemoveSingleIterationLoopPass());

  // Add the vector lowering expert.
  {
    OpPassManager &nestedFuncPassManager = nestedModulePM.nest<func::FuncOp>();
    LinalgVectorLoweringPassOptions options;
    options.splitVectorTransfersTo = "linalg-copy";
    addLowerToVectorTransforms(nestedFuncPassManager, options);
  }
}

void addDoubleTilingExpertPassPipeline(OpPassManager &passManager,
                                       bool enablePeeling, bool lowerToAVX2) {
  addTileAndDistributePasses(passManager);

  OpPassManager &nestedModulePM = passManager.nest<ModuleOp>();
  // Run LinalgFusePass firstly in case that we have fill + matmul + generic
  // ops. At this stage, we do not apply vectorization. The reduction dim won't
  // get tiled if the case is matmul + generic op. In this case, we have to tile
  // along reduction dim again, which needs them to be Linalg ops form.
  {
    LinalgFusePassOptions options;
    options.tilingLevel =
        static_cast<int64_t>(StrategyTilingLevel::ParallelTiles);
    nestedModulePM.addNestedPass<func::FuncOp>(createLinalgFusePass(options));
    nestedModulePM.addNestedPass<func::FuncOp>(createCanonicalizerPass());
    nestedModulePM.addNestedPass<func::FuncOp>(createCSEPass());
  }

  // Add the sandbox single tiling expert to tile and vectorize.
  {
    LinalgSingleTilingExpertPassOptions options;
    options.peel = enablePeeling;
    options.vectorize = true;
    options.tilingLevel =
        static_cast<int64_t>(StrategyTilingLevel::ReductionTiles);
    nestedModulePM.addNestedPass<func::FuncOp>(
        createLinalgSingleTilingExpertPass(options));
    nestedModulePM.addNestedPass<func::FuncOp>(createCanonicalizerPass());
    nestedModulePM.addNestedPass<func::FuncOp>(createCSEPass());
  }

  addBufferizePasses(nestedModulePM);

  // Run IREE specific passes before vector lowering expert.
  nestedModulePM.addNestedPass<func::FuncOp>(
      createRemoveSingleIterationLoopPass());

  // Add the vector lowering expert.
  {
    OpPassManager &nestedFuncPassManager = nestedModulePM.nest<func::FuncOp>();
    LinalgVectorLoweringPassOptions options;
    options.lowerVectorTransposeToAVX2 = lowerToAVX2;
    options.splitVectorTransfersTo = "linalg-copy";
    addLowerToVectorTransforms(nestedFuncPassManager, options);
  }
}

void addConvTileAndDecomposeExpertPassPipeline(OpPassManager &passManager) {
  addTileAndDistributePasses(passManager);

  OpPassManager &nestedModulePM = passManager.nest<ModuleOp>();
  // Run LinalgFusePass firstly in case that we have fill + conv + generic
  // ops. At this stage, we do not apply vectorization. The reduction dim won't
  // get tiled if the case is conv + generic op. In this case, we have to tile
  // along reduction dim again, which needs them to be Linalg ops form.
  {
    LinalgFusePassOptions options;
    options.tilingLevel =
        static_cast<int64_t>(StrategyTilingLevel::ParallelTiles);
    nestedModulePM.addNestedPass<func::FuncOp>(createLinalgFusePass(options));
    nestedModulePM.addNestedPass<func::FuncOp>(createCanonicalizerPass());
    nestedModulePM.addNestedPass<func::FuncOp>(createCSEPass());
  }

  // Add the sandbox single tiling expert to tile.
  {
    LinalgSingleTilingExpertPassOptions options;
    options.decomposeToLowerDimOp = true;
    options.tilingLevel =
        static_cast<int64_t>(StrategyTilingLevel::ReductionTiles);
    nestedModulePM.addNestedPass<func::FuncOp>(
        createLinalgSingleTilingExpertPass(options));
    nestedModulePM.addNestedPass<func::FuncOp>(createCanonicalizerPass());
    nestedModulePM.addNestedPass<func::FuncOp>(createCSEPass());
  }

  // Add the sandbox single tiling expert to vectorize.
  // We can't do the vectorization in the tiling expert above due to an issue in
  // codegen strategy pipeline. Since we are moving to the transform dialect, we
  // choose to have a workaround here by splitting them into two stages.
  {
    LinalgSingleTilingExpertPassOptions options;
    options.vectorize = true;
    options.vectorizePadding = true;
    nestedModulePM.addNestedPass<func::FuncOp>(
        createLinalgSingleTilingExpertPass(options));
    nestedModulePM.addNestedPass<func::FuncOp>(createCanonicalizerPass());
    nestedModulePM.addNestedPass<func::FuncOp>(createCSEPass());
  }

  addBufferizePasses(nestedModulePM);
  nestedModulePM.addNestedPass<func::FuncOp>(createCSEPass());
  nestedModulePM.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  nestedModulePM.addNestedPass<func::FuncOp>(
      createOptimizeVectorTransferPass());

  // Run IREE specific passes before vector lowering expert.
  nestedModulePM.addNestedPass<func::FuncOp>(
      createRemoveSingleIterationLoopPass());

  // Add the vector lowering expert.
  {
    OpPassManager &nestedFuncPassManager = nestedModulePM.nest<func::FuncOp>();
    LinalgVectorLoweringPassOptions options;
    options.splitVectorTransfersTo = "shuffle";
    addLowerToVectorTransforms(nestedFuncPassManager, options);
  }
}

void addTileFuseAndVectorizePassPipeline(OpPassManager &passManager,
                                         bool lowerToVectors) {
  addTileAndDistributePasses(passManager);

  OpPassManager &nestedModulePM = passManager.nest<ModuleOp>();

  nestedModulePM.addNestedPass<func::FuncOp>(
      createLLVMCPUTileFuseAndVectorizePass(lowerToVectors));
  nestedModulePM.addNestedPass<func::FuncOp>(createCSEPass());
  nestedModulePM.addNestedPass<func::FuncOp>(createCanonicalizerPass());

  addBufferizePasses(nestedModulePM);
  nestedModulePM.addNestedPass<func::FuncOp>(createCSEPass());
  nestedModulePM.addNestedPass<func::FuncOp>(createCanonicalizerPass());

  // TODO: This produces ~300k instructions for vmvx !?
  // nestedModulePM.addNestedPass<func::FuncOp>(
  //     createLLVMCPUAArch64VectorLoweringPass());
  nestedModulePM.addNestedPass<func::FuncOp>(
      createOptimizeVectorTransferPass());
}

void addCPUDefaultPassPipeline(OpPassManager &passManager) {
  addTileAndDistributePasses(passManager);
  OpPassManager &nestedModulePM = passManager.nest<ModuleOp>();
  addBufferizePasses(nestedModulePM);
}

void addTransformDialectInterpreterPasses(OpPassManager &passManager) {
  // Give control to the transform dialect.
  passManager.addPass(createTransformDialectInterpreterPass(
      clCPUCodegenTransformDialectFileName));

  // Dropping the schedule is only needed if we want to embed the transform in
  // the module: we should drop the schedule once applied.
  // This pass does nothing in the case where we apply a separate policy
  // through a file.
  passManager.addPass(createDropSchedulePass());
}

static void addLowerToLLVMPasses(OpPassManager &passManager) {
  // LinalgExt -> SCF
  passManager.addNestedPass<func::FuncOp>(
      IREE::LinalgExt::createLinalgExtToLoopsPass());

  // Linalg -> SCF
  passManager.addNestedPass<func::FuncOp>(createMemrefCopyToLinalgPass());
  if (clCheckLinalgVectorization) {
    passManager.addNestedPass<func::FuncOp>(
        createLLVMCPUEmitVectorizationRemarksPass());
  }
  passManager.addNestedPass<func::FuncOp>(createConvertLinalgToLoopsPass());
  passManager.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  passManager.addNestedPass<func::FuncOp>(createCSEPass());

  // Handled tensor-type constants.
  passManager.addPass(arith::createConstantBufferizePass());
  passManager.addPass(createFoldTensorExtractOpPass());

  // math dialect elementry functions -> polynomial form.
  passManager.addNestedPass<func::FuncOp>(createPolynomialApproximationPass());

  // Checking stack allocation before converting to CF dialect is easier.
  // Do not check allocation if hoist-padding is enabled. It intends to allocate
  // big stack buffers for better accessing.
  if (clCheckIRBeforeLLVMConversion && !clEnableHoistPadding) {
    passManager.addPass(createLLVMCPUCheckIRBeforeLLVMConversionPass());
  }

  // SCF -> CF
  passManager.addNestedPass<func::FuncOp>(createConvertSCFToCFPass());
  passManager.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  passManager.addNestedPass<func::FuncOp>(createCSEPass());

  // (HAL, IREE, Linalg, CF) -> LLVM
  passManager.addNestedPass<func::FuncOp>(
      arith::createArithmeticExpandOpsPass());
  passManager.addNestedPass<func::FuncOp>(memref::createExpandOpsPass());
  passManager.addPass(createConvertToLLVMPass());
  passManager.addPass(createReconcileUnrealizedCastsPass());

  // We rely on MLIR symbol visibility being correct after this point and need
  // to mirror the LLVM linkage that was assigned during conversion.
  passManager.addPass(createLLVMCPUSynchronizeSymbolVisibilityPass());

  passManager.addPass(createCanonicalizerPass());
  passManager.addPass(createCSEPass());
}

void buildLLVMCPUCodegenPassPipeline(OpPassManager &passManager) {
  passManager.addNestedPass<ModuleOp>(
      createVerifyLinalgTransformLegalityPass());
  passManager.nest<ModuleOp>().addNestedPass<func::FuncOp>(
      createTypePropagationPass());
  passManager.addNestedPass<ModuleOp>(createBufferizeCopyOnlyDispatchesPass());

  passManager.addPass(createLLVMCPULowerExecutableTargetPass());
  OpPassManager &nestedModulePM = passManager.nest<ModuleOp>();
  addLowerToLLVMPasses(nestedModulePM);
}

}  // namespace iree_compiler
}  // namespace mlir
