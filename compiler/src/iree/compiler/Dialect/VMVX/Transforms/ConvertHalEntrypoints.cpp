// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/VMVX/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/VMVX/Transforms/Passes.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Matchers.h"

namespace mlir::iree_compiler::IREE::VMVX {
namespace {

// Ordered indices of arguments to the entry point function.
// This is what the VM will receive at runtime from the HAL.
enum EntryArgOrdinals {
  kEntryArgLocalMemory,
  kEntryArgConstants,
  kEntryArgBindings,
  kEntryArgWorkgroupX,
  kEntryArgWorkgroupY,
  kEntryArgWorkgroupZ,
  kEntryArgWorkgroupSizeX,
  kEntryArgWorkgroupSizeY,
  kEntryArgWorkgroupSizeZ,
  kEntryArgWorkgroupCountX,
  kEntryArgWorkgroupCountY,
  kEntryArgWorkgroupCountZ,
};

/// Rewrites entry functions to have a vmvx.interface, local memory, and an XYZ
/// workgroup ID. The runtime will provide these values during invocation.
///
/// Source:
///   func.func @entry()
///
/// Target:
///   func.func @entry(
///       %local_memory: !vmvx.buffer,
///       %constants: !vmvx.buffer,
///       %bindings: !util.list<!vmvx.buffer>,
///       %workgroup_id_x: index,
///       %workgroup_id_y: index,
///       %workgroup_id_z: index,
///       %workgroup_size_x: index,
///       %workgroup_size_y: index,
///       %workgroup_size_z: index,
///       %workgroup_count_x: index,
///       %workgroup_count_y: index,
///       %workgroup_count_z: index
///   )
LogicalResult updateHALToVMVXEntryFuncOp(func::FuncOp funcOp,
                                         TypeConverter &typeConverter) {
  auto originalType = funcOp.getFunctionType();
  if (originalType.getNumInputs() != 0 || originalType.getNumResults() != 0) {
    return funcOp.emitError() << "exported functions must have no I/O";
  }

  auto bufferType = IREE::Util::BufferType::get(funcOp.getContext());
  auto bindingsType = IREE::Util::ListType::get(bufferType);  // of i8
  auto indexType = IndexType::get(funcOp.getContext());
  auto newType = FunctionType::get(funcOp.getContext(),
                                   {
                                       /*local_memory=*/bufferType,  // of i8
                                       /*constants=*/bufferType,     // of i32
                                       /*bindings=*/bindingsType,
                                       /*workgroup_id_x=*/indexType,
                                       /*workgroup_id_y=*/indexType,
                                       /*workgroup_id_z=*/indexType,
                                       /*workgroup_size_x=*/indexType,
                                       /*workgroup_size_y=*/indexType,
                                       /*workgroup_size_z=*/indexType,
                                       /*workgroup_count_x=*/indexType,
                                       /*workgroup_count_y=*/indexType,
                                       /*workgroup_count_z=*/indexType,
                                   },
                                   {});

  funcOp.setType(newType);
  SmallVector<Location> locs(newType.getNumInputs(), funcOp.getLoc());
  funcOp.front().addArguments(newType.getInputs(), locs);

  return success();
}

/// Rewrites hal.interface.workgroup.id to use the arguments injected onto the
/// function.
struct ConvertHALInterfaceWorkgroupIDOp
    : public OpConversionPattern<IREE::HAL::InterfaceWorkgroupIDOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::HAL::InterfaceWorkgroupIDOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    uint64_t dim = op.getDimension().getZExtValue();
    if (dim >= 3) {
      return op.emitOpError() << "out of bounds workgroup ID dimension";
    }

    // Get the argument to the function corresponding to the workgroup dim.
    auto workgroupDim = op->getParentOfType<mlir::func::FuncOp>().getArgument(
        kEntryArgWorkgroupX + dim);
    rewriter.replaceOp(op, workgroupDim);
    return success();
  }
};

/// Rewrites hal.interface.workgroup.size to use the arguments injected onto the
/// function.
struct ConvertHALInterfaceWorkgroupSizeOp
    : public OpConversionPattern<IREE::HAL::InterfaceWorkgroupSizeOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::HAL::InterfaceWorkgroupSizeOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    uint64_t dim = op.getDimension().getZExtValue();
    if (dim >= 3) {
      return op.emitOpError() << "out of bounds workgroup size dimension";
    }

    // Get the argument to the function corresponding to the workgroup dim.
    auto workgroupDim = op->getParentOfType<mlir::func::FuncOp>().getArgument(
        kEntryArgWorkgroupSizeX + dim);
    rewriter.replaceOp(op, workgroupDim);
    return success();
  }
};

/// Rewrites hal.interface.workgroup.count to use the arguments injected onto
/// the function.
struct ConvertHALInterfaceWorkgroupCountOp
    : public OpConversionPattern<IREE::HAL::InterfaceWorkgroupCountOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::HAL::InterfaceWorkgroupCountOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    uint64_t dim = op.getDimension().getZExtValue();
    if (dim >= 3) {
      return op.emitOpError() << "out of bounds workgroup count dimension";
    }

    // Get the argument to the function corresponding to the workgroup dim.
    auto workgroupDim = op->getParentOfType<mlir::func::FuncOp>().getArgument(
        kEntryArgWorkgroupCountX + dim);
    rewriter.replaceOp(op, workgroupDim);
    return success();
  }
};

/// Rewrites hal.interface.constant.load to ops loading from the ABI structs.
struct ConvertHALInterfaceConstantLoadOp
    : public OpConversionPattern<IREE::HAL::InterfaceConstantLoadOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::HAL::InterfaceConstantLoadOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // Find the vmvx.interface argument to the function.
    auto constantsArg = op->getParentOfType<mlir::func::FuncOp>().getArgument(
        kEntryArgConstants);
    assert(constantsArg && "entry point not conforming to requirements");
    auto constantType =
        constantsArg.getType().cast<MemRefType>().getElementType();

    auto resultType = getTypeConverter()->convertType(op.getResult().getType());

    auto constantIndex = rewriter.createOrFold<arith::ConstantIndexOp>(
        op.getLoc(), op.getIndex().getZExtValue());
    auto loadedValue = rewriter.createOrFold<memref::LoadOp>(
        op.getLoc(), constantType, constantsArg, ValueRange{constantIndex});
    rewriter.replaceOpWithNewOp<mlir::UnrealizedConversionCastOp>(
        op, resultType, loadedValue);
    return success();
  }
};

/// Rewrites hal.interface.binding.subspan to ops loading from the ABI structs.
struct ConvertHALInterfaceBindingSubspanOp
    : public OpConversionPattern<IREE::HAL::InterfaceBindingSubspanOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::HAL::InterfaceBindingSubspanOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // Find the vmvx.interface argument to the function.
    auto bindingsArg = op->getParentOfType<mlir::func::FuncOp>().getArgument(
        kEntryArgBindings);
    assert(bindingsArg && bindingsArg.getType().isa<IREE::Util::ListType>() &&
           "entry point not conforming to requirements");

    // TODO(benvanik): compact the indices - the bindings we have on the ABI
    // interface are dense.
    if (op.getSet().getZExtValue() != 0) {
      return op.emitOpError() << "sparse binding sets not yet implemented";
    }

    auto bindingType =
        bindingsArg.getType().cast<IREE::Util::ListType>().getElementType();
    auto sourceBuffer =
        rewriter
            .create<IREE::Util::ListGetOp>(
                op.getLoc(), bindingType, bindingsArg,
                rewriter.createOrFold<arith::ConstantIndexOp>(
                    op.getLoc(), op.getBinding().getZExtValue()))
            .getResult();
    Value storageOffsetBytes;
    auto memRefType = op.getResult().getType().cast<MemRefType>();
    auto flatMemRefType =
        MemRefType::get({ShapedType::kDynamicSize}, rewriter.getI8Type(),
                        /*layout=*/nullptr, memRefType.getMemorySpace());
    // Get the buffer subspan.
    Value sourceSize = rewriter.createOrFold<IREE::Util::BufferSizeOp>(
        op.getLoc(), sourceBuffer);
    auto storageOp = rewriter.create<IREE::Util::BufferStorageOp>(
        op.getLoc(), flatMemRefType, rewriter.getIndexType(), sourceBuffer,
        sourceSize);
    storageOffsetBytes = storageOp.getOffset();

    // Construct the view.
    rewriter.replaceOpWithNewOp<memref::ViewOp>(
        op, memRefType, storageOp.getResult(), storageOffsetBytes,
        op.getDynamicDims());
    return success();
  }
};

void populateHALToVMVXPatterns(MLIRContext *context,
                               RewritePatternSet &patterns,
                               TypeConverter &typeConverter) {
  patterns.insert<ConvertHALInterfaceWorkgroupIDOp>(typeConverter, context);
  patterns.insert<ConvertHALInterfaceWorkgroupSizeOp>(typeConverter, context);
  patterns.insert<ConvertHALInterfaceWorkgroupCountOp>(typeConverter, context);
  patterns.insert<ConvertHALInterfaceConstantLoadOp>(typeConverter, context);
  patterns.insert<ConvertHALInterfaceBindingSubspanOp>(typeConverter, context);
}

class ConvertHalEntrypointsPass
    : public ConvertHalEntrypointsBase<ConvertHalEntrypointsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::Util::UtilDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override {
    auto *context = &getContext();

    TypeConverter typeConverter;

    typeConverter.addConversion([](Type type) { return type; });

    // Run a pre-pass that updates the entry function signature.
    for (auto funcOp : getOperation().getOps<func::FuncOp>()) {
      if (funcOp.isPublic()) {
        if (failed(updateHALToVMVXEntryFuncOp(funcOp, typeConverter))) {
          return signalPassFailure();
        }
      }
    }

    ConversionTarget conversionTarget(*context);
    conversionTarget.addIllegalDialect<IREE::HAL::HALDialect>();
    conversionTarget.markUnknownOpDynamicallyLegal(
        [](Operation *op) { return true; });
    RewritePatternSet patterns(&getContext());
    populateHALToVMVXPatterns(context, patterns, typeConverter);

    if (failed(applyPartialConversion(getOperation(), conversionTarget,
                                      std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<mlir::ModuleOp>>
createConvertHalEntrypointsPass() {
  return std::make_unique<ConvertHalEntrypointsPass>();
}

}  // namespace mlir::iree_compiler::IREE::VMVX
