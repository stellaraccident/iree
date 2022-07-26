// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_MODULES_VMVX_OPS
#define IREE_DIALECT_MODULES_VMVX_OPS

include "iree/compiler/Dialect/VMVX/IR/VMVXBase.td"
include "iree/compiler/Dialect/VMVX/IR/VMVXInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class VMVX_PureOp<string mnemonic, list<Trait> traits = []> :
    VMVX_Op<mnemonic, !listconcat(traits, [NoSideEffect])>;

//===----------------------------------------------------------------------===//
// Utility Ops
//===----------------------------------------------------------------------===//

def VMVX_DefBaseBufferOp : VMVX_PureOp<"def_base_buffer"> {
  let summary = "Defines a base buffer after lowering to the VMVX ABI";
  let description = [{
    When lowering from a binding, captures the original buffer and offset
    for later use.
  }];

  let arguments = (ins
    Util_BufferType:$original_buffer,
    VMVX_Index:$byte_offset
  );
  let results = (outs
    Util_BufferType:$buffer
  );

  let assemblyFormat = [{
    $original_buffer `offset` $byte_offset
    attr-dict
  }];
}

def VMVX_ResolveBufferOp : VMVX_PureOp<"resolve_buffer"> {
  let summary = "Gets a base buffer and adjusted offset of a memref";
  let description = [{
    Given a base memref and an element-type sized offset into it, returns
    a !util.buffer and adjusted offset to the actual backing buffer.

    This is needed to bridge between memrefs and the packed buffers that are
    passed at the VMVX ABI boundary, where a memref may be backed into an
    arbitrary slab of memory at a byte offset within that memory (which is
    aligned).
  }];

  let arguments = (ins
    Util_BufferType:$original_buffer,
    VMVX_Index:$original_offset,
    VMVX_Index:$element_size
  );
  let results = (outs
    Util_BufferType:$base_buffer,
    VMVX_Index:$adjusted_offset
  );

  let assemblyFormat = [{
    $original_buffer `offset` $original_offset `by` $element_size
    attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// VMVX Ops: ABI
//===----------------------------------------------------------------------===//

def VMVX_AddOp : VMVX_Op<"add", [SameVariadicOperandSize]> {
  let summary = "Performs a strided elementwise add of two same-rank buffers";
  let description = [{
    Performs addition in-place as if:
      OUT = LHS + RHS

    All operands have the same rank.
  }];
  let arguments = (ins
    // LHS.
    Util_BufferType:$lhs_buffer,
    VMVX_Index:$lhs_offset,
    Variadic<VMVX_Index>:$lhs_strides,
    // RHS.
    Util_BufferType:$rhs_buffer,
    VMVX_Index:$rhs_offset,
    Variadic<VMVX_Index>:$rhs_strides,
    // OUT.
    Util_BufferType:$out_buffer,
    VMVX_Index:$out_offset,
    Variadic<VMVX_Index>:$out_strides,

    // Dimensions.
    Variadic<VMVX_Index>:$sizes,

    // Attributes.
    VMVX_ElementTypeAttr:$element_type
  );

  let assemblyFormat = [{
    `lhs` `` `(` $lhs_buffer `offset` $lhs_offset `strides` `[` $lhs_strides `]``)`
    `rhs` `` `(` $rhs_buffer `offset` $rhs_offset `strides` `[` $rhs_strides `]``)`
    `out` `` `(` $out_buffer `offset` $out_offset `strides` `[` $out_strides `]``)`
    `sizes` `` `(` $sizes `)`
    `:` $element_type
    attr-dict
  }];
}

def VMVX_CopyOp : VMVX_Op<"copy", [SameVariadicOperandSize]> {
  let summary = "Copy from one buffer to another";
  let arguments = (ins
    // LHS.
    Util_BufferType:$in_buffer,
    VMVX_Index:$in_offset,
    Variadic<VMVX_Index>:$in_strides,
    // OUT.
    Util_BufferType:$out_buffer,
    VMVX_Index:$out_offset,
    Variadic<VMVX_Index>:$out_strides,

    // Dimensions.
    Variadic<VMVX_Index>:$sizes,

    // Attributes.
    VMVX_ElementTypeAttr:$element_type
  );
  let assemblyFormat = [{
    `in` `` `(` $in_buffer `offset` $in_offset `strides` `[` $in_strides `]``)`
    `out` `` `(` $out_buffer `offset` $out_offset `strides` `[` $out_strides `]``)`
    `sizes` `` `(` $sizes `)`
    `:` $element_type
    attr-dict
  }];
}

def VMVX_Fill2DOp : VMVX_Op<"fill2d"> {
  let summary = "Fill a tile with a scalar";
  let description = [{
    Fills a tile with dimensions [m, n] with a scalar.
  }];
  let arguments = (ins
    VMVX_ElementType:$scalar,
    Util_BufferType:$out_buffer,
    VMVX_Index:$out_offset,
    VMVX_Index:$out_row_stride,

    // Dimensions.
    VMVX_Index:$m,
    VMVX_Index:$n
  );

  let assemblyFormat = [{
    `scalar` `` `(` $scalar `:` type($scalar) `)`
    `out` `` `(` $out_buffer `offset` $out_offset `row_stride` $out_row_stride`)`
    `sizes` `` `(` $m `,` $n `)`
    attr-dict
  }];
}

def VMVX_MatmulOp : VMVX_Op<"matmul"> {
  let summary = "Matmul";
  let description = [{
    General matrix-multiply of the form:

      OUT = alpha * (LHS * RHS) + beta * OUT
  }];
  let arguments = (ins
    // Lhs buffer.
    Util_BufferType:$lhs_buffer,
    VMVX_Index:$lhs_offset,
    VMVX_Index:$lhs_row_stride,
    // Rhs buffer.
    Util_BufferType:$rhs_buffer,
    VMVX_Index:$rhs_offset,
    VMVX_Index:$rhs_row_stride,
    // Out buffer.
    Util_BufferType:$out_buffer,
    VMVX_Index:$out_offset,
    VMVX_Index:$out_row_stride,

    // Dimensions.
    VMVX_Index:$m,
    VMVX_Index:$n,
    VMVX_Index:$k,

    // Scale factors.
    VMVX_ElementType:$alpha,
    VMVX_ElementType:$beta,

    // Type and flag attributes.
    VMVX_ElementTypeAttr:$lhs_type,
    VMVX_ElementTypeAttr:$rhs_type,
    VMVX_ElementTypeAttr:$out_type,
    I32Attr:$flags
  );

  let assemblyFormat = [{
    `lhs` `` `(` $lhs_buffer `offset` $lhs_offset `row_stride` $lhs_row_stride `)`
    `rhs` `` `(` $rhs_buffer `offset` $rhs_offset `row_stride` $rhs_row_stride `)`
    `out` `` `(` $out_buffer `offset` $out_offset `row_stride` $out_row_stride `)`
    `mnk` `` `(` $m `,` $n `,` $k `)`
    `scale` `` `(` $alpha `:` type($alpha) `,` $beta `:` type($beta) `)`
    `flags` `` `(` $flags `)`
    `:` `(` $lhs_type `,` $rhs_type `,` $out_type `)`
    attr-dict
  }];
}

#endif  // IREE_DIALECT_MODULES_VMVX_OPS
