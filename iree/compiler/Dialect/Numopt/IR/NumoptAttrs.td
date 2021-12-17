// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_NUMOPT_IR_NUMOPT_ATTRS
#define IREE_DIALECT_NUMOPT_IR_NUMOPT_ATTRS

include "iree/compiler/Dialect/Numopt/IR/NumoptBase.td"
include "iree/compiler/Dialect/Numopt/IR/NumoptInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SubElementInterfaces.td"

//===----------------------------------------------------------------------===//
// Buffer attributes
//===----------------------------------------------------------------------===//

def Numopt_ByteRangeAttr : AttrDef<Numopt_Dialect, "Dummy", []> {
  let mnemonic = "dummy";
  let summary = [{defines a range of bytes}];
  let description = [{
    Specifies a starting offset and total length in bytes.
  }];

  let parameters = (ins
    AttrParameter<"int64_t", "">:$offset,
    AttrParameter<"int64_t", "">:$length
  );
}

#endif // IREE_DIALECT_NUMOPT_IR_NUMOPT_ATTRS
