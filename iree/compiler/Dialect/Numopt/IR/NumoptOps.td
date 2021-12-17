// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_NUMOPT_IR_NUMOPT_OPS
#define IREE_DIALECT_NUMOPT_IR_NUMOPT_OPS

include "iree/compiler/Dialect/Numopt/IR/NumoptBase.td"
include "iree/compiler/Dialect/Numopt/IR/NumoptInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Op types
//===----------------------------------------------------------------------===//

class Numopt_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<Numopt_Dialect, mnemonic, traits> {
  let parser = [{ return parse$cppClass(parser, result); }];
  let printer = [{ print$cppClass(p, *this); }];
}

class Numopt_PureOp<string mnemonic, list<OpTrait> traits = []> :
    Numopt_Op<mnemonic, !listconcat(traits, [NoSideEffect])>;

//===----------------------------------------------------------------------===//
// Ops
//===----------------------------------------------------------------------===//

def Numopt_DummyOp : Numopt_PureOp<"dummy"> {
  let summary = [{dummy op}];
  let description = [{
    dummy op
  }];

  let results = (outs
    AnyType:$result
  );

  let assemblyFormat = [{
    attr-dict `:` type($result)
  }];
}

#endif // IREE_DIALECT_NUMOPT_IR_NUMOPT_OPS
