# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//build_tools/bazel:iree_tablegen_doc.bzl", "iree_tablegen_doc")
load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")
load("//build_tools/bazel:enforce_glob.bzl", "enforce_glob")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

exports_files(["NumoptBase.td"])

td_library(
    name = "td_files",
    srcs = enforce_glob(
        [
            "NumoptAttrs.td",
            "NumoptBase.td",
            "NumoptInterfaces.td",
            "NumoptOps.td",
        ],
        include = ["*.td"],
    ),
    deps = [
        "@llvm-project//mlir:CallInterfacesTdFiles",
        "@llvm-project//mlir:ControlFlowInterfacesTdFiles",
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectTdFiles",
        "@llvm-project//mlir:SubElementInterfacesTdFiles",
    ],
)

cc_library(
    name = "IR",
    srcs = [
        "NumoptDialect.cpp",
        "NumoptOps.cpp",
    ],
    hdrs = [
        "NumoptDialect.h",
        "NumoptOps.h",
    ],
    textual_hdrs = [
    ],
    deps = [
        ":NumoptAttrsGen",
        ":NumoptInterfacesGen",
        ":NumoptOpsGen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithmeticDialect",
        "@llvm-project//mlir:ControlFlowInterfaces",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:SideEffectInterfaces",
        "@llvm-project//mlir:SideEffects",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:Transforms",
    ],
)

gentbl_cc_library(
    name = "NumoptAttrsGen",
    tbl_outs = [
        (
            ["-gen-attrdef-decls"],
            "NumoptAttrs.h.inc",
        ),
        (
            ["-gen-attrdef-defs"],
            "NumoptAttrs.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "NumoptAttrs.td",
    deps = [":td_files"],
)

gentbl_cc_library(
    name = "NumoptInterfacesGen",
    tbl_outs = [
        (
            ["-gen-op-interface-decls"],
            "NumoptOpInterfaces.h.inc",
        ),
        (
            ["-gen-op-interface-defs"],
            "NumoptOpInterfaces.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "NumoptInterfaces.td",
    deps = [":td_files"],
)

gentbl_cc_library(
    name = "NumoptOpsGen",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "NumoptOps.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "NumoptOps.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "NumoptOps.td",
    deps = [":td_files"],
)

iree_tablegen_doc(
    name = "NumoptDialectDocGen",
    tbl_outs = [
        (
            ["-gen-dialect-doc"],
            "NumoptDialect.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "NumoptOps.td",
    deps = [":td_files"],
)
