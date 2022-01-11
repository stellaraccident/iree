# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Generates a ".run" file in the corresponding test tree for a file under
# this directory. This is a convenience for bootstrapping new test files.
# Usage:
#   python generate_runner.py llvmaot "--target_backends=iree_llvmaot" \
#     iree_tf_tests/uncategorized/batch_norm_test.py
#
# The first argument is the lit feature that this test is gated on and will
# be prepended to the test name as "{variant}__testfile".
# The second argument is the flag string to include when running the test.
# All remaining arguments are relative paths to python files under this
# directory that should have a .run file created for them.

import os
import sys


def main(args):
  variant = args[0]
  flags = args[1]
  src_files = args[2:]
  module_names = [transform_src_file_to_module(f) for f in src_files]
  run_files = [transform_src_file_to_run_file(f, variant) for f in src_files]
  for src_file, module, run_file in zip(src_files, module_names, run_files):
    if os.path.exists(run_file):
      print(f"SKIPPING (exists): {run_file}")
    print(f"CREATE RUN FILE: {module} -> {run_file}")
    os.makedirs(os.path.dirname(run_file), exist_ok=True)
    with open(run_file, "wt") as f:
      print(f"# REQUIRES: {variant}", file=f)
      print(f"# RUN: %PYTHON -m {module} {flags}", file=f)


def transform_src_file_to_module(file_name):
  module_name = file_name.replace("/", ".")
  if (module_name.endswith(".py")):
    module_name = module_name[0:-3]
  return module_name


def transform_src_file_to_run_file(file_path, variant):
  main_test_dir = os.path.join(os.path.dirname(__file__), "..")
  file_name = os.path.basename(file_path)
  parent_path = os.path.dirname(file_path)
  if file_name.endswith(".py"):
    file_name = file_name[0:-3]
  if file_name.endswith("_test"):
    file_name = file_name[0:-5]

  file_name = f"{variant}__{file_name}.run"
  run_file = os.path.join(main_test_dir, parent_path, file_name)
  return run_file


if __name__ == "__main__":
  main(sys.argv[1:])
