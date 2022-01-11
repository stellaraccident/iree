# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import sys

import lit.formats
import lit.util

import lit.llvm

# Configuration file for the 'lit' test runner.
lit.llvm.initialize(lit_config, config)
from lit.llvm import llvm_config

llvm_config.with_system_environment("PYTHONPATH")
llvm_config.with_system_environment("VK_ICD_FILENAMES")

# name: The name of this test suite.
config.name = 'TENSORFLOW_TESTS'

config.test_format = lit.formats.ShTest()

# suffixes: A list of file extensions to treat as test files.
config.suffixes = ['.py', '.run']

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

#config.use_default_substitutions()
config.excludes = [
    'lit.cfg.py',
    'lit.site.cfg.py',
    'test_util.py',
    'manual_test.py',
]

config.substitutions.extend([
    ('%PYTHON', sys.executable),
])

# Add our local projects to the PYTHONPATH
python_projects_dir = os.path.join(os.path.dirname(__file__), "..",
                                   "python_projects")
test_src_dir = os.path.join(os.path.dirname(__file__), "python")
llvm_config.with_environment("PYTHONPATH", [
    test_src_dir,
    os.path.join(python_projects_dir, "iree_tf"),
    os.path.join(python_projects_dir, "iree_tflite"),
    os.path.join(python_projects_dir, "iree_xla"),
],
                                      append_path=True)

# Enable features based on -D FEATURES=hugetest,vulkan
# syntax.
# We always allow 'llvmaot'. It can be disabled with -D DISABLE_FEATURES=llvmaot
disable_features_param = lit_config.params.get('DISABLE_FEATURES')
disable_features = []
if disable_features_param:
    disable_features = disable_features_param.split(',')
if 'llvmaot' not in disable_features:
    config.available_features.add('llvmaot')
features_param = lit_config.params.get('FEATURES')
if features_param:
  config.available_features.update(features_param.split(','))
