# Lint-as: python3
"""TensorFlow compiler interface."""

# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from enum import Enum
import logging
import tempfile
from typing import List, Optional, Sequence, Set, Union

from .tools import find_tool, invoke_immediate, invoke_pipeline
from .core import CompilerOptions, DEFAULT_TESTING_BACKENDS, build_compile_command_line

__all__ = [
    "compile_saved_model",
    "compile_module",
    "is_available",
    "DEFAULT_TESTING_BACKENDS",
    "ImportOptions",
    "ImportType",
]

_TF_IMPORT_TOOL = "iree-tf-import"


def is_available():
  """Determine if TensorFlow and the compiler are available."""
  try:
    import tensorflow as tf
  except ModuleNotFoundError:
    logging.warn("Unable to import tensorflow")
    return False
  try:
    find_tool(_TF_IMPORT_TOOL)
  except ValueError:
    logging.warning("Unable to find IREE tool %s", _TF_IMPORT_TOOL)
    return False
  return True


class ImportType(Enum):
  """Import type of the model."""
  OBJECT_GRAPH = "savedmodel_v2"
  V2 = "savedmodel_v2"
  SIGNATURE_DEF = "savedmodel_v1"
  V1 = "savedmodel_v1"

  @staticmethod
  def parse(spec: Union[str, "ImportType"]) -> "ImportType":
    """Parses or returns an ImportType.

    Args:
      spec: An ImportType instance or the case-insensitive name of one of
        the enum values.
    Returns:
      An ImportType instance.
    """
    if isinstance(spec, ImportType):
      return spec
    spec = spec.upper()
    if spec not in ImportType.__members__:
      raise ValueError(f"For import_type= argument, expected one of: "
                       f"{', '.join(ImportType.__members__.keys())}")
    return ImportType[spec]


class ImportOptions(CompilerOptions):
  """Import options layer on top of the backend compiler options."""

  def __init__(self,
               exported_names: Sequence[str] = (),
               import_only: bool = False,
               import_type: Union[ImportType, str] = ImportType.OBJECT_GRAPH,
               saved_model_tags: Set[str] = set(),
               import_extra_args: Sequence[str] = (),
               **kwargs):
    """Initialize options from keywords.

    Args:
      exported_names: Optional sequence representing the exported names to
        keep (object graph/v2 models only).
      import_only: Only import the module. If True, the result will be textual
        MLIR that can be further fed to the IREE compiler. If False (default),
        the result will be the fully compiled IREE binary. In both cases,
        bytes-like output is returned. Note that if the output_file= is
        specified and import_only=True, then the MLIR form will be written to
        the output file.
      import_type: Type of import to perform. See ImportType enum.
      saved_model_tags: Set of tags to export (signature def/v1 saved models
        only).
      import_extra_args: Extra arguments to pass to the iree-tf-import tool.
    """
    super().__init__(**kwargs)
    self.exported_names = exported_names
    self.import_only = import_only
    self.import_type = ImportType.parse(import_type)
    self.saved_model_tags = saved_model_tags
    self.import_extra_args = import_extra_args


def build_import_command_line(input_path: str,
                              options: ImportOptions) -> List[str]:
  """Builds a command line for invoking the import stage.

  Args:
    input_path: The input path.
    options: Import options.
  Returns:
    List of strings of command line.
  """
  tf_import = find_tool(_TF_IMPORT_TOOL)
  cl = [
      tf_import,
      input_path,
      f"--tf-import-type={options.import_type.value}",
      f"--tf-savedmodel-exported-names={','.join(options.exported_names)}",
      f"--tf-savedmodel-tags={','.join(options.saved_model_tags)}",
  ]
  if options.import_only and options.output_file:
    # Import stage directly outputs.
    if options.output_file:
      cl.append(f"-o={options.output_file}")
  cl.extend(options.import_extra_args)
  return cl


def compile_saved_model(saved_model_dir: str, **kwargs):
  """Compiles an on-disk saved model to an IREE binary.

  Args:
    saved_model_dir: Path to directory where the model was saved.
    **kwargs: Keyword args corresponding to ImportOptions or CompilerOptions.
  Returns:
    A bytes-like object with the compiled output or None if output_file=
    was specified.
  """
  options = ImportOptions(**kwargs)
  import_cl = build_import_command_line(saved_model_dir, options)
  if options.import_only:
    # One stage tool pipeline.
    result = invoke_immediate(import_cl)
    if options.output_file:
      return None
    return result

  # Full compilation pipeline.
  compile_cl = build_compile_command_line("-", options)
  result = invoke_pipeline([import_cl, compile_cl])
  if options.output_file:
    return None
  return result


def compile_module(module, saved_model_dir: Optional[str] = None, **kwargs):
  """Compiles a tf.Module to an IREE binary (by saving to disk).

  Args:
    module: The tf.Module instance to convert to MLIR
    saved_model_dir: Optional path to save the tf.Module to. The module will not
      be persisted on disk outside of this call if this is not provided.
    **kwargs: Keyword args corresponding to ImportOptions or CompilerOptions.
  Returns:
    Same as compile_saved_model().
  """

  def do_it(saved_model_dir):
    import tensorflow as tf
    options = tf.saved_model.SaveOptions(save_debug_info=True)
    tf.saved_model.save(module, saved_model_dir, options=options)
    return compile_saved_model(saved_model_dir, **kwargs)

  if saved_model_dir:
    return do_it(saved_model_dir)
  else:
    with tempfile.TemporaryDirectory(suffix=".sm") as td:
      return do_it(td)
