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

set(IREE_BAZEL_EXECUTABLE "bazel"
    CACHE STRING "Bazel executable to use for bazel builds")

# iree_configure_bazel
#
# Configures the CMake binary directory to also contain a bazel build root.
# The following files will be created:
#   bazel (shell script): Shell wrapper to invoke bazel
#   bazel.bat: Windows batch file to invoke bazel
#   bazelrc: The bazelrc to use for the build
#   bazel-out/: Bazel output directory
#   bazel-bin/: Symlink to the bin directory appropriate for the build mode
#
# Variables will be set in the parent scope:
#   IREE_BAZEL_WRAPPER: Executable wrapper to invoke to run bazel
#   IREE_BAZEL_BIN: Path to the bazel-bin directory
function(iree_configure_bazel)
  set(_bazel_output_base "${CMAKE_BINARY_DIR}/bazel-out")
  set(_bazel_src_root "${CMAKE_SOURCE_DIR}")

  # Use the utility to emit _bazelrc_file configuration options.
  set(_bazelrc_file "${CMAKE_BINARY_DIR}/bazelrc")
  execute_process(
    RESULT_VARIABLE RC
    COMMAND
      "${Python3_EXECUTABLE}"
      "${_bazel_src_root}/configure_bazel.py"
      "${_bazelrc_file}"
  )
  if(NOT RC EQUAL 0)
    message(FATAL_ERROR "Error running ${_bazel_src_root}/configure_bazel.py script")
  endif()

  # Now add an import to the configured.bazelrc to load the project-wide
  # bazelrc file.
  file(APPEND "${_bazelrc_file}" "
import ${_bazel_src_root}/build_tools/bazel/iree.bazelrc
")

  # Note that we do allow a .bazelrc in the home directory (otherwise we
  # would have --nohome_rc). This is mainly about disabling interference from
  # interactive builds in the workspace.
  set(_bazel_startup_options "--nosystem_rc --noworkspace_rc '--bazelrc=${_bazelrc_file}' '--output_base=${_bazel_output_base}'")

  # And emit scripts to delegate to bazel.
  set(IREE_BAZEL_WRAPPER "${CMAKE_BINARY_DIR}/bazel")
  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/build_tools/cmake/bazel.sh.in"
    "${IREE_BAZEL_WRAPPER}"
  )
  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/build_tools/cmake/bazel.bat.in"
    "${IREE_BAZEL_WRAPPER}.bat"
  )
  if(NOT WIN32)
    execute_process(
      COMMAND chmod a+x "${IREE_BAZEL_WRAPPER}"
    )
  endif()

  # Now ready to start bazel and ask it things.
  message(STATUS "Detecting bazel version...")
  execute_process(
    RESULT_VARIABLE RC
    OUTPUT_VARIABLE BAZEL_RELEASE
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND
      "${IREE_BAZEL_WRAPPER}" info release
  )
  if(NOT RC EQUAL 0)
    message(FATAL_ERROR "Failed to launch bazel using wrapper ${IREE_BAZEL_WRAPPER}. Inspect that script and ensure bazel is installed properly.")
  endif()
  execute_process(
    RESULT_VARIABLE RC
    OUTPUT_VARIABLE IREE_BAZEL_BIN
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND
      "${IREE_BAZEL_WRAPPER}" info bazel-bin
  )
  if(NOT RC EQUAL 0)
    message(FATAL_ERROR "Failed to run 'info bazel-bin' via ${IREE_BAZEL_WRAPPER}. Inspect that script and ensure bazel is installed properly.")
  endif()
  message(STATUS "Found bazel ${BAZEL_RELEASE}, bin directory: ${IREE_BAZEL_BIN}")
  message(STATUS "Bazel wrapper script generated at: ${IREE_BAZEL_WRAPPER}")

  # Build automation will use the IREE_BAZEL_BIN variable, but also drop a
  # convenience symlink, since that is what people expect.
  # And since bazel isn't nice enough to create it...
  if(NOT WIN32)
    execute_process(
      RESULT_VARIABLE RC
      COMMAND
        ln -sf "${IREE_BAZEL_BIN}" "${CMAKE_CURRENT_BINARY_DIR}/bazel-bin"
    )
    if(NOT RC EQUAL 0)
      message(WARNING "Failed to create convenience bazel-bin symlink")
    endif()
  endif()

  set(IREE_BAZEL_WRAPPER "${IREE_BAZEL_WRAPPER}" PARENT_SCOPE)
  set(IREE_BAZEL_BIN "${IREE_BAZEL_BIN}" PARENT_SCOPE)
endfunction()

# iree_add_bazel_invocation
#
# Adds a target to perform a bazel invocation, building a list of targets
# and exporting pseudo targets for some results of the build.
#
# Parameters:
#   INVOCATION_TARGET: The target name for the custom invocation target.
#   BAZEL_TARGETS: List of bazel targets to build.
#   EXECUTABLE_PATHS: Paths under bazel-bin for executables. An equivalent
#     CMake imported executable target will be created for each by replacing
#     the "/" with "_".
function(iree_add_bazel_invocation)
  cmake_parse_arguments(ARG
    ""
    "INVOCATION_TARGET"
    "BAZEL_TARGETS;EXECUTABLE_PATHS"
    ${ARGN}
  )

  add_custom_target(${ARG_INVOCATION_TARGET}
    USES_TERMINAL
    COMMAND ${CMAKE_COMMAND} -E echo
        "Starting bazel build of targets ${ARG_BAZEL_TARGETS}"
    COMMAND "${IREE_BAZEL_WRAPPER}" build ${ARG_BAZEL_TARGETS}
    COMMAND ${CMAKE_COMMAND} -E echo "Bazel build complete."
  )

  # Create an imported executable target for each binary path.
  # Since the bazel directory namespace lines up with the cmake namespace,
  # generate a cmake target name for each.
  foreach(_executable_path ${ARG_EXECUTABLE_PATHS})
    string(REPLACE "/" "_" _executable_target "${_executable_path}")
    message(STATUS "Add bazel executable target ${_executable_target}")
    add_executable(${_executable_target} IMPORTED GLOBAL)
    set_target_properties(${_executable_target}
        PROPERTIES IMPORTED_LOCATION
            "${IREE_BAZEL_BIN}/${_executable_path}${CMAKE_EXECUTABLE_SUFFIX}"
    )
    add_dependencies(${_executable_target} ${ARG_INVOCATION_TARGET})
  endforeach()
endfunction()
