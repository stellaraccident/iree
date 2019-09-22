# Copyright 2019 Google LLC
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

include(CMakeParseArguments)

# iree_cc_binary()
#
# CMake function to imitate Bazel's cc_binary rule.
#
# Parameters:
# NAME: name of target (see Usage below)
# SRCS: List of source files for the binary
# DEPS: List of other libraries to be linked in to the binary targets
# COPTS: List of private compile options
# DEFINES: List of public defines
# LINKOPTS: List of link options
#
# Note:
# By default, iree_cc_binary will always create a binary named iree_${NAME}.
#
# Usage:
# iree_cc_library(
#   NAME
#     awesome
#   HDRS
#     "a.h"
#   SRCS
#     "a.cc"
#   PUBLIC
# )
#
# iree_cc_binary(
#   NAME
#     awesome_tool
#   OUT
#     awesome-tool
#   SRCS
#     "awesome_tool_main.cc"
#   DEPS
#     iree::awesome
# )
function(iree_cc_binary)
  cmake_parse_arguments(
    _RULE
    ""
    "NAME;OUT"
    "SRCS;COPTS;DEFINES;LINKOPTS;DEPS"
    ${ARGN}
  )

  # Prefix the library with the package name, so we get: iree_package_name
  iree_package_name(_PACKAGE_NAME)
  set(_NAME "${_PACKAGE_NAME}_${_RULE_NAME}")

  add_executable(${_NAME} "")
  if(_RULE_SRCS)
    target_sources(${_NAME}
      PRIVATE
        ${_RULE_SRCS}
    )
  else()
    set(_DUMMY_SRC "${CMAKE_CURRENT_BINARY_DIR}/${_NAME}_dummy.cc")
    file(WRITE ${_DUMMY_SRC} "")
    target_sources(${_NAME}
      PRIVATE
        ${_DUMMY_SRC}
    )
  endif()
  if(_RULE_OUT)
    set_target_properties(${_NAME} PROPERTIES OUTPUT_NAME "${_RULE_OUT}")
  endif()
  target_include_directories(${_NAME}
    PUBLIC
      ${IREE_COMMON_INCLUDE_DIRS}
    PRIVATE
      ${GTEST_INCLUDE_DIRS}
  )
  target_compile_definitions(${_NAME}
    PUBLIC
      ${_RULE_DEFINES}
  )
  target_compile_options(${_NAME}
    PRIVATE
      ${_RULE_COPTS}
  )
  target_link_libraries(${_NAME}
    PUBLIC
      ${_RULE_DEPS}
    PRIVATE
      ${_RULE_LINKOPTS}
  )

  # Add all IREE targets to a a folder in the IDE for organization.
  set_property(TARGET ${_NAME} PROPERTY FOLDER ${IREE_IDE_FOLDER}/binaries)

  set_property(TARGET ${_NAME} PROPERTY CXX_STANDARD ${IREE_CXX_STANDARD})
  set_property(TARGET ${_NAME} PROPERTY CXX_STANDARD_REQUIRED ON)

  # DO NOT SUBMIT
  set(_LINK_DEPS)
  get_property(_L GLOBAL PROPERTY IREE_CC_LIBRARY_ALWAYSLINK_LIST)
  message("alwayslink list ${_L}")
  foreach(_TARGET IN LISTS _RULE_DEPS)
    message("DEP ${_TARGET}")
    get_target_property(_ALIASED_TARGET ${_TARGET} ALIASED_TARGET)
    if(_ALIASED_TARGET)
      set(_TARGET_NAME ${_ALIASED_TARGET})
    else()
      string(REPLACE "::" "_" _TARGET_NAME ${_TARGET})
    endif()
    get_target_property(_TARGET_TYPE ${_TARGET_NAME} TYPE)
    # if(NOT "${_TARGET_TYPE}" STREQUAL "INTERFACE_LIBRARY")
    # endif()
    message("DEP ${_TARGET} == ${_TARGET_NAME} of type ${_TARGET_TYPE}")
    #get_target_property(_TARGET_DEPS ${_TARGET} IREE_CC_LIBRARY_TRANSITIVE_LINK_DEPS)
    #list(APPEND _LINK_DEPS "$<TARGET_PROPERTY:${_TARGET},IREE_CC_LIBRARY_ALWAYSLINK>")
    #list(APPEND _LINK_DEPS ${_LINK_FLAGS} "/WHOLEARCHIVE:${_TARGET} ")
    set(_prop "$<TARGET_PROPERTY:${_TARGET},IREE_CC_LIBRARY_ALWAYSLINK>")
    list(APPEND _LINK_DEPS ${_LINK_FLAGS} "$<$<BOOL:${_prop}>:/WHOLEARCHIVE:$<JOIN:$<TARGET_LINKER_FILE:${_TARGET_NAME}>, /WHOLEARCHIVE:>>")
    #list(APPEND _LINK_DEPS ${_TARGET_DEPS})
  endforeach()
  message("WHOLE ARCHIVE DEPS FOR ${_NAME}: ${_LINK_DEPS}")
  target_link_options(${_NAME} PUBLIC "$<TARGET_GENEX_EVAL:${_NAME},${_LINK_DEPS}>")
  add_custom_target(genexdebug COMMAND ${CMAKE_COMMAND} -E echo "$<TARGET_GENEX_EVAL:${_NAME},${_LINK_DEPS}>")

endfunction()
