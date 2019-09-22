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

# DO NOT SUBMIT
define_property(
  TARGET
  PROPERTY
    IREE_CC_LIBRARY_ALWAYSLINK
  BRIEF_DOCS
    "List of all libraries (and their aliases) with alwayslink=1."
  FULL_DOCS
    "Each iree_cc_library with ALWAYSLINK set will be added to this list along"
    "with any aliases that target it. This list can be used as a lookup whether"
    "the library needs to be linked into binaries."
)

# iree_cc_library()
#
# CMake function to imitate Bazel's cc_library rule.
#
# Parameters:
# NAME: name of target (see Note)
# HDRS: List of public header files for the library
# SRCS: List of source files for the library
# DEPS: List of other libraries to be linked in to the binary targets
# COPTS: List of private compile options
# DEFINES: List of public defines
# LINKOPTS: List of link options
# ALWAYSLINK: Always link the library into any binary with a transitive dep.
# PUBLIC: Add this so that this library will be exported under iree::
# Also in IDE, target will appear in IREE folder while non PUBLIC will be in IREE/internal.
# TESTONLY: When added, this target will only be built if user passes -DIREE_BUILD_TESTS=ON to CMake.
#
# Note:
# By default, iree_cc_library will always create a library named iree_${NAME},
# and alias target iree::${NAME}. The iree:: form should always be used.
# This is to reduce namespace pollution.
#
# iree_cc_library(
#   NAME
#     awesome
#   HDRS
#     "a.h"
#   SRCS
#     "a.cc"
# )
# iree_cc_library(
#   NAME
#     fantastic_lib
#   SRCS
#     "b.cc"
#   DEPS
#     iree::package::awesome # not "awesome" !
#   PUBLIC
# )
#
# iree_cc_library(
#   NAME
#     main_lib
#   ...
#   DEPS
#     iree::package::fantastic_lib
# )
function(iree_cc_library)
  cmake_parse_arguments(
    _RULE
    "PUBLIC;ALWAYSLINK;TESTONLY"
    "NAME"
    "HDRS;SRCS;COPTS;DEFINES;LINKOPTS;DEPS"
    ${ARGN}
  )

  if(NOT _RULE_TESTONLY OR IREE_BUILD_TESTS)
    # Prefix the library with the package name, so we get: iree_package_name.
    iree_package_name(_PACKAGE_NAME)
    set(_NAME "${_PACKAGE_NAME}_${_RULE_NAME}")

    # Check if this is a header-only library.
    # Note that as of February 2019, many popular OS's (for example, Ubuntu
    # 16.04 LTS) only come with cmake 3.5 by default.  For this reason, we can't
    # use list(FILTER...)
    set(_CC_SRCS "${_RULE_SRCS}")
    foreach(src_file IN LISTS _CC_SRCS)
      if(${src_file} MATCHES ".*\\.(h|inc)")
        list(REMOVE_ITEM _CC_SRCS "${src_file}")
      endif()
    endforeach()
    if("${_CC_SRCS}" STREQUAL "")
      set(_RULE_IS_INTERFACE 1)
    else()
      set(_RULE_IS_INTERFACE 0)
    endif()

    if(NOT _RULE_IS_INTERFACE)
      add_library(${_NAME} STATIC "")
      target_sources(${_NAME}
        PRIVATE
          ${_RULE_SRCS}
          ${_RULE_HDRS}
      )
      target_include_directories(${_NAME}
        PUBLIC
          "$<BUILD_INTERFACE:${IREE_COMMON_INCLUDE_DIRS}>"
      )
      target_compile_options(${_NAME}
        PRIVATE
          ${_RULE_COPTS}
          ${IREE_DEFAULT_COPTS}
      )
      target_link_libraries(${_NAME}
        PUBLIC
          ${_RULE_DEPS}
        PRIVATE
          ${_RULE_LINKOPTS}
          ${IREE_DEFAULT_LINKOPTS}
      )
      target_compile_definitions(${_NAME}
        PUBLIC
          ${_RULE_DEFINES}
      )

      # Add all IREE targets to a a folder in the IDE for organization.
      if(_RULE_PUBLIC)
        set_property(TARGET ${_NAME} PROPERTY FOLDER ${IREE_IDE_FOLDER})
      elseif(_RULE_TESTONLY)
        set_property(TARGET ${_NAME} PROPERTY FOLDER ${IREE_IDE_FOLDER}/test)
      else()
        set_property(TARGET ${_NAME} PROPERTY FOLDER ${IREE_IDE_FOLDER}/internal)
      endif()

      # INTERFACE libraries can't have the CXX_STANDARD property set.
      set_property(TARGET ${_NAME} PROPERTY CXX_STANDARD ${IREE_CXX_STANDARD})
      set_property(TARGET ${_NAME} PROPERTY CXX_STANDARD_REQUIRED ON)

      # Setup the transitive link dependencies for alwayslink libraries.
      # set(_LINK_DEPS)
      # if(${_RULE_ALWAYSLINK})
      #   list(APPEND _LINK_DEPS ${_NAME})
      # endif()
      # foreach(_TARGET IN LISTS _RULE_DEPS)
      #   get_target_property(_ALIASED_TARGET ${_TARGET} ALIASED_TARGET)
      #   if(_ALIASED_TARGET)
      #     set(_TARGET_NAME ${_ALIASED_TARGET})
      #   else()
      #     string(REPLACE "::" "_" _TARGET_NAME ${_TARGET})
      #   endif()
      #   get_target_property(_TARGET_TYPE ${_TARGET_NAME} TYPE)
      #   message(${_TARGET_TYPE})
      #   if(NOT "${_TARGET_TYPE}" STREQUAL "INTERFACE_LIBRARY")
      #     #get_target_property(_TARGET_DEPS ${_TARGET_NAME} IREE_CC_LIBRARY_TRANSITIVE_LINK_DEPS)
      #     #if(NOT "${_TARGET_DEPS}" STREQUAL "_TARGET_DEPS-NOTFOUND")
      #       list(APPEND _LINK_DEPS "$<TARGET_PROPERTY:${_TARGET_NAME},IREE_CC_LIBRARY_TRANSITIVE_LINK_DEPS>")
      #       #list(APPEND _LINK_DEPS ${_TARGET_DEPS})
      #     #endif()
      #   endif()
      # endforeach()
      # set_property(
      #   TARGET ${_NAME}
      #   PROPERTY IREE_CC_LIBRARY_TRANSITIVE_LINK_DEPS
      #   ${_LINK_DEPS}
      # )
      # message("link deps for ${_NAME} = ${_LINK_DEPS}")
      if(${_RULE_ALWAYSLINK})
        message("SET ALWAYSLINK ON ${_NAME}")
        set_target_properties(
          ${_NAME}
          PROPERTIES
          IREE_CC_LIBRARY_ALWAYSLINK ON
        )
      else()
        message("SET ALWAYSLINK OFF ${_NAME}")
        set_target_properties(
          ${_NAME}
          PROPERTIES
          IREE_CC_LIBRARY_ALWAYSLINK OFF
        )
      endif()
    else()
      # Generating header-only library.
      add_library(${_NAME} INTERFACE)
      target_include_directories(${_NAME}
        INTERFACE
          "$<BUILD_INTERFACE:${IREE_COMMON_INCLUDE_DIRS}>"
      )
      target_compile_options(${_NAME}
        INTERFACE
          ${_RULE_COPTS}
          ${IREE_DEFAULT_COPTS}
      )
      target_link_libraries(${_NAME}
        INTERFACE
          ${_RULE_DEPS}
          ${_RULE_LINKOPTS}
          ${IREE_DEFAULT_LINKOPTS}
      )
      target_compile_definitions(${_NAME}
        INTERFACE
          ${_RULE_DEFINES}
      )
    endif()

    # Alias the iree_package_name library to iree::package::name.
    # This lets us more clearly map to Bazel and makes it possible to
    # disambiguate the underscores in paths vs. the separators.
    iree_package_ns(_PACKAGE_NS)
    add_library(${_PACKAGE_NS}::${_RULE_NAME} ALIAS ${_NAME})
    if(${_RULE_ALWAYSLINK})
      set_property(GLOBAL APPEND PROPERTY IREE_CC_LIBRARY_ALWAYSLINK_LIST "${_NAME}")
      set_property(GLOBAL APPEND PROPERTY IREE_CC_LIBRARY_ALWAYSLINK_LIST "${_PACKAGE_NS}::${_RULE_NAME}")
    endif()
    iree_package_dir(_PACKAGE_DIR)
    if(${_RULE_NAME} STREQUAL ${_PACKAGE_DIR})
      # If the library name matches the package then treat it as a default.
      # For example, foo/bar/ library 'bar' would end up as 'foo::bar'.
      add_library(${_PACKAGE_NS} ALIAS ${_NAME})
      if(${_RULE_ALWAYSLINK})
        set_property(GLOBAL APPEND PROPERTY IREE_CC_LIBRARY_ALWAYSLINK_LIST "${_PACKAGE_NS}")
      endif()
    endif()
  endif()
endfunction()
