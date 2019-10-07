// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef IREE_RT_API_H_
#define IREE_RT_API_H_

#include <stdint.h>

struct iree_rt_instance_t;
struct iree_rt_context_t;
struct iree_rt_module_t;
struct iree_rt_function_t;
struct iree_rt_invocation_t;
struct iree_rt_resource_state_t;

//===----------------------------------------------------------------------===//
// Types and Enums
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// iree::rt::Instance
//===----------------------------------------------------------------------===//

// Creates a new instance. This should be shared with all contexts in an
// application to ensure that resources are tracked properly and threads are
// managed correctly.
iree_status_t iree_rt_instance_create(iree_allocator_t allocator,
                                      iree_rt_instance_t** out_instance);

// Retains the given |instance| for the caller.
iree_status_t iree_rt_instance_retain(iree_rt_instance_t* instance);

// Release the given |instance| from the caller.
iree_status_t iree_rt_instance_release(iree_rt_instance_t* instance);

//===----------------------------------------------------------------------===//
// iree::rt::Context
//===----------------------------------------------------------------------===//

// Creates a new context that uses the given |instance| for device management.
iree_status_t iree_rt_context_create(iree_rt_instance_t* instance,
                                     iree_allocator_t allocator,
                                     iree_rt_context_t** out_context);

// Retains the given |context| for the caller.
iree_status_t iree_rt_context_retain(iree_rt_context_t* context);

// Release the given |context| from the caller.
iree_status_t iree_rt_context_release(iree_rt_context_t* context);

// Returns a process-unique ID for the |context|.
int32_t iree_rt_context_id(const iree_rt_context_t* context);

// Sets |out_module| to a module with the given |module_name| or returns
// IREE_STATUS_NOT_FOUND.
iree_status_t iree_rt_context_lookup_module_by_name(
    const iree_rt_context_t* context, iree_string_view_t module_name,
    iree_rt_module_t** out_module);

// Sets |out_function| to a function with the given
// |module_name|::|function_name| or returns IREE_STATUS_NOT_FOUND.
iree_status_t iree_rt_context_lookup_function_by_name(
    const iree_rt_context_t* context, iree_string_view_t module_name,
    iree_string_view_t function_name, iree_rt_function_t** out_function);

// DO NOT SUBMIT variants and invocation
iree_status_t iree_rt_context_invoke(iree_rt_context_t* context);
iree_status_t iree_rt_context_begin_invoke(iree_rt_context_t* context);

//===----------------------------------------------------------------------===//
// iree::rt::Module
//===----------------------------------------------------------------------===//

// Defines an external module that can be used to reflect and execute functions.
// Modules must be thread-safe as lookups and executions may occur in any order
// from any thread.
typedef struct {
  // User-defined pointer passed to all functions.
  void* self;
  // Destroys |self| when all references to the module have been released.
  iree_status_t (*destroy)(void* self);
  // Returns the name of the module (used during resolution).
  iree_string_view_t (*name)(void* self);
  // Sets |out_function| a resolved function by exported ordinal, if found.
  iree_status_t (*lookup_function_by_ordinal)(
      void* self, int32_t function_ordinal, iree_rt_function_t** out_function);
  // Sets |out_function| a resolved function by exported name, if found.
  iree_status_t (*lookup_function_by_name)(void* self,
                                           iree_string_view_t function_name,
                                           iree_rt_function_t** out_function);
  // Executes the |function| with the given |invocation| context.
  iree_status_t (*execute)(void* self, iree_rt_function_t* function,
                           iree_rt_invocation_t* invocation);
} iree_rt_extern_module_t;

// Creates a module with an external backing store.
// The provided |extern_module| definition will be used to query the module
// state as needed. No caching occurs within the implementation to allow calls
// to return different values per-invocation.
//
// iree_rt_extern_module_t::destroy is called when the last reference to the
// iree_rt_module_t is released.
iree_status_t iree_rt_module_create_extern(
    iree_rt_extern_module_t extern_module, iree_allocator_t allocator,
    iree_rt_module_t** out_module);

// TODO(benvanik): iree_rt_module_create_native() for codegen.

// Creates a module from a mapped ModuleDef FlatBuffer.
// The provided |file_mapping| will be retained for the life of the module and
// the contents will be accessed by reference.
// TODO(benvanik): disable function in impl if VM is not compiled in.
iree_status_t iree_rt_module_create_from_flatbuffer(
    iree_file_mapping_t* file_mapping, iree_allocator_t allocator,
    iree_rt_module_t** out_module);

// Retains the given |module| for the caller.
iree_status_t iree_rt_module_retain(iree_rt_module_t* module);

// Release the given |module| from the caller.
iree_status_t iree_rt_module_release(iree_rt_module_t* module);

// Returns the name of the module.
iree_string_view_t iree_rt_module_name(const iree_rt_module_t* module);

// Sets |out_function| to to an exported function with |function_ordinal| or
// returns IREE_STATUS_NOT_FOUND.
iree_status_t iree_rt_module_lookup_function_by_ordinal(
    const iree_rt_module_t* module, int32_t function_ordinal,
    iree_rt_function_t** out_function);

// Sets |out_function| to an exported function with |function_name| or returns
// IREE_STATUS_NOT_FOUND.
iree_status_t iree_rt_module_lookup_function_by_name(
    const iree_rt_module_t* module, iree_string_view_t function_name,
    iree_rt_function_t** out_function);

//===----------------------------------------------------------------------===//
// iree::rt::Function
//===----------------------------------------------------------------------===//

// TODO(benvanik): create extern function.

// Returns the module the function is a member of.
iree_rt_module_t* iree_rt_function_module(const iree_rt_function_t* function);

// Returns the name of the function as exported from the module.
iree_string_view_t iree_rt_function_name(const iree_rt_function_t* function);

//===----------------------------------------------------------------------===//
// iree::rt::Invocation
//===----------------------------------------------------------------------===//

// Retains the given |invocation| for the caller.
iree_status_t iree_rt_invocation_retain(iree_rt_invocation_t* invocation);

// Release the given |invocation| from the caller.
iree_status_t iree_rt_invocation_state_release(
    iree_rt_invocation_t* invocation);

// DO NOT SUBMIT
iree_status_t iree_rt_invocation_wait(iree_rt_invocation_t* invocation);

// DO NOT SUBMIT
iree_status_t iree_rt_invocation_abort(iree_rt_invocation_t* invocation);

//===----------------------------------------------------------------------===//
// iree::rt::ResourceState
//===----------------------------------------------------------------------===//

// Allocates the |resource_state_ordinal| from the given |module|.
iree_status_t iree_rt_resource_state_create(
    iree_rt_module_t* module, int32_t resource_state_ordinal,
    iree_allocator_t allocator, iree_rt_resource_state_t** out_resource_state);

// Retains the given |resource_state| for the caller.
iree_status_t iree_rt_resource_state_retain(
    iree_rt_resource_state_t* resource_state);

// Release the given |resource_state| from the caller.
iree_status_t iree_rt_resource_state_release(
    iree_rt_resource_state_t* resource_state);

iree_status_t iree_rt_resource_state_wait_idle(
    iree_rt_resource_state_t* resource_state);

iree_status_t iree_rt_resource_state_multi_map(
    iree_rt_resource_state_t* resource_state);

iree_status_t iree_rt_resource_state_read(
    iree_rt_resource_state_t* resource_state);

iree_status_t iree_rt_resource_state_write(
    iree_rt_resource_state_t* resource_state);

iree_status_t iree_rt_resource_state_fill(
    iree_rt_resource_state_t* resource_state);

iree_status_t iree_rt_resource_state_assign(
    iree_rt_resource_state_t* resource_state);

iree_status_t iree_rt_resource_state_copy(
    iree_rt_resource_state_t* resource_state);

iree_status_t iree_rt_resource_state_copy_all(
    iree_rt_resource_state_t* resource_state);

iree_status_t iree_rt_resource_state_reset(
    iree_rt_resource_state_t* resource_state);

iree_status_t iree_rt_resource_state_reset_all(
    iree_rt_resource_state_t* resource_state);

#endif  // IREE_RT_API_H_
