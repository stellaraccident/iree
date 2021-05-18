// Copyright 2021 Google LLC
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

#ifndef IREE_HAL_ROCM_ARRAY_COMMAND_BUFFER_H_
#define IREE_HAL_ROCM_ARRAY_COMMAND_BUFFER_H_

#include "experimental/rocm/context_wrapper.h"
#include "experimental/rocm/dynamic_symbols.h"
#include "experimental/rocm/rocm_headers.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// ROCM Kernel Information Structure
typedef struct {
  hipFunction_t func;
  unsigned int gridDimX;
  unsigned int gridDimY;
  unsigned int gridDimZ;
  unsigned int blockDimX;
  unsigned int blockDimY;
  unsigned int blockDimZ;
  void **kernelParams;
} hip_launch_params;

typedef struct {
  size_t num_of_kernels;
  size_t total_size;
  hip_launch_params kernels[];
} kernelArrayType;

// Creates a rocm array.
iree_status_t iree_hal_rocm_array_command_buffer_allocate(
    iree_hal_rocm_context_wrapper_t *context,
    iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity,
    iree_hal_command_buffer_t **out_command_buffer);

// Returns the array of kernel associated to the command buffer.
kernelArrayType *iree_hal_rocm_array_command_buffer_exec(
    const iree_hal_command_buffer_t *command_buffer);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_ROCM_ARRAY_COMMAND_BUFFER_H_
