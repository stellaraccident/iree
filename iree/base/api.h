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

#ifndef IREE_BASE_API_H_
#define IREE_BASE_API_H_

#include <stddef.h>
#include <stdint.h>

//===----------------------------------------------------------------------===//
// Types and Enums
//===----------------------------------------------------------------------===//

// Well-known status codes matching absl::StatusCode.
typedef enum {
  IREE_STATUS_SUCCESS = 0,
  IREE_STATUS_CANCELLED = 1,
  IREE_STATUS_UNKNOWN = 2,
  IREE_STATUS_INVALID_ARGUMENT = 3,
  IREE_STATUS_DEADLINE_EXCEEDED = 4,
  IREE_STATUS_NOT_FOUND = 5,
  IREE_STATUS_ALREADY_EXISTS = 6,
  IREE_STATUS_PERMISSION_DENIED = 7,
  IREE_STATUS_RESOURCE_EXHAUSTED = 8,
  IREE_STATUS_FAILED_PRECONDITION = 9,
  IREE_STATUS_ABORTED = 10,
  IREE_STATUS_OUT_OF_RANGE = 11,
  IREE_STATUS_UNIMPLEMENTED = 12,
  IREE_STATUS_INTERNAL = 13,
  IREE_STATUS_UNAVAILABLE = 14,
  IREE_STATUS_DATA_LOSS = 15,
  IREE_STATUS_UNAUTHENTICATED = 16,
} iree_status_t;

// TODO(benvanik): add ABSL_MUST_USE_RESULT to iree_status_t.

// Size, in bytes, of a buffer on the host.
typedef size_t iree_host_size_t;

// Size, in bytes, of a buffer on devices.
typedef uint64_t iree_device_size_t;
// Whole length of the underlying buffer.
#define IREE_WHOLE_BUFFER (iree_device_size_t(-1))

// An allocator for host-memory allocations.
typedef struct {
  // User-defined pointer passed to all functions.
  void* self;
  // Allocates |byte_length| of memory and stores the pointer in |out_ptr|.
  iree_status_t (*alloc)(void* self, iree_host_size_t byte_length,
                         void** out_ptr);
  // Frees |ptr| from a previous alloc call.
  iree_status_t (*free)(void* self, void* ptr);
} iree_allocator_t;

// A span of bytes (ala std::span of uint8_t).
typedef struct {
  void* data;
  iree_host_size_t data_length;
} iree_byte_span_t;

// A string view (ala std::string_view) into a non-NUL-terminated string.
typedef struct {
  const char* data;
  size_t data_length;
} iree_string_view_t;

// Known versions of the API that can be referenced in code.
// Out-of-bounds values are possible in forward-versioned changes.
typedef enum {
  IREE_API_VERSION_0 = 0,
} iree_api_version_t;

struct iree_file_mapping_t;

//===----------------------------------------------------------------------===//
// iree Core API
//===----------------------------------------------------------------------===//

// Checks whether the |expected_version| of the caller matches the implemented
// version of |out_actual_version|. Forward compatibility of the API is
// supported but backward compatibility is not: newer binaries using older
// shared libraries of the runtime will fail.
iree_status_t iree_api_version_check(iree_api_version_t expected_version,
                                     iree_api_version_t* out_actual_version);

//===----------------------------------------------------------------------===//
// iree::FileMapping
//===----------------------------------------------------------------------===//

// Opens a file at |path| for read-only access via a file mapping.
iree_status_t iree_file_mapping_open_read(
    iree_string_view_t path, iree_allocator_t allocator,
    iree_file_mapping_t** out_file_mapping);

// Retains the given |file_mapping| for the caller.
iree_status_t iree_file_mapping_retain(iree_file_mapping_t* file_mapping);

// Releases the given |file_mapping| from the caller.
iree_status_t iree_file_mapping_release(iree_file_mapping_t* file_mapping);

// Returns a reference to the byte buffer the |file_mapping| backs.
// Though the returned buffer is non-const behavior is undefined if read-only
// mappings are written to (exceptions, segfaults, etc).
iree_byte_span_t iree_file_mapping_data(iree_file_mapping_t* file_mapping);

#endif  // IREE_BASE_API_H_
