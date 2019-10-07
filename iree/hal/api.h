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

#ifndef IREE_HAL_API_H_
#define IREE_HAL_API_H_

#include <stdint.h>

struct iree_hal_buffer_t;
struct iree_hal_semaphore_t;
struct iree_hal_fence_t;
struct iree_hal_buffer_view_t;

//===----------------------------------------------------------------------===//
// Types and Enums
//===----------------------------------------------------------------------===//

typedef struct {
  // TODO(benvanik): flesh out memory mapping info.
  // Want to avoid needing an allocation, can we presize this to fit a
  // MappedMemory<uint8_t>?
} iree_hal_mapped_memory_t;

// A bitfield specifying how memory will be accessed in a mapped memory region.
typedef enum {
  // Memory is not mapped.
  IREE_HAL_MEMORY_ACCESS_NONE = 0,
  // Memory will be read.
  // If a buffer is only mapped for reading it may still be possible to write to
  // it but the results will be undefined (as it may present coherency issues).
  IREE_HAL_MEMORY_ACCESS_READ = 1 << 0,
  // Memory will be written.
  // If a buffer is only mapped for writing it may still be possible to read
  // from it but the results will be undefined or incredibly slow (as it may
  // be mapped by the driver as uncached).
  IREE_HAL_MEMORY_ACCESS_WRITE = 1 << 1,
  // Memory will be discarded prior to mapping.
  // The existing contents will be undefined after mapping and must be written
  // to ensure validity.
  IREE_HAL_MEMORY_ACCESS_DISCARD = 1 << 2,
  // Memory will be discarded and completely overwritten in a single operation.
  IREE_HAL_MEMORY_ACCESS_DISCARD_WRITE =
      IREE_HAL_MEMORY_ACCESS_WRITE | IREE_HAL_MEMORY_ACCESS_DISCARD,
  // Memory may have any operation performed on it.
  IREE_HAL_MEMORY_ACCESS_ALL = IREE_HAL_MEMORY_ACCESS_READ |
                               IREE_HAL_MEMORY_ACCESS_WRITE |
                               IREE_HAL_MEMORY_ACCESS_DISCARD,
} iree_hal_memory_access_t;

//===----------------------------------------------------------------------===//
// iree::hal::Buffer
//===----------------------------------------------------------------------===//

// Returns a reference to a subspan of the |buffer|.
// If |byte_length| is IREE_WHOLE_BUFFER the remaining bytes in the buffer after
// |byte_offset| (possibly 0) will be selected.
//
// The parent buffer will remain alive for the lifetime of the subspan
// returned. If the subspan is a small portion this may cause additional
// memory to remain allocated longer than required.
//
// Returns the given |buffer| if the requested span covers the entire range.
iree_status_t iree_hal_buffer_subspan(iree_hal_buffer_t* buffer,
                                      iree_device_size_t byte_offset,
                                      iree_device_size_t byte_length,
                                      iree_allocator_t allocator,
                                      iree_hal_buffer_t** out_buffer);

// Retains the given |buffer| for the caller.
iree_status_t iree_hal_buffer_retain(iree_hal_buffer_t* buffer);

// Releases the given |buffer| from the caller.
iree_status_t iree_hal_buffer_release(iree_hal_buffer_t* buffer);

// Returns the size in bytes of the buffer.
iree_device_size_t iree_hal_buffer_byte_length(const iree_hal_buffer_t* buffer);

// Sets a range of the buffer to the given value.
iree_status_t iree_hal_buffer_fill(iree_hal_buffer_t* buffer,
                                   iree_device_size_t byte_offset,
                                   iree_device_size_t byte_length,
                                   uint32_t pattern);

// Reads a block of data from the buffer at the given offset.
iree_status_t iree_hal_buffer_read(iree_hal_buffer_t* buffer,
                                   iree_device_size_t source_offset,
                                   void* target_buffer,
                                   iree_device_size_t data_length);

// Writes a block of byte data into the buffer at the given offset.
iree_status_t iree_hal_buffer_write(iree_hal_buffer_t* buffer,
                                    iree_device_size_t target_offset,
                                    const void* source_buffer,
                                    iree_device_size_t data_length);

// Maps the buffer to be accessed as a host pointer into |out_mapped_memory|.
iree_status_t iree_hal_buffer_map(iree_hal_buffer_t* buffer,
                                  iree_hal_memory_access_t memory_access,
                                  iree_device_size_t element_offset,
                                  iree_device_size_t element_length,
                                  iree_hal_mapped_memory_t* out_mapped_memory);

// Unmaps the buffer as was previously mapped to |mapped_memory|.
iree_status_t iree_hal_buffer_unmap(iree_hal_buffer_t* buffer,
                                    iree_hal_mapped_memory_t* mapped_memory);

//===----------------------------------------------------------------------===//
// iree::hal::BufferView
//===----------------------------------------------------------------------===//

// Creates a buffer view with the given |buffer|, which may be nullptr.
iree_status_t iree_hal_buffer_view_create(
    iree_hal_buffer_t* buffer, iree_byte_span_t shape, int8_t element_size,
    iree_allocator_t allocator, iree_hal_buffer_view_t** out_buffer_view);

// Retains the given |buffer_view| for the caller.
iree_status_t iree_hal_buffer_view_retain(iree_hal_buffer_view_t* buffer_view);

// Releases the given |buffer_view| from the caller.
iree_status_t iree_hal_buffer_view_release(iree_hal_buffer_view_t* buffer_view);

// Sets the buffer view to point at the new |buffer| with the given metadata.
// To clear a buffer_view to empty use iree_hal_buffer_view_reset.
iree_status_t iree_hal_buffer_view_assign(iree_hal_buffer_view_t* buffer_view,
                                          iree_hal_buffer_t* buffer,
                                          iree_byte_span_t shape,
                                          int8_t element_size);

// Resets the buffer view to have an empty buffer and shape.
iree_status_t iree_hal_buffer_view_reset(iree_hal_buffer_view_t* buffer_view);

// Returns the buffer underlying the buffer view.
iree_hal_buffer_t* iree_hal_buffer_view_buffer(
    iree_hal_buffer_view_t* buffer_view);

// Returns the shape of the buffer view in |out_shape|.
iree_status_t iree_hal_buffer_view_shape(iree_hal_buffer_view_t* buffer_view,
                                         iree_byte_span_t* out_shape);

// Returns the size of each element in the buffer view in bytes.
int8_t iree_hal_buffer_view_element_size(iree_hal_buffer_view_t* buffer_view);

//===----------------------------------------------------------------------===//
// iree::hal::Semaphore
//===----------------------------------------------------------------------===//

// Retains the given |semaphore| for the caller.
iree_status_t iree_hal_semaphore_retain(iree_hal_semaphore_t* semaphore);

// Releases the given |semaphore| from the caller.
iree_status_t iree_hal_semaphore_release(iree_hal_semaphore_t* semaphore);

//===----------------------------------------------------------------------===//
// iree::hal::Fence
//===----------------------------------------------------------------------===//

// Retains the given |fence| for the caller.
iree_status_t iree_hal_fence_retain(iree_hal_fence_t* fence);

// Releases the given |fence| from the caller.
iree_status_t iree_hal_fence_release(iree_hal_fence_t* fence);

#endif  // IREE_HAL_API_H_
