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

#ifndef IREE_RT_RESOURCE_STATE_H_
#define IREE_RT_RESOURCE_STATE_H_

#include "iree/base/ref_ptr.h"

namespace iree {
namespace rt {

// DO NOT SUBMIT
// need to think of synchronization call scheme
// low level: command buffer to fill with commands
// mid level: wait semaphores, signal semaphores
// high level: synchronous?
//
// fence per pool? timeline increments on each mutate?
//
// semaphore per resource or per group (possibly per pool)
//
// most manip happens within modules from IR/c++
// need to integrate well with that
// call into function: capture current resource values (buffers)
// use Map(MemoryAccess) to get the versioned value/bump version?
//
// MultiMap to map many - may return resource exhausted of too many in-flight?
// useful for rate limiting (block caller/insert into wait queue), some kind of
// backpressure mechanism
// use SchedulingPolicy to decide? or queries on resource pool?
//   CanMap(...) before attempting to invoke?
//
// external manip happens from calling code
// maybe always sync or map? worth having helpers?
//
// how to define epochs/ringbuffer
// when to advance time?
// what kind of commands required?
//  - read: possibly release from queue, transfer to new queue
//  - write: possibly release from queue, transfer to new queue
//         release may not be needed if discard
//  - assign: just take buffer directly, drop old
//  - copy: some kind of queue transfer? how to ensure coherent? move to source
//       queue and then insert a transfer/acquisition in target?
//  - fill: fill on current queue?
//
// sync Foo(rp, in0, in1, out0):
//   cres0 = rp.Map(0, Read) -- get buffer
//   mres1 = rp.Map(1, Write) -- get buffer
//   FooInner(cres0, mres1, in0, in1, out0)
//
// async Foo(wait_semas, rp, in0, in1, out0, signal_semas):
//   {cres0} = rp.Map(0, Read) -- get buffer (no version needed, constant)
//   {mres1, sema, wait_value, signal_value} = rp.Map(1, Write) -- get buffer
//       and sema when avail/done
//   wait_semas += {sema, wait_value}
//   signal_semas += {sema, signal_value}
//   FooInner(wait_semas, cres0, mres1, in0, in1, out0, signal_semas)
//
// ** NEED MULTIMAP ** to reduce overhead - may return just a single sema value
// or multiple semaphores
// maybe MultiMap takes a list of wait/signal semas to append to

// Thread-compatible.
class ResourceState : public RefObject<ResourceState> {
 public:
  ~ResourceState();

  // requires a device that owns it? or just allocator?
  // pass devices to functions? or command buffers only?
  // query associated devices, used for invocation? or not?

  // query devices

  // reflect resources (names? current types? etc)

  // waits until all outstanding writes/assigns have completed?
  Status WaitIdle(absl::Time deadline);

  // pins memory until buffer reference is dropped
  // could lead to exhaustion
  StatusOr<hal::BufferView> Map(int resource_id);

  // copies buffer into scratch and returns
  // DO NOT SUBMIT subrange with indices and with offsets
  StatusOr<hal::BufferView> Read(int source_resource_id);

  // copies into target buffer and returns a view describing current shape/etc
  // returned buffer may be a subset of the |target_buffer|
  // DO NOT SUBMIT subrange with indices and with offsets
  StatusOr<hal::BufferView> Read(int source_resource_id,
                                 ref_ptr<Buffer> target_buffer);

  //
  // DO NOT SUBMIT subrange with indices and with offsets
  Status Write(int target_resource_id, ref_ptr<Buffer> source_buffer);

  // fills existing contents in-place
  // DO NOT SUBMIT subrange with indices and with offsets
  Status Fill(int target_resource_id, uint32_t value);

  // fills contents of resource with a given value.
  // may result in a splat?
  Status Assign(int target_resource_id, Shape shape, int8_t element_size,
                uint32_t value);

  // assigns the resource to the given buffer, taking ownership
  Status Assign(int target_resource_id, hal::BufferView buffer_view);

  // copy all resources (assumes template id matches)
  static Status Copy(ref_ptr<ResourcePool> source_pool,
                     ref_ptr<ResourcePool> target_pool);

  // copy a single resource
  // DO NOT SUBMIT subrange with indices
  static Status Copy(ref_ptr<ResourcePool> source_pool, int source_resource_id,
                     ref_ptr<ResourcePool> target_pool, int target_resource_id);

  // resets the resource to unassigned
  Status Reset(int target_resource_id);

  // resets all resources to unassigned
  Status Reset();

  // serialize - async and sync
  // deserialize

 private:
  ResourceState();
};

}  // namespace rt
}  // namespace iree

#endif  // IREE_RT_RESOURCE_STATE_H_
