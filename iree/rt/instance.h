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

#ifndef IREE_RT_INSTANCE_H_
#define IREE_RT_INSTANCE_H_

#include "iree/base/ref_ptr.h"
#include "iree/hal/device_manager.h"

namespace iree {
namespace rt {

// Shared runtime instance responsible for routing Context events, enumerating
// and creating hardware device interfaces, and managing thread pools.
//
// A single runtime instance can service multiple contexts and hosting
// applications should try to reuse a runtime as much as possible. This ensures
// that resource allocation across contexts is handled and extraneous device
// interaction is avoided.
class Instance final : public RefObject<Instance> {
 public:
  Instance();
  ~Instance();
  Instance(const Instance&) = delete;
  Instance& operator=(const Instance&) = delete;

  hal::DeviceManager* device_manager() const { return &device_manager_; }

 private:
  mutable hal::DeviceManager device_manager_;
};

}  // namespace rt
}  // namespace iree

#endif  // IREE_RT_INSTANCE_H_
