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

#ifndef IREE_RT_MODULE_H_
#define IREE_RT_MODULE_H_

#include <ostream>

#include "absl/strings/string_view.h"
#include "iree/base/ref_ptr.h"

namespace iree {
namespace rt {

// DO NOT SUBMIT abstract source map resolver accessible from here

class Module : public RefObject<Module> {
 public:
  Module(const Module&) = delete;
  Module& operator=(const Module&) = delete;
  virtual ~Module() = default;

  virtual absl::string_view name() const = 0;

  virtual std::string DebugStringShort() const = 0;

  // resolve function
  // can be used to wildcard/regex by implementations
  // like 'always resolve tf.foo' by using tf lookup

  // VariableState allocator/initializer
  // - allocate a variable state by ordinal/name?
  // - provide with device list and/or policy?

  // Execute(Invocation)

 protected:
  Module() = default;
};

inline std::ostream& operator<<(std::ostream& stream, const Module& module) {
  stream << module.DebugStringShort();
  return stream;
}

}  // namespace rt
}  // namespace iree

#endif  // IREE_RT_MODULE_H_
