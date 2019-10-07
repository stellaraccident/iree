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

#ifndef IREE_RT_FUNCTION_H_
#define IREE_RT_FUNCTION_H_

#include <ostream>

#include "absl/strings/string_view.h"
#include "iree/base/ref_ptr.h"


// DO NOT SUBMIT how to make value typed?
// some module-specific ordinal that can be looked up again?
// want to avoid allocating name/etc on query - just a reference?

namespace iree {
namespace rt {

class Module;

class Function {
 public:
  Function(const Function&) = delete;
  Function& operator=(const Function&) = delete;
  virtual ~Function() = default;

  virtual const Module& module() const = 0;
  virtual absl::string_view name() const = 0;

  virtual std::string DebugStringShort() const = 0;

 protected:
  Function() = default;
};

inline std::ostream& operator<<(std::ostream& stream,
                                const Function& function) {
  stream << function.DebugStringShort();
  return stream;
}

}  // namespace rt
}  // namespace iree

#endif  // IREE_RT_FUNCTION_H_
