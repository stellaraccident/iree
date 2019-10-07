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

#ifndef IREE_RT_STACK_H_
#define IREE_RT_STACK_H_

#include "iree/rt/stack_frame.h"

namespace iree {
namespace rt {

// Current runtime callstack.
//
// Stacks are thread-compatible. Execution on one thread and stack manipulation
// on another must be externally synchronized by the caller.
class Stack {
 public:
  Stack(const Stack&) = delete;
  Stack& operator=(const Stack&) = delete;
  virtual ~Stack();

  virtual absl::Span<const StackFrame> frames() const = 0;
  virtual absl::Span<StackFrame> mutable_frames() = 0;

  virtual StackFrame* current_frame() = 0;
  virtual const StackFrame* current_frame() const = 0;
  virtual StackFrame* caller_frame() = 0;
  virtual const StackFrame* caller_frame() const = 0;

  StatusOr<StackFrame*> PushFrame(Function function);
  Status PopFrame();

  virtual std::string DebugString() const = 0;
  virtual std::string DebugStringShort() const = 0;

 protected:
  Stack();
};

}  // namespace rt
}  // namespace iree

#endif  // IREE_RT_STACK_H_
