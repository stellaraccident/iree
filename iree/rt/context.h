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

#ifndef IREE_RT_CONTEXT_H_
#define IREE_RT_CONTEXT_H_

#include <ostream>

#include "absl/strings/string_view.h"
#include "iree/base/ref_ptr.h"
#include "iree/base/status.h"

// load module

// context:
//   need to set policy per context
//   policy changes may require device changes
//   context holds master device table? use to keep modules in sync?
//
// module:
//   gets list of available devices from context (when policy/devs change)
//   updates device table, reindexes executable/function variants
// resource table:
//   templates for resource pools?
// device table:
//   one per module?
// executable table:
//   has index map of executable ordinal to selected variant ordinal
//   has hal executable cache
// function table:
//   has index map of function ordinal to selected variant ordinal
//

// iree::rt::
// make Instance/Context/Module/Function abstract
// rt::Instance - is the scheduler?
//   deferred calls, etc? fibers? or as DeferredCall interface?

// how much of debugger can be seen?
// remove concept of fibers? replace with just system threads?
// can still set breakpoints/step/etc
// could have stack frame interface:
//   native: use native debugging features/stack dump
//   vm: actual vm stack
//   could still track live buffers/etc in codegen models

// caller:
//   create context
//   load modules
//   set policy
//   create resource pools (per module), retain
// instance, context[], resource_pools[]

namespace iree {
namespace rt {

// An isolated execution context.
// Effectively a sandbox where modules can be loaded and run with restricted
// visibility. Each context may have its own set of imports that modules can
// access and its own resource constraints.
//
// Modules have imports resolved automatically when loaded by searching existing
// modules. This means that load order is important to ensure overrides are
// respected. For example, target-specific modules should be loaded prior to
// generic modules that may import functions defined there and if a function is
// not available in the target-specific modules the fallback provided by the
// generic module will be used.
//
// Thread-safe due to immutability after construction.
class Context final : public RefObject<Context> {
 public:
  // DO NOT SUBMIT provide list of modules
  static StatusOr<ref_ptr<Context>> Create();

  Context();
  Context(const Context&) = delete;
  Context& operator=(const Context&) = delete;
  Context(Context&&) = default;
  Context& operator=(Context&&) = default;
  ~Context();

  // A process-unique ID for the context.
  int id() const { return id_; }

  std::string DebugStringShort() const;

  StatusOr<const Module*> LookupModule(absl::string_view module_name) const;
  StatusOr<Module*> LookupModule(absl::string_view module_name);
  StatusOr<const Function> LookupExport(absl::string_view export_name) const;

  // import/export functions
  // import set or something that allows fast reuse of fixed tables
  //   FFI version, etc
  // or maybe imports are just modules with system impl? no need for individual
  // functions!!
  // can version modules independently
  // can polyfill if needed

  // Module exposes 'Execute' that takes invocation and directly evaluates
  // - routes to C++ function
  // - runs VM

  //    BeginInvoke(target, policy, semaphores/fences) -> ref Invocation
  //    Invoke(target, policy, semaphores/fences) (sync)

  // variant for all supported types + opaque
  // args provided as list of variants
  // results returned in Invocation struct or as variants from invoke
  // variant: semaphore, fence, resource pool, command buffer, buffer_view
  // opaque: void*? RefObject base? something else?
  // template magic for packing/unpacking and validation
  Status Invoke();
  StatusOr<ref_ptr<Invocation>> BeginInvoke();

 private:
  int id_;
};

inline std::ostream& operator<<(std::ostream& stream, const Context& context) {
  stream << context.DebugStringShort();
  return stream;
}

}  // namespace rt
}  // namespace iree

#endif  // IREE_RT_CONTEXT_H_
