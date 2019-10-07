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

#include "iree/rt/context.h"

#include "iree/base/tracing.h"

namespace iree {
namespace rt {

namespace {

int NextUniqueContextId() {
  static int next_id = 0;
  return ++next_id;
}

}  // namespace

Context::Context() : id_(NextUniqueContextId()) {
  IREE_TRACE_SCOPE("Context::ctor", int32_t)(id_);
}

Context::~Context() { IREE_TRACE_SCOPE("Context::dtor", int32_t)(id_); }

}  // namespace rt
}  // namespace iree
