# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from absl.testing import absltest

from pyiree import binding as binding


def create_simple_mul_module():
    blob = binding.compiler.compile_module_from_asm("""
    func @simple_mul(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32>
          attributes { iree.module.export } {
        %0 = "xla_hlo.mul"(%arg0, %arg1) {name = "mul.1"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
        return %0 : tensor<4xf32>
    }
    """)
    m = binding.vm.create_module_from_blob(blob)
    return m


def create_host_buffer_view():
  b = binding.hal.Buffer.allocate_heap(
    memory_type=int(binding.hal.MemoryType.HOST_LOCAL),
    usage=int(binding.hal.BufferUsage.ALL),
    allocation_size=16)
  b.fill_zero(0, 16)
  bv = b.create_view(binding.hal.Shape([4]), 4)
  print("BUFFER VIEW:", bv)
  return bv


class RuntimeTest(absltest.TestCase):

  def testModuleAndFunction(self):
    m = create_simple_mul_module()
    print("Module:", m)
    print("Module name:", m.name)
    self.assertEqual("module", m.name)

    # Function 0.
    f = m.lookup_function_by_ordinal(0)
    print("Function 0:", f)
    self.assertEqual("simple_mul", f.name)
    sig = f.signature
    self.assertEqual(2, sig.argument_count)
    self.assertEqual(1, sig.result_count)

    # Function 1.
    f = m.lookup_function_by_ordinal(1)
    self.assertIs(f, None)

    # By name.
    f = m.lookup_function_by_name("simple_mul")
    self.assertEqual("simple_mul", f.name)
    sig = f.signature
    self.assertEqual(2, sig.argument_count)
    self.assertEqual(1, sig.result_count)

    # By name not found.
    f = m.lookup_function_by_name("not_here")
    self.assertIs(f, None)

  def testInitialization(self):
    policy = binding.rt.Policy()
    print("policy =", policy)
    instance = binding.rt.Instance()
    print("instance =", instance)
    context = binding.rt.Context(instance=instance, policy=policy)
    print("context =", context)
    context_id = context.context_id
    print("context_id =", context.context_id)
    self.assertGreater(context_id, 0)

  def testRegisterModule(self):
    policy = binding.rt.Policy()
    instance = binding.rt.Instance()
    context = binding.rt.Context(instance=instance, policy=policy)
    m = create_simple_mul_module()
    context.register_module(m)
    self.assertIsNot(context.lookup_module_by_name("module"), None)
    self.assertIs(context.lookup_module_by_name("nothere"), None)
    f = context.resolve_function("module.simple_mul")
    self.assertIsNot(f, None)
    print("Resolved function:", f.name)
    self.assertIs(context.resolve_function("module.nothere"), None)

  def testInvoke(self):
    policy = binding.rt.Policy()
    instance = binding.rt.Instance()
    context = binding.rt.Context(instance=instance, policy=policy)
    m = create_simple_mul_module()
    context.register_module(m)
    f = context.resolve_function("module.simple_mul")
    print("INVOKE F:", f)
    arg0 = create_host_buffer_view()
    arg1 = create_host_buffer_view()
    result = create_host_buffer_view()
    inv = context.invoke(f, policy, [arg0, arg1], [result])

if __name__ == "__main__":
  absltest.main()
