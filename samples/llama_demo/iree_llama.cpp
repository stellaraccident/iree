#include <stdio.h>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/modules/hal/module.h"
#include "iree/vm/api.h"
#include "iree/vm/bytecode/module.h"
#include <iree/runtime/api.h>

#include "iree/runtime/api.h"

iree_status_t llama_func(iree_runtime_session_t *session, const int *values,
                         int values_length, int *out_result) {
  iree_runtime_call_t call;
  IREE_RETURN_IF_ERROR(iree_runtime_call_initialize_by_name(
      session, iree_make_cstring_view("module.first_vicuna_forward"), &call));

  iree_hal_buffer_view_t *arg0 = NULL;
  const iree_hal_dim_t arg0_shape[2] = {(iree_hal_dim_t)values_length,
                                        (iree_hal_dim_t)values_length};

  iree_status_t status = iree_ok_status();
  iree_hal_buffer_params_t params;
  memset(&params, 0, sizeof(params));
  params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_buffer_view_allocate_buffer_copy(
        iree_runtime_session_device(session),
        iree_runtime_session_device_allocator(session),
        IREE_ARRAYSIZE(arg0_shape), arg0_shape, IREE_HAL_ELEMENT_TYPE_INT_64,
        IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
        params,
        iree_make_const_byte_span((void *)values, sizeof(int) * values_length),
        &arg0);
  }

  if (iree_status_is_ok(status)) {
    status = iree_runtime_call_inputs_push_back_buffer_view(&call, arg0);
  }
  iree_hal_buffer_view_release(arg0);
  if (iree_status_is_ok(status)) {
    status = iree_runtime_call_invoke(&call, /*flags=*/0);
  }

  iree_hal_buffer_view_t *buffer_view = NULL;
  if (iree_status_is_ok(status)) {
    status =
        iree_runtime_call_outputs_pop_front_buffer_view(&call, &buffer_view);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_transfer_d2h(
        iree_runtime_session_device(session),
        iree_hal_buffer_view_buffer(buffer_view), 0, out_result,
        sizeof(*out_result), IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
        iree_infinite_timeout());
  }
  iree_hal_buffer_view_release(buffer_view);

  iree_runtime_call_deinitialize(&call);
  return status;
}

iree_status_t run_sample(iree_string_view_t bytecode_module_path,
                         iree_string_view_t driver_name) {
  iree_status_t status = iree_ok_status();

  //===-------------------------------------------------------------------===//
  // Instance configuration (this should be shared across sessions).

  fprintf(stdout, "Configuring IREE runtime instance and '%s' device\n",
          driver_name.data);
  iree_runtime_instance_options_t instance_options;
  iree_runtime_instance_options_initialize(&instance_options);
  iree_runtime_instance_options_use_all_available_drivers(&instance_options);
  iree_runtime_instance_t *instance = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_runtime_instance_create(&instance_options,
                                          iree_allocator_system(), &instance);
  }

  iree_hal_device_t *device = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_runtime_instance_try_create_default_device(
        instance, driver_name, &device);
  }
  //===-------------------------------------------------------------------===//

  //===-------------------------------------------------------------------===//
  // Session configuration (one per loaded module to hold module state).

  fprintf(stdout, "Creating IREE runtime session\n");
  iree_runtime_session_options_t session_options;
  iree_runtime_session_options_initialize(&session_options);
  iree_runtime_session_t *session = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_runtime_session_create_with_device(
        instance, &session_options, device,
        iree_runtime_instance_host_allocator(instance), &session);
  }
  iree_hal_device_release(device);

  fprintf(stdout, "Loading bytecode module at '%s'\n",
          bytecode_module_path.data);
  if (iree_status_is_ok(status)) {
    status = iree_runtime_session_append_bytecode_module_from_file(
        session, bytecode_module_path.data);
  }
  //===-------------------------------------------------------------------===//

  //===-------------------------------------------------------------------===//
  // Call the exported sample functions with some test inputs
  fprintf(stdout, "Calling functions\n\n");

  // llama_func([100])
  if (iree_status_is_ok(status)) {
    const int input[1] = {100};
    int result = -1;
    status = llama_func(session, input, 1, &result);
    fprintf(stdout, "llama_func([100]): %d\n", result);
  }

  return status;
}

int main(int argc, char **argv) {
  if (argc != 3) {
    fprintf(stderr, "Usage: llama-app.exe </path/to/llama2_7b_int4_cpu.vmfb> "
                    "<driver_name>\n");
    return -1;
  }

  iree_string_view_t bytecode_module_path = iree_make_cstring_view(argv[1]);
  iree_string_view_t driver_name = iree_make_cstring_view(argv[2]);

  iree_status_t result = run_sample(bytecode_module_path, driver_name);
  if (!iree_status_is_ok(result)) {
    fprintf(stdout, "Failed!\n");
    iree_status_fprint(stderr, result);
    iree_status_ignore(result);
    return -1;
  }
  fprintf(stdout, "\nSuccess!\n");
  return 0;
}