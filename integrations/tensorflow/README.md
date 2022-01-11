# IREE TensorFlow Importers

This project contains IREE frontends for importing various forms of TensorFlow
formats.

## Quick Development Setup

This assumes that you have an appropriate `bazel` installed.

TODO: Remove the need for these symlinks by copying iree-dialects in tree
and switching to sha-hash based tensorflow loading.

```
ln -s $IREE_DIR/llvm-external-projects/iree-dialects .
ln -s $IREE_DIR/third_party/tensorflow third_party/
```

Build the importer binaries:

```
# All of them (takes a long time).
bazel build iree_tf_compiler:importer-binaries

# Or individuals:
bazel build iree_tf_compiler:iree-import-tflite
bazel build iree_tf_compiler:iree-import-xla
bazel build iree_tf_compiler:iree-import-tf
```

Symlink binaries into python packages (only needs to be done once):

```
./symlink_binaries.sh
```

Pip install editable (recommend to do this in a virtual environment):

```
pip install -e python_projects/iree_tflite
pip install -e python_projects/iree_xla
pip install -e python_projects/iree_tf
```

Test installed:

```
iree-import-tflite -help
iree-import-xla -help
iree-import-tf -help
```
