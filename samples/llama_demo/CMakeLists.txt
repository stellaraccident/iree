iree_cc_binary(
  NAME
    iree_llama_sample
  SRCS
    "iree_llama.cpp"
  DEPS
    iree::runtime
)
