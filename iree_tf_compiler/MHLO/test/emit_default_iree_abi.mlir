// RUN: iree-tf-opt %s -iree-mhlo-emit-default-iree-abi -split-input-file -verify-diagnostics | IreeFileCheck %s

// CHECK-LABEL: func @valid
// CHECK-SAME{LITERAL}: iree.abi = "{\22a\22:[[\22ndarray\22,\22f32\22,2,2,3],[\22ndarray\22,\22f32\22,1,3]],\22r\22:[[\22ndarray\22,\22f32\22,1,3],[\22ndarray\22,\22f32\22,2,2,3]],\22v\22:1}"
func @valid(%arg0: tensor<2x3xf32>, %arg1: tensor<3xf32>) -> (tensor<3xf32>, tensor<2x3xf32>) {
  return %arg1, %arg0 : tensor<3xf32>, tensor<2x3xf32>
}
