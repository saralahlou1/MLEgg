module {
  %0 = tensor.empty() : tensor<3x2xf32>
  %1 = tensor.empty() : tensor<2x4xf32>
  %2 = tensor.empty() : tensor<3x4xf32>
  %3 = tensor.empty() : tensor<4x3xf32>
  %4 = tensor.empty() : tensor<3x3xf32>
  %5 = tensor.empty() : tensor<2x3xf32>
  %6 = linalg.matmul ins(%1, %3 : tensor<2x4xf32>, tensor<4x3xf32>) outs(%5 : tensor<2x3xf32>) -> tensor<2x3xf32>
  %7 = tensor.empty() : tensor<3x3xf32>
  %8 = linalg.matmul ins(%0, %6 : tensor<3x2xf32>, tensor<2x3xf32>) outs(%7 : tensor<3x3xf32>) -> tensor<3x3xf32>
}

