module {
  %0 = tensor.empty() : tensor<3x2xf32>
  %1 = tensor.empty() : tensor<2x4xf32>
  %2 = tensor.empty() : tensor<3x4xf32>
  %3 = tensor.empty() : tensor<4x3xf32>
  %4 = tensor.empty() : tensor<3x3xf32>
  %5 = tensor.empty() : tensor<3x4xf32>
  %6 = tensor.empty() : tensor<2x3xf32>
  %7 = linalg.matmul ins(%1, %3 : tensor<2x4xf32>, tensor<4x3xf32>) outs(%6 : tensor<2x3xf32>) -> tensor<2x3xf32>
  %8 = tensor.empty() : tensor<3x3xf32>
  %9 = linalg.matmul ins(%0, %7 : tensor<3x2xf32>, tensor<2x3xf32>) outs(%8 : tensor<3x3xf32>) -> tensor<3x3xf32>
  %10 = tensor.empty() : tensor<3x2xf32>
  %11 = linalg.add ins(%0, %0 : tensor<3x2xf32>, tensor<3x2xf32>) outs(%10 : tensor<3x2xf32>) -> tensor<3x2xf32>
  %12 = tensor.empty() : tensor<3x4xf32>
  %13 = linalg.matmul ins(%11, %1 : tensor<3x2xf32>, tensor<2x4xf32>) outs(%12 : tensor<3x4xf32>) -> tensor<3x4xf32>
}

