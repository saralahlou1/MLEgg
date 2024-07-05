module {
  %0 = tensor.empty() : tensor<3x2xf32>
  %1 = tensor.empty() : tensor<2x4xf32>
  %2 = tensor.empty() : tensor<3x4xf32>
  %3 = tensor.empty() : tensor<3x4xf32>
  %4 = tensor.empty() : tensor<2x4xf32>
  %5 = linalg.matmul ins(%0, %1 : tensor<3x2xf32>, tensor<2x4xf32>) outs(%2 : tensor<3x4xf32>) -> tensor<3x4xf32>
  %6 = tensor.empty() : tensor<3x4xf32>
  %7 = tensor.empty() : tensor<2x4xf32>
  %8 = linalg.add ins(%1, %4 : tensor<2x4xf32>, tensor<2x4xf32>) outs(%7 : tensor<2x4xf32>) -> tensor<2x4xf32>
  %9 = tensor.empty() : tensor<3x4xf32>
  %10 = linalg.matmul ins(%0, %8 : tensor<3x2xf32>, tensor<2x4xf32>) outs(%9 : tensor<3x4xf32>) -> tensor<3x4xf32>
}

