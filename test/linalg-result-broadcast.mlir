module {
  %0 = tensor.empty() : tensor<3xf32>
  %1 = tensor.empty() : tensor<3x2xf32>
  %broadcasted = linalg.broadcast ins(%0 : tensor<3xf32>) outs(%1 : tensor<3x2xf32>) dimensions = [1] 
  %2 = tensor.empty() : tensor<2x4xf32>
  %3 = tensor.empty() : tensor<3x4xf32>
  %4 = tensor.empty() : tensor<4x3xf32>
  %5 = tensor.empty() : tensor<3x3xf32>
  %6 = tensor.empty() : tensor<2x3xf32>
  %7 = linalg.matmul ins(%2, %4 : tensor<2x4xf32>, tensor<4x3xf32>) outs(%6 : tensor<2x3xf32>) -> tensor<2x3xf32>
  %8 = tensor.empty() : tensor<3x3xf32>
  %9 = linalg.matmul ins(%broadcasted, %7 : tensor<3x2xf32>, tensor<2x3xf32>) outs(%8 : tensor<3x3xf32>) -> tensor<3x3xf32>
}

