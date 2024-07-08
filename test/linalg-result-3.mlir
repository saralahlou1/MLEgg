module {
  %0 = tensor.empty() : tensor<2x3xf32>
  %1 = tensor.empty() : tensor<2x3xf32>
  %2 = linalg.add ins(%0, %0 : tensor<2x3xf32>, tensor<2x3xf32>) outs(%1 : tensor<2x3xf32>) -> tensor<2x3xf32>
  %3 = tensor.empty() : tensor<3x2xf32>
  %transposed = linalg.transpose ins(%2 : tensor<2x3xf32>) outs(%3 : tensor<3x2xf32>) permutation = [1, 0] 
  %4 = tensor.empty() : tensor<2x3xf32>
  %5 = tensor.empty() : tensor<2x3xf32>
  %6 = tensor.empty() : tensor<2x3xf32>
  %7 = linalg.add ins(%0, %0 : tensor<2x3xf32>, tensor<2x3xf32>) outs(%6 : tensor<2x3xf32>) -> tensor<2x3xf32>
  %8 = tensor.empty() : tensor<2x3xf32>
  %9 = linalg.add ins(%0, %0 : tensor<2x3xf32>, tensor<2x3xf32>) outs(%8 : tensor<2x3xf32>) -> tensor<2x3xf32>
  %10 = tensor.empty() : tensor<2x3xf32>
  %11 = linalg.add ins(%9, %9 : tensor<2x3xf32>, tensor<2x3xf32>) outs(%10 : tensor<2x3xf32>) -> tensor<2x3xf32>
}

