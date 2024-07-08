module {
  %0 = tensor.empty() : tensor<2x3xf32>
  %1 = tensor.empty() : tensor<3x2xf32>
  %transposed = linalg.transpose ins(%0 : tensor<2x3xf32>) outs(%1 : tensor<3x2xf32>) permutation = [1, 0] 
  %2 = tensor.empty() : tensor<2x3xf32>
  %3 = tensor.empty() : tensor<2x3xf32>
  %4 = tensor.empty() : tensor<2x3xf32>
  %5 = linalg.add ins(%0, %0 : tensor<2x3xf32>, tensor<2x3xf32>) outs(%4 : tensor<2x3xf32>) -> tensor<2x3xf32>
}

