module {
  %0 = tensor.empty() : tensor<2xf32>
  %1 = tensor.empty() : tensor<f32>
  %2 = tensor.empty() : tensor<1x2xf32>
  %transposed = linalg.transpose ins(%2 : tensor<1x2xf32>) outs(%transposed : tensor<2x1xf32>) permutation = [1, 0] 
  %extracted_slice = tensor.extract_slice %transposed[0, 0] [2, 1] [1, 1] : tensor<2x1xf32> to tensor<2xf32>
  %expanded = tensor.expand_shape %0 [[0, 1]] : tensor<2xf32> into tensor<2x1xf32>
  %3 = tensor.empty() : tensor<1x1xf32>
  %4 = linalg.matmul ins(%2, %expanded : tensor<1x2xf32>, tensor<2x1xf32>) outs(%3 : tensor<1x1xf32>) -> tensor<1x1xf32>
  %extracted_slice_0 = tensor.extract_slice %4[0, 0] [1, 1] [1, 1] : tensor<1x1xf32> to tensor<f32>
}

