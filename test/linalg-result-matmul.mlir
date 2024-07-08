module {
  %0 = tensor.empty() : tensor<5x1xf32>
  %1 = tensor.empty() : tensor<1x1xf32>
  %2 = tensor.empty() : tensor<5x1xf32>
  %transposed = linalg.transpose ins(%2 : tensor<5x1xf32>) outs(%transposed : tensor<1x5xf32>) permutation = [1, 0] 
  %extracted_slice = tensor.extract_slice %2[0, 0] [5, 1] [1, 1] : tensor<5x1xf32> to tensor<5xf32>
  %extracted_slice_0 = tensor.extract_slice %0[0, 0] [5, 1] [1, 1] : tensor<5x1xf32> to tensor<5xf32>
  %3 = tensor.empty() : tensor<f32>
  %4 = linalg.dot ins(%extracted_slice, %extracted_slice_0 : tensor<5xf32>, tensor<5xf32>) outs(%3 : tensor<f32>) -> tensor<f32>
  %cst = arith.constant dense<1> : tensor<2xi64>
  %reshape = tensor.reshape %4(%cst) : (tensor<f32>, tensor<2xi64>) -> tensor<1x1xf32>
}

