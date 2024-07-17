module {
  %0 = tensor.empty() : tensor<3x2xf32>
  %1 = tensor.empty() : tensor<2x4xf32>
  %2 = tensor.empty() : tensor<3x4xf32>
  %3 = tensor.empty() : tensor<4x3xf32>
  %4 = tensor.empty() : tensor<3x3xf32>
  %5 = tensor.empty() : tensor<3x2xf32>
  %6 = tensor.empty() : tensor<2x4xf32>
  %7 = tensor.empty() : tensor<3x4xf32>
  %8 = tensor.empty() : tensor<2x4xf32>
  %9 = tensor.empty() : tensor<3x4xf32>
  %10 = linalg.matmul ins(%5, %8 : tensor<3x2xf32>, tensor<2x4xf32>) outs(%9 : tensor<3x4xf32>) -> tensor<3x4xf32>
  %11 = tensor.empty() : tensor<3x4xf32>
  %12 = tensor.empty() : tensor<2xf32>
  %13 = tensor.empty() : tensor<f32>
  %14 = tensor.empty() : tensor<1x2xf32>
  %transposed = linalg.transpose ins(%14 : tensor<1x2xf32>) outs(%transposed : tensor<2x1xf32>) permutation = [1, 0] 
  %extracted_slice = tensor.extract_slice %transposed[0, 0] [2, 1] [1, 1] : tensor<2x1xf32> to tensor<2xf32>
  %15 = tensor.empty() : tensor<2x3xf32>
  %16 = linalg.matmul ins(%1, %3 : tensor<2x4xf32>, tensor<4x3xf32>) outs(%15 : tensor<2x3xf32>) -> tensor<2x3xf32>
  %17 = tensor.empty() : tensor<3x3xf32>
  %18 = linalg.matmul ins(%0, %16 : tensor<3x2xf32>, tensor<2x3xf32>) outs(%17 : tensor<3x3xf32>) -> tensor<3x3xf32>
  %19 = tensor.empty() : tensor<2x4xf32>
  %20 = linalg.add ins(%6, %8 : tensor<2x4xf32>, tensor<2x4xf32>) outs(%19 : tensor<2x4xf32>) -> tensor<2x4xf32>
  %21 = tensor.empty() : tensor<3x4xf32>
  %22 = linalg.matmul ins(%5, %20 : tensor<3x2xf32>, tensor<2x4xf32>) outs(%21 : tensor<3x4xf32>) -> tensor<3x4xf32>
  %expanded = tensor.expand_shape %12 [[0, 1]] : tensor<2xf32> into tensor<2x1xf32>
  %23 = tensor.empty() : tensor<1x1xf32>
  %24 = linalg.matmul ins(%14, %expanded : tensor<1x2xf32>, tensor<2x1xf32>) outs(%23 : tensor<1x1xf32>) -> tensor<1x1xf32>
  %extracted_slice_0 = tensor.extract_slice %24[0, 0] [1, 1] [1, 1] : tensor<1x1xf32> to tensor<f32>
}

