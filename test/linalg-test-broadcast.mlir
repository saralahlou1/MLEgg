module {
    %lhs0 = tensor.empty() : tensor<3xf32>
    %init0 = tensor.empty() : tensor<3x2xf32>
    %lhs = linalg.broadcast ins(%lhs0 : tensor<3xf32>) outs(%init0: tensor<3x2xf32>) dimensions = [1]
    %rhs = tensor.empty() : tensor<2x4xf32>
    %init = tensor.empty() : tensor<3x4xf32>
    %matmul = linalg.matmul ins(%lhs, %rhs: tensor<3x2xf32>, tensor<2x4xf32>)
                            outs(%init: tensor<3x4xf32>) -> tensor<3x4xf32>
    %new = tensor.empty() : tensor<4x3xf32>
    %init2 = tensor.empty() : tensor<3x3xf32>
    %next = linalg.matmul ins(%matmul, %new: tensor<3x4xf32>, tensor<4x3xf32>)
                          outs(%init2: tensor<3x3xf32>) -> tensor<3x3xf32>
}
