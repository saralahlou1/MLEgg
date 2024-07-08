module {
    %lhs = tensor.empty() : tensor<3x2xf32>
    %rhs1 = tensor.empty() : tensor<2x4xf32>
    %init = tensor.empty() : tensor<3x4xf32>
    %init0 = tensor.empty() : tensor<3x4xf32>
    %rhs2 = tensor.empty() : tensor<2x4xf32>
    %matmul1 = linalg.matmul ins(%lhs, %rhs1: tensor<3x2xf32>, tensor<2x4xf32>)
                            outs(%init: tensor<3x4xf32>) -> tensor<3x4xf32>
    %matmul2 = linalg.matmul ins(%lhs, %rhs2: tensor<3x2xf32>, tensor<2x4xf32>)
                            outs(%init0: tensor<3x4xf32>) -> tensor<3x4xf32>
    
    %init2 = tensor.empty() : tensor<3x4xf32>
    %next = linalg.add ins(%matmul1, %matmul2: tensor<3x4xf32>, tensor<3x4xf32>)
                          outs(%init2: tensor<3x4xf32>) -> tensor<3x4xf32>
}
