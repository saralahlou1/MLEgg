module {

    %1 = tensor.empty() : tensor<2xf32>
    %3 = tensor.empty() : tensor<f32>

    %init3 = tensor.empty() : tensor<1x2xf32>
    %transpose1_result = linalg.transpose ins(%init3: tensor<1x2xf32>) outs(%transpose1_result: tensor<2x1xf32>) permutation = [1, 0]

    %2 = tensor.extract_slice %transpose1_result[0, 0] [2, 1] [1, 1] : tensor<2x1xf32> to tensor<2xf32>


    %dot = linalg.dot ins(%2, %1 : tensor<2xf32>, tensor<2xf32>)
            outs(%3 : tensor<f32>) -> tensor<f32>

    %lhs1 = tensor.empty() : tensor<3x2xf32>
    %rhs1 = tensor.empty() : tensor<2x4xf32>
    %init1 = tensor.empty() : tensor<3x4xf32>
    %rhs2 = tensor.empty() : tensor<2x4xf32>
    %7 = tensor.empty() : tensor<3x4xf32>
    %matmul1 = linalg.matmul ins(%lhs1, %rhs1: tensor<3x2xf32>, tensor<2x4xf32>)
                            outs(%init1: tensor<3x4xf32>) -> tensor<3x4xf32>
    %matmul2 = linalg.matmul ins(%lhs1, %rhs2: tensor<3x2xf32>, tensor<2x4xf32>)
                            outs(%7: tensor<3x4xf32>) -> tensor<3x4xf32>

    %init2 = tensor.empty() : tensor<3x4xf32>
    %next = linalg.add ins(%matmul1, %matmul2: tensor<3x4xf32>, tensor<3x4xf32>)
                          outs(%init2: tensor<3x4xf32>) -> tensor<3x4xf32>


    %lhs = tensor.empty() : tensor<3x2xf32>
    %rhs = tensor.empty() : tensor<2x4xf32>
    %init = tensor.empty() : tensor<3x4xf32>
    %matmul = linalg.matmul ins(%lhs, %rhs: tensor<3x2xf32>, tensor<2x4xf32>)
                            outs(%init: tensor<3x4xf32>) -> tensor<3x4xf32>
    %new = tensor.empty() : tensor<4x3xf32>
    %init0 = tensor.empty() : tensor<3x3xf32>
    %next0 = linalg.matmul ins(%matmul, %new: tensor<3x4xf32>, tensor<4x3xf32>)
                          outs(%init0: tensor<3x3xf32>) -> tensor<3x3xf32>



}

