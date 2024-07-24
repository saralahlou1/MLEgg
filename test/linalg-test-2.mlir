module {
    %init = tensor.empty() : tensor<2x3xf32>
    %transpose1 = tensor.empty() : tensor<3x2xf32>
    %transpose1_result = linalg.transpose ins(%init: tensor<2x3xf32>) outs(%transpose1: tensor<3x2xf32>) permutation = [1, 0]
    %transpose2 = tensor.empty() : tensor<2x3xf32>
    %transpose2_result = linalg.transpose ins(%transpose1_result: tensor<3x2xf32>) outs(%transpose2: tensor<2x3xf32>) permutation = [1, 0]
    %init2 = tensor.empty() : tensor<2x3xf32>
    %test = linalg.add ins(%transpose2_result, %transpose2_result: tensor<2x3xf32>, tensor<2x3xf32>)
                          outs(%init2: tensor<2x3xf32>) -> tensor<2x3xf32>
}

