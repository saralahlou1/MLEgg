module {
    // Define the input matrix
    %init = tensor.empty() : tensor<2x3xf32>
    %out = tensor.empty() : tensor<2x3xf32>
    
    %0 = linalg.add ins(%init, %init: tensor<2x3xf32>, tensor<2x3xf32>)
                          outs(%out: tensor<2x3xf32>) -> tensor<2x3xf32>

    // Allocate a tensor for the first transpose (3x2)
    %transpose1 = tensor.empty() : tensor<3x2xf32>
    
    // First transpose: transpose the 2x3 matrix to 3x2
    %transpose1_result = linalg.transpose ins(%0: tensor<2x3xf32>) outs(%transpose1: tensor<3x2xf32>) permutation = [1, 0]
    
    // Allocate a tensor for the second transpose (2x3)
    %transpose2 = tensor.empty() : tensor<2x3xf32>
    
    // Second transpose: transpose the 3x2 matrix back to 2x3
    %transpose2_result = linalg.transpose ins(%transpose1_result: tensor<3x2xf32>) outs(%transpose2: tensor<2x3xf32>) permutation = [1, 0]
    
    %init2 = tensor.empty() : tensor<2x3xf32>

    %test = linalg.add ins(%transpose2_result, %transpose2_result: tensor<2x3xf32>, tensor<2x3xf32>)
                          outs(%init2: tensor<2x3xf32>) -> tensor<2x3xf32>
}

