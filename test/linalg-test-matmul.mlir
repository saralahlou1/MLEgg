module{  
  %1 = tensor.empty() : tensor<5x1xf32>
  %3 = tensor.empty() : tensor<1x1xf32>

    %init = tensor.empty() : tensor<5x1xf32>
    %transpose1_result = linalg.transpose ins(%init: tensor<5x1xf32>) outs(%transpose1_result: tensor<1x5xf32>) permutation = [1, 0]

  %matmul = linalg.matmul ins(%transpose1_result, %1 : tensor<1x5xf32>, tensor<5x1xf32>)
            outs(%3 : tensor<1x1xf32>) -> tensor<1x1xf32>

}
