module{  
  %1 = tensor.empty() : tensor<2xf32>
  %3 = tensor.empty() : tensor<f32>

    %init = tensor.empty() : tensor<1x2xf32>
    %transpose1_result = linalg.transpose ins(%init: tensor<1x2xf32>) outs(%transpose1_result: tensor<2x1xf32>) permutation = [1, 0]

  %2 = tensor.extract_slice %transpose1_result[0, 0] [2, 1] [1, 1] : tensor<2x1xf32> to tensor<2xf32>


  %dot = linalg.dot ins(%2, %1 : tensor<2xf32>, tensor<2xf32>)
            outs(%3 : tensor<f32>) -> tensor<f32>

}

