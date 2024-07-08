// this was a simple exemple to check that writing back works for dot
// there is no optimisation to be applied
module{  
  %1 = tensor.empty() : tensor<2xf32>
  %2 = tensor.empty() : tensor<2xf32>
  %3 = tensor.empty() : tensor<f32>

  %dot = linalg.dot ins(%2, %1 : tensor<2xf32>, tensor<2xf32>)
            outs(%3 : tensor<f32>) -> tensor<f32>

}

