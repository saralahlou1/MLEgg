module {
    %0 = linalg.matmul ins(%lhs, %rhs: tensor<3x2xf32>, tensor<2x4xf32>)
                       outs(%huh: tensor<3x2xf32xf32>) -> tensor<3x4xf32>
}
