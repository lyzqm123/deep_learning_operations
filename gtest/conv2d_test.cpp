#include "gtest/gtest.h"
#include "conv2d.hpp"

TEST(conv2dTest, CheckConv2DOutputDimensionIsFine)
{
    using TensorType = float;
    Tensor<TensorType> inputs("inputs",
                              {3, 3},
                              {-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0});
    Tensor<TensorType> kernel_3d("kernel_3d", {3, 3}, "ones");
    Tensor<TensorType> conv2d_3d_3d_output = conv::conv2d(inputs, kernel_3d, "conv2d_3d_3d_output");

    const std::vector<int> conv2d_3d_3d_output_dimension = conv2d_3d_3d_output.get_dimension();
    const std::vector<int> conv2d_3d_3d_output_dimension_answer = {1, 1, 1, 1};
    EXPECT_EQ(conv2d_3d_3d_output_dimension, conv2d_3d_3d_output_dimension_answer);

    Tensor<TensorType> kernel_1d("kernel_1d", {1, 1}, "ones");
    Tensor<TensorType> conv2d_3d_1d_output = conv::conv2d(inputs, kernel_1d, "conv2d_3d_1d_output");
    const std::vector<int> conv2d_3d_1d_output_dimension = conv2d_3d_1d_output.get_dimension();
    const std::vector<int> conv2d_3d_1d_output_dimension_answer = {1, 3, 3, 1};
    EXPECT_EQ(conv2d_3d_1d_output_dimension, conv2d_3d_1d_output_dimension_answer);
}