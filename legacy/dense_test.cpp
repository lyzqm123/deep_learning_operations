#include "gtest/gtest.h"
#include "dense.hpp"

TEST(denseTest, CheckDenseLogicIsFine)
{
    using TensorType = float;
    Tensor<TensorType> inputs("inputs",
                              {3, 3},
                              "ones");
    Tensor<TensorType> dense_weight("dense_weight", {3, 5},
                                    {-0.04568075, 0.00012321, -0.04472231, 0.04860416, 0.02075065,
                                     0.02035464, 0.00869691, -0.00689282, -0.01703328, -0.04589733,
                                     -0.0409387, -0.02341675, 0.0471049, 0.00569559, 0.0215294});

    Tensor<TensorType> dense_result("dense_result", {3, 5},
                                    {-0.06626481, -0.01459662, -0.00451023, 0.03726648, -0.00361728,
                                     -0.06626481, -0.01459662, -0.00451023, 0.03726648, -0.00361728,
                                     -0.06626481, -0.01459662, -0.00451023, 0.03726648, -0.00361728});

    auto dense_pred = dense::dense(inputs, dense_weight, "dense_pred");
    ASSERT_EQ(dense_pred.get_dimension(), dense_result.get_dimension());

    std::cout << dense_pred << "\n";
    std::cout << dense_result << "\n";

    auto dense_1d_pred = dense_pred.get_tensor();
    auto dense_1d_result = dense_result.get_tensor();
    for (int i = 0; i < (int)dense_1d_pred.size(); i++)
    {
        EXPECT_NEAR(dense_1d_pred[i], dense_1d_result[i], 0.00001);
    }
}