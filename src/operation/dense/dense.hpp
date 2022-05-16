#pragma once

#include "tensor.hpp"

namespace dense
{
    template <typename T>
    Tensor<T> dense(const Tensor<T> &inputs, const Tensor<T> &kernel, const std::string &name = "")
    {
        try
        {
            const std::vector<std::vector<T>> inputs_2d = inputs.get_2d_tensor();
            const std::vector<std::vector<T>> kernel_2d = kernel.get_2d_tensor();

            int inputs_2d_row = inputs_2d.size();
            int inputs_2d_col = inputs_2d[0].size();
            int kernel_2d_row = kernel_2d.size();
            int kernel_2d_col = kernel_2d[0].size();

            Tensor<T> dense_output(name, kernel.get_dimension());
            for (int i_row = 0; i_row < inputs_2d_row; i_row++)
            {
                for (int k_col = 0; k_col < kernel_2d_col; k_col++)
                {
                    T sum = T();
                    for (int i_col = 0; i_col < inputs_2d_col; i_col++)
                    {
                        T value = inputs_2d[i_row][i_col] * kernel_2d[i_col][k_col];
                        sum += value;
                    }
                    dense_output.push_back(sum);
                }
            }
            return dense_output;
        }
        catch (const std::exception &e)
        {
            std::cerr << e.what() << '\n';
        }
        return Tensor<T>("");
    }
}