#pragma once

#include "tensor.hpp"

namespace conv
{
    template <typename T>
    Tensor<T> conv2d(const Tensor<T> &inputs, const Tensor<T> &kernel, const std::string &name = "")
    {
        try
        {
            const int padding = 0, stride = 1;
            const std::vector<std::vector<T>> inputs_2d = inputs.get_2d_tensor();
            const std::vector<std::vector<T>> kernel_2d = kernel.get_2d_tensor();

            int inputs_2d_row = inputs_2d.size();
            int inputs_2d_col = inputs_2d[0].size();
            int kernel_2d_row = kernel_2d.size();
            int kernel_2d_col = kernel_2d[0].size();

            int output_dimension = (inputs_2d_row + 2 * padding - kernel_2d_row) / stride + 1;
            Tensor<T> conv2d_output(name, {output_dimension, output_dimension});
            for (int i_row = 0; i_row <= inputs_2d_row - kernel_2d_row; i_row += stride)
            {
                for (int i_col = 0; i_col <= inputs_2d_col - kernel_2d_col; i_col += stride)
                {
                    T sum = 0;
                    for (int k_row = 0; k_row < kernel_2d_row; k_row++)
                    {
                        for (int k_col = 0; k_col < kernel_2d_col; k_col++)
                        {
                            T value = inputs_2d[i_row + k_row][i_col + k_col] * kernel_2d[k_row][k_col];
                            sum += value;
                        }
                    }
                    conv2d_output.push_back(sum);
                }
            }
            return conv2d_output;
        }
        catch (const std::exception &e)
        {
            std::cerr << e.what() << '\n';
        }

        return Tensor<T>("");
    }
};