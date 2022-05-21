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
            const auto &inputs_4d = inputs.get_tensor();
            const auto &inputs_dim = inputs.get_dimension();

            const auto &kernel_4d = kernel.get_tensor();
            const auto &kernel_dim = kernel.get_dimension();

            int inputs_4d_batch = inputs_dim[0];
            int inputs_4d_row = inputs_dim[1];
            int inputs_4d_col = inputs_dim[2];
            int inputs_4d_channel = inputs_dim[3];

            int kernel_4d_row = kernel_dim[1];
            int kernel_4d_col = kernel_dim[2];

            std::vector<T> output;
            for (int i_batch = 0; i_batch < inputs_4d_batch; i_batch++)
            {
                for (int i_row = 0; i_row <= inputs_4d_row - kernel_4d_row; i_row += stride)
                {
                    for (int i_col = 0; i_col <= inputs_4d_col - kernel_4d_col; i_col += stride)
                    {
                        T sum = 0;
                        for (int i_channel = 0; i_channel < inputs_4d_channel; i_channel++)
                        {
                            for (int k_row = 0; k_row < kernel_4d_row; k_row++)
                            {
                                for (int k_col = 0; k_col < kernel_4d_col; k_col++)
                                {
                                    T value = inputs_4d[i_batch][i_row + k_row][i_col + k_col][i_channel] * kernel_4d[i_batch][k_row][k_col][i_channel];
                                    sum += value;
                                }
                            }
                        }
                        output.push_back(sum);
                    }
                }
            }
            int output_dimension = (inputs_4d_row + 2 * padding - kernel_4d_row) / stride + 1;
            Tensor<T> conv2d_output(name, {inputs_4d_batch, output_dimension, output_dimension, 1}, output);
            return conv2d_output;
        }
        catch (const std::exception &e)
        {
            std::cerr << e.what() << '\n';
        }

        return Tensor<T>("");
    }
};