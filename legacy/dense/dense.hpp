#pragma once

#include "tensor.hpp"

namespace dense
{
    template <typename T>
    Tensor<T> dense(const Tensor<T> &inputs, const Tensor<T> &kernel, const std::string &name = "")
    {
        try
        {
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
            int kernel_4d_channel = kernel_dim[3];

            std::vector<T> output;
            for (int i_batch = 0; i_batch < inputs_4d_batch; i_batch++)
            {
                for (int i_row = 0; i_row < inputs_4d_row; i_row++)
                {
                    for (int k_col = 0; k_col < kernel_4d_col; k_col++)
                    {
                        for (int i_channel = 0; i_channel < inputs_4d_channel; i_channel++)
                        {
                            T sum = T();
                            for (int i_col = 0; i_col < inputs_4d_col; i_col++)
                            {
                                T value = inputs_2d[i_batch][i_row][i_col][i_channel] * kernel_2d[i_batch][i_col][k_col][i_channel];
                                sum += value;
                            }
                        }
                        output.push_back(sum);
                    }
                }
            }

            Tensor<T> dense_output(name, {inputs_4d_batch, inputs_4d_row, inputs_4d_col, kernel_4d_channel}, output);
            return dense_output;
        }
        catch (const std::exception &e)
        {
            std::cerr << e.what() << '\n';
        }
        return Tensor<T>("");
    }
}