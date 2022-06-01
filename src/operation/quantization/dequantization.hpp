#pragma once

#include "quantized_tensor.hpp"

namespace quantization
{
    template <typename DQType, typename QType>
    Tensor<QType> dequantization(const QuantizedTensor<QType, DQType> &quantized_weights, const std::string &name)
    {
        try
        {

            QType scale = quantized_weights.get_scale();
            DQType offset = quantized_weights.get_offset();

            QType min_value = -(QType)(offset)*scale;
            const auto &weights_vector = quantized_weights.get_serialized_tensor();

            std::vector<QType> output;
            for (const auto &weight : weights_vector)
            {
                QType dequantized_weight = (DQType)(weight)*scale + min_value;
                output.push_back(dequantized_weight);
            }
            Tensor<QType> dequantized_tensor(name, quantized_weights.get_dimension(), output);
            return std::move(dequantized_tensor);
        }
        catch (const std::exception &e)
        {
            std::cerr << e.what() << '\n';
        }
        return Tensor<QType>("");
    }
};