#pragma once

#include "quantized_tensor.hpp"

template <typename DQType, typename QType>
Tensor<QType> dequantization(const QuantizedTensor<QType, DQType> &quantized_weights, const std::string &name)
{
	try
	{
		Tensor<QType> dequantized_tensor(name, quantized_weights.get_dimension());

		QType scale = quantized_weights.get_scale();
		DQType offset = quantized_weights.get_offset();

		QType min_value = -(QType)(offset)*scale;
		auto weights_vector = quantized_weights.get_tensor();
		for (const auto &weight : weights_vector)
		{
			QType dequantized_weight = (DQType)(weight)*scale + min_value;
			dequantized_tensor.push_back(dequantized_weight);
		}
		return dequantized_tensor;
	}
	catch (const std::exception &e)
	{
		std::cerr << e.what() << '\n';
	}
	return Tensor<QType>("");
}