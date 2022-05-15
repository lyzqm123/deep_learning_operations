#pragma once

#include <vector>
#include <iostream>
#include <algorithm>
#include "../../tensor/tensor.hpp"
#include "../../tensor/quantized_tensor.hpp"

template <typename QType>
int get_bit()
{	
	int default_bit = 8;
	if (std::is_same<QType, uint8_t>::value)
	{
		default_bit = 8;
	}
	else if (std::is_same<QType, uint16_t>::value)
	{
		default_bit = 16;
	}
	else if (std::is_same<QType, float>::value)
	{
		default_bit = 32;
	}
	else if (std::is_same<QType, double>::value)
	{
		default_bit = 64;
	}
	else
	{
		std::string log_message = "[ERROR] This type is not supported yet...";
		throw std::runtime_error(log_message);
	}
	return default_bit;
}

template <typename QType, typename DQType>
QuantizedTensor<QType, DQType> quantization(const Tensor<QType> &weights, const std::string &name, QType min_value, QType max_value)
{
	try
	{
		QuantizedTensor<QType, DQType> quantized_tensor(name, weights.get_dimension());

		int bit = get_bit<DQType>();

		QType scale = (QType)((max_value - min_value) / (QType)((1 << bit) - 1));
		DQType offset = (DQType)((-min_value) / scale + 0.5);
		quantized_tensor.set_scale(scale);
		quantized_tensor.set_offset(offset);

		auto weights_vector = weights.get_tensor();
		for (const auto &weight : weights_vector)
		{
			DQType quantized_weight = (DQType)(weight / scale + (float)offset + 0.5);
			quantized_tensor.push_back(quantized_weight);
		}
		return quantized_tensor;
	}
	catch (const std::exception &e)
	{
		std::cerr << e.what() << '\n';
		exit(0);
	}
	return QuantizedTensor<QType, DQType>("");
}

template <typename QType, typename DQType>
QuantizedTensor<QType, DQType> quantization(const Tensor<QType> &weights, const std::string &name)
{

	try
	{
		const auto &weights_vector = weights.get_tensor();
		QType min_value = *std::min_element(weights_vector.begin(), weights_vector.end());
		QType max_value = *std::max_element(weights_vector.begin(), weights_vector.end());
		return quantization<QType, DQType>(weights, name, min_value, max_value);
	}
	catch (const std::exception &e)
	{
		std::cerr << e.what() << '\n';
	}
	return QuantizedTensor<QType, DQType>("");
}
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