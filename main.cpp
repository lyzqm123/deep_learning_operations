#include "quantization.hpp"
#include <vector>
#include <iostream>

int main()
{
	Tensor<float> weights("weights", {3, 3}, "gaussian");
	Tensor<float> bias("bias", {1}, "ones");

	std::cout << "[Before quantization] <origin float32 weights>\n";
	std::cout << weights << "\n"
			  << bias << "\n";

	auto uint8_weights = quantization<float, uint8_t>(weights, "quantized_weights");
	std::cout << "[After quantization] <quantized uint8 weights>\n";
	std::cout << uint8_weights << "\n";

	auto dequantized_weights = dequantization<uint8_t, float>(uint8_weights, "dequantized_weights");
	std::cout << "[After dequantization] <dequantized float32 weights>\n";
	std::cout << dequantized_weights << "\n";
	return 0;
}
