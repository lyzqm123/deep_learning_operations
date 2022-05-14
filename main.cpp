#include "quantization.hpp"
#include "conv2d.hpp"
#include <vector>
#include <iostream>

void quantization_and_dequantization()
{
	using QType = float;
	using DQType = uint8_t;

	Tensor<QType> weights("weights",
						  {3, 3},
						  {-1.3731, -0.480495, -1.48554, -0.543328, 0.58587, 0.645656, 0.712967, -1.28438, 1.57578});
	Tensor<QType> bias("bias", {1}, "ones");

	std::cout << "[Before quantization] <origin weights>\n";
	std::cout << weights << "\n"
			  << bias << "\n";

	auto uint8_weights = quantization<QType, DQType>(weights, "quantized_weights");
	std::cout << "[After quantization] <quantized weights>\n";
	std::cout << uint8_weights << "\n";

	auto dequantized_weights = dequantization<DQType, QType>(uint8_weights, "dequantized_weights");
	std::cout << "[After dequantization] <dequantized weights>\n";
	std::cout << dequantized_weights << "\n";
}

int main()
{
	quantization_and_dequantization();
	return 0;
}
