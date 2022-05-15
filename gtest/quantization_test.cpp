#include "gtest/gtest.h"
#include "quantization.hpp"

TEST(fp32_to_uint8_quantization_and_dequantization_test, CheckQuantizationDequantizationLogic_Fp32ToUint8_Gaussian)
{
    using QType = float;
    using DQType = uint8_t;

    Tensor<QType> weights("weights",
                          {3, 3},
                          "gaussian");

    std::cout << "[Before quantization] <origin weights>\n";
    std::cout << weights << "\n";

    const QuantizedTensor<QType, DQType> uint8_weights = quantization<QType, DQType>(weights, "quantized_weights");
    const auto uint8_weights_vector = uint8_weights.get_tensor();
    for (const auto &uint8_weight : uint8_weights_vector){
        EXPECT_TRUE(0 <= uint8_weight && uint8_weight <= 255);
    }
    std::cout << "[After quantization] <quantized weights>\n";
    std::cout << uint8_weights << "\n";
    
    Tensor<QType> dequantized_weights = dequantization<DQType, QType>(uint8_weights, "dequantized_weights");
    std::cout << "[After dequantization] <dequantized weights>\n";
    std::cout << dequantized_weights << "\n";
}

TEST(fp32_to_uint8_quantization_and_dequantization_test, CheckQuantizationDequantizationLogic_Fp32ToUint8_HugeNumber)
{
    using QType = float;
    using DQType = uint8_t;

    Tensor<QType> weights("weights",
                          {3, 3},
                          {371.2f, 60.3f, 80.4f, 20.1f, 0.5f, -20.2f, -277.7f, -1.f, 3.7f});

    std::cout << "[Before quantization] <origin weights>\n";
    std::cout << weights << "\n";

    const QuantizedTensor<QType, DQType> uint8_weights = quantization<QType, DQType>(weights, "quantized_weights");
    const auto uint8_weights_vector = uint8_weights.get_tensor();
    for (const auto &uint8_weight : uint8_weights_vector){
        EXPECT_TRUE(0 <= uint8_weight && uint8_weight <= 255);
    }
    std::cout << "[After quantization] <quantized weights>\n";
    std::cout << uint8_weights << "\n";
    
    Tensor<QType> dequantized_weights = dequantization<DQType, QType>(uint8_weights, "dequantized_weights");
    std::cout << "[After dequantization] <dequantized weights>\n";
    std::cout << dequantized_weights << "\n";
}