#pragma once

#include "tensor.hpp"

namespace conv{
template<typename T>
Tensor<T> conv2d(const Tensor<T> &weights, const Tensor<T> &filters, std::string name="") {
    try
    {
        Tensor<T> conv2d_out(name);
        const std::vector<std::vector<T>> weights_2d = weights.get_2d_tensor();
        const std::vector<std::vector<T>> filters_2d = filters.get_2d_tensor();
        return conv2d_out;
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
    
    return Tensor<T>("");
}
};