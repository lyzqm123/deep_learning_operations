#pragma once

#include "tensor.hpp"

template <typename Q, typename DQ>
class QuantizedTensor : public Tensor<DQ>
{
public:
    QuantizedTensor(const std::string &name) : Tensor<DQ>(name) {}
    QuantizedTensor(const std::string &name, const std::vector<int> &dimension) : Tensor<DQ>(name, dimension) {}
    QuantizedTensor(const std::string &name, const std::vector<int> &dimension, const std::string &distribution_type) : Tensor<DQ>(name, dimension, distribution_type) {}
    QuantizedTensor(const std::string &name, const std::vector<int> &dimension, const std::vector<DQ> &init_tensor) : Tensor<DQ>(name, dimension, init_tensor) {}
    QuantizedTensor(const std::string &name, const std::initializer_list<int> &dimension, const std::initializer_list<DQ> &init_tensor) : Tensor<DQ>(name, dimension, init_tensor) {}

    void set_scale(Q scale)
    {
        this->scale_ = scale;
    }
    void set_offset(DQ offset)
    {
        this->offset_ = offset;
    }
    Q get_scale() const
    {
        return this->scale_;
    }
    DQ get_offset() const
    {
        return this->offset_;
    }

private:
    Q scale_ = (Q)0.0;
    DQ offset_ = (DQ)0;
};