#pragma once

#include "tensor.hpp"

template <typename Q, typename DQ>
class QuantizedTensor : public Tensor<Q>
{
public:
    QuantizedTensor(const std::string &name) : Tensor<Q>(name) {}
    QuantizedTensor(const std::string &name, const std::vector<int> &dimension) : Tensor<Q>(name, dimension) {}

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