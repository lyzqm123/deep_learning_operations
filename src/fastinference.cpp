#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "tensor.hpp"
#include "conv2d.hpp"

namespace py = pybind11;

template <typename T>
void declare_tensor(py::module &m, std::string pyclass_name)
{
    using Class = Tensor<T>;
    using TensorArray = std::vector<T>;
    py::class_<Class>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
        .def(py::init<const std::string &>())
        .def(py::init<const std::string &, const std::vector<int> &, const TensorArray &>())
        .def(py::init<const std::string &, const std::vector<int> &, const std::string &>())
        .def("get_dimension", &Class::get_dimension)
        .def("get_tensor", &Class::get_tensor)
        .def("get_name", &Class::get_name)
        .def("size", &Class::size);
}

PYBIND11_MODULE(fastinference, m)
{
    declare_tensor<float>(m, "TensorFloat");

    m.def("Conv2dFloat", &conv::conv2d<float>);
    m.def("Conv2dInt", &conv::conv2d<int>);
}

