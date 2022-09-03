# Fast deep learning inference (C++, Python)
- **Requirement**
  - CMake (https://cmake.org/)
  - GoogleTest (https://github.com/google/googletest)
  - pybind11 (https://github.com/pybind/pybind11)
  - Eigen (https://eigen.tuxfamily.org/index.php?title=Main_Page)

- **Build**
  1. `git clone https://github.com/lyzqm123/fast_deep_learning_inference.git`
  2. `cd fast_deep_learning_inference/src`
  3. `git clone https://github.com/pybind/pybind11.git`
  4. `cd ../` 
  5. `cmake CMakeLists.txt`
  6. `make`
<br></br>

- **Supported operations**
  |Operation|Location|
  |:--:|:--:|
  |Matrix multiplication|[src/tensor/tensor.hpp](https://github.com/lyzqm123/fast_deep_learning_inference/blob/master/src/tensor/tensor.hpp#L278-L307) |
  |Conv2D|[src/operation/conv/conv2d.hpp](https://github.com/lyzqm123/deep_learning_operations/blob/master/src/operation/conv/conv2d.hpp)|
  |Quantization|[src/operation/quantization/quantization.hpp](https://github.com/lyzqm123/deep_learning_operations/blob/master/src/operation/quantization/quantization.hpp)|
  |DeQuantization|[src/operation/quantization/dequantization.hpp](https://github.com/lyzqm123/deep_learning_operations/blob/master/src/operation/quantization/dequantization.hpp)|


- **Google test**
  - `./dl_operations_gtest`
  - ![unittest](https://user-images.githubusercontent.com/22426868/188266006-53435974-7856-483f-925a-3379555a6ff0.png)


- **Pybind module**
  - If the build succeeds, you can import the "fastinference" module.
  - ```
    import fastinference
        
    input = fastinference.TensorFloat("input", (3, 3), (1, 2, 3, 4, 5, 6, 7, 8, 9))
    feature = fastinference.TensorFloat("feature", (3, 3), "ones")  
    output = fastinference.Conv2dFloat(input, feature, "output")
    
    print(output.get_tensor())  # [[[[45.0]]]]
    ```
