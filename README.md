# Deep learning operations (C++)
- **Requirement**
  - CMake (https://cmake.org/)
  - GoogleTest (https://github.com/google/googletest)

- **Build**
  1. `git clone https://github.com/lyzqm123/deep_learning_operations.git`
  2. `cd deep_learning_operations`
  3. `cmake CMakeLists.txt`
  4. `make`
<br></br>

- **Supported operations**
  |Operation|Location|
  |:--:|:--:|
  |Conv2D|[src/operation/conv/conv2d.hpp](https://github.com/lyzqm123/deep_learning_operations/blob/master/src/operation/conv/conv2d.hpp)|
  |Dense| [src/operation/dense/dense.hpp](https://github.com/lyzqm123/deep_learning_operations/blob/master/src/operation/dense/dense.hpp)|
  |Quantization|[src/operation/quantization/quantization.hpp](https://github.com/lyzqm123/deep_learning_operations/blob/master/src/operation/quantization/quantization.hpp)|
  |DeQuantization|[src/operation/quantization/dequantization.hpp](https://github.com/lyzqm123/deep_learning_operations/blob/master/src/operation/quantization/dequantization.hpp)|


- **Google test**
  - `./dl_operations_gtest` 
  - ![unittest](https://user-images.githubusercontent.com/22426868/168610945-c180ed16-7519-480f-b99d-34be85930c6e.png)
