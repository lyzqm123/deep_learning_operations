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
  |Quantization|[src/operation/quantization/quantization.hpp](https://github.com/lyzqm123/deep_learning_operations/blob/master/src/operation/quantization/quantization.hpp)|


- **Google test**
  - `./dl_operations_gtest` 
  - ![unittest](https://user-images.githubusercontent.com/22426868/168457065-c2a873e7-8bf9-4a2f-a403-48f4c832fec4.png)
