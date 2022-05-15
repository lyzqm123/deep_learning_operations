cmake_minimum_required(VERSION 3.10)
set(BINARY ${CMAKE_PROJECT_NAME})
set(CMAKE_CXX_STANDARD 17)
set(SRC_DIR ${CMAKE_SOURCE_DIR}/src/)
set(LIBRARY_OUTPUT_PATH ${CMAKE_SOURCE_DIR} )

set(DIR
	${SRC_DIR}/operation/quantization
	${SRC_DIR}/operation/conv
	${SRC_DIR}/tensor)
set(SRC_FILES
    ${SRC_DIR}/operation/quantization/quantization.hpp
    ${SRC_DIR}/operation/conv/conv2d.hpp
    ${SRC_DIR}/tensor/tensor.hpp
    ${SRC_DIR}/tensor/quantized_tensor.hpp
    ${SRC_DIR}/main.cpp
)
include_directories(${DIR})

add_executable (${BINARY} ${SRC_FILES})
add_library(${BINARY}_lib STATIC ${SRC_FILES})

target_compile_options(${BINARY} PUBLIC -Wall -Werror)
target_include_directories(${BINARY} PUBLIC ${DIR})