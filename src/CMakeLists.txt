cmake_minimum_required(VERSION 3.10)
set (CMAKE_CXX_STANDARD 17)

set(SRC_DIR ${CMAKE_SOURCE_DIR}/src/)
project(Operations
	VERSION 0.1
	DESCRIPTION "Operations"
	LANGUAGES CXX)
add_executable (run ${SRC_DIR}/main.cpp)

set(DIR
	${SRC_DIR}/operation/quantization
	${SRC_DIR}/operation/conv
	${SRC_DIR}/tensor)

include_directories(${DIR})

target_compile_options(run PUBLIC -Wall -Werror)
target_include_directories(run PUBLIC ${DIR})

