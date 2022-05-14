cmake_minimum_required(VERSION 3.10)
set (CMAKE_CXX_STANDARD 17)

project(Operations
	VERSION 0.1
	DESCRIPTION "Operations"
	LANGUAGES CXX)
add_executable (run main.cpp)

set(DIR
	${CMAKE_SOURCE_DIR}/operation/quantization
	${CMAKE_SOURCE_DIR}/tensor)

include_directories(${DIR})

target_compile_options(run PUBLIC -Wall -Werror)
target_include_directories(run PUBLIC ${DIR})

