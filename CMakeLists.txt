cmake_minimum_required(VERSION 3.10)
set (CMAKE_CXX_STANDARD 17)

project(dl_operations
	VERSION 0.1
	DESCRIPTION "Deep learning operations"
	LANGUAGES CXX)
	
add_subdirectory(src)
add_subdirectory(gtest)