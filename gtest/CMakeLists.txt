set(BINARY ${CMAKE_PROJECT_NAME}_gtest)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR} )

file(GLOB_RECURSE TEST_SOURCES LIST_DIRECTORIES false *.hpp *.cpp)

include_directories(${CMAKE_SOURCE_DIR}/src/
                    ${CMAKE_SOURCE_DIR}/src/operation
                    ${CMAKE_SOURCE_DIR}/src/operation/quantization
                    ${CMAKE_SOURCE_DIR}/src/operation/conv
                    ${CMAKE_SOURCE_DIR}/src/tensor)

add_executable(${BINARY} ${TEST_SOURCES})
add_test(NAME ${BINARY} COMMAND ${BINARY})
target_link_libraries(${BINARY} PUBLIC ${CMAKE_PROJECT_NAME}_lib gtest pthread)
