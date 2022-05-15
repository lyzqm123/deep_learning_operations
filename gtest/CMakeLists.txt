set(BINARY ${CMAKE_PROJECT_NAME}_gtest)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR} )

file(GLOB_RECURSE TEST_SOURCES LIST_DIRECTORIES false *.hpp *.cpp)
set(SOURCES ${TEST_SOURCES})

add_executable(${BINARY} ${TEST_SOURCES})
add_test(NAME ${BINARY} COMMAND ${BINARY})
target_link_libraries(${BINARY} PUBLIC ${CMAKE_PROJECT_NAME}_lib gtest pthread)
