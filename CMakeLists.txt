cmake_minimum_required(VERSION 3.17)
project(syclten)

set(CMAKE_CXX_STANDARD 17)

find_package(hipSYCL CONFIG REQUIRED)


add_library(syclten src/tensor.cpp)
target_include_directories(syclten PUBLIC include)
target_link_libraries(syclten PRIVATE hipSYCL::hipSYCL-rt)

add_executable(testexe main.cpp)
target_link_libraries(testexe PUBLIC syclten)


add_subdirectory(test)

