cmake_minimum_required(VERSION 3.17)
project(syclten)

set(CMAKE_CXX_STANDARD 20)

find_package(hipSYCL CONFIG REQUIRED)

add_library(syclten INTERFACE)
target_include_directories(syclten INTERFACE include)
target_link_libraries(syclten INTERFACE hipSYCL::hipSYCL-rt)

add_executable(testexe main.cpp)
target_link_libraries(testexe PUBLIC syclten)


add_subdirectory(test)

