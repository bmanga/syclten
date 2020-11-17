# For conan's doctest
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_BINARY_DIR})

find_package(doctest CONFIG)

add_executable(testsuite test_tensor.cpp)
target_link_libraries(testsuite PUBLIC syclten doctest::doctest)
