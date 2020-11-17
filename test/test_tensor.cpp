#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <sten/tensor.hpp>

using namespace sten;

TEST_CASE("Tensor constructors")
{
  auto t1 = tensor({1, 2, 3}, kCPU);
  float data[] = {1, 2, 3};
  auto t2 = tensor(data, {1, 3}, kGPU);
  auto t3 = tensor(3, kCPU);
  auto t4 = tensor(data, 3, kGPU);

  CHECK(t1.is_cpu());
  CHECK(t1.num_elems() == 1 * 2 * 3);

  CHECK(t2.is_gpu());
  CHECK(t2.num_elems() == 3);

  CHECK(t3.is_cpu());
  CHECK(t4.is_gpu());
}

TEST_CASE("Tensor math operations")
{
  SUBCASE("sum")
  {
    float data1[] = {1, 2, 3, 4};
    float data2[] = {4, 3, 2, 1};
    auto t1cpu = tensor(data1, {4}, kCPU);
    auto t2cpu = tensor(data2, {4}, kCPU);
    auto sumcpu = t1cpu + t2cpu;
    CHECK(sumcpu.get_at(0) == 5);
    CHECK(sumcpu.get_at(1) == 5);
    auto t1gpu = tensor(data1, {4}, kGPU);
    auto t2gpu = tensor(data2, {4}, kGPU);
    auto sumgpu = t1gpu + t2gpu;
    CHECK(sumgpu.get_at(2) == 5);
    CHECK(sumgpu.get_at(3) == 5);
  }
}
