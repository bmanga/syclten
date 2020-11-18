#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <sten/tensor.hpp>

using namespace sten;

TEST_CASE("Tensor constructor data")
{
  auto t1 = tensor({1, 2, 3}, kCPU);
  auto t2 = tensor(3, kGPU);

  float data[] = {1, 2, 3};
  auto t3 = tensor(data, {1, 3}, kCPU);
  auto t4 = tensor(data, 3, kGPU);

  SUBCASE("Device defaults")
  {
    CHECK(t1.is_default_cpu());
    CHECK(t2.is_default_gpu());
    CHECK(t3.is_default_cpu());
    CHECK(t4.is_default_gpu());
  }

  SUBCASE("Number of elements")
  {
    CHECK(t1.num_elems() == 1 * 2 * 3);
    CHECK(t3.num_elems() == 3);
  }

  SUBCASE("Intial data availability")
  {
    // No operations performed, it should be lazily allocated.
    CHECK(!t1.is_available_on_cpu());
    CHECK(!t1.is_available_on_gpu());

    // No operations performed, it should be lazily allocated.
    CHECK(!t2.is_available_on_cpu());
    CHECK(!t2.is_available_on_gpu());

    CHECK(t3.is_available_on_cpu());
    CHECK(!t3.is_available_on_gpu());

    CHECK(t3.is_available_on_cpu());
    CHECK(!t3.is_available_on_gpu());
  }
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

TEST_CASE("Tensor logical operations")
{
  float data1[] = {1, 2, 3, 4};
  float data2[] = {1, 3, 2, 4};
  auto t1cpu = tensor(data1, {4}, kCPU);
  auto t2cpu = tensor(data2, {4}, kCPU);
  auto t1gpu = tensor(data1, {4}, kGPU);
  auto t2gpu = tensor(data2, {4}, kGPU);
  SUBCASE("gt")
  {
    auto gt1cpu = t1cpu > t2cpu;
    CHECK(!gt1cpu.get_at(0));
    CHECK(!gt1cpu.get_at(1));
    CHECK(gt1cpu.get_at(2));
    CHECK(!gt1cpu.get_at(3));

    auto gt2gpu = t2gpu > t1gpu;
    CHECK(!gt2gpu.get_at(0));
    CHECK(gt2gpu.get_at(1));
    CHECK(!gt2gpu.get_at(2));
    CHECK(!gt2gpu.get_at(3));
  }
  SUBCASE("lt")
  {
    auto gt1cpu = t1cpu < t2cpu;
    CHECK(!gt1cpu.get_at(0));
    CHECK(gt1cpu.get_at(1));
    CHECK(!gt1cpu.get_at(2));
    CHECK(!gt1cpu.get_at(3));

    auto gt2gpu = t2gpu < t1gpu;
    CHECK(!gt2gpu.get_at(0));
    CHECK(!gt2gpu.get_at(1));
    CHECK(gt2gpu.get_at(2));
    CHECK(!gt2gpu.get_at(3));
  }
}
