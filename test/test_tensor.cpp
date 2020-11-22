#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <sten/sten.hpp>

using namespace sten;

static auto cpu = kCPU;
// There is a bug in clang preventing the use of C++20 with
// CUDA. Until that is fixed, run all tests on CPU.
static auto gpu = kCPU;

TEST_CASE("Tensor constructor data")
{
  auto t1 = tensor({1, 2, 3}, cpu);
  auto t2 = tensor(3, gpu);

  float data[] = {1, 2, 3};
  auto t3 = tensor(data, {1, 3}, cpu);
  auto t4 = tensor(data, 3, gpu);

  SUBCASE("Device defaults")
  {
    CHECK(t1.is_default_cpu());
    // CHECK(t2.is_default_gpu());
    CHECK(t3.is_default_cpu());
    // CHECK(t4.is_default_gpu());
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
    auto t1cpu = tensor(data1, {4}, cpu);
    auto t2cpu = tensor(data2, {4}, cpu);
    tensor sumcpu = t1cpu + t2cpu;
    CHECK(sumcpu.get_at(0) == 5);
    CHECK(sumcpu.get_at(1) == 5);
    auto t1gpu = tensor(data1, {4}, gpu);
    auto t2gpu = tensor(data2, {4}, gpu);
    tensor sumgpu = t1gpu + t2gpu;
    CHECK(sumgpu.get_at(2) == 5);
    CHECK(sumgpu.get_at(3) == 5);
  }
}

TEST_CASE("Tensor logical operations")
{
  float data1[] = {1, 2, 3, 4};
  float data2[] = {1, 3, 2, 4};
  auto t1cpu = tensor(data1, {4}, cpu);
  auto t2cpu = tensor(data2, {4}, cpu);
  auto t1gpu = tensor(data1, {4}, gpu);
  auto t2gpu = tensor(data2, {4}, gpu);
  SUBCASE("gt")
  {
    tensor gt1cpu = t1cpu > t2cpu;
    CHECK(!gt1cpu.get_at(0));
    CHECK(!gt1cpu.get_at(1));
    CHECK(gt1cpu.get_at(2));
    CHECK(!gt1cpu.get_at(3));

    tensor gt2gpu = t2gpu > t1gpu;
    CHECK(!gt2gpu.get_at(0));
    CHECK(gt2gpu.get_at(1));
    CHECK(!gt2gpu.get_at(2));
    CHECK(!gt2gpu.get_at(3));
  }
  SUBCASE("lt")
  {
    tensor gt1cpu = t1cpu < t2cpu;
    CHECK(!gt1cpu.get_at(0));
    CHECK(gt1cpu.get_at(1));
    CHECK(!gt1cpu.get_at(2));
    CHECK(!gt1cpu.get_at(3));

    tensor gt2gpu = t2gpu < t1gpu;
    CHECK(!gt2gpu.get_at(0));
    CHECK(!gt2gpu.get_at(1));
    CHECK(gt2gpu.get_at(2));
    CHECK(!gt2gpu.get_at(3));
  }
}

TEST_CASE("Tensor expressions")
{
  float data1[] = {1, 2, 3, 4};
  float data2[] = {1, 3, 2, 4};
  auto t1 = tensor(data1, {4}, cpu);
  auto t2 = tensor(data2, {4}, cpu);

  auto expr = (sin(t1) + sin(t2)) > (sin(t1 + t2));

  float expected_data[] = {1, 1, 1, 0};
  auto expected = tensor(expected_data, {4}, cpu);
  tensor actual1 = expr;
  tensor actual2 = expr == expected;
  for (auto j = 0ul; j < 4; ++j) {
    CHECK(actual1.get_at(j) == expected_data[j]);
    CHECK(actual2.get_at(j) == 1);
  }
}
