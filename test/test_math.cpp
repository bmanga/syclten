#include <doctest/doctest.h>

#include <sten/sten.hpp>

using namespace sten;

TEST_CASE("Rand")
{
  float zeros[] = {0, 0, 0, 0, 0, 0};
  float ones[] = {1, 1, 1, 1, 1, 1};
  auto tensor0 = sten::tensor(zeros, {6}, kCPU);
  auto tensor1 = sten::tensor(ones, {6}, kCPU);
  tensor result1 = sten::rand() / constant<RAND_MAX> >= tensor0 &&
                   sten::rand() / constant<RAND_MAX> <= tensor1;

  tensor result2 = sten::rand() / constant<RAND_MAX>> tensor1 ||
                   sten::rand() / constant<RAND_MAX> < tensor0;

  for (auto j = 0ul; j < 6; ++j) {
    CHECK(result1.get_at(j) == 1);
    CHECK(result2.get_at(j) == 0);
  }
}
