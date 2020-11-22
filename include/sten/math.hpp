#ifndef SYCLTEN_MATH_HPP
#define SYCLTEN_MATH_HPP

#include "tensorexpr.hpp"

namespace sten {
inline nullary_expression<nullary_op::rand_op> rand()
{
  return {};
}
}  // namespace sten

#endif  // SYCLTEN_MATH_HPP
