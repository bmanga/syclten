#ifndef SYCLTEN_TENSOROPS_HPP
#define SYCLTEN_TENSOROPS_HPP

#include "tensorexpr.hpp"

namespace sten {
template <sten_expr Expr>
unary_expression<unary_op::sin_op, Expr> sin(Expr e)
{
  return {e};
}

template <sten_expr Expr1, sten_expr Expr2>
binary_expression<binary_op::plus_op, Expr1, Expr2> operator+(Expr1 e1,
                                                              Expr2 e2)
{
  return {e1, e2};
}

template <sten_expr Expr1, sten_expr Expr2>
binary_expression<binary_op::times_op, Expr1, Expr2> operator*(Expr1 e1,
                                                               Expr2 e2)
{
  return {e1, e2};
}

template <sten_expr Expr1, sten_expr Expr2>
binary_expression<binary_op::gt_op, Expr1, Expr2> operator>(Expr1 e1, Expr2 e2)
{
  return {e1, e2};
}

template <sten_expr Expr1, sten_expr Expr2>
binary_expression<binary_op::lt_op, Expr1, Expr2> operator<(Expr1 e1, Expr2 e2)
{
  return {e1, e2};
}

template <sten_expr Expr1, sten_expr Expr2>
binary_expression<binary_op::eq_op, Expr1, Expr2> operator==(Expr1 e1, Expr2 e2)
{
  return {e1, e2};
}

template <sten_expr Expr1, sten_expr Expr2>
binary_expression<binary_op::eq_op, Expr1, Expr2> operator!=(Expr1 e1, Expr2 e2)
{
  return {e1, e2};
}
}  // namespace sten
#endif  // SYCLTEN_TENSOROPS_HPP
