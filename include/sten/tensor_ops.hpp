#ifndef SYCLTEN_TENSOROPS_INC
#define SYCLTEN_TENSOROPS_INC

#include "tensorexpr.hpp"

namespace sten {
template <sten_expr Expr>
unary_expression<unary_op::sin_op, Expr> sin(Expr e)
{
  return {e};
}

template <sten_expr Expr>
unary_expression<unary_op::cos_op, Expr> cos(Expr e)
{
  return {e};
}

#define XMACRO(name, op_symbol)                                             \
  template <sten_expr Expr1, sten_expr Expr2>                               \
  binary_expression<binary_op::name##_op, Expr1, Expr2> operator op_symbol( \
      Expr1 e1, Expr2 e2)                                                   \
  {                                                                         \
    return {e1, e2};                                                        \
  }
BINARY_OPS_XMACRO
#undef XMACRO

}  // namespace sten
#endif  // SYCLTEN_TENSOROPS_INC
