#ifndef SYCLTEN_KERNELEXPR_HPP
#define SYCLTEN_KERNELEXPR_HPP

#include <cmath>
#include <tuple>
#include <utility>

namespace sten {
namespace nullary_op {
struct rand_op {
  static auto op() { return rand(); }
};
}  // namespace nullary_op
namespace unary_op {
struct sin_op {
  template <class T>
  static auto op(T t)
  {
    return sin(t);
  }
};

struct cos_op {
  template <class T>
  static auto op(T t)
  {
    return sin(t);
  }
};
}  // namespace unary_op

#define BINARY_OPS_XMACRO \
  XMACRO(plus, +)         \
  XMACRO(minus, -)        \
  XMACRO(times, *)        \
  XMACRO(div, /)          \
  XMACRO(gt, >)           \
  XMACRO(lt, <)           \
  XMACRO(le, <=)          \
  XMACRO(ge, >=)          \
  XMACRO(eq, ==)          \
  XMACRO(neq, !=)         \
  XMACRO(logic_and, &&)   \
  XMACRO(logic_or, ||)

namespace binary_op {

#define XMACRO(name, op_symbol)   \
  struct name##_op {              \
    template <class T1, class T2> \
    static auto op(T1 a, T2 b)    \
    {                             \
      return a op_symbol b;       \
    }                             \
  };

BINARY_OPS_XMACRO
#undef XMACRO

};  // namespace binary_op

template <auto Value>
struct kernel_constant_operand {
  static constexpr int NUM_OPERANDS = 0;
  static constexpr auto execute(std::tuple<>) { return Value; }
};

struct kernel_operand_identity {
  static constexpr int NUM_OPERANDS = 1;
  template <class Operands>
  static constexpr auto execute(Operands ops)
  {
    return std::get<0>(ops);
  }
};

template <class Op, class... KExprs>
struct kernel_expression {
  static constexpr int NUM_EXPR_OPERANDS[sizeof...(KExprs) + 1] = {
      0, KExprs::NUM_OPERANDS...};
  static constexpr int NUM_OPERANDS = (KExprs::NUM_OPERANDS + ... + 0);

  template <size_t SubExprIdx, class Operands, size_t... Ids>
  static constexpr auto execute_subexpr(Operands ops,
                                        std::index_sequence<Ids...>)
  {
    using SubExpr =
        typename std::tuple_element<SubExprIdx, std::tuple<KExprs...>>::type;
    auto operands =
        std::make_tuple(std::get<Ids + NUM_EXPR_OPERANDS[SubExprIdx]>(ops)...);
    return SubExpr::execute(operands);
  }

  template <class Operands, size_t... ExprIndices>
  static constexpr auto execute_impl(Operands ops,
                                     std::index_sequence<ExprIndices...>)
  {
    return Op::op(execute_subexpr<ExprIndices>(
        ops,
        std::make_index_sequence<NUM_EXPR_OPERANDS[ExprIndices + 1]>())...);
  }

  template <class... Operands>
  static constexpr auto execute(std::tuple<Operands...> operands)
  {
    return execute_impl(operands,
                        std::make_index_sequence<sizeof...(KExprs)>());
  }
};

}  // namespace sten

#endif  // SYCLTEN_KERNELEXPR_HPP
