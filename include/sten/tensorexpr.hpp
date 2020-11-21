#ifndef SYCLTEN_TENSOREXPR_HPP
#define SYCLTEN_TENSOREXPR_HPP

#include "ops/kernelexpr.hpp"
#include "tensor.hpp"

#include <concepts>
#include <tuple>
#include <utility>

namespace sten {
using tensor_expr = tensor;

template <auto Value>
struct constant_t {
  static constexpr bool IS_CONSTANT_T = true;
};

template <auto Value>
auto constant = constant_t<Value>{};

template <class>
struct is_constant : public std::false_type {
};

template <auto Value>
struct is_constant<constant_t<Value>> : public std::true_type {
  static constexpr decltype(Value) constant = Value;
};

namespace detail {
template <class Expr>
constexpr auto collect_tensors(Expr e)
{
  if constexpr (is_constant<Expr>::value) {
    return std::tuple{};
  }
  else if constexpr (std::is_same_v<Expr, tensor_expr>) {
    return std::tuple{e};
  }
  else
    return e.collect_tensors();
}

template <class Expr>
constexpr auto get_kernel()
{
  if constexpr (is_constant<Expr>::value) {
    return kernel_constant_operand<is_constant<Expr>::constant>{};
  }
  else if constexpr (std::is_same_v<Expr, tensor_expr>) {
    return kernel_operand_identity{};
  }
  else
    return Expr::get_kernel();
}
}  // namespace detail

template <class Op>
struct nullary_expression {
  auto collect_tensors() { return std::tuple{}; }
  static constexpr auto get_kernel() { return kernel_expression<Op>{}; }
};

template <class Op, class Expr>
struct unary_expression {
  Expr e;

  auto collect_tensors() { return detail::collect_tensors(e); }

  static constexpr auto get_kernel()
  {
    return kernel_expression<Op, decltype(detail::get_kernel<Expr>())>{};
  }
};

template <class Op, class Expr1, class Expr2>
struct binary_expression {
  Expr1 e1;
  Expr2 e2;

  auto collect_tensors()
  {
    return std::tuple_cat(detail::collect_tensors(e1),
                          detail::collect_tensors(e2));
  }

  static constexpr auto get_kernel()
  {
    return kernel_expression<Op, decltype(detail::get_kernel<Expr1>()),
                             decltype(detail::get_kernel<Expr2>())>{};
  }
};

template <class T, template <class...> class U>
concept instance_of = requires(T x)
{
  {
    U { x }
  }
  ->std::same_as<T>;
};

template <class T>
concept sten_expr =
    std::is_same_v<T, tensor_expr> || is_constant<T>::value ||
    instance_of<T, nullary_expression> || instance_of<T, unary_expression> ||
    instance_of<T, binary_expression>;

}  // namespace sten

#endif  // SYCLTEN_TENSOREXPR_HPP
