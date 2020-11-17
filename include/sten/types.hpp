#ifndef SYCLTEN_TYPES_HPP
#define SYCLTEN_TYPES_HPP

#include <cstddef>
#include <utility>

namespace sten {
template <size_t N>
using size = int[N];

namespace detail {
template <size_t... Is>
size_t get_size_count_impl(const size<sizeof...(Is)> &dims,
                           std::index_sequence<Is...>)
{
  return (... * (dims[Is]));
}
}  // namespace detail

template <size_t N>
size_t get_count(const size<N> &sz)
{
  return detail::get_size_count_impl(sz, std::make_index_sequence<N>{});
}

enum device_kind { kCPU, kGPU };

}  // namespace sten

#endif  // SYCLTEN_TYPES_HPP
