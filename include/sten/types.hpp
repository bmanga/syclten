#ifndef SYCLTEN_TYPES_HPP
#define SYCLTEN_TYPES_HPP

#include <cstddef>
#include <numeric>
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

class dimensions {
 public:
  dimensions(std::initializer_list<size_t> dims) : m_dims(dims)
  {
    m_num_elems =
        std::accumulate(m_dims.begin(), m_dims.end(), 1, std::multiplies<>{});
  }
  dimensions(size_t num_elems) : dimensions({num_elems}) {}

  size_t num_elems() const { return m_num_elems; }

 private:
  std::vector<size_t> m_dims;
  size_t m_num_elems;
};

}  // namespace sten

#endif  // SYCLTEN_TYPES_HPP
