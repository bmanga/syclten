#ifndef SYCLTEN_TENSOR_HPP
#define SYCLTEN_TENSOR_HPP

#include <sten/types.hpp>
namespace sten {
namespace detail {
using tensor_storage = std::aligned_storage_t<128, 4>;
class sten_tensor;
}  // namespace detail
class tensor {
 public:
  template <unsigned Dimensions>
  tensor(const size<Dimensions> &dimensions, device_kind default_device)
      : tensor(get_count(dimensions), default_device)
  {
  }
  template <unsigned Dimensions>
  tensor(float *data,
         const size<Dimensions> &dimensions,
         device_kind default_device)
      : tensor(data, get_count(dimensions), default_device)
  {
  }
  explicit tensor(size_t num_elems, device_kind default_device);
  explicit tensor(float *data, size_t num_elems, device_kind default_device);
  ~tensor();

  bool is_default_gpu() const;
  bool is_default_cpu() const;

  bool is_available_on_cpu() const;
  bool is_available_on_gpu() const;

  size_t num_elems() const;
  float get_at(size_t idx);

  friend tensor operator+(tensor &a, tensor &b);
  friend tensor operator>(tensor &a, tensor &b);
  friend tensor operator<(tensor &a, tensor &b);

 private:
  tensor(const detail::sten_tensor &);
  detail::sten_tensor &get_impl();
  const detail::sten_tensor &get_impl() const;

 private:
  detail::tensor_storage m_storage;
};
}  // namespace sten

#endif  // SYCLTEN_TENSOR_HPP
