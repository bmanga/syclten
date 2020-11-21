#ifndef SYCLTEN_TENSOR_HPP
#define SYCLTEN_TENSOR_HPP

#include "ops/dispatch.hpp"
#include "ops/kernels.hpp"

#include <CL/sycl.hpp>

namespace sten {

class tensor {
  template <class Kernel, class KernelName, class Tensor0, class... Tensors>
  friend void sten::dispatch(device, Tensor0 &, Tensors &...);

 public:
  tensor(float *data, dimensions dims, device default_device)
      : m_buffer(data, cl::sycl::range<1>(dims.num_elems())),
        m_default_device(default_device)
  {
    m_available_on_cpu = true;
  }

  tensor(dimensions dims, device default_device)
      : m_buffer(sycl::range<1>(dims.num_elems())),
        m_default_device(default_device)
  {
  }
  bool is_default_gpu() const { return m_default_device.is_gpu(); }
  bool is_default_cpu() const { return m_default_device.is_host(); }

  sten::device get_default_device() const { return m_default_device; }

  sycl::buffer<float, 1> &get_buffer() { return m_buffer; }

  const sycl::buffer<float, 1> &get_buffer() const { return m_buffer; }

  float get_at(size_t idx)
  {
    float result = get_single(*this, idx);
    return result;
  }

  template <class TensorExpr>
  tensor(TensorExpr expr)
      : tensor(std::get<0>(expr.collect_tensors()).num_elems(),
               std::get<0>(expr.collect_tensors()).get_default_device())
  {
    dispatch_expression(get_default_device(), *this, expr);
  }

  size_t num_elems() const { return m_buffer.get_count(); }
  bool is_available_on_cpu() const { return m_available_on_cpu; }
  bool is_available_on_gpu() const { return m_avaiable_on_gpu; }

 private:
  void set_available_on(device device)
  {
    m_available_on_cpu |= device.is_cpu();
    m_avaiable_on_gpu |= device.is_gpu();
  }

 private:
  sycl::buffer<float, 1> m_buffer;
  sten::device m_default_device;
  bool m_available_on_cpu = false;
  bool m_avaiable_on_gpu = false;
};

static_assert(sizeof(tensor) <= 128);

}  // namespace sten

#include "tensorops.inc"

#endif  // SYCLTEN_TENSOR_HPP
