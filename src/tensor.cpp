#include <sten/tensor.hpp>

#include "device.hpp"
#include "ops/dispatch.hpp"
#include "ops/kernels.hpp"

#include <CL/sycl.hpp>

namespace sten {
namespace detail {
class sten_tensor {
  template <class Kernel, class Tensor0, class Tensor1, class Tensor2>
  friend void sten::dispatch(device, Tensor0 &, Tensor1 &, Tensor2 &);

 public:
  sten_tensor(float *data, size_t num_elems, device default_device)
      : m_buffer(data, cl::sycl::range<1>(num_elems)),
        m_default_device(default_device)
  {
    m_available_on_cpu = true;
  }

  sten_tensor(size_t num_elems, device default_device)
      : m_buffer(sycl::range<1>(num_elems)), m_default_device(default_device)
  {
  }
  bool is_default_gpu() const { return m_default_device.is_gpu(); }
  bool is_default_cpu() const { return m_default_device.is_host(); }

  sten::device get_default_device() const { return m_default_device; }

  sycl::buffer<float, 1> &get_buffer() { return m_buffer; }

  const sycl::buffer<float, 1> &get_buffer() const { return m_buffer; }

  sten_tensor sum(sten_tensor &b)
  {
    auto device = get_default_device();
    sten_tensor result(get_buffer().get_count(), device);
    dispatch<sten::SumKernel>(device, result, *this, b);
    return result;
  }

  sten_tensor gt(sten_tensor &b)
  {
    auto device = get_default_device();
    sten_tensor result(get_buffer().get_count(), device);
    dispatch<sten::GtKernel>(device, result, *this, b);
    return result;
  }

  sten_tensor lt(sten_tensor &b)
  {
    auto device = get_default_device();
    sten_tensor result(get_buffer().get_count(), device);
    dispatch<sten::GtKernel>(device, result, b, *this);
    return result;
  }

  float get_at(size_t idx)
  {
    float result = get_single(*this, idx);
    return result;
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

static_assert(sizeof(sten_tensor) <= 128);
}  // namespace detail

tensor::tensor(size_t num_elems, device_kind default_device)
{
  new (&m_storage)
      detail::sten_tensor(num_elems, sten::get_device(default_device));
}

tensor::tensor(float *data, size_t num_elems, device_kind default_device)
{
  new (&m_storage)
      detail::sten_tensor(data, num_elems, sten::get_device(default_device));
}

tensor::tensor(const detail::sten_tensor &other)
{
  new (&m_storage)
      detail::sten_tensor(other.num_elems(), other.get_default_device());
  get_impl() = other;
}

tensor::~tensor()
{
  get_impl().~sten_tensor();
}

detail::sten_tensor &tensor::get_impl()
{
  return *reinterpret_cast<detail::sten_tensor *>(&m_storage);
}

bool tensor::is_default_gpu() const
{
  return get_impl().is_default_gpu();
}

bool tensor::is_default_cpu() const
{
  return get_impl().is_default_cpu();
}

tensor operator+(tensor &a, tensor &b)
{
  return a.get_impl().sum(b.get_impl());
}
tensor operator>(tensor &a, tensor &b)
{
  return a.get_impl().gt(b.get_impl());
}
tensor operator<(tensor &a, tensor &b)
{
  return a.get_impl().lt(b.get_impl());
}

float tensor::get_at(size_t idx)
{
  return get_impl().get_at(idx);
}
size_t tensor::num_elems() const
{
  return get_impl().get_buffer().get_count();
}
const detail::sten_tensor &tensor::get_impl() const
{
  return *reinterpret_cast<const detail::sten_tensor *>(&m_storage);
}
bool tensor::is_available_on_cpu() const
{
  return get_impl().is_available_on_cpu();
}
bool tensor::is_available_on_gpu() const
{
  return get_impl().is_available_on_gpu();
}

}  // namespace sten
