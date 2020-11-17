#include <sten/tensor.hpp>

#include "device.hpp"
#include "ops/dispatch.hpp"
#include "ops/kernels.hpp"

#include <CL/sycl.hpp>

namespace sten {
namespace detail {
class sten_tensor {
 public:
  sten_tensor(float *data, size_t num_elems, device device)
      : m_buffer(data, cl::sycl::range<1>(num_elems)), m_device(device)
  {
  }

  sten_tensor(size_t num_elems, device device)
      : m_buffer(sycl::range<1>(num_elems)), m_device(device)
  {
  }
  bool is_gpu() const { return m_device.is_gpu(); }
  bool is_cpu() const { return m_device.is_host(); }

  sten::device get_device() const { return m_device; }

  sycl::buffer<float, 1> &get_buffer() { return m_buffer; }

  const sycl::buffer<float, 1> &get_buffer() const { return m_buffer; }

  sten_tensor sum(sten_tensor &b)
  {
    sten_tensor result(get_buffer().get_count(), b.get_device());
    dispatch<sten::SumKernel>(result, *this, b);
    return result;
  }

  sten_tensor gt(sten_tensor &b)
  {
    sten_tensor result(get_buffer().get_count(), b.get_device());
    dispatch<sten::GtKernel>(result, *this, b);
    return result;
  }

  float get_at(size_t idx)
  {
    float result = get_single(*this, idx);
    return result;
  }

  size_t num_elems() const { return m_buffer.get_count(); }

 private:
  sycl::buffer<float, 1> m_buffer;
  sten::device m_device;
};

static_assert(sizeof(sten_tensor) <= 128);
}  // namespace detail

tensor::tensor(size_t num_elems, device_kind device)
{
  new (&m_storage) detail::sten_tensor(num_elems, sten::get_device(device));
}

tensor::tensor(float *data, size_t num_elems, device_kind device)
{
  new (&m_storage)
      detail::sten_tensor(data, num_elems, sten::get_device(device));
}

tensor::tensor(const detail::sten_tensor &other)
{
  new (&m_storage) detail::sten_tensor(other.num_elems(), other.get_device());
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

bool tensor::is_gpu() const
{
  return get_impl().is_gpu();
}

bool tensor::is_cpu() const
{
  return get_impl().is_cpu();
}

tensor operator+(tensor &a, tensor &b)
{
  return a.get_impl().sum(b.get_impl());
}
tensor operator>(tensor &a, tensor &b)
{
  return a.get_impl().gt(b.get_impl());
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

}  // namespace sten
