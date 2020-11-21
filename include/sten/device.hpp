#ifndef SYCLTEN_DEVICE_SELECTOR_HPP
#define SYCLTEN_DEVICE_SELECTOR_HPP

#include <sten/types.hpp>

#include <CL/sycl.hpp>

using namespace cl::sycl;

namespace sten {
namespace sycl = cl::sycl;

inline const sycl::device_selector &get_device_selector(device_kind kind)
{
  static auto cpu_selector = sycl::cpu_selector{};
  static auto gpu_selector = sycl::gpu_selector{};
  switch (kind) {
    case kCPU:
      return cpu_selector;
    case kGPU:
      return gpu_selector;
    default:
      __builtin_unreachable();
  }
}

struct device : cl::sycl::device {
  using cl::sycl::device::device;
  device(device_kind kind) : device(get_device_selector(kind)) {}
};

}  // namespace sten

#endif  // SYCLTEN_DEVICE_SELECTOR_HPP
