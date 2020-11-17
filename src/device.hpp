#ifndef SYCLTEN_DEVICE_SELECTOR_HPP
#define SYCLTEN_DEVICE_SELECTOR_HPP

#include <CL/sycl.hpp>

using namespace cl::sycl;

namespace sten {
namespace sycl = cl::sycl;

using device = cl::sycl::device;

inline device get_device(device_kind kind)
{
  switch (kind) {
    case kCPU:
      return sycl::cpu_selector{}.select_device();
    case kGPU:
      return sycl::gpu_selector{}.select_device();
    default:
      __builtin_unreachable();
  }
}
}  // namespace sten

#endif  // SYCLTEN_DEVICE_SELECTOR_HPP
