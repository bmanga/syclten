#ifndef SYCLTEN_KERNELS_HPP
#define SYCLTEN_KERNELS_HPP

#include <CL/sycl.hpp>

namespace sten {

template <class ValueType>
using read_accessor =
    cl::sycl::accessor<ValueType,
                       1,
                       cl::sycl::access::mode::read,
                       cl::sycl::access::target::global_buffer>;
template <class ValueType>
using write_accessor =
    cl::sycl::accessor<ValueType,
                       1,
                       cl::sycl::access::mode::write,
                       cl::sycl::access::target::global_buffer>;
template <class ValueType>
using readwrite_accessor =
    cl::sycl::accessor<ValueType,
                       1,
                       cl::sycl::access::mode::read_write,
                       cl::sycl::access::target::global_buffer>;

template <class KernelExpr>
struct ExpressionKernel {
  template <class ValueType, class... ReadAccessors>
  static void op(const cl::sycl::nd_item<1> &item,
                 write_accessor<ValueType> dest,
                 ReadAccessors... read_accessors)
  {
    ValueType result = KernelExpr::execute(
        std::make_tuple(read_accessors[item.get_global_id()]...));
    dest[item.get_global_id()] = result;
  }
};
}  // namespace sten

#endif
