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

struct SumKernel {
  template <class ValueType>
  static void op(const nd_item<1> &item,
                 write_accessor<ValueType> dest,
                 read_accessor<ValueType> a,
                 read_accessor<ValueType> b)
  {
    ValueType result = a[item.get_global_id()] + b[item.get_global_id()];
    dest[item.get_global_id()] = result;
  }
};

struct GtKernel {
  template <class ValueType>
  static void op(const nd_item<1> &item,
                 write_accessor<ValueType> dest,
                 read_accessor<ValueType> a,
                 read_accessor<ValueType> b)
  {
    ValueType result = a[item.get_global_id()] > b[item.get_global_id()];
    dest[item.get_global_id()] = result;
  }
};
}  // namespace sten

#endif
