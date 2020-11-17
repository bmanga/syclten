#ifndef SYCLTEN_DISPATCH_HPP
#define SYCLTEN_DISPATCH_HPP

#include "queue.hpp"

#include "kernels.hpp"

template <class Kernel, class... Accessors>
auto make_callable(Accessors... accessors)
{
  return [=](nd_item<1> item) { return Kernel::op(item, accessors...); };
}

template <class Kernel, class Tensor0, class Tensor1, class Tensor2>
void dispatch(Tensor0 &t0, Tensor1 &t1, Tensor2 &t2)
{
  using namespace cl::sycl;
  auto &queue = queue_for(t1.get_device());
  queue.submit([&](handler &cgh) {
    auto t0access =
        t0.get_buffer().template get_access<access::mode::write>(cgh);
    auto t1access =
        t1.get_buffer().template get_access<access::mode::read>(cgh);
    auto t2access =
        t2.get_buffer().template get_access<access::mode::read>(cgh);

    auto nElems = t0.get_buffer().get_count();

    auto myRange = nd_range<1>(range<1>(nElems), range<1>(nElems / 4));

    cgh.parallel_for<Kernel>(
        myRange, make_callable<Kernel>(t0access, t1access, t2access));
  });
}

template <class Tensor>
auto get_single(Tensor &t0, size_t idx)
{
  using namespace cl::sycl;
  auto accessor = t0.get_buffer().template get_access<access::mode::read>();
  return accessor[idx];
}

#endif  // SYCLTEN_DISPATCH_HPP
