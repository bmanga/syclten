#ifndef SYCLTEN_DISPATCH_HPP
#define SYCLTEN_DISPATCH_HPP

#include "kernels.hpp"
#include "queue.hpp"

namespace sten {

template <class Kernel, class OutAccessor, class AccessorsTuple, size_t... Ids>
auto make_callable_impl(OutAccessor accessor,
                        AccessorsTuple accessors,
                        std::index_sequence<Ids...>)
{
  return [=](nd_item<1> item) {
    return Kernel::op(item, accessor, std::get<Ids>(accessors)...);
  };
}

template <class Kernel, class OutAccessor, class AccessorsTuple>
auto make_callable(OutAccessor out_accessor, AccessorsTuple accessors)
{
  return make_callable_impl<Kernel>(
      out_accessor, accessors,
      std::make_index_sequence<std::tuple_size<AccessorsTuple>::value>());
}

template <class Kernel,
          class KernelName = Kernel,
          class TensorOut,
          class... TensorsIn>
void dispatch(device device, TensorOut &result, TensorsIn &... tensors)
{
  using namespace cl::sycl;
  auto &queue = queue_for(device);
  (tensors.set_available_on(device), ...);

  queue.submit([&](handler &cgh) {
    auto t0access =
        result.get_buffer().template get_access<access::mode::write>(cgh);
    auto input_accessors = std::make_tuple(
        tensors.get_buffer().template get_access<access::mode::read>(cgh)...);

    auto nElems = result.get_buffer().get_count();

    auto myRange = nd_range<1>(range<1>(nElems), range<1>(nElems / 4));

    cgh.parallel_for<Kernel>(myRange,
                             make_callable<Kernel>(t0access, input_accessors));
  });
}

template <class KernelExpr, class TensorTuple, class Tensor, size_t... Ids>
void dispatch_expression_impl(device device,
                              Tensor &result,
                              TensorTuple &inputs,
                              std::index_sequence<Ids...>)
{
  dispatch<ExpressionKernel<KernelExpr>, KernelExpr>(device, result,
                                                     std::get<Ids>(inputs)...);
}

template <class TensorExpr, class Tensor>
void dispatch_expression(device device, Tensor &result, TensorExpr expr)
{
  auto input_tensors = expr.collect_tensors();
  using kernel_expr = decltype(TensorExpr::get_kernel());
  dispatch_expression_impl<kernel_expr>(
      device, result, input_tensors,
      std::make_index_sequence<
          std::tuple_size<decltype(input_tensors)>::value>());
}

template <class Tensor>
auto get_single(Tensor &t0, size_t idx)
{
  using namespace cl::sycl;
  auto accessor = t0.get_buffer().template get_access<access::mode::read>();
  return accessor[idx];
}
}  // namespace sten

#endif  // SYCLTEN_DISPATCH_HPP
