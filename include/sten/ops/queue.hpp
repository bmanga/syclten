#ifndef SYCLTEN_QUEUE_HPP
#define SYCLTEN_QUEUE_HPP

#include <CL/sycl.hpp>
#include "../device.hpp"

#include <unordered_map>

inline cl::sycl::queue &queue_for(const sten::device &device)
{
  static std::unordered_map<int, cl::sycl::queue> queues;
  static struct at_scope_exit {
    ~at_scope_exit()
    {
      for (auto &[id, queue] : queues) {
        queue.wait();
      }
    }
  } queue_waiter;

  auto type = (int)device.get_info<info::device::device_type>();
  auto it = queues.find(type);
  if (it == queues.end()) {
    it = queues.emplace(type, cl::sycl::queue(device)).first;
  }
  return it->second;
}

#endif  // SYCLTEN_QUEUE_HPP
