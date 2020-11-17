#include <sten/tensor.hpp>

#include <iostream>
#include <numeric>
#include <vector>

int main()
{
  int n = 512 * 4;
  std::vector<float> vf(n);
  std::iota(vf.begin(), vf.end(), 0);
  sten::tensor t(vf.data(), {n}, sten::kGPU);
  sten::tensor t2(vf.data(), {n}, sten::kGPU);

  sten::tensor t3(t + t2);
  std::cout << t.is_cpu() << t.is_gpu() << std::endl;
  std::cout << t.get_at(400) << " + " << t2.get_at(400) << " = "
            << t3.get_at(400) << std::endl;
}
