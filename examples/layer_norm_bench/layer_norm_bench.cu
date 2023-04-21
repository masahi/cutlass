#include <cmath>
#include <iostream>
#include <vector>
#include <limits>

#include "cutlass/util/device_layernorm.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/constants.h"

using ElementType = cutlass::half_t;
using Layout = cutlass::layout::RowMajor;

void layernorm_host(cutlass::MatrixCoord tensor_size,
		    cutlass::HostTensor<ElementType, Layout> output,
		    cutlass::HostTensor<ElementType, Layout> input,
		    cutlass::HostTensor<ElementType, Layout> gamma,
		    cutlass::HostTensor<ElementType, Layout> beta) {
  const int M = tensor_size.row();
  const int N = tensor_size.column();

  for (int m = 0; m < M; ++m) {
    float sum{0};
    float square_sum{0};

    for (int n = 0; n < N; ++n) {
      float inp = static_cast<float>(input.at({m, n}));
      sum += inp;
      square_sum += inp * inp;
    }

    ElementType mean = static_cast<ElementType>(sum / (float)N);
    ElementType sq_mean = static_cast<ElementType>(square_sum / (float)N);
    float sqrt_var = cutlass::fast_sqrt(static_cast<float>(sq_mean - mean * mean + ElementType(1e-6)));
    ElementType inv_sqrt_var = cutlass::constants::one<ElementType>() / static_cast<ElementType>(sqrt_var);

    for (int n = 0; n < N; ++n) {
      output.at({m, n}) = (input.at({m, n}) - mean) * inv_sqrt_var * gamma.at({0, n}) + beta.at({0, n});
    }
  }


}

int main(int argc, const char **argv) {
  cutlass::HostTensor<ElementType, Layout> input, output, gamma, beta;


  return 0;
}
