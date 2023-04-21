#include <cmath>
#include <iostream>
#include <vector>
#include <limits>

#include "cutlass/util/device_layernorm.h"
#include "cutlass/util/host_tensor.h"

using ElementType = cutlass::half_t;
using Layout = cutlass::layout::RowMajor;

void layernorm_host(cutlass::MatrixCoord tensor_size,
		    cutlass::HostTensor<ElementType, Layout> output,
		    cutlass::HostTensor<ElementType, Layout> input,
		    cutlass::HostTensor<ElementType, Layout> gamma,
		    cutlass::HostTensor<ElementType, Layout> beta) {
  cutlass::HostTensor<ElementType, Layout> mean, variance;
}

int main(int argc, const char **argv) {
  cutlass::HostTensor<ElementType, Layout> input, output, gamma, beta;



  return 0;
}
