#include <cmath>
#include <iostream>
#include <vector>
#include <limits>

#include "cutlass/util/device_rms_norm.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/constants.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_compare.h"


using ElementType = cutlass::half_t;
using Layout = cutlass::layout::RowMajor;

void rmsnorm_host(cutlass::MatrixCoord tensor_size,
		  cutlass::TensorRef<ElementType, Layout> output,
		  cutlass::TensorRef<ElementType, Layout> input,
		  cutlass::TensorRef<ElementType, Layout> weight) {
  const int M = tensor_size.row();
  const int N = tensor_size.column();

  for (int m = 0; m < M; ++m) {
    float square_sum{0};

    for (int n = 0; n < N; ++n) {
      float inp = static_cast<float>(input.at({m, n}));
      square_sum += inp * inp;
    }

    float sq_mean = square_sum / (float)N;
    float sqrt_var = cutlass::fast_sqrt(sq_mean + (float)1e-6);

    for (int n = 0; n < N; ++n) {
      float inp = static_cast<float>(input.at({m, n}));
      float g = static_cast<float>(weight.at({0, n}));
      float res_fp32 = inp / sqrt_var * g;
      output.at({m, n}) = ElementType(res_fp32);
    }
  }
}

template <typename Func>
float benchmark(Func f) {

  cudaEvent_t events[2];
  for (cudaEvent_t &evt : events) {
    cudaEventCreate(&evt);
  }

  cudaEventRecord(events[0]);

  const int n_iters = 100;
  for (int i = 0; i < n_iters; ++i) {
    f();
  }

  cudaEventRecord(events[1]);

  cudaDeviceSynchronize();

  float elapsed_ms = 0;
  cudaEventElapsedTime(&elapsed_ms, events[0], events[1]);

  return elapsed_ms / float(n_iters) * 1e3;
}

int main(int argc, const char **argv) {
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  cutlass::HostTensor<ElementType, Layout> input, output_ref, output, weight;

  const int M = std::stoi(argv[1]);
  const int N = std::stoi(argv[2]);

  input.reset({M, N});
  output.reset({M, N});
  output_ref.reset({M, N});
  weight.reset({1, N});

  const unsigned seed = 2022;

  cutlass::reference::host::TensorFillRandomUniform(input.host_view(),
						    seed,
						    ElementType(5),
						    ElementType(-5),
						    0);

  cutlass::reference::host::TensorFillRandomUniform(weight.host_view(),
						    seed,
						    ElementType(5),
						    ElementType(-5),
						    0);

  input.sync_device();
  weight.sync_device();

  rmsnorm_host({M, N}, output_ref.host_ref(), input.host_ref(), weight.host_ref());
  cutlass::rmsnorm_half8({M, N}, output.device_ref(),
			 input.device_ref(), weight.device_ref(), stream);

  output.sync_host();

  float max_abs_diff = -1;
  float mean_abs_diff = 0;
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      auto diff = abs(static_cast<float>(output_ref.at({m, n}) - output.at({m, n})));
      mean_abs_diff += diff;
      max_abs_diff = max(max_abs_diff, diff);
    }
  }

  mean_abs_diff /= float(M * N);

  std::cout << "Max and mean abs diff: " << max_abs_diff << ", " << mean_abs_diff << "\n\n";

  cudaStreamDestroy(stream);

  return 0;
}
