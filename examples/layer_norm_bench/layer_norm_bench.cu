#include <cmath>
#include <iostream>
#include <vector>
#include <limits>

#include "cutlass/util/device_layernorm.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/constants.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_compare.h"

#include "layer_norm.h"

using ElementType = cutlass::half_t;
using Layout = cutlass::layout::RowMajor;

void layernorm_host(cutlass::MatrixCoord tensor_size,
		    cutlass::TensorRef<ElementType, Layout> output,
		    cutlass::TensorRef<ElementType, Layout> input,
		    cutlass::TensorRef<ElementType, Layout> gamma,
		    cutlass::TensorRef<ElementType, Layout> beta) {
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

    float mean = sum / (float)N;
    float sq_mean = square_sum / (float)N;
    float sqrt_var = cutlass::fast_sqrt(sq_mean - mean * mean + (float)1e-6);

    for (int n = 0; n < N; ++n) {
      float inp = static_cast<float>(input.at({m, n}));
      float g = static_cast<float>(gamma.at({0, n}));
      float b = static_cast<float>(beta.at({0, n}));
      float res_fp32 = (inp - mean) / sqrt_var * g + b;
      output.at({m, n}) = ElementType(res_fp32);
    }
  }
}

template <typename Func>
void benchmark(const std::string& name, Func f) {

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

  std::cout << name << ", elapsed us: " << elapsed_ms / float(n_iters) * 1e3 << std::endl;
}

int main(int argc, const char **argv) {
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  cutlass::HostTensor<ElementType, Layout> input, output_ref, output, gamma, beta;

  const int M = std::stoi(argv[1]);
  const int N = std::stoi(argv[2]);

  input.reset({M, N});
  output.reset({M, N});
  output_ref.reset({M, N});
  gamma.reset({1, N});
  beta.reset({1, N});

  const unsigned seed = 2022;

  cutlass::reference::host::TensorFillRandomUniform(input.host_view(),
						    seed,
						    ElementType(5),
						    ElementType(-5),
						    0);

  cutlass::reference::host::TensorFillRandomUniform(gamma.host_view(),
						    seed,
						    ElementType(5),
						    ElementType(-5),
						    0);

  cutlass::reference::host::TensorFillRandomUniform(beta.host_view(),
						    seed,
						    ElementType(5),
						    ElementType(-5),
						    0);

  input.sync_device();
  gamma.sync_device();
  beta.sync_device();

  layernorm_host({M, N}, output_ref.host_ref(), input.host_ref(), gamma.host_ref(), beta.host_ref());
  layernorm_half_smem_async({M, N}, output.device_ref(),
			    input.device_ref(), gamma.device_ref(), beta.device_ref(), stream);

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

  //  std::cout << cutlass::reference::host::TensorEquals(output_ref.host_view(), output.host_view()) << std::endl;
  std::cout << "Max and mean abs diff: " << max_abs_diff << ", " << mean_abs_diff << "\n\n";

  benchmark("CUTLASS layer norm", [&]() {
      layernorm({M, N}, output.device_ref(),
		input.device_ref(), gamma.device_ref(), beta.device_ref(), stream);
    });

  benchmark("Simple half8 kernel", [&]() {
      layernorm_half8({M, N}, output.device_ref(),
		      input.device_ref(), gamma.device_ref(), beta.device_ref(), stream);
    });

  benchmark("half kernel with smem", [&]() {
      layernorm_half8_smem({M, N}, output.device_ref(),
			   input.device_ref(), gamma.device_ref(), beta.device_ref(), stream);
    });

  benchmark("half kernel with smem and async", [&]() {
      layernorm_half_smem_async({M, N}, output.device_ref(),
				input.device_ref(), gamma.device_ref(), beta.device_ref(), stream);
    });

  cudaStreamDestroy(stream);

  return 0;
}
