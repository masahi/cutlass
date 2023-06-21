#pragma once

#include "cutlass/arch/memory_sm80.h"
#include "cutlass/cutlass.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/numeric_types.h"
#include "cutlass/tensor_coord.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/util/device_utils.h"
#include <float.h>

namespace cutlass {

__global__ void rmsnorm_twoPassAlgo_e8(float4 *output, const float4 *input,
				       const float4 *weight,
				       const int m, const int n) {
  const int m_idx = blockIdx.x;
  const int tid = threadIdx.x;
  const int bdimx = blockDim.x;
  __shared__ float s_mean;
  float local_sums[1] = {0.0f};
  const int n_8 = n / 8;
  int offset = m_idx * n_8;
  input += offset;
  output += offset;

  for (int index = tid; index < n_8; index += bdimx) {
    const float4 local_val = input[index];
    const half2 *h1 = (half2 *)&local_val.x;
    const half2 *h2 = (half2 *)&local_val.y;
    const half2 *h3 = (half2 *)&local_val.z;
    const half2 *h4 = (half2 *)&local_val.w;
    local_sums[0] += static_cast<float>(h1->x) * static_cast<float>(h1->x) +
                     static_cast<float>(h1->y) * static_cast<float>(h1->y) +
                     static_cast<float>(h2->x) * static_cast<float>(h2->x) +
                     static_cast<float>(h2->y) * static_cast<float>(h2->y) +
                     static_cast<float>(h3->x) * static_cast<float>(h3->x) +
                     static_cast<float>(h3->y) * static_cast<float>(h3->y) +
                     static_cast<float>(h4->x) * static_cast<float>(h4->x) +
                     static_cast<float>(h4->y) * static_cast<float>(h4->y);
  }

  if (blockDim.x <= 32) {
    warpReduceSum<float, 1>(local_sums);
  } else {
    blockReduceSum<float, 1>(local_sums);
  }
  if (threadIdx.x == 0) {
    s_mean = rsqrtf(local_sums[0] / n + 1e-6);
  }
  __syncthreads();

  for (int index = tid; index < n_8; index += bdimx) {
    const float4 local_val = input[index];
    const float4 weight_val = weight[index];

    const half2 *l1 = (half2 *)&local_val.x;
    const half2 *l2 = (half2 *)&local_val.y;
    const half2 *l3 = (half2 *)&local_val.z;
    const half2 *l4 = (half2 *)&local_val.w;

    const half2 *g1 = (half2 *)&weight_val.x;
    const half2 *g2 = (half2 *)&weight_val.y;
    const half2 *g3 = (half2 *)&weight_val.z;
    const half2 *g4 = (half2 *)&weight_val.w;

    float4 tmp;
    half2 *h1 = (half2 *)&tmp.x;
    half2 *h2 = (half2 *)&tmp.y;
    half2 *h3 = (half2 *)&tmp.z;
    half4 *h4 = (half4 *)&tmp.w;

    h1->x = half(static_cast<float>(l1->x) * s_mean * static_cast<float>(g1->x));
    h1->y = half(static_cast<float>(l1->y) * s_mean * static_cast<float>(g1->y));
    h2->x = half(static_cast<float>(l2->x) * s_mean * static_cast<float>(g2->x));
    h2->y = half(static_cast<float>(l2->y) * s_mean * static_cast<float>(g2->y));
    h3->x = half(static_cast<float>(l3->x) * s_mean * static_cast<float>(g3->x));
    h3->y = half(static_cast<float>(l3->y) * s_mean * static_cast<float>(g3->y));
    h4->x = half(static_cast<float>(l4->x) * s_mean * static_cast<float>(g4->x));
    h4->y = half(static_cast<float>(l4->y) * s_mean * static_cast<float>(g4->y));

    output[index] = tmp;
  }
}

void rmsnorm_half8(cutlass::MatrixCoord tensor_size,
                     TensorRef<cutlass::half_t, layout::RowMajor> ref_output,
                     TensorRef<cutlass::half_t, layout::RowMajor> ref_input,
                     TensorRef<cutlass::half_t, layout::RowMajor> ref_weight,
                     cudaStream_t stream) {

  const int m = tensor_size.row();
  const int n = tensor_size.column();
  cutlass::half_t *output = ref_output.data();
  const cutlass::half_t *input = ref_input.data();
  const cutlass::half_t *weight = ref_weight.data();

  dim3 grid(m);

  if (n % 8 == 0) {
    dim3 block(min(1024, (n / 8 + 31) / 32 * 32));

    rmsnorm_twoPassAlgo_e8<<<grid, block, 0, stream>>>(
        (float4 *)output, (const float4 *)input, (const float4 *)weight, m, n);
  } else {
    std::cerr << "Not divisible by 8" << std::endl;
  }

  auto result = cudaGetLastError();
  if (result != cudaSuccess) {
    std::cerr << "CUDA error: " << cudaGetErrorString(result) << std::endl;
    abort();
  }
}

} // namespace cutlass
