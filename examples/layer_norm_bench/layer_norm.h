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

__global__ void layernorm_twoPassAlgo_e8(float4 *output, const float4 *input,
                                         const float4 *gamma,
                                         const float4 *beta, const int m,
                                         const int n) {
  const int m_idx = blockIdx.x;
  const int tid = threadIdx.x;
  const int bdimx = blockDim.x;
  __shared__ float s_mean, s_variance;
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
    local_sums[0] += static_cast<float>(h1->x) + static_cast<float>(h1->y) +
                     static_cast<float>(h2->x) + static_cast<float>(h2->y) +
                     static_cast<float>(h3->x) + static_cast<float>(h3->y) +
                     static_cast<float>(h4->x) + static_cast<float>(h4->y);
  }

  if (blockDim.x <= 32) {
    warpReduceSum<float, 1>(local_sums);
  } else {
    blockReduceSum<float, 1>(local_sums);
  }
  if (threadIdx.x == 0) {
    s_mean = local_sums[0] / n;
  }
  __syncthreads();

  local_sums[0] = 0.0f;
  for (int index = tid; index < n_8; index += bdimx) {
    const float4 local_val = input[index];
    const half2 *h1 = (half2 *)&local_val.x;
    const half2 *h2 = (half2 *)&local_val.y;
    const half2 *h3 = (half2 *)&local_val.z;
    const half2 *h4 = (half2 *)&local_val.w;

    local_sums[0] += (static_cast<float>(h1->x) - s_mean) *
                     (static_cast<float>(h1->x) - s_mean);
    local_sums[0] += (static_cast<float>(h1->y) - s_mean) *
                     (static_cast<float>(h1->y) - s_mean);
    local_sums[0] += (static_cast<float>(h2->x) - s_mean) *
                     (static_cast<float>(h2->x) - s_mean);
    local_sums[0] += (static_cast<float>(h2->y) - s_mean) *
                     (static_cast<float>(h2->y) - s_mean);
    local_sums[0] += (static_cast<float>(h3->x) - s_mean) *
                     (static_cast<float>(h3->x) - s_mean);
    local_sums[0] += (static_cast<float>(h3->y) - s_mean) *
                     (static_cast<float>(h3->y) - s_mean);
    local_sums[0] += (static_cast<float>(h4->x) - s_mean) *
                     (static_cast<float>(h4->x) - s_mean);
    local_sums[0] += (static_cast<float>(h4->y) - s_mean) *
                     (static_cast<float>(h4->y) - s_mean);
  }

  if (blockDim.x <= 32) {
    warpReduceSum<float, 1>(local_sums);
  } else {
    blockReduceSum<float, 1>(local_sums);
  }
  if (threadIdx.x == 0) {
    s_variance = rsqrtf(local_sums[0] / n + 1e-5);
  }
  __syncthreads();

  for (int index = tid; index < n_8; index += bdimx) {
    const float4 local_val = input[index];
    const float4 gamma_val = gamma[index];
    const float4 beta_val = beta[index];

    const half2 *l1 = (half2 *)&local_val.x;
    const half2 *l2 = (half2 *)&local_val.y;
    const half2 *l3 = (half2 *)&local_val.z;
    const half2 *l4 = (half2 *)&local_val.w;

    const half2 *g1 = (half2 *)&gamma_val.x;
    const half2 *g2 = (half2 *)&gamma_val.y;
    const half2 *g3 = (half2 *)&gamma_val.z;
    const half2 *g4 = (half2 *)&gamma_val.w;

    const half2 *b1 = (half2 *)&beta_val.x;
    const half2 *b2 = (half2 *)&beta_val.y;
    const half2 *b3 = (half2 *)&beta_val.z;
    const half2 *b4 = (half2 *)&beta_val.w;

    float4 tmp;
    half2 *h1 = (half2 *)&tmp.x;
    half2 *h2 = (half2 *)&tmp.y;
    half2 *h3 = (half2 *)&tmp.z;
    half4 *h4 = (half4 *)&tmp.w;

    h1->x = half((static_cast<float>(l1->x) - s_mean) * s_variance *
                     static_cast<float>(g1->x) +
                 static_cast<float>(b1->x));
    h1->y = half((static_cast<float>(l1->y) - s_mean) * s_variance *
                     static_cast<float>(g1->y) +
                 static_cast<float>(b1->y));
    h2->x = half((static_cast<float>(l2->x) - s_mean) * s_variance *
                     static_cast<float>(g2->x) +
                 static_cast<float>(b2->x));
    h2->y = half((static_cast<float>(l2->y) - s_mean) * s_variance *
                     static_cast<float>(g2->y) +
                 static_cast<float>(b2->y));
    h3->x = half((static_cast<float>(l3->x) - s_mean) * s_variance *
                     static_cast<float>(g3->x) +
                 static_cast<float>(b3->x));
    h3->y = half((static_cast<float>(l3->y) - s_mean) * s_variance *
                     static_cast<float>(g3->y) +
                 static_cast<float>(b3->y));
    h4->x = half((static_cast<float>(l4->x) - s_mean) * s_variance *
                     static_cast<float>(g4->x) +
                 static_cast<float>(b4->x));
    h4->y = half((static_cast<float>(l4->y) - s_mean) * s_variance *
                     static_cast<float>(g4->y) +
                 static_cast<float>(b4->y));

    output[index] = tmp;
  }
}

__global__ void layernorm_twoPassAlgo_e8_smem(float4 *output,
                                              const float4 *input,
                                              const float4 *gamma,
                                              const float4 *beta, const int m,
                                              const int n) {
  const int m_idx = blockIdx.x;
  const int tid = threadIdx.x;
  const int bdimx = blockDim.x;
  __shared__ float s_mean, s_variance;
  extern __shared__ float4 smem[];

  float local_sums[1] = {0.0f};
  const int n_8 = n / 8;
  int offset = m_idx * n_8;
  input += offset;
  output += offset;

  for (int index = tid; index < n_8; index += bdimx) {
    smem[index] = input[index];
    const half2 *h1 = (half2 *)&smem[index].x;
    const half2 *h2 = (half2 *)&smem[index].y;
    const half2 *h3 = (half2 *)&smem[index].z;
    const half2 *h4 = (half2 *)&smem[index].w;
    local_sums[0] += static_cast<float>(h1->x) + static_cast<float>(h1->y) +
                     static_cast<float>(h2->x) + static_cast<float>(h2->y) +
                     static_cast<float>(h3->x) + static_cast<float>(h3->y) +
                     static_cast<float>(h4->x) + static_cast<float>(h4->y);
  }

  if (blockDim.x <= 32) {
    warpReduceSum<float, 1>(local_sums);
  } else {
    blockReduceSum<float, 1>(local_sums);
  }
  if (threadIdx.x == 0) {
    s_mean = local_sums[0] / n;
  }
  __syncthreads();

  local_sums[0] = 0.0f;
  for (int index = tid; index < n_8; index += bdimx) {
    const half2 *h1 = (half2 *)&smem[index].x;
    const half2 *h2 = (half2 *)&smem[index].y;
    const half2 *h3 = (half2 *)&smem[index].z;
    const half2 *h4 = (half2 *)&smem[index].w;

    local_sums[0] += (static_cast<float>(h1->x) - s_mean) *
                     (static_cast<float>(h1->x) - s_mean);
    local_sums[0] += (static_cast<float>(h1->y) - s_mean) *
                     (static_cast<float>(h1->y) - s_mean);
    local_sums[0] += (static_cast<float>(h2->x) - s_mean) *
                     (static_cast<float>(h2->x) - s_mean);
    local_sums[0] += (static_cast<float>(h2->y) - s_mean) *
                     (static_cast<float>(h2->y) - s_mean);
    local_sums[0] += (static_cast<float>(h3->x) - s_mean) *
                     (static_cast<float>(h3->x) - s_mean);
    local_sums[0] += (static_cast<float>(h3->y) - s_mean) *
                     (static_cast<float>(h3->y) - s_mean);
    local_sums[0] += (static_cast<float>(h4->x) - s_mean) *
                     (static_cast<float>(h4->x) - s_mean);
    local_sums[0] += (static_cast<float>(h4->y) - s_mean) *
                     (static_cast<float>(h4->y) - s_mean);
  }

  if (blockDim.x <= 32) {
    warpReduceSum<float, 1>(local_sums);
  } else {
    blockReduceSum<float, 1>(local_sums);
  }
  if (threadIdx.x == 0) {
    s_variance = rsqrtf(local_sums[0] / n + 1e-5);
  }
  __syncthreads();

  for (int index = tid; index < n_8; index += bdimx) {
    const float4 gamma_val = gamma[index];
    const float4 beta_val = beta[index];

    const half2 *l1 = (half2 *)&smem[index].x;
    const half2 *l2 = (half2 *)&smem[index].y;
    const half2 *l3 = (half2 *)&smem[index].z;
    const half2 *l4 = (half2 *)&smem[index].w;

    const half2 *g1 = (half2 *)&gamma_val.x;
    const half2 *g2 = (half2 *)&gamma_val.y;
    const half2 *g3 = (half2 *)&gamma_val.z;
    const half2 *g4 = (half2 *)&gamma_val.w;

    const half2 *b1 = (half2 *)&beta_val.x;
    const half2 *b2 = (half2 *)&beta_val.y;
    const half2 *b3 = (half2 *)&beta_val.z;
    const half2 *b4 = (half2 *)&beta_val.w;

    float4 tmp;
    half2 *h1 = (half2 *)&tmp.x;
    half2 *h2 = (half2 *)&tmp.y;
    half2 *h3 = (half2 *)&tmp.z;
    half4 *h4 = (half4 *)&tmp.w;

    h1->x = half((static_cast<float>(l1->x) - s_mean) * s_variance *
                     static_cast<float>(g1->x) +
                 static_cast<float>(b1->x));
    h1->y = half((static_cast<float>(l1->y) - s_mean) * s_variance *
                     static_cast<float>(g1->y) +
                 static_cast<float>(b1->y));
    h2->x = half((static_cast<float>(l2->x) - s_mean) * s_variance *
                     static_cast<float>(g2->x) +
                 static_cast<float>(b2->x));
    h2->y = half((static_cast<float>(l2->y) - s_mean) * s_variance *
                     static_cast<float>(g2->y) +
                 static_cast<float>(b2->y));
    h3->x = half((static_cast<float>(l3->x) - s_mean) * s_variance *
                     static_cast<float>(g3->x) +
                 static_cast<float>(b3->x));
    h3->y = half((static_cast<float>(l3->y) - s_mean) * s_variance *
                     static_cast<float>(g3->y) +
                 static_cast<float>(b3->y));
    h4->x = half((static_cast<float>(l4->x) - s_mean) * s_variance *
                     static_cast<float>(g4->x) +
                 static_cast<float>(b4->x));
    h4->y = half((static_cast<float>(l4->y) - s_mean) * s_variance *
                     static_cast<float>(g4->y) +
                 static_cast<float>(b4->y));

    output[index] = tmp;
  }
}

template <int ILP = 8>
__global__ void
layernorm_twoPassAlgo_e8_smem_async(float4 *output, const float4 *input,
                                    const float4 *gamma, const float4 *beta,
                                    const int m, const int n) {
  const int m_idx = blockIdx.x;
  const int tid = threadIdx.x;
  const int bdimx = blockDim.x;
  __shared__ float s_mean, s_variance;
  extern __shared__ float4 smem[];

  float local_sums[1] = {0.0f};
  const int n_8 = n / 8;
  int offset = m_idx * n_8;
  input += offset;
  output += offset;

  for (int i = tid; i < n_8; i += bdimx * ILP) {
#pragma unroll ILP
    for (int ii = 0; ii < ILP; ++ii) {
      int index = i + ii * bdimx;
      if (index < n_8)
        cutlass::arch::cp_async<sizeof(float4)>(&smem[index], &input[index],
                                                true);
    }
  }

  cutlass::arch::cp_async_wait<0>();

  for (int index = tid; index < n_8; index += bdimx) {
    const half2 *h1 = (half2 *)&smem[index].x;
    const half2 *h2 = (half2 *)&smem[index].y;
    const half2 *h3 = (half2 *)&smem[index].z;
    const half2 *h4 = (half2 *)&smem[index].w;
    local_sums[0] += static_cast<float>(h1->x) + static_cast<float>(h1->y) +
                     static_cast<float>(h2->x) + static_cast<float>(h2->y) +
                     static_cast<float>(h3->x) + static_cast<float>(h3->y) +
                     static_cast<float>(h4->x) + static_cast<float>(h4->y);
  }

  if (blockDim.x <= 32) {
    warpReduceSum<float, 1>(local_sums);
  } else {
    blockReduceSum<float, 1>(local_sums);
  }
  if (threadIdx.x == 0) {
    s_mean = local_sums[0] / n;
  }
  __syncthreads();

  local_sums[0] = 0.0f;
  for (int index = tid; index < n_8; index += bdimx) {
    const half2 *h1 = (half2 *)&smem[index].x;
    const half2 *h2 = (half2 *)&smem[index].y;
    const half2 *h3 = (half2 *)&smem[index].z;
    const half2 *h4 = (half2 *)&smem[index].w;

    local_sums[0] += (static_cast<float>(h1->x) - s_mean) *
                     (static_cast<float>(h1->x) - s_mean);
    local_sums[0] += (static_cast<float>(h1->y) - s_mean) *
                     (static_cast<float>(h1->y) - s_mean);
    local_sums[0] += (static_cast<float>(h2->x) - s_mean) *
                     (static_cast<float>(h2->x) - s_mean);
    local_sums[0] += (static_cast<float>(h2->y) - s_mean) *
                     (static_cast<float>(h2->y) - s_mean);
    local_sums[0] += (static_cast<float>(h3->x) - s_mean) *
                     (static_cast<float>(h3->x) - s_mean);
    local_sums[0] += (static_cast<float>(h3->y) - s_mean) *
                     (static_cast<float>(h3->y) - s_mean);
    local_sums[0] += (static_cast<float>(h4->x) - s_mean) *
                     (static_cast<float>(h4->x) - s_mean);
    local_sums[0] += (static_cast<float>(h4->y) - s_mean) *
                     (static_cast<float>(h4->y) - s_mean);
  }

  if (blockDim.x <= 32) {
    warpReduceSum<float, 1>(local_sums);
  } else {
    blockReduceSum<float, 1>(local_sums);
  }
  if (threadIdx.x == 0) {
    s_variance = rsqrtf(local_sums[0] / n + 1e-5);
  }
  __syncthreads();

  for (int index = tid; index < n_8; index += bdimx) {
    const float4 gamma_val = gamma[index];
    const float4 beta_val = beta[index];

    const half2 *l1 = (half2 *)&smem[index].x;
    const half2 *l2 = (half2 *)&smem[index].y;
    const half2 *l3 = (half2 *)&smem[index].z;
    const half2 *l4 = (half2 *)&smem[index].w;

    const half2 *g1 = (half2 *)&gamma_val.x;
    const half2 *g2 = (half2 *)&gamma_val.y;
    const half2 *g3 = (half2 *)&gamma_val.z;
    const half2 *g4 = (half2 *)&gamma_val.w;

    const half2 *b1 = (half2 *)&beta_val.x;
    const half2 *b2 = (half2 *)&beta_val.y;
    const half2 *b3 = (half2 *)&beta_val.z;
    const half2 *b4 = (half2 *)&beta_val.w;

    float4 tmp;
    half2 *h1 = (half2 *)&tmp.x;
    half2 *h2 = (half2 *)&tmp.y;
    half2 *h3 = (half2 *)&tmp.z;
    half4 *h4 = (half4 *)&tmp.w;

    h1->x = half((static_cast<float>(l1->x) - s_mean) * s_variance *
                     static_cast<float>(g1->x) +
                 static_cast<float>(b1->x));
    h1->y = half((static_cast<float>(l1->y) - s_mean) * s_variance *
                     static_cast<float>(g1->y) +
                 static_cast<float>(b1->y));
    h2->x = half((static_cast<float>(l2->x) - s_mean) * s_variance *
                     static_cast<float>(g2->x) +
                 static_cast<float>(b2->x));
    h2->y = half((static_cast<float>(l2->y) - s_mean) * s_variance *
                     static_cast<float>(g2->y) +
                 static_cast<float>(b2->y));
    h3->x = half((static_cast<float>(l3->x) - s_mean) * s_variance *
                     static_cast<float>(g3->x) +
                 static_cast<float>(b3->x));
    h3->y = half((static_cast<float>(l3->y) - s_mean) * s_variance *
                     static_cast<float>(g3->y) +
                 static_cast<float>(b3->y));
    h4->x = half((static_cast<float>(l4->x) - s_mean) * s_variance *
                     static_cast<float>(g4->x) +
                 static_cast<float>(b4->x));
    h4->y = half((static_cast<float>(l4->y) - s_mean) * s_variance *
                     static_cast<float>(g4->y) +
                 static_cast<float>(b4->y));

    output[index] = tmp;
  }
}

void layernorm_half8(cutlass::MatrixCoord tensor_size,
                     TensorRef<cutlass::half_t, layout::RowMajor> ref_output,
                     TensorRef<cutlass::half_t, layout::RowMajor> ref_input,
                     TensorRef<cutlass::half_t, layout::RowMajor> ref_gamma,
                     TensorRef<cutlass::half_t, layout::RowMajor> ref_beta,
                     cudaStream_t stream) {

  const int m = tensor_size.row();
  const int n = tensor_size.column();
  cutlass::half_t *output = ref_output.data();
  const cutlass::half_t *input = ref_input.data();
  const cutlass::half_t *gamma = ref_gamma.data();
  const cutlass::half_t *beta = ref_beta.data();

  dim3 grid(m);
  dim3 block((n / 8 + 31) / 32 * 32);

  if (block.x > 1024) {
    block.x = 1024;
  }

  layernorm_twoPassAlgo_e8<<<grid, block, 0, stream>>>(
      (float4 *)output, (const float4 *)input, (const float4 *)gamma,
      (const float4 *)beta, m, n);
}

void layernorm_half8_smem(
    cutlass::MatrixCoord tensor_size,
    TensorRef<cutlass::half_t, layout::RowMajor> ref_output,
    TensorRef<cutlass::half_t, layout::RowMajor> ref_input,
    TensorRef<cutlass::half_t, layout::RowMajor> ref_gamma,
    TensorRef<cutlass::half_t, layout::RowMajor> ref_beta,
    cudaStream_t stream) {

  const int m = tensor_size.row();
  const int n = tensor_size.column();
  cutlass::half_t *output = ref_output.data();
  const cutlass::half_t *input = ref_input.data();
  const cutlass::half_t *gamma = ref_gamma.data();
  const cutlass::half_t *beta = ref_beta.data();

  dim3 grid(m);
  dim3 block((n / 8 + 31) / 32 * 32);

  if (block.x > 1024) {
    block.x = 1024;
  }

  const size_t smem_size_bytes = n * 2;

  layernorm_twoPassAlgo_e8_smem<<<grid, block, smem_size_bytes, stream>>>(
      (float4 *)output, (const float4 *)input, (const float4 *)gamma,
      (const float4 *)beta, m, n);
}

void layernorm_half8_smem_async(
    cutlass::MatrixCoord tensor_size,
    TensorRef<cutlass::half_t, layout::RowMajor> ref_output,
    TensorRef<cutlass::half_t, layout::RowMajor> ref_input,
    TensorRef<cutlass::half_t, layout::RowMajor> ref_gamma,
    TensorRef<cutlass::half_t, layout::RowMajor> ref_beta,
    cudaStream_t stream) {

  const int m = tensor_size.row();
  const int n = tensor_size.column();
  cutlass::half_t *output = ref_output.data();
  const cutlass::half_t *input = ref_input.data();
  const cutlass::half_t *gamma = ref_gamma.data();
  const cutlass::half_t *beta = ref_beta.data();

  dim3 grid(m);
  dim3 block((n / 8 + 31) / 32 * 32);

  if (block.x > 1024) {
    block.x = 1024;
  }

  const size_t smem_size_bytes = n * 2;

  layernorm_twoPassAlgo_e8_smem_async<<<grid, block, smem_size_bytes, stream>>>(
      (float4 *)output, (const float4 *)input, (const float4 *)gamma,
      (const float4 *)beta, m, n);
}

} // namespace cutlass
