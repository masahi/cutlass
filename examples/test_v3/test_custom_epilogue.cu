/***************************************************************************************************
 * Copyright (c) 2023 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#include <iostream>

#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/thread/scale_type.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_mma.hpp"

#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/device/tensor_fill.h"

#include "helper.h"

using namespace cute;
using namespace cutlass;
using namespace cutlass::gemm;

template <
  class StrideC_,
  class StrideD_,
  class ThreadEpilogueOp_
>
class EpilogueThreeSources {
public:
  using ThreadEpilogueOp = ThreadEpilogueOp_;
  using ElementOutput = typename ThreadEpilogueOp::ElementOutput;
  using ElementAccumulator = typename ThreadEpilogueOp::ElementAccumulator;
  using ElementCompute = typename ThreadEpilogueOp::ElementCompute;
  using ElementScalar = ElementCompute;
  using ElementC = typename ThreadEpilogueOp::ElementC;
  using StrideC = StrideC_;
  using ElementD = typename ThreadEpilogueOp::ElementD;
  using StrideD = StrideD_;

  static const int kOutputAlignment = ThreadEpilogueOp::kCount;
  using AlignmentType = typename cute::uint_bit<cutlass::sizeof_bits<ElementOutput>::value * kOutputAlignment>::type;

  static_assert(rank(StrideC{}) == 3, "StrideCD must be rank-3: [M, N, L]");
  static_assert(rank(StrideD{}) == 3, "StrideCD must be rank-3: [M, N, L]");

  struct SharedStorage { };

  struct Params {
    ElementC const* ptr_C1 = nullptr;
    ElementC const* ptr_C2 = nullptr;
    ElementC const* ptr_C3 = nullptr;
    StrideC dC{};
    ElementD* ptr_D = nullptr;
    StrideD dD{};
    typename ThreadEpilogueOp::Params thread_params{};
  };

  template <class Args>
  static constexpr Params
  to_underlying_arguments(Args const& args, void* workspace) {
    (void) workspace;
    return {args.epilogue_params};
  }

  CUTLASS_HOST_DEVICE
  EpilogueThreeSources(Params const& params_) : params(params_) { }

  template<
    class ProblemShapeMNKL,
    class BlockShapeMNK,
    class BlockCoordMNKL,
    class FrgEngine, class FrgLayout,
    class TiledMma,
    class ResidueMNK
  >
  CUTLASS_HOST_DEVICE void
  operator()(
      ProblemShapeMNKL problem_shape_mnkl,
      BlockShapeMNK blk_shape_MNK,
      BlockCoordMNKL blk_coord_mnkl,
      cute::Tensor<FrgEngine, FrgLayout> const& accumulators,
      TiledMma tiled_mma,
      ResidueMNK residue_mnk,
      int thread_idx,
      char* smem_buf)
  {
    using X = Underscore;

    static_assert(rank(ProblemShapeMNKL{}) == 4, "ProblemShapeMNKL must be rank 4");
    static_assert(is_static<BlockShapeMNK>::value, "ThreadBlock tile shape must be static");
    static_assert(rank(BlockShapeMNK{}) == 3, "BlockShapeMNK must be rank 3");
    static_assert(rank(BlockCoordMNKL{}) == 4, "BlockCoordMNKL must be rank 3");

    (void) smem_buf;
    ThreadEpilogueOp epilogue_op{params.thread_params};

    // Separate out problem shape for convenience
    auto M = get<0>(problem_shape_mnkl);
    auto N = get<1>(problem_shape_mnkl);
    auto L = get<3>(problem_shape_mnkl);
    // Slice to get the tile this CTA is responsible for
    auto [m_coord, n_coord, k_coord, l_coord] = blk_coord_mnkl;

    // Represent the full output tensor
    Tensor mD_mnl = make_tensor(make_gmem_ptr(params.ptr_D), make_shape(M,N,L), params.dD);                // (m,n,l)
    Tensor gD_mnl = local_tile(mD_mnl, blk_shape_MNK, make_coord(_,_,_), Step<_1,_1, X>{});    // (BLK_M,BLK_N,m,n,l)
    Tensor gD = gD_mnl(_,_,m_coord,n_coord,l_coord);                                                 // (BLK_M,BLK_N)

    // Partition source and destination tiles to match the accumulator partitioning
    auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
    Tensor tCgD = thr_mma.partition_C(gD);                                       // (VEC,THR_M,THR_N)

    // Make an identity coordinate tensor for predicating our output MN tile
    auto cD = make_identity_tensor(make_shape(unwrap(shape<0>(gD)), unwrap(shape<1>(gD))));
    Tensor tCcD = thr_mma.partition_C(cD);

    static_assert(is_static<FrgLayout>::value, "Accumulator layout must be static");
    CUTE_STATIC_ASSERT_V(size(tCgD) == size(accumulators),
        "Accumulator count must have the same destination element count.");

    Tensor mC1_mnl = make_tensor(make_gmem_ptr(params.ptr_C1), make_shape(M,N,L), params.dC);                // (m,n,l)
    Tensor gC1_mnl = local_tile(mC1_mnl, blk_shape_MNK, make_coord(_,_,_), Step<_1,_1, X>{});    // (BLK_M,BLK_N,m,n,l)
    Tensor gC1 = gC1_mnl(_,_,m_coord,n_coord,l_coord);                                                 // (BLK_M,BLK_N)
    Tensor tCgC1 = thr_mma.partition_C(gC1);                                       // (VEC,THR_M,THR_N)

    Tensor mC2_mnl = make_tensor(make_gmem_ptr(params.ptr_C2), make_shape(M,N,L), params.dC);                // (m,n,l)
    Tensor gC2_mnl = local_tile(mC2_mnl, blk_shape_MNK, make_coord(_,_,_), Step<_1,_1, X>{});    // (BLK_M,BLK_N,m,n,l)
    Tensor gC2 = gC2_mnl(_,_,m_coord,n_coord,l_coord);                                                 // (BLK_M,BLK_N)
    Tensor tCgC2 = thr_mma.partition_C(gC2);                                       // (VEC,THR_M,THR_N)

    Tensor mC3_mnl = make_tensor(make_gmem_ptr(params.ptr_C3), make_shape(M,N,L), params.dC);                // (m,n,l)
    Tensor gC3_mnl = local_tile(mC3_mnl, blk_shape_MNK, make_coord(_,_,_), Step<_1,_1, X>{});    // (BLK_M,BLK_N,m,n,l)
    Tensor gC3 = gC3_mnl(_,_,m_coord,n_coord,l_coord);                                                 // (BLK_M,BLK_N)
    Tensor tCgC3 = thr_mma.partition_C(gC3);                                       // (VEC,THR_M,THR_N)

    CUTE_STATIC_ASSERT_V(size(tCgC1) == size(tCgD),
        "Source and destination must have the same number of elements.");

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(accumulators); ++i) {
      if (elem_less(tCcD(i), make_coord(get<0>(residue_mnk), get<1>(residue_mnk)))) {
        tCgD(i) = epilogue_op(accumulators(i), tCgC1(i) + tCgC2(i) + tCgC3(i));
      }
    }
  }

private:
  Params params;
};

// A matrix configuration
using         ElementA    = cutlass::half_t;                                          // Element type for A matrix operand
using         LayoutA     = cutlass::layout::RowMajor;                      // Layout type for A matrix operand
constexpr int AlignmentA  = 128 / cutlass::sizeof_bits<ElementA>::value;    // Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)

// B matrix configuration
using         ElementB    = cutlass::half_t;                                          // Element type for B matrix operand
using         LayoutB     = cutlass::layout::ColumnMajor;                   // Layout type for B matrix operand
constexpr int AlignmentB  = 128 / cutlass::sizeof_bits<ElementB>::value;    // Memory access granularity/alignment of B matrix in units of elements (up to 16 bytes)

// C/D matrix configuration
using         ElementC    = float;                                          // Element type for C and D matrix operands
using         LayoutC     = cutlass::layout::ColumnMajor;                   // Layout type for C and D matrix operands

// Core kernel configurations
using ElementAccumulator  = float;                                          // Element type for internal accumulation
using ElementOutput = cutlass::half_t;

using DispatchPolicy = MainloopSm80CpAsync<3>;

using TiledMma = TiledMMA<
      MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,	Layout<Shape<_2,_2,_1>>,  // 2x2x1 thread group
	Layout<Shape<_1,_2,_1>>>; // 1x2x1 value group for 16x16x16 MMA and LDSM

template <typename Element, typename Layout, int Alignment, int SizeK>
struct DefaultGemm_TensorOpSm80_OperandA;

template <typename Element, typename Layout, int Alignment, int SizeK>
struct DefaultGemm_TensorOpSm80_OperandB;

template <>
struct DefaultGemm_TensorOpSm80_OperandA<half_t, layout::RowMajor, 8, 64>
{
  // Smem
  using SmemLayoutAtom = decltype(
				  composition(Swizzle<3,3,3>{},
					      Layout<Shape < _8,_64>,
					      Stride<_64, _1>>{}));
  using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, half_t>;

  // Gmem
  using GmemTiledCopy = decltype(
				 make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<cute::uint128_t>, half_t>{},
						 Layout<Shape <_16,_8>,
						 Stride< _8,_1>>{},
						 Layout<Shape < _1,_8>>{}));
};

// Because the F32F16 TiledMMA is A-B symmetric, we can reuse the DefaultOperands

// Operand B - Column-Major (K-major)
template <int Alignment, int SizeK>
struct DefaultGemm_TensorOpSm80_OperandB<half_t, layout::ColumnMajor, Alignment, SizeK>
  : DefaultGemm_TensorOpSm80_OperandA<half_t, layout::RowMajor,    Alignment, SizeK>
{};

// Operand B - Row-Major (N-major)
template <int Alignment, int SizeK>
struct DefaultGemm_TensorOpSm80_OperandB<half_t, layout::RowMajor,    Alignment, SizeK>
  : DefaultGemm_TensorOpSm80_OperandA<half_t, layout::ColumnMajor, Alignment, SizeK>
{};

//
// F16: 128-by-128-by-32 (small k-block)
//

/// Operand A - Row-major (K-Major)
template <>
struct DefaultGemm_TensorOpSm80_OperandA<half_t, layout::RowMajor, 8, 32>
{
  // Smem
  using SmemLayoutAtom = decltype(
				  composition(Swizzle<2,3,3>{},
					      Layout<Shape < _8,_32>,
					      Stride<_32, _1>>{}));
  using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, half_t>;

  // Gmem
  using GmemTiledCopy = decltype(
				 make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<cute::uint128_t>, half_t>{},
						 Layout<Shape <_32,_4>,
						 Stride< _4,_1>>{},
						 Layout<Shape < _1,_8>>{}));
};

using DefaultOperandA = DefaultGemm_TensorOpSm80_OperandA<
    half_t, LayoutA, AlignmentA, 32>;

using SmemLayoutAtomA = DefaultOperandA::SmemLayoutAtom; // M, K
using SmemCopyAtomA = DefaultOperandA::SmemCopyAtom;
using GmemTiledCopyA = DefaultOperandA::GmemTiledCopy;

// B
using DefaultOperandB = DefaultGemm_TensorOpSm80_OperandB<
    half_t, LayoutB, AlignmentB, 32>;
using SmemLayoutAtomB =  DefaultOperandB::SmemLayoutAtom; // N, K
using SmemCopyAtomB =  DefaultOperandB::SmemCopyAtom;
using GmemTiledCopyB =  DefaultOperandB::GmemTiledCopy;

using CollectiveEpilogue = EpilogueThreeSources<
    cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::epilogue::thread::LinearCombination<ElementC, 1, ElementAccumulator, ElementAccumulator,
						     epilogue::thread::ScaleType::Default,
						     FloatRoundStyle::round_to_nearest,
						     ElementOutput>>;

using TileShape          = Shape<_128,_128,_32>;                           // Threadblock-level tile size

using CollectiveMainloop = collective::CollectiveMma<
    DispatchPolicy, TileShape,
      half_t, TagToStrideA_t<LayoutA>,
      half_t, TagToStrideB_t<LayoutB>,
      TiledMma,
      GmemTiledCopyA, SmemLayoutAtomA, SmemCopyAtomA, cute::identity,  // A
      GmemTiledCopyB, SmemLayoutAtomB, SmemCopyAtomB, cute::identity   // B
      >;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int>, // Indicates ProblemShape
      CollectiveMainloop,
      CollectiveEpilogue
      >;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using DeviceGemmReference = cutlass::reference::device::Gemm<
  ElementA,
    LayoutA,
    ElementB,
    LayoutB,
    ElementC,
    LayoutC,
    ElementAccumulator,
    ElementAccumulator>;

using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;

//
// Data members
//

StrideA stride_A;
StrideB stride_B;
StrideC stride_C;
StrideD stride_D;
uint64_t seed;

cutlass::DeviceAllocation<typename Gemm::ElementA> block_A;
cutlass::DeviceAllocation<typename Gemm::ElementB> block_B;
cutlass::DeviceAllocation<typename Gemm::ElementC> block_C1;
cutlass::DeviceAllocation<typename Gemm::ElementC> block_C2;
cutlass::DeviceAllocation<typename Gemm::ElementC> block_C3;
cutlass::DeviceAllocation<typename Gemm::EpilogueOutputOp::ElementOutput> block_D;
cutlass::DeviceAllocation<typename Gemm::EpilogueOutputOp::ElementOutput> block_ref_D;

// Command line options parsing
struct Options {

  bool help;

  float alpha, beta;
  int iterations;
  int m, n, k;

  Options():
    help(false),
    m(5120), n(4096), k(4096),
    alpha(1.f), beta(0.f),
    iterations(1000)
  { }

  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }

    cmd.get_cmd_line_argument("m", m);
    cmd.get_cmd_line_argument("n", n);
    cmd.get_cmd_line_argument("k", k);
    cmd.get_cmd_line_argument("alpha", alpha, 1.f);
    cmd.get_cmd_line_argument("beta", beta, 0.f);
    cmd.get_cmd_line_argument("iterations", iterations);
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "48_hopper_warp_specialized_gemm\n\n"
	<< "  Hopper FP32 GEMM using a Warp Specialized kernel.\n\n"
	<< "Options:\n\n"
	<< "  --help                      If specified, displays this usage statement\n\n"
	<< "  --m=<int>                   Sets the M extent of the GEMM\n"
	<< "  --n=<int>                   Sets the N extent of the GEMM\n"
	<< "  --k=<int>                   Sets the K extent of the GEMM\n"
	<< "  --alpha=<f32>               Epilogue scalar alpha\n"
	<< "  --beta=<f32>                Epilogue scalar beta\n\n"
	<< "  --iterations=<int>          Number of profiling iterations to perform.\n\n";

    out
      << "\n\nExamples:\n\n"
      << "$ " << "48_hopper_warp_specialized_gemm" << " --m=1024 --n=512 --k=1024 --alpha=2 --beta=0.707 \n\n";

    return out;
  }

  /// Compute performance in GFLOP/s
  double gflops(double runtime_s) const
  {
    // Two flops per multiply-add
    uint64_t flop = uint64_t(2) * m * n * k;
    double gflop = double(flop) / double(1.0e9);
    return gflop / runtime_s;
  }
};

/// Result structure
struct Result
{
  double avg_runtime_ms;
  double gflops;
  cutlass::Status status;
  cudaError_t error;
  bool passed;

  Result(
	 double avg_runtime_ms = 0,
	 double gflops = 0,
	 cutlass::Status status = cutlass::Status::kSuccess,
	 cudaError_t error = cudaSuccess)
    :
    avg_runtime_ms(avg_runtime_ms), gflops(gflops), status(status), error(error), passed(false)
  {}

};

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM setup and evaluation
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Helper to initialize a block of device data
template <class Element>
bool initialize_block(
		      cutlass::DeviceAllocation<Element>& block,
		      uint64_t seed=2023) {

  Element scope_max, scope_min;
  int bits_input = cutlass::sizeof_bits<Element>::value;

  if (bits_input == 1) {
    scope_max = 2;
    scope_min = 0;
  } else if (bits_input <= 8) {
    scope_max = 2;
    scope_min = -2;
  } else {
    scope_max = 8;
    scope_min = -8;
  }

  cutlass::reference::device::BlockFillRandomUniform(
						     block.get(), block.size(), seed, scope_max, scope_min, 0);

  return true;
}

/// Initialize operands to be used in the GEMM and reference GEMM
void initialize(const Options &options) {

  stride_A = make_cute_packed_stride(StrideA{}, cute::make_shape(options.m, options.k, Int<1>{}));
  stride_B = make_cute_packed_stride(StrideB{}, cute::make_shape(options.n, options.k, Int<1>{}));
  stride_C = make_cute_packed_stride(StrideC{}, cute::make_shape(options.m, options.n, Int<1>{}));
  stride_D = make_cute_packed_stride(StrideD{}, cute::make_shape(options.m, options.n, Int<1>{}));

  block_A.reset(options.m * options.k);
  block_B.reset(options.k * options.n);
  block_C1.reset(options.m * options.n);
  block_C2.reset(options.m * options.n);
  block_C3.reset(options.m * options.n);
  block_D.reset(options.m * options.n);
  block_ref_D.reset(options.m * options.n);

  initialize_block(block_A, seed + 2023);
  initialize_block(block_B, seed + 2022);
  initialize_block(block_C1, seed + 2021);
  initialize_block(block_C2, seed + 2021);
  initialize_block(block_C3, seed + 2021);
}

/// Populates a Gemm::Arguments structure from the given commandline options
typename Gemm::Arguments args_from_options(const Options &options)
{
  typename Gemm::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
      {options.m, options.n, options.k},
      block_A.get(),
      stride_A,
      block_B.get(),
      stride_B,
      {block_C1.get(), block_C2.get(), block_C3.get(), stride_C, block_D.get(), stride_D, {options.alpha, options.beta}}
  };

  return arguments;
}

/// Execute a given example GEMM computation
template <typename Gemm>
int run(Options &options)
{
  initialize(options);

  // Instantiate CUTLASS kernel depending on templates
  Gemm gemm;

  // Create a structure of gemm kernel arguments suitable for invoking an instance of Gemm
  auto arguments = args_from_options(options);

  // Using the arguments, query for extra workspace required for matrix multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Check if the problem size is supported or not
  CUTLASS_CHECK(gemm.can_implement(arguments));

  // Initialize CUTLASS kernel with arguments and workspace pointer
  CUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));

  // Correctness / Warmup iteration
  CUTLASS_CHECK(gemm.run());

  return 0;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char const **args) {
  Options options;
  options.parse(argc, args);

  run<Gemm>(options);
  return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
