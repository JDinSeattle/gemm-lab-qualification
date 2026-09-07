#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <string>

namespace gemm_lab {

__device__ __forceinline__ uint32_t shared_address(const void* pointer) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(pointer));
}

__device__ __forceinline__ void ldmatrix_x4(
    uint32_t address,
    uint32_t& value0,
    uint32_t& value1,
    uint32_t& value2,
    uint32_t& value3) {
  asm volatile(
      "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
      : "=r"(value0), "=r"(value1), "=r"(value2), "=r"(value3)
      : "r"(address));
}

__device__ __forceinline__ void ldmatrix_x2_trans(
    uint32_t address,
    uint32_t& value0,
    uint32_t& value1) {
  asm volatile(
      "ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, [%2];\n"
      : "=r"(value0), "=r"(value1)
      : "r"(address));
}

__device__ __forceinline__ void mma_m16n8k16(
    float& d0,
    float& d1,
    float& d2,
    float& d3,
    uint32_t a0,
    uint32_t a1,
    uint32_t a2,
    uint32_t a3,
    uint32_t b0,
    uint32_t b1) {
  const float c0 = d0;
  const float c1 = d1;
  const float c2 = d2;
  const float c3 = d3;
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0, %1, %2, %3}, "
      "{%4, %5, %6, %7}, "
      "{%8, %9}, "
      "{%10, %11, %12, %13};\n"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
        "f"(c0), "f"(c1), "f"(c2), "f"(c3));
}

// Element-type traits for the pipelined kernel.
//
// ldmatrix is a `.b16` move and does not interpret its payload, so the fragment
// loading path is identical for FP16 and BF16. Only the MMA opcode and the
// float/packed conversions differ.
template <typename T>
struct MmaElement;

template <>
struct MmaElement<__half> {
  using Packed = __half2;
  static constexpr const char* name = "fp16";
  __device__ static __half zero() { return __float2half(0.0f); }
  __device__ static __half from_float(float value) { return __float2half_rn(value); }
  __device__ static Packed pack(float low, float high) {
    return __floats2half2_rn(low, high);
  }
  __device__ static void mma(
      float& d0, float& d1, float& d2, float& d3,
      uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
      uint32_t b0, uint32_t b1) {
    const float c0 = d0;
    const float c1 = d1;
    const float c2 = d2;
    const float c3 = d3;
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3));
  }
};

template <>
struct MmaElement<__nv_bfloat16> {
  using Packed = __nv_bfloat162;
  static constexpr const char* name = "bf16";
  __device__ static __nv_bfloat16 zero() { return __float2bfloat16(0.0f); }
  __device__ static __nv_bfloat16 from_float(float value) {
    return __float2bfloat16_rn(value);
  }
  __device__ static Packed pack(float low, float high) {
    return __floats2bfloat162_rn(low, high);
  }
  __device__ static void mma(
      float& d0, float& d1, float& d2, float& d3,
      uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
      uint32_t b0, uint32_t b1) {
    const float c0 = d0;
    const float c1 = d1;
    const float c2 = d2;
    const float c3 = d3;
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3));
  }
};

// One warp computes one 16x8 output tile. This intentionally small kernel is
// a fragment-mapping reference: it stages row-major A/B tiles in shared
// memory, uses x4 normal ldmatrix for A, x2 transposed ldmatrix for B, and
// issues explicit m16n8k16 FP32-accumulating MMA instructions.
__global__ void ptx_mma_reference_kernel(
    const __half* a,
    const __half* b,
    __half* c,
    int m,
    int n,
    int k) {
  __shared__ __align__(16) __half shared_a[16][16];
  __shared__ __align__(16) __half shared_b[16][8];

  const int lane = threadIdx.x & 31;
  const int output_row = static_cast<int>(blockIdx.y) * 16;
  const int output_column = static_cast<int>(blockIdx.x) * 8;
  float accum0 = 0.0f;
  float accum1 = 0.0f;
  float accum2 = 0.0f;
  float accum3 = 0.0f;

  for (int k_base = 0; k_base < k; k_base += 16) {
    for (int linear = lane; linear < 16 * 16; linear += 32) {
      const int row = linear / 16;
      const int column = linear % 16;
      const int global_row = output_row + row;
      const int global_column = k_base + column;
      shared_a[row][column] =
          global_row < m && global_column < k ? a[global_row * k + global_column] : __float2half(0.0f);
    }
    for (int linear = lane; linear < 16 * 8; linear += 32) {
      const int row = linear / 8;
      const int column = linear % 8;
      const int global_row = k_base + row;
      const int global_column = output_column + column;
      shared_b[row][column] =
          global_row < k && global_column < n ? b[global_row * n + global_column] : __float2half(0.0f);
    }
    __syncwarp();

    // Four logical 8x8 A matrices are [M0K0, M1K0, M0K1, M1K1].
    const int a_matrix = lane / 8;
    const int a_row = (lane % 8) + ((a_matrix & 1) * 8);
    const int a_column = (a_matrix >> 1) * 8;
    const uint32_t a_address = shared_address(&shared_a[a_row][a_column]);

    // B is a 16x8 KxN tile. The transposed ldmatrix form produces the
    // column-major B fragment expected by mma.sync.row.col.
    const int b_address_lane = lane & 15;
    const uint32_t b_address = shared_address(&shared_b[b_address_lane][0]);

    uint32_t fragment_a0;
    uint32_t fragment_a1;
    uint32_t fragment_a2;
    uint32_t fragment_a3;
    uint32_t fragment_b0;
    uint32_t fragment_b1;
    ldmatrix_x4(a_address, fragment_a0, fragment_a1, fragment_a2, fragment_a3);
    ldmatrix_x2_trans(b_address, fragment_b0, fragment_b1);
    mma_m16n8k16(
        accum0,
        accum1,
        accum2,
        accum3,
        fragment_a0,
        fragment_a1,
        fragment_a2,
        fragment_a3,
        fragment_b0,
        fragment_b1);
    __syncwarp();
  }

  const int group = lane >> 2;
  const int thread_in_group = lane & 3;
  const int column0 = output_column + thread_in_group * 2;
  const int column1 = column0 + 1;
  const int row0 = output_row + group;
  const int row1 = row0 + 8;
  if (row0 < m && column0 < n) {
    c[row0 * n + column0] = __float2half_rn(accum0);
  }
  if (row0 < m && column1 < n) {
    c[row0 * n + column1] = __float2half_rn(accum1);
  }
  if (row1 < m && column0 < n) {
    c[row1 * n + column0] = __float2half_rn(accum2);
  }
  if (row1 < m && column1 < n) {
    c[row1 * n + column1] = __float2half_rn(accum3);
  }
}

inline void launch_ptx_mma_reference(
    const __half* a,
    const __half* b,
    __half* c,
    int m,
    int n,
    int k,
    cudaStream_t stream) {
  const dim3 grid((n + 7) / 8, (m + 15) / 16);
  ptx_mma_reference_kernel<<<grid, 32, 0, stream>>>(a, b, c, m, n, k);
}

// Static launch characteristics of a kernel, read from the driver rather than
// asserted in a comment. Hardware counters need elevated privileges on this
// host, but occupancy and resource usage do not, so these belong in the result
// artifact.
struct OccupancyReport {
  int threads_per_block = 0;
  int registers_per_thread = 0;
  int shared_bytes_per_block = 0;
  int blocks_per_multiprocessor = 0;
  int active_threads_per_multiprocessor = 0;
  double occupancy = 0.0;
};

template <typename Kernel>
inline OccupancyReport occupancy_of(Kernel kernel, int threads_per_block) {
  OccupancyReport report;
  report.threads_per_block = threads_per_block;
  cudaFuncAttributes attributes{};
  if (cudaFuncGetAttributes(&attributes, kernel) != cudaSuccess) {
    return report;
  }
  report.registers_per_thread = attributes.numRegs;
  report.shared_bytes_per_block = static_cast<int>(attributes.sharedSizeBytes);
  int blocks = 0;
  if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &blocks, kernel, threads_per_block, 0) != cudaSuccess) {
    return report;
  }
  report.blocks_per_multiprocessor = blocks;
  report.active_threads_per_multiprocessor = blocks * threads_per_block;
  int device = 0;
  cudaGetDevice(&device);
  int max_threads_per_sm = 0;
  cudaDeviceGetAttribute(
      &max_threads_per_sm, cudaDevAttrMaxThreadsPerMultiProcessor, device);
  if (max_threads_per_sm > 0) {
    report.occupancy =
        static_cast<double>(report.active_threads_per_multiprocessor) /
        static_cast<double>(max_threads_per_sm);
  }
  return report;
}

// Asynchronous 16-byte global-to-shared copy. Both addresses must be 16-byte
// aligned and fully inside their allocations; boundary groups use the predicated
// scalar path instead.
__device__ __forceinline__ void cp_async_16(
    uint32_t destination,
    const void* source) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
               :
               : "r"(destination), "l"(source)
               : "memory");
}

__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n" ::: "memory");
}

template <int PENDING>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(PENDING) : "memory");
}

// Hierarchical three-level tiled Tensor Core GEMM.
//
// The reference kernel above assigns one warp to one 16x8 output tile, which
// leaves the machine almost entirely idle: a 4096-cube launch becomes 131072
// blocks of 32 threads, each re-reading A and B rows from L2. This kernel keeps
// the same m16n8k16 mma.sync + ldmatrix fragment mapping but adds the three
// levels of tiling that make it a usable kernel:
//
//   block tile   BLOCK_M x BLOCK_N x BLOCK_K, staged in shared memory
//   warp tile    (BLOCK_M/WARPS_M) x (BLOCK_N/WARPS_N), held in registers
//   MMA tile     16x8x16, issued as explicit PTX
//
// Global loads are 16-byte vectorized when the trailing dimension permits it,
// and shared-memory strides are padded so the eight row addresses of one
// ldmatrix 8x8 block land in distinct banks.
template <
    int BLOCK_M,
    int BLOCK_N,
    int BLOCK_K,
    int WARPS_M,
    int WARPS_N,
    int A_PAD,
    int B_PAD>
__global__ __launch_bounds__(WARPS_M* WARPS_N * 32) void ptx_mma_tiled_kernel(
    const __half* __restrict__ a,
    const __half* __restrict__ b,
    __half* __restrict__ c,
    int m,
    int n,
    int k) {
  constexpr int kThreads = WARPS_M * WARPS_N * 32;
  constexpr int kWarpM = BLOCK_M / WARPS_M;
  constexpr int kWarpN = BLOCK_N / WARPS_N;
  constexpr int kTilesM = kWarpM / 16;
  constexpr int kTilesN = kWarpN / 8;
  constexpr int kKSteps = BLOCK_K / 16;
  constexpr int kAStride = BLOCK_K + A_PAD;
  constexpr int kBStride = BLOCK_N + B_PAD;
  constexpr int kAVectorsPerRow = BLOCK_K / 8;
  constexpr int kAVectors = BLOCK_M * kAVectorsPerRow;
  constexpr int kBVectorsPerRow = BLOCK_N / 8;
  constexpr int kBVectors = BLOCK_K * kBVectorsPerRow;

  static_assert(BLOCK_K % 16 == 0, "BLOCK_K must cover whole MMA k-steps");
  static_assert(kWarpM % 16 == 0 && kWarpN % 8 == 0, "warp tile must be MMA aligned");
  // Padded strides must stay 16-byte aligned or ldmatrix addresses are invalid.
  static_assert((kAStride * sizeof(__half)) % 16 == 0, "A stride must be 16B aligned");
  static_assert((kBStride * sizeof(__half)) % 16 == 0, "B stride must be 16B aligned");

  __shared__ __align__(16) __half shared_a[BLOCK_M * kAStride];
  __shared__ __align__(16) __half shared_b[BLOCK_K * kBStride];

  const int thread = static_cast<int>(threadIdx.x);
  const int lane = thread & 31;
  const int warp = thread >> 5;
  const int warp_row = (warp / WARPS_N) * kWarpM;
  const int warp_column = (warp % WARPS_N) * kWarpN;
  const int block_row = static_cast<int>(blockIdx.y) * BLOCK_M;
  const int block_column = static_cast<int>(blockIdx.x) * BLOCK_N;

  // A 16-byte load is only legal when the row pitch keeps eight-element groups
  // aligned, which requires the trailing dimension to be a multiple of eight.
  const bool vector_a = (k & 7) == 0;
  const bool vector_b = (n & 7) == 0;

  float accumulator[kTilesM][kTilesN][4];
#pragma unroll
  for (int row_tile = 0; row_tile < kTilesM; ++row_tile) {
#pragma unroll
    for (int column_tile = 0; column_tile < kTilesN; ++column_tile) {
#pragma unroll
      for (int element = 0; element < 4; ++element) {
        accumulator[row_tile][column_tile][element] = 0.0f;
      }
    }
  }

  const int a_matrix = lane >> 3;
  const int a_fragment_row = (lane & 7) + ((a_matrix & 1) << 3);
  const int a_fragment_column = (a_matrix >> 1) << 3;
  const int b_fragment_row = lane & 15;

  for (int k_base = 0; k_base < k; k_base += BLOCK_K) {
    for (int index = thread; index < kAVectors; index += kThreads) {
      const int row = index / kAVectorsPerRow;
      const int column = (index % kAVectorsPerRow) * 8;
      const int global_row = block_row + row;
      const int global_column = k_base + column;
      __half* destination = &shared_a[row * kAStride + column];
      if (vector_a && global_row < m && global_column + 8 <= k) {
        *reinterpret_cast<uint4*>(destination) = *reinterpret_cast<const uint4*>(
            &a[static_cast<size_t>(global_row) * k + global_column]);
      } else {
#pragma unroll
        for (int element = 0; element < 8; ++element) {
          destination[element] =
              (global_row < m && global_column + element < k)
                  ? a[static_cast<size_t>(global_row) * k + global_column + element]
                  : __float2half(0.0f);
        }
      }
    }
    for (int index = thread; index < kBVectors; index += kThreads) {
      const int row = index / kBVectorsPerRow;
      const int column = (index % kBVectorsPerRow) * 8;
      const int global_row = k_base + row;
      const int global_column = block_column + column;
      __half* destination = &shared_b[row * kBStride + column];
      if (vector_b && global_row < k && global_column + 8 <= n) {
        *reinterpret_cast<uint4*>(destination) = *reinterpret_cast<const uint4*>(
            &b[static_cast<size_t>(global_row) * n + global_column]);
      } else {
#pragma unroll
        for (int element = 0; element < 8; ++element) {
          destination[element] =
              (global_row < k && global_column + element < n)
                  ? b[static_cast<size_t>(global_row) * n + global_column + element]
                  : __float2half(0.0f);
        }
      }
    }
    __syncthreads();

#pragma unroll
    for (int k_step = 0; k_step < kKSteps; ++k_step) {
      const int k_offset = k_step * 16;
      uint32_t fragment_a[kTilesM][4];
      uint32_t fragment_b[kTilesN][2];
#pragma unroll
      for (int row_tile = 0; row_tile < kTilesM; ++row_tile) {
        const uint32_t address = shared_address(
            &shared_a[(warp_row + row_tile * 16 + a_fragment_row) * kAStride +
                      k_offset + a_fragment_column]);
        ldmatrix_x4(
            address,
            fragment_a[row_tile][0],
            fragment_a[row_tile][1],
            fragment_a[row_tile][2],
            fragment_a[row_tile][3]);
      }
#pragma unroll
      for (int column_tile = 0; column_tile < kTilesN; ++column_tile) {
        const uint32_t address = shared_address(
            &shared_b[(k_offset + b_fragment_row) * kBStride + warp_column +
                      column_tile * 8]);
        ldmatrix_x2_trans(address, fragment_b[column_tile][0], fragment_b[column_tile][1]);
      }
#pragma unroll
      for (int row_tile = 0; row_tile < kTilesM; ++row_tile) {
#pragma unroll
        for (int column_tile = 0; column_tile < kTilesN; ++column_tile) {
          mma_m16n8k16(
              accumulator[row_tile][column_tile][0],
              accumulator[row_tile][column_tile][1],
              accumulator[row_tile][column_tile][2],
              accumulator[row_tile][column_tile][3],
              fragment_a[row_tile][0],
              fragment_a[row_tile][1],
              fragment_a[row_tile][2],
              fragment_a[row_tile][3],
              fragment_b[column_tile][0],
              fragment_b[column_tile][1]);
        }
      }
    }
    __syncthreads();
  }

  const int group = lane >> 2;
  const int thread_in_group = lane & 3;
#pragma unroll
  for (int row_tile = 0; row_tile < kTilesM; ++row_tile) {
#pragma unroll
    for (int column_tile = 0; column_tile < kTilesN; ++column_tile) {
      const int row0 = block_row + warp_row + row_tile * 16 + group;
      const int row1 = row0 + 8;
      const int column0 = block_column + warp_column + column_tile * 8 + thread_in_group * 2;
      const int column1 = column0 + 1;
      const float* values = accumulator[row_tile][column_tile];
      if (row0 < m) {
        if (column0 < n) {
          c[static_cast<size_t>(row0) * n + column0] = __float2half_rn(values[0]);
        }
        if (column1 < n) {
          c[static_cast<size_t>(row0) * n + column1] = __float2half_rn(values[1]);
        }
      }
      if (row1 < m) {
        if (column0 < n) {
          c[static_cast<size_t>(row1) * n + column0] = __float2half_rn(values[2]);
        }
        if (column1 < n) {
          c[static_cast<size_t>(row1) * n + column1] = __float2half_rn(values[3]);
        }
      }
    }
  }
}

// Double-buffered variant of the kernel above.
//
// The single-buffered loop is strictly serial: every warp waits at a barrier for
// the block's global loads to land before any MMA can issue, so global latency is
// never hidden behind Tensor Core work. This version stages tile i+1 with
// `cp.async` into the alternate shared buffer while tile i is being multiplied,
// and only the boundary tiles fall back to predicated scalar copies.
template <
    typename T,
    int BLOCK_M,
    int BLOCK_N,
    int BLOCK_K,
    int WARPS_M,
    int WARPS_N,
    int A_PAD,
    int B_PAD,
    bool SPLIT_K = false>
__global__ __launch_bounds__(WARPS_M* WARPS_N * 32) void ptx_mma_pipelined_kernel(
    const T* __restrict__ a,
    const T* __restrict__ b,
    T* __restrict__ c,
    int m,
    int n,
    int k,
    float* __restrict__ partial,
    int splits) {
  using Element = MmaElement<T>;
  constexpr int kThreads = WARPS_M * WARPS_N * 32;
  constexpr int kWarpM = BLOCK_M / WARPS_M;
  constexpr int kWarpN = BLOCK_N / WARPS_N;
  constexpr int kTilesM = kWarpM / 16;
  constexpr int kTilesN = kWarpN / 8;
  constexpr int kKSteps = BLOCK_K / 16;
  constexpr int kAStride = BLOCK_K + A_PAD;
  constexpr int kBStride = BLOCK_N + B_PAD;
  constexpr int kAVectorsPerRow = BLOCK_K / 8;
  constexpr int kAVectors = BLOCK_M * kAVectorsPerRow;
  constexpr int kBVectorsPerRow = BLOCK_N / 8;
  constexpr int kBVectors = BLOCK_K * kBVectorsPerRow;
  constexpr int kAElements = BLOCK_M * kAStride;
  constexpr int kBElements = BLOCK_K * kBStride;

  static_assert((kAStride * sizeof(T)) % 16 == 0, "A stride must be 16B aligned");
  static_assert((kBStride * sizeof(T)) % 16 == 0, "B stride must be 16B aligned");
  static_assert(kAVectors % kThreads == 0, "A tile must divide evenly across threads");
  static_assert(kBVectors % kThreads == 0, "B tile must divide evenly across threads");

  __shared__ __align__(16) T shared_a[2 * kAElements];
  __shared__ __align__(16) T shared_b[2 * kBElements];

  const int thread = static_cast<int>(threadIdx.x);
  const int lane = thread & 31;
  const int warp = thread >> 5;
  const int warp_row = (warp / WARPS_N) * kWarpM;
  const int warp_column = (warp % WARPS_N) * kWarpN;
  const int block_row = static_cast<int>(blockIdx.y) * BLOCK_M;
  const int block_column = static_cast<int>(blockIdx.x) * BLOCK_N;
  const bool vector_a = (k & 7) == 0;
  const bool vector_b = (n & 7) == 0;

  float accumulator[kTilesM][kTilesN][4];
#pragma unroll
  for (int row_tile = 0; row_tile < kTilesM; ++row_tile) {
#pragma unroll
    for (int column_tile = 0; column_tile < kTilesN; ++column_tile) {
#pragma unroll
      for (int element = 0; element < 4; ++element) {
        accumulator[row_tile][column_tile][element] = 0.0f;
      }
    }
  }

  // Per-thread load assignment is loop invariant.
  constexpr int kALoads = kAVectors / kThreads;
  constexpr int kBLoads = kBVectors / kThreads;
  int a_row[kALoads];
  int a_column[kALoads];
  int b_row[kBLoads];
  int b_column[kBLoads];
#pragma unroll
  for (int load = 0; load < kALoads; ++load) {
    const int index = thread + load * kThreads;
    a_row[load] = index / kAVectorsPerRow;
    a_column[load] = (index % kAVectorsPerRow) * 8;
  }
#pragma unroll
  for (int load = 0; load < kBLoads; ++load) {
    const int index = thread + load * kThreads;
    b_row[load] = index / kBVectorsPerRow;
    b_column[load] = (index % kBVectorsPerRow) * 8;
  }

  const auto stage_tile = [&](int k_base, int buffer) {
    T* tile_a = &shared_a[buffer * kAElements];
    T* tile_b = &shared_b[buffer * kBElements];
#pragma unroll
    for (int load = 0; load < kALoads; ++load) {
      const int row = a_row[load];
      const int column = a_column[load];
      const int global_row = block_row + row;
      const int global_column = k_base + column;
      T* destination = &tile_a[row * kAStride + column];
      if (vector_a && global_row < m && global_column + 8 <= k) {
        cp_async_16(
            shared_address(destination),
            &a[static_cast<size_t>(global_row) * k + global_column]);
      } else {
#pragma unroll
        for (int element = 0; element < 8; ++element) {
          destination[element] =
              (global_row < m && global_column + element < k)
                  ? a[static_cast<size_t>(global_row) * k + global_column + element]
                  : Element::zero();
        }
      }
    }
#pragma unroll
    for (int load = 0; load < kBLoads; ++load) {
      const int row = b_row[load];
      const int column = b_column[load];
      const int global_row = k_base + row;
      const int global_column = block_column + column;
      T* destination = &tile_b[row * kBStride + column];
      if (vector_b && global_row < k && global_column + 8 <= n) {
        cp_async_16(
            shared_address(destination),
            &b[static_cast<size_t>(global_row) * n + global_column]);
      } else {
#pragma unroll
        for (int element = 0; element < 8; ++element) {
          destination[element] =
              (global_row < k && global_column + element < n)
                  ? b[static_cast<size_t>(global_row) * n + global_column + element]
                  : Element::zero();
        }
      }
    }
    cp_async_commit();
  };

  const int a_matrix = lane >> 3;
  const int a_fragment_row = (lane & 7) + ((a_matrix & 1) << 3);
  const int a_fragment_column = (a_matrix >> 1) << 3;
  const int b_fragment_row = lane & 15;

  const auto multiply_tile = [&](int buffer) {
    const T* tile_a = &shared_a[buffer * kAElements];
    const T* tile_b = &shared_b[buffer * kBElements];
#pragma unroll
    for (int k_step = 0; k_step < kKSteps; ++k_step) {
      const int k_offset = k_step * 16;
      uint32_t fragment_a[kTilesM][4];
      uint32_t fragment_b[kTilesN][2];
#pragma unroll
      for (int row_tile = 0; row_tile < kTilesM; ++row_tile) {
        ldmatrix_x4(
            shared_address(
                &tile_a[(warp_row + row_tile * 16 + a_fragment_row) * kAStride +
                        k_offset + a_fragment_column]),
            fragment_a[row_tile][0],
            fragment_a[row_tile][1],
            fragment_a[row_tile][2],
            fragment_a[row_tile][3]);
      }
#pragma unroll
      for (int column_tile = 0; column_tile < kTilesN; ++column_tile) {
        ldmatrix_x2_trans(
            shared_address(
                &tile_b[(k_offset + b_fragment_row) * kBStride + warp_column +
                        column_tile * 8]),
            fragment_b[column_tile][0],
            fragment_b[column_tile][1]);
      }
#pragma unroll
      for (int row_tile = 0; row_tile < kTilesM; ++row_tile) {
#pragma unroll
        for (int column_tile = 0; column_tile < kTilesN; ++column_tile) {
          Element::mma(
              accumulator[row_tile][column_tile][0],
              accumulator[row_tile][column_tile][1],
              accumulator[row_tile][column_tile][2],
              accumulator[row_tile][column_tile][3],
              fragment_a[row_tile][0],
              fragment_a[row_tile][1],
              fragment_a[row_tile][2],
              fragment_a[row_tile][3],
              fragment_b[column_tile][0],
              fragment_b[column_tile][1]);
        }
      }
    }
  };

  // Split along whole BLOCK_K tiles rather than raw k, so every split stays
  // tile-aligned and the inner loop is unchanged. A split that lands past the end
  // of k does no work and writes a zero partial, which keeps the reduction exact.
  int tile_begin = 0;
  int tile_end = (k + BLOCK_K - 1) / BLOCK_K;
  if constexpr (SPLIT_K) {
    const int total = tile_end;
    const int per_split = (total + splits - 1) / splits;
    tile_begin = min(static_cast<int>(blockIdx.z) * per_split, total);
    tile_end = min(tile_begin + per_split, total);
  }

  if (tile_begin < tile_end) {
    stage_tile(tile_begin * BLOCK_K, 0);
  }
  for (int tile = tile_begin; tile < tile_end; ++tile) {
    const int buffer = (tile - tile_begin) & 1;
    if (tile + 1 < tile_end) {
      stage_tile((tile + 1) * BLOCK_K, buffer ^ 1);
      cp_async_wait<1>();
    } else {
      cp_async_wait<0>();
    }
    __syncthreads();
    multiply_tile(buffer);
    // Tile i+2 reuses this buffer, so its staging must not start until every
    // warp has finished reading it.
    __syncthreads();
  }

  const int group = lane >> 2;
  const int thread_in_group = lane & 3;
  // A 32-bit packed store needs an even element index. The column offset is
  // always even, so this holds for every row exactly when the row pitch is even.
  const bool packed_epilogue = (n & 1) == 0;
#pragma unroll
  for (int row_tile = 0; row_tile < kTilesM; ++row_tile) {
#pragma unroll
    for (int column_tile = 0; column_tile < kTilesN; ++column_tile) {
      const int row0 = block_row + warp_row + row_tile * 16 + group;
      const int row1 = row0 + 8;
      const int column0 =
          block_column + warp_column + column_tile * 8 + thread_in_group * 2;
      const float* values = accumulator[row_tile][column_tile];
      // The two accumulator lanes are adjacent columns, so a full in-bounds pair
      // becomes one 32-bit store instead of two 16-bit stores.
#pragma unroll
      for (int half_tile = 0; half_tile < 2; ++half_tile) {
        const int row = half_tile == 0 ? row0 : row1;
        const float low = values[half_tile * 2];
        const float high = values[half_tile * 2 + 1];
        if (row >= m) {
          continue;
        }
        const size_t base = static_cast<size_t>(row) * n + column0;
        if constexpr (SPLIT_K) {
          // FP32 partials, reduced in FP32 and converted once. Deterministic:
          // no atomics, so the result does not depend on block completion order.
          float* slice = partial + static_cast<size_t>(blockIdx.z) *
                                       static_cast<size_t>(m) * static_cast<size_t>(n);
          if (column0 < n) {
            slice[base] = low;
          }
          if (column0 + 1 < n) {
            slice[base + 1] = high;
          }
          continue;
        }
        if (packed_epilogue && column0 + 1 < n) {
          *reinterpret_cast<typename Element::Packed*>(&c[base]) =
              Element::pack(low, high);
        } else {
          if (column0 < n) {
            c[base] = Element::from_float(low);
          }
          if (column0 + 1 < n) {
            c[base + 1] = Element::from_float(high);
          }
        }
      }
    }
  }
}

inline void launch_ptx_mma_tiled(
    const __half* a,
    const __half* b,
    __half* c,
    int m,
    int n,
    int k,
    cudaStream_t stream) {
  constexpr int block_m = 128;
  constexpr int block_n = 128;
  constexpr int block_k = 32;
  constexpr int warps_m = 2;
  constexpr int warps_n = 4;
  // 40-half A rows and 136-half B rows spread one ldmatrix block's eight row
  // addresses over eight distinct 4-byte banks.
  constexpr int a_pad = 8;
  constexpr int b_pad = 8;
  constexpr int threads = warps_m * warps_n * 32;
  const dim3 grid((n + block_n - 1) / block_n, (m + block_m - 1) / block_m);
  ptx_mma_tiled_kernel<block_m, block_n, block_k, warps_m, warps_n, a_pad, b_pad>
      <<<grid, threads, 0, stream>>>(a, b, c, m, n, k);
}

template <typename T>
inline void launch_ptx_mma_pipelined(
    const T* a,
    const T* b,
    T* c,
    int m,
    int n,
    int k,
    cudaStream_t stream) {
  constexpr int block_m = 128;
  constexpr int block_n = 128;
  constexpr int block_k = 32;
  constexpr int warps_m = 2;
  constexpr int warps_n = 4;
  constexpr int a_pad = 8;
  constexpr int b_pad = 8;
  constexpr int threads = warps_m * warps_n * 32;
  // Occupancy is register bound, not shared-memory bound: 128 registers across
  // 256 threads is exactly half the 64K register file, so two blocks per SM is
  // the ceiling and 37888 bytes of shared memory already fits twice. Requesting
  // a larger L1/shared carveout was measured to change nothing.
  const dim3 grid((n + block_n - 1) / block_n, (m + block_m - 1) / block_m);
  ptx_mma_pipelined_kernel<T, block_m, block_n, block_k, warps_m, warps_n, a_pad, b_pad>
      <<<grid, threads, 0, stream>>>(a, b, c, m, n, k, nullptr, 1);
}

// Unpadded twin of the small-tile kernel, kept only to measure what the padding
// actually buys. At A_PAD=B_PAD=0 the strides are 32 and 128 halves, so the eight
// row addresses of one ldmatrix block collapse onto two 4-byte bank groups
// instead of spreading over eight. Both strides stay 16-byte aligned, so this is
// a legal kernel and the only difference is the bank mapping.
template <typename T>
inline void launch_ptx_mma_small_unpadded(
    const T* a,
    const T* b,
    T* c,
    int m,
    int n,
    int k,
    cudaStream_t stream) {
  constexpr int block_m = 64;
  constexpr int block_n = 64;
  constexpr int block_k = 32;
  constexpr int warps_m = 2;
  constexpr int warps_n = 2;
  constexpr int threads = warps_m * warps_n * 32;
  const dim3 grid((n + block_n - 1) / block_n, (m + block_m - 1) / block_m);
  ptx_mma_pipelined_kernel<T, block_m, block_n, block_k, warps_m, warps_n, 0, 0>
      <<<grid, threads, 0, stream>>>(a, b, c, m, n, k, nullptr, 1);
}

template <typename T>
inline void launch_ptx_mma_pipelined_small(
    const T* a, const T* b, T* c, int m, int n, int k, cudaStream_t stream);

// Second pass of the two-pass Split-K: sum the FP32 partials and convert once.
template <typename T>
__global__ void ptx_mma_splitk_reduce_kernel(
    const float* __restrict__ partial,
    T* __restrict__ c,
    int elements,
    int splits) {
  using Element = MmaElement<T>;
  const int index = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
                    static_cast<int>(threadIdx.x);
  if (index >= elements) {
    return;
  }
  float total = 0.0f;
  for (int split = 0; split < splits; ++split) {
    total += partial[static_cast<size_t>(split) * static_cast<size_t>(elements) + index];
  }
  c[index] = Element::from_float(total);
}

// How many K splits are worth using.
//
// Counters showed `square-512` running at 0.10 waves per SM with zero shared-load
// bank conflicts and a long-scoreboard ratio of 0.16: the inner loop is clean and
// the machine is simply idle, because a 64x64 tile over 512x512 yields 64 blocks
// for 128 SMs that can hold five each.
//
// But parallelism starvation alone does not justify splitting. The reduction pass
// reads `splits * m * n` floats, and that traffic is paid whether or not the split
// helped. A first version of this function split whenever the grid was small,
// which cost `square-512` 26 percentage points -- its reduction traffic was 4x the
// GEMM's own input traffic. The governing quantity is therefore the ratio
//
//     reduction bytes / GEMM input bytes  =  4 * splits * m * n / (2 * (m*k + k*n))
//
// which measured 0.00 and 0.12 on the two shapes where splitting won, against 2.00
// and 4.00 where it lost. The 0.25 bound below sits in that gap. Splits stay powers
// of two so the reduction loop stays cheap, and stop once the grid alone can fill
// the device.
inline int ptx_mma_split_count(int m, int n, int k, int block_m, int block_n, int block_k) {
  const long long tiles = static_cast<long long>((m + block_m - 1) / block_m) *
                          static_cast<long long>((n + block_n - 1) / block_n);
  const int k_tiles = (k + block_k - 1) / block_k;
  int device = 0;
  int multiprocessors = 0;
  cudaGetDevice(&device);
  cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, device);
  if (multiprocessors <= 0) {
    return 1;
  }
  const long long target = static_cast<long long>(multiprocessors) * 2;
  const double input_bytes =
      2.0 * (static_cast<double>(m) * k + static_cast<double>(k) * n);
  constexpr double kReductionBudget = 0.25;

  int splits = 1;
  while (splits < 16) {
    const int next = splits * 2;
    if (k_tiles / next < 1) {
      break;  // no whole K tile left for each split
    }
    if (tiles * next > target) {
      // Splitting past a full grid buys no parallelism but still pays the
      // reduction. Overshooting cost `decode-m8-mlp-up` 3.9 percentage points at
      // 344 blocks against a 256 target, so the split must land inside it.
      break;
    }
    const double reduce_bytes = 4.0 * next * static_cast<double>(m) * n;
    if (reduce_bytes > kReductionBudget * input_bytes) {
      break;  // the reduction would cost more than the parallelism is worth
    }
    splits = next;
  }
  return splits;
}

inline size_t ptx_mma_splitk_workspace_bytes(int m, int n, int splits) {
  return static_cast<size_t>(splits) * static_cast<size_t>(m) * static_cast<size_t>(n) *
         sizeof(float);
}

template <typename T>
inline void launch_ptx_mma_splitk(
    const T* a,
    const T* b,
    T* c,
    int m,
    int n,
    int k,
    float* partial,
    int splits,
    cudaStream_t stream) {
  constexpr int block_m = 64;
  constexpr int block_n = 64;
  constexpr int block_k = 32;
  constexpr int warps_m = 2;
  constexpr int warps_n = 2;
  constexpr int a_pad = 8;
  constexpr int b_pad = 8;
  constexpr int threads = warps_m * warps_n * 32;
  if (splits <= 1) {
    launch_ptx_mma_pipelined_small(a, b, c, m, n, k, stream);
    return;
  }
  const dim3 grid(
      (n + block_n - 1) / block_n,
      (m + block_m - 1) / block_m,
      static_cast<unsigned int>(splits));
  ptx_mma_pipelined_kernel<T, block_m, block_n, block_k, warps_m, warps_n, a_pad, b_pad, true>
      <<<grid, threads, 0, stream>>>(a, b, c, m, n, k, partial, splits);
  const int elements = m * n;
  constexpr int reduce_threads = 256;
  ptx_mma_splitk_reduce_kernel<T>
      <<<(elements + reduce_threads - 1) / reduce_threads, reduce_threads, 0, stream>>>(
          partial, c, elements, splits);
}

// Eight-warp instantiation of the same 64x64 tile: a 32x16 warp tile instead of
// 32x32, so 16 accumulator registers per thread instead of 64.
//
// Investigation 0009 measured this as faster below 512 and slower from 1024 up,
// which reads like a clean tile/warp trade-off. Both margins were 8-9%, inside the
// 11% clock envelope, and the two configurations were measured either side of a
// rebuild -- so the comparison is reported as unresolved rather than as a
// crossover. It stays a runnable implementation specifically so that the
// comparison can be settled once clocks are locked; a documented open question
// with no way to execute it is not an open question, it is a note.
template <typename T>
inline void launch_ptx_mma_small_8warp(
    const T* a,
    const T* b,
    T* c,
    int m,
    int n,
    int k,
    cudaStream_t stream) {
  constexpr int block_m = 64;
  constexpr int block_n = 64;
  constexpr int block_k = 32;
  constexpr int warps_m = 2;
  constexpr int warps_n = 4;
  constexpr int a_pad = 8;
  constexpr int b_pad = 8;
  constexpr int threads = warps_m * warps_n * 32;
  const dim3 grid((n + block_n - 1) / block_n, (m + block_m - 1) / block_m);
  ptx_mma_pipelined_kernel<T, block_m, block_n, block_k, warps_m, warps_n, a_pad, b_pad>
      <<<grid, threads, 0, stream>>>(a, b, c, m, n, k, nullptr, 1);
}

// Occupancy of the inline-PTX kernels, keyed by the CLI implementation name.
inline OccupancyReport ptx_mma_occupancy(const char* implementation) {
  const std::string name(implementation);
  if (name == "ptx_mma") {
    return occupancy_of(ptx_mma_reference_kernel, 32);
  }
  if (name == "ptx_mma_tiled") {
    return occupancy_of(ptx_mma_tiled_kernel<128, 128, 32, 2, 4, 8, 8>, 256);
  }
  if (name == "ptx_mma_pipelined") {
    return occupancy_of(ptx_mma_pipelined_kernel<__half, 128, 128, 32, 2, 4, 8, 8>, 256);
  }
  if (name == "ptx_mma_small") {
    return occupancy_of(ptx_mma_pipelined_kernel<__half, 64, 64, 32, 2, 2, 8, 8>, 128);
  }
  if (name == "ptx_mma_small_8warp") {
    return occupancy_of(ptx_mma_pipelined_kernel<__half, 64, 64, 32, 2, 4, 8, 8>, 256);
  }
  if (name == "ptx_mma_small_unpadded") {
    return occupancy_of(ptx_mma_pipelined_kernel<__half, 64, 64, 32, 2, 2, 0, 0>, 128);
  }
  return OccupancyReport{};
}

// Small-tile instantiation of the same pipelined kernel.
//
// A 128x128 block tile cannot fill this GPU below roughly 2048 rows: a 512-cube
// GEMM produces a 4x4 grid, which is 16 blocks for 128 SMs. Measured throughput
// tracks that grid occupancy almost exactly, so the fix is a smaller tile rather
// than a different inner loop. 64x64x32 across four warps quadruples the block
// count for the same output.
template <typename T>
inline void launch_ptx_mma_pipelined_small(
    const T* a,
    const T* b,
    T* c,
    int m,
    int n,
    int k,
    cudaStream_t stream) {
  constexpr int block_m = 64;
  constexpr int block_n = 64;
  constexpr int block_k = 32;
  constexpr int warps_m = 2;
  constexpr int warps_n = 2;
  constexpr int a_pad = 8;
  constexpr int b_pad = 8;
  constexpr int threads = warps_m * warps_n * 32;
  const dim3 grid((n + block_n - 1) / block_n, (m + block_m - 1) / block_m);
  ptx_mma_pipelined_kernel<T, block_m, block_n, block_k, warps_m, warps_n, a_pad, b_pad>
      <<<grid, threads, 0, stream>>>(a, b, c, m, n, k, nullptr, 1);
}


}  // namespace gemm_lab
