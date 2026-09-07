#include "gemm_lab/cuda_check.hpp"
#include "ptx_mma.cuh"

#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

struct Options {
  int64_t m = 512;
  int64_t n = 512;
  int64_t k = 512;
  std::string dtype = "fp16";
  std::string implementation = "cublas";
  std::string cache_mode = "steady";
  int warmup = 25;
  double warmup_seconds = 0.0;
  int samples = 100;
  double measurement_seconds = 0.0;
  double min_sample_ms = 0.25;
  int max_inner_repetitions = 100;
  int seed = 17;
  int correctness_samples = 64;
  int split_k_override = 0;  // 0 = let the cost model choose
  bool pretty = false;
};

struct ErrorMetrics {
  bool passed = false;
  double atol = 0.0;
  double rtol = 0.0;
  double max_abs = 0.0;
  double max_relative = 0.0;
  double normalized_l2 = 0.0;
  int64_t nan_count = 0;
  int64_t inf_count = 0;
  int64_t elements_checked = 0;
  std::string reference;
};

struct TimingMetrics {
  double p05_ms = 0.0;
  double p50_ms = 0.0;
  double p95_ms = 0.0;
  double mean_ms = 0.0;
  double stddev_ms = 0.0;
  double mad_ms = 0.0;
  double cv = 0.0;
  double tflops_p50 = 0.0;
};

template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(size_t count) : count_(count) {
    if (count_ > 0) {
      GEMM_LAB_CUDA_CHECK(cudaMalloc(&pointer_, count_ * sizeof(T)));
    }
  }

  ~DeviceBuffer() {
    if (pointer_ != nullptr) {
      cudaFree(pointer_);
    }
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  DeviceBuffer(DeviceBuffer&& other) noexcept
      : pointer_(std::exchange(other.pointer_, nullptr)), count_(std::exchange(other.count_, 0)) {}

  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this == &other) {
      return *this;
    }
    if (pointer_ != nullptr) {
      cudaFree(pointer_);
    }
    pointer_ = std::exchange(other.pointer_, nullptr);
    count_ = std::exchange(other.count_, 0);
    return *this;
  }

  T* get() { return pointer_; }
  const T* get() const { return pointer_; }
  size_t size() const { return count_; }
  size_t bytes() const { return count_ * sizeof(T); }

 private:
  T* pointer_ = nullptr;
  size_t count_ = 0;
};

class CudaStream {
 public:
  CudaStream() { GEMM_LAB_CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking)); }
  ~CudaStream() { cudaStreamDestroy(stream_); }
  CudaStream(const CudaStream&) = delete;
  CudaStream& operator=(const CudaStream&) = delete;
  cudaStream_t get() const { return stream_; }

 private:
  cudaStream_t stream_ = nullptr;
};

class CudaEvent {
 public:
  CudaEvent() { GEMM_LAB_CUDA_CHECK(cudaEventCreate(&event_)); }
  ~CudaEvent() { cudaEventDestroy(event_); }
  CudaEvent(const CudaEvent&) = delete;
  CudaEvent& operator=(const CudaEvent&) = delete;
  cudaEvent_t get() const { return event_; }

 private:
  cudaEvent_t event_ = nullptr;
};

class CublasHandle {
 public:
  explicit CublasHandle(cudaStream_t stream) {
    GEMM_LAB_CUBLAS_CHECK(cublasCreate(&handle_));
    GEMM_LAB_CUBLAS_CHECK(cublasSetStream(handle_, stream));
  }
  ~CublasHandle() { cublasDestroy(handle_); }
  CublasHandle(const CublasHandle&) = delete;
  CublasHandle& operator=(const CublasHandle&) = delete;
  cublasHandle_t get() const { return handle_; }

 private:
  cublasHandle_t handle_ = nullptr;
};

template <typename T>
struct CudaDataType;

template <typename T>
class CublasLtPlan {
 public:
  CublasLtPlan(int m, int n, int k, size_t max_workspace_bytes) {
    GEMM_LAB_CUBLAS_CHECK(cublasLtCreate(&handle_));
    GEMM_LAB_CUBLAS_CHECK(cublasLtMatmulDescCreate(
        &operation_, CudaDataType<T>::compute, CUDA_R_32F));
    GEMM_LAB_CUBLAS_CHECK(cublasLtMatrixLayoutCreate(
        &a_layout_, CudaDataType<T>::value, m, k, k));
    GEMM_LAB_CUBLAS_CHECK(cublasLtMatrixLayoutCreate(
        &b_layout_, CudaDataType<T>::value, k, n, n));
    GEMM_LAB_CUBLAS_CHECK(cublasLtMatrixLayoutCreate(
        &c_layout_, CudaDataType<T>::value, m, n, n));

    const cublasLtOrder_t row_order = CUBLASLT_ORDER_ROW;
    GEMM_LAB_CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
        a_layout_, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)));
    GEMM_LAB_CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
        b_layout_, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)));
    GEMM_LAB_CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
        c_layout_, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)));

    GEMM_LAB_CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference_));
    GEMM_LAB_CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
        preference_,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &max_workspace_bytes,
        sizeof(max_workspace_bytes)));

    cublasLtMatmulHeuristicResult_t heuristic{};
    int returned_results = 0;
    GEMM_LAB_CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(
        handle_,
        operation_,
        a_layout_,
        b_layout_,
        c_layout_,
        c_layout_,
        preference_,
        1,
        &heuristic,
        &returned_results));
    if (returned_results == 0 || heuristic.state != CUBLAS_STATUS_SUCCESS) {
      throw std::runtime_error("cuBLASLt did not return a usable matmul heuristic");
    }
    algorithm_ = heuristic.algo;
    workspace_bytes_ = heuristic.workspaceSize;
    if (workspace_bytes_ > 0) {
      GEMM_LAB_CUDA_CHECK(cudaMalloc(&workspace_, workspace_bytes_));
    }
  }

  ~CublasLtPlan() {
    if (workspace_ != nullptr) {
      cudaFree(workspace_);
    }
    if (preference_ != nullptr) {
      cublasLtMatmulPreferenceDestroy(preference_);
    }
    if (c_layout_ != nullptr) {
      cublasLtMatrixLayoutDestroy(c_layout_);
    }
    if (b_layout_ != nullptr) {
      cublasLtMatrixLayoutDestroy(b_layout_);
    }
    if (a_layout_ != nullptr) {
      cublasLtMatrixLayoutDestroy(a_layout_);
    }
    if (operation_ != nullptr) {
      cublasLtMatmulDescDestroy(operation_);
    }
    if (handle_ != nullptr) {
      cublasLtDestroy(handle_);
    }
  }

  CublasLtPlan(const CublasLtPlan&) = delete;
  CublasLtPlan& operator=(const CublasLtPlan&) = delete;

  void run(const T* a, const T* b, T* c, cudaStream_t stream) const {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    GEMM_LAB_CUBLAS_CHECK(cublasLtMatmul(
        handle_,
        operation_,
        &alpha,
        a,
        a_layout_,
        b,
        b_layout_,
        &beta,
        c,
        c_layout_,
        c,
        c_layout_,
        &algorithm_,
        workspace_,
        workspace_bytes_,
        stream));
  }

  size_t workspace_bytes() const { return workspace_bytes_; }

 private:
  cublasLtHandle_t handle_ = nullptr;
  cublasLtMatmulDesc_t operation_ = nullptr;
  cublasLtMatrixLayout_t a_layout_ = nullptr;
  cublasLtMatrixLayout_t b_layout_ = nullptr;
  cublasLtMatrixLayout_t c_layout_ = nullptr;
  cublasLtMatmulPreference_t preference_ = nullptr;
  cublasLtMatmulAlgo_t algorithm_{};
  void* workspace_ = nullptr;
  size_t workspace_bytes_ = 0;
};

[[noreturn]] void fail_option(const std::string& message) {
  throw std::invalid_argument("argument error: " + message);
}

int64_t parse_positive_int64(const std::string& name, const std::string& value) {
  size_t consumed = 0;
  long long parsed = 0;
  try {
    parsed = std::stoll(value, &consumed);
  } catch (const std::exception&) {
    fail_option(name + " must be an integer, received '" + value + "'");
  }
  if (consumed != value.size() || parsed <= 0) {
    fail_option(name + " must be a positive integer, received '" + value + "'");
  }
  return static_cast<int64_t>(parsed);
}

int parse_nonnegative_int(const std::string& name, const std::string& value) {
  size_t consumed = 0;
  long parsed = 0;
  try {
    parsed = std::stol(value, &consumed);
  } catch (const std::exception&) {
    fail_option(name + " must be an integer, received '" + value + "'");
  }
  if (consumed != value.size() || parsed < 0 || parsed > std::numeric_limits<int>::max()) {
    fail_option(name + " must be a non-negative integer, received '" + value + "'");
  }
  return static_cast<int>(parsed);
}

double parse_nonnegative_double(const std::string& name, const std::string& value) {
  size_t consumed = 0;
  double parsed = 0.0;
  try {
    parsed = std::stod(value, &consumed);
  } catch (const std::exception&) {
    fail_option(name + " must be a number, received '" + value + "'");
  }
  if (consumed != value.size() || !std::isfinite(parsed) || parsed < 0.0) {
    fail_option(name + " must be a non-negative finite number, received '" + value + "'");
  }
  return parsed;
}

void print_help() {
  std::cout
      << "LLM GEMM Qualification Lab native benchmark\n\n"
      << "Usage: gemm_bench [options]\n\n"
      << "  --m INT                    Rows of A/C (default: 512)\n"
      << "  --n INT                    Columns of B/C (default: 512)\n"
      << "  --k INT                    Reduction dimension (default: 512)\n"
      << "  --dtype fp16|bf16|fp32     Input/output dtype (default: fp16)\n"
      << "  --impl cublas|cublaslt|naive|wmma|wmma_tiled|wmma_tiled_padded|ptx_mma|\n"
      << "         ptx_mma_tiled|ptx_mma_pipelined|ptx_mma_small|\n"
      << "         ptx_mma_small_unpadded|ptx_mma_splitk|ptx_mma_small_8warp\n"
      << "                             Implementation (default: cublas)\n"
      << "  --cache steady|cold        Cache protocol (default: steady)\n"
      << "  --warmup INT               Warmup launches (default: 25)\n"
      << "  --warmup-seconds NUMBER    Minimum warmup duration (default: 0)\n"
      << "  --samples INT              Timed samples (default: 100)\n"
      << "  --measurement-seconds NUM  Minimum timed duration (default: 0)\n"
      << "  --min-sample-ms NUM        Minimum aggregate event duration (default: 0.25)\n"
      << "  --max-inner-repetitions N  Maximum launches per event sample (default: 100)\n"
      << "  --seed INT                 Deterministic input seed (default: 17)\n"
      << "  --correctness-samples INT  CPU FP64 spot checks for cuBLAS (default: 64)\n"
      << "  --split-k INT              Override the Split-K cost model (0 = auto)\n"
      << "  --pretty                    Pretty-print JSON\n"
      << "  --help                      Show this help\n";
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--help") {
      print_help();
      std::exit(0);
    }
    if (argument == "--pretty") {
      options.pretty = true;
      continue;
    }
    if (index + 1 >= argc) {
      fail_option("missing value for " + argument);
    }
    const std::string value = argv[++index];
    if (argument == "--m") {
      options.m = parse_positive_int64(argument, value);
    } else if (argument == "--n") {
      options.n = parse_positive_int64(argument, value);
    } else if (argument == "--k") {
      options.k = parse_positive_int64(argument, value);
    } else if (argument == "--dtype") {
      options.dtype = value;
    } else if (argument == "--impl") {
      options.implementation = value;
    } else if (argument == "--cache") {
      options.cache_mode = value;
    } else if (argument == "--warmup") {
      options.warmup = parse_nonnegative_int(argument, value);
    } else if (argument == "--warmup-seconds") {
      options.warmup_seconds = parse_nonnegative_double(argument, value);
    } else if (argument == "--samples") {
      options.samples = parse_nonnegative_int(argument, value);
    } else if (argument == "--measurement-seconds") {
      options.measurement_seconds = parse_nonnegative_double(argument, value);
    } else if (argument == "--min-sample-ms") {
      options.min_sample_ms = parse_nonnegative_double(argument, value);
    } else if (argument == "--max-inner-repetitions") {
      options.max_inner_repetitions = parse_nonnegative_int(argument, value);
    } else if (argument == "--seed") {
      options.seed = parse_nonnegative_int(argument, value);
    } else if (argument == "--correctness-samples") {
      options.correctness_samples = parse_nonnegative_int(argument, value);
    } else if (argument == "--split-k") {
      options.split_k_override = parse_nonnegative_int(argument, value);
    } else {
      fail_option("unknown option " + argument);
    }
  }

  if (options.dtype != "fp16" && options.dtype != "bf16" && options.dtype != "fp32") {
    fail_option("--dtype must be fp16, bf16, or fp32");
  }
  if (options.implementation != "cublas" && options.implementation != "cublaslt" &&
      options.implementation != "naive" && options.implementation != "wmma" &&
      options.implementation != "wmma_tiled" && options.implementation != "wmma_tiled_padded" &&
      options.implementation != "ptx_mma" && options.implementation != "ptx_mma_tiled" &&
      options.implementation != "ptx_mma_pipelined" &&
      options.implementation != "ptx_mma_small" &&
      options.implementation != "ptx_mma_small_unpadded" &&
      options.implementation != "ptx_mma_splitk" &&
      options.implementation != "ptx_mma_small_8warp") {
    fail_option("unsupported --impl value; run --help for the implementation list");
  }
  if ((options.implementation == "wmma" || options.implementation == "wmma_tiled") &&
      options.dtype != "fp16") {
    fail_option("WMMA implementations currently support --dtype fp16 only");
  }
  if (options.implementation == "wmma_tiled_padded" && options.dtype != "fp16") {
    fail_option("WMMA implementations currently support --dtype fp16 only");
  }
  if ((options.implementation == "ptx_mma" || options.implementation == "ptx_mma_tiled") &&
      options.dtype != "fp16") {
    fail_option("the PTX reference and single-buffered kernels support --dtype fp16 only");
  }
  if ((options.implementation == "ptx_mma_pipelined" ||
       options.implementation == "ptx_mma_small" ||
       options.implementation == "ptx_mma_small_unpadded") &&
      options.dtype != "fp16" && options.dtype != "bf16") {
    fail_option("pipelined inline PTX kernels support --dtype fp16 or bf16");
  }
  if (options.cache_mode != "steady" && options.cache_mode != "cold") {
    fail_option("--cache must be steady or cold");
  }
  if (options.samples == 0) {
    fail_option("--samples must be greater than zero");
  }
  if (options.max_inner_repetitions == 0) {
    fail_option("--max-inner-repetitions must be greater than zero");
  }
  if (options.correctness_samples == 0) {
    fail_option("--correctness-samples must be greater than zero");
  }
  if (options.m > std::numeric_limits<int>::max() || options.n > std::numeric_limits<int>::max() ||
      options.k > std::numeric_limits<int>::max()) {
    fail_option("dimensions must fit in a signed 32-bit cuBLAS dimension");
  }
  return options;
}

template <typename T>
float to_float(T value);

template <>
float to_float<float>(float value) {
  return value;
}

template <>
float to_float<__half>(__half value) {
  return __half2float(value);
}

template <>
float to_float<__nv_bfloat16>(__nv_bfloat16 value) {
  return __bfloat162float(value);
}

template <typename T>
T from_float(float value);

template <>
float from_float<float>(float value) {
  return value;
}

template <>
__half from_float<__half>(float value) {
  return __float2half_rn(value);
}

template <>
__nv_bfloat16 from_float<__nv_bfloat16>(float value) {
  return __float2bfloat16_rn(value);
}

template <typename T>
std::vector<T> make_input(size_t count, int seed, int salt) {
  std::mt19937 generator(static_cast<uint32_t>(seed * 131 + salt));
  std::uniform_real_distribution<float> distribution(-0.25f, 0.25f);
  std::vector<T> values(count);
  for (T& value : values) {
    value = from_float<T>(distribution(generator));
  }
  return values;
}

template <typename T>
__global__ void naive_gemm_kernel(const T* a, const T* b, T* c, int m, int n, int k) {
  const int row = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  const int column = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= m || column >= n) {
    return;
  }
  float accumulator = 0.0f;
  for (int inner = 0; inner < k; ++inner) {
    accumulator += static_cast<float>(a[static_cast<size_t>(row) * k + inner]) *
                   static_cast<float>(b[static_cast<size_t>(inner) * n + column]);
  }
  c[static_cast<size_t>(row) * n + column] = static_cast<T>(accumulator);
}

template <>
__global__ void naive_gemm_kernel<__half>(
    const __half* a, const __half* b, __half* c, int m, int n, int k) {
  const int row = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  const int column = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= m || column >= n) {
    return;
  }
  float accumulator = 0.0f;
  for (int inner = 0; inner < k; ++inner) {
    accumulator += __half2float(a[static_cast<size_t>(row) * k + inner]) *
                   __half2float(b[static_cast<size_t>(inner) * n + column]);
  }
  c[static_cast<size_t>(row) * n + column] = __float2half_rn(accumulator);
}

template <>
__global__ void naive_gemm_kernel<__nv_bfloat16>(
    const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* c, int m, int n, int k) {
  const int row = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  const int column = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= m || column >= n) {
    return;
  }
  float accumulator = 0.0f;
  for (int inner = 0; inner < k; ++inner) {
    accumulator += __bfloat162float(a[static_cast<size_t>(row) * k + inner]) *
                   __bfloat162float(b[static_cast<size_t>(inner) * n + column]);
  }
  c[static_cast<size_t>(row) * n + column] = __float2bfloat16_rn(accumulator);
}

__global__ void evict_l2_kernel(uint32_t* buffer, size_t elements) {
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) {
    buffer[index] = static_cast<uint32_t>(index);
  }
}

__global__ void wmma_fp16_gemm_kernel(
    const __half* a,
    const __half* b,
    __half* c,
    int m,
    int n,
    int k) {
  using namespace nvcuda;
  constexpr int tile_m = 16;
  constexpr int tile_n = 16;
  constexpr int tile_k = 16;
  constexpr int elements_per_tile = tile_m * tile_n;
  constexpr int warps_per_block = 4;

  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int tiles_n = (n + tile_n - 1) / tile_n;
  const int total_tiles = ((m + tile_m - 1) / tile_m) * tiles_n;
  const int output_tile = static_cast<int>(blockIdx.x) * warps_per_block + warp;
  if (output_tile >= total_tiles) {
    return;
  }

  const int tile_row = output_tile / tiles_n;
  const int tile_column = output_tile % tiles_n;
  const int row_origin = tile_row * tile_m;
  const int column_origin = tile_column * tile_n;

  extern __shared__ unsigned char shared_raw[];
  __half* shared_half = reinterpret_cast<__half*>(shared_raw);
  __half* shared_a = shared_half + warp * (2 * elements_per_tile);
  __half* shared_b = shared_a + elements_per_tile;
  float* shared_float = reinterpret_cast<float*>(
      shared_half + warps_per_block * (2 * elements_per_tile));
  float* shared_c = shared_float + warp * elements_per_tile;

  wmma::fragment<wmma::matrix_a, tile_m, tile_n, tile_k, __half, wmma::row_major> a_fragment;
  wmma::fragment<wmma::matrix_b, tile_m, tile_n, tile_k, __half, wmma::row_major> b_fragment;
  wmma::fragment<wmma::accumulator, tile_m, tile_n, tile_k, float> accumulator;
  wmma::fill_fragment(accumulator, 0.0f);

  for (int inner_origin = 0; inner_origin < k; inner_origin += tile_k) {
    for (int index = lane; index < elements_per_tile; index += 32) {
      const int tile_row_index = index / tile_k;
      const int tile_column_index = index % tile_k;
      const int global_a_row = row_origin + tile_row_index;
      const int global_a_column = inner_origin + tile_column_index;
      const int global_b_row = inner_origin + tile_row_index;
      const int global_b_column = column_origin + tile_column_index;
      shared_a[index] = (global_a_row < m && global_a_column < k)
                            ? a[static_cast<size_t>(global_a_row) * k + global_a_column]
                            : __float2half(0.0f);
      shared_b[index] = (global_b_row < k && global_b_column < n)
                            ? b[static_cast<size_t>(global_b_row) * n + global_b_column]
                            : __float2half(0.0f);
    }
    __syncwarp();
    wmma::load_matrix_sync(a_fragment, shared_a, tile_k);
    wmma::load_matrix_sync(b_fragment, shared_b, tile_n);
    wmma::mma_sync(accumulator, a_fragment, b_fragment, accumulator);
    __syncwarp();
  }

  wmma::store_matrix_sync(shared_c, accumulator, tile_n, wmma::mem_row_major);
  __syncwarp();
  for (int index = lane; index < elements_per_tile; index += 32) {
    const int tile_row_index = index / tile_n;
    const int tile_column_index = index % tile_n;
    const int global_row = row_origin + tile_row_index;
    const int global_column = column_origin + tile_column_index;
    if (global_row < m && global_column < n) {
      c[static_cast<size_t>(global_row) * n + global_column] = __float2half_rn(shared_c[index]);
    }
  }
}

template <int shared_a_stride, int shared_b_stride>
__global__ void wmma_tiled_fp16_gemm_kernel(
    const __half* a,
    const __half* b,
    __half* c,
    int m,
    int n,
    int k) {
  using namespace nvcuda;
  constexpr int block_m = 64;
  constexpr int block_n = 64;
  constexpr int block_k = 16;
  constexpr int warp_m = 32;
  constexpr int warp_n = 32;
  constexpr int mma_m = 16;
  constexpr int mma_n = 16;
  constexpr int mma_k = 16;

  const int warp = static_cast<int>(threadIdx.x) / 32;
  const int block_row = static_cast<int>(blockIdx.y) * block_m;
  const int block_column = static_cast<int>(blockIdx.x) * block_n;
  const int warp_row = (warp / 2) * warp_m;
  const int warp_column = (warp % 2) * warp_n;

  extern __shared__ unsigned char shared_raw[];
  __half* shared_a = reinterpret_cast<__half*>(shared_raw);
  __half* shared_b = shared_a + block_m * shared_a_stride;
  float* shared_c = reinterpret_cast<float*>(shared_raw);

  wmma::fragment<wmma::accumulator, mma_m, mma_n, mma_k, float> accumulator[2][2];
#pragma unroll
  for (int row_fragment = 0; row_fragment < 2; ++row_fragment) {
#pragma unroll
    for (int column_fragment = 0; column_fragment < 2; ++column_fragment) {
      wmma::fill_fragment(accumulator[row_fragment][column_fragment], 0.0f);
    }
  }

  for (int inner_origin = 0; inner_origin < k; inner_origin += block_k) {
    for (int index = static_cast<int>(threadIdx.x); index < block_m * block_k;
         index += static_cast<int>(blockDim.x)) {
      const int local_row = index / block_k;
      const int local_column = index % block_k;
      const int global_row = block_row + local_row;
      const int global_column = inner_origin + local_column;
      shared_a[local_row * shared_a_stride + local_column] =
          (global_row < m && global_column < k)
              ? a[static_cast<size_t>(global_row) * k + global_column]
              : __float2half(0.0f);
    }
    for (int index = static_cast<int>(threadIdx.x); index < block_k * block_n;
         index += static_cast<int>(blockDim.x)) {
      const int local_row = index / block_n;
      const int local_column = index % block_n;
      const int global_row = inner_origin + local_row;
      const int global_column = block_column + local_column;
      shared_b[local_row * shared_b_stride + local_column] =
          (global_row < k && global_column < n)
              ? b[static_cast<size_t>(global_row) * n + global_column]
              : __float2half(0.0f);
    }
    __syncthreads();

    wmma::fragment<wmma::matrix_a, mma_m, mma_n, mma_k, __half, wmma::row_major> a_fragment[2];
    wmma::fragment<wmma::matrix_b, mma_m, mma_n, mma_k, __half, wmma::row_major> b_fragment[2];
#pragma unroll
    for (int row_fragment = 0; row_fragment < 2; ++row_fragment) {
      wmma::load_matrix_sync(
          a_fragment[row_fragment],
          shared_a + (warp_row + row_fragment * mma_m) * shared_a_stride,
          shared_a_stride);
    }
#pragma unroll
    for (int column_fragment = 0; column_fragment < 2; ++column_fragment) {
      wmma::load_matrix_sync(
          b_fragment[column_fragment],
          shared_b + warp_column + column_fragment * mma_n,
          shared_b_stride);
    }
#pragma unroll
    for (int row_fragment = 0; row_fragment < 2; ++row_fragment) {
#pragma unroll
      for (int column_fragment = 0; column_fragment < 2; ++column_fragment) {
        wmma::mma_sync(
            accumulator[row_fragment][column_fragment],
            a_fragment[row_fragment],
            b_fragment[column_fragment],
            accumulator[row_fragment][column_fragment]);
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int row_fragment = 0; row_fragment < 2; ++row_fragment) {
#pragma unroll
    for (int column_fragment = 0; column_fragment < 2; ++column_fragment) {
      wmma::store_matrix_sync(
          shared_c + (warp_row + row_fragment * mma_m) * block_n +
              warp_column + column_fragment * mma_n,
          accumulator[row_fragment][column_fragment],
          block_n,
          wmma::mem_row_major);
    }
  }
  __syncthreads();

  for (int index = static_cast<int>(threadIdx.x); index < block_m * block_n;
       index += static_cast<int>(blockDim.x)) {
    const int local_row = index / block_n;
    const int local_column = index % block_n;
    const int global_row = block_row + local_row;
    const int global_column = block_column + local_column;
    if (global_row < m && global_column < n) {
      c[static_cast<size_t>(global_row) * n + global_column] = __float2half_rn(shared_c[index]);
    }
  }
}

template <>
struct CudaDataType<float> {
  static constexpr cudaDataType_t value = CUDA_R_32F;
  static constexpr cublasComputeType_t compute = CUBLAS_COMPUTE_32F_PEDANTIC;
};

template <>
struct CudaDataType<__half> {
  static constexpr cudaDataType_t value = CUDA_R_16F;
  static constexpr cublasComputeType_t compute = CUBLAS_COMPUTE_32F;
};

template <>
struct CudaDataType<__nv_bfloat16> {
  static constexpr cudaDataType_t value = CUDA_R_16BF;
  static constexpr cublasComputeType_t compute = CUBLAS_COMPUTE_32F;
};

template <typename T>
void launch_cublas(
    cublasHandle_t handle,
    const T* a,
    const T* b,
    T* c,
    int m,
    int n,
    int k) {
  const float alpha = 1.0f;
  const float beta = 0.0f;
  // cuBLAS is column-major. Row-major C=A*B is evaluated as C^T=B^T*A^T
  // by swapping A/B and the M/N dimensions without materializing transposes.
  GEMM_LAB_CUBLAS_CHECK(cublasGemmEx(
      handle,
      CUBLAS_OP_N,
      CUBLAS_OP_N,
      n,
      m,
      k,
      &alpha,
      b,
      CudaDataType<T>::value,
      n,
      a,
      CudaDataType<T>::value,
      k,
      &beta,
      c,
      CudaDataType<T>::value,
      n,
      CudaDataType<T>::compute,
      CUBLAS_GEMM_DEFAULT));
}

template <typename T>
void launch_naive(const T* a, const T* b, T* c, int m, int n, int k, cudaStream_t stream) {
  const dim3 block(16, 16);
  const dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
  naive_gemm_kernel<T><<<grid, block, 0, stream>>>(a, b, c, m, n, k);
  GEMM_LAB_CUDA_CHECK(cudaGetLastError());
}

void launch_wmma(
    const __half* a,
    const __half* b,
    __half* c,
    int m,
    int n,
    int k,
    cudaStream_t stream) {
  constexpr int warps_per_block = 4;
  constexpr int threads = warps_per_block * 32;
  constexpr int elements_per_tile = 16 * 16;
  const int output_tiles = ((m + 15) / 16) * ((n + 15) / 16);
  const int blocks = (output_tiles + warps_per_block - 1) / warps_per_block;
  const size_t shared_bytes = warps_per_block *
                              (2 * elements_per_tile * sizeof(__half) +
                               elements_per_tile * sizeof(float));
  wmma_fp16_gemm_kernel<<<blocks, threads, shared_bytes, stream>>>(a, b, c, m, n, k);
  GEMM_LAB_CUDA_CHECK(cudaGetLastError());
}

void launch_wmma_tiled(
    const __half* a,
    const __half* b,
    __half* c,
    int m,
    int n,
    int k,
    cudaStream_t stream) {
  constexpr int threads = 4 * 32;
  constexpr int block_m = 64;
  constexpr int block_n = 64;
  constexpr int block_k = 16;
  constexpr int shared_a_stride = block_k;
  constexpr int shared_b_stride = block_n;
  const dim3 grid((n + block_n - 1) / block_n, (m + block_m - 1) / block_m);
  const size_t ab_shared_bytes =
      (block_m * shared_a_stride + block_k * shared_b_stride) * sizeof(__half);
  const size_t c_shared_bytes = block_m * block_n * sizeof(float);
  const size_t shared_bytes = std::max(ab_shared_bytes, c_shared_bytes);
  wmma_tiled_fp16_gemm_kernel<shared_a_stride, shared_b_stride>
      <<<grid, threads, shared_bytes, stream>>>(a, b, c, m, n, k);
  GEMM_LAB_CUDA_CHECK(cudaGetLastError());
}

void launch_wmma_tiled_padded(
    const __half* a,
    const __half* b,
    __half* c,
    int m,
    int n,
    int k,
    cudaStream_t stream) {
  constexpr int threads = 4 * 32;
  constexpr int block_m = 64;
  constexpr int block_n = 64;
  constexpr int block_k = 16;
  constexpr int shared_a_stride = block_k + 8;
  constexpr int shared_b_stride = block_n + 8;
  const dim3 grid((n + block_n - 1) / block_n, (m + block_m - 1) / block_m);
  const size_t ab_shared_bytes =
      (block_m * shared_a_stride + block_k * shared_b_stride) * sizeof(__half);
  const size_t c_shared_bytes = block_m * block_n * sizeof(float);
  const size_t shared_bytes = std::max(ab_shared_bytes, c_shared_bytes);
  wmma_tiled_fp16_gemm_kernel<shared_a_stride, shared_b_stride>
      <<<grid, threads, shared_bytes, stream>>>(a, b, c, m, n, k);
  GEMM_LAB_CUDA_CHECK(cudaGetLastError());
}

template <typename T>
void launch_implementation(
    const Options& options,
    cublasHandle_t handle,
    const T* a,
    const T* b,
    T* c,
    cudaStream_t stream,
    const CublasLtPlan<T>* lt_plan,
    float* splitk_workspace,
    int splitk_splits) {
  const int m = static_cast<int>(options.m);
  const int n = static_cast<int>(options.n);
  const int k = static_cast<int>(options.k);
  if (options.implementation == "cublas") {
    launch_cublas(handle, a, b, c, m, n, k);
  } else if (options.implementation == "cublaslt") {
    if (lt_plan == nullptr) {
      throw std::logic_error("cuBLASLt implementation requires a prepared plan");
    }
    lt_plan->run(a, b, c, stream);
  } else if (options.implementation == "wmma") {
    if constexpr (std::is_same<T, __half>::value) {
      launch_wmma(a, b, c, m, n, k, stream);
    } else {
      throw std::logic_error("WMMA launch reached an unsupported dtype");
    }
  } else if (options.implementation == "wmma_tiled") {
    if constexpr (std::is_same<T, __half>::value) {
      launch_wmma_tiled(a, b, c, m, n, k, stream);
    } else {
      throw std::logic_error("tiled WMMA launch reached an unsupported dtype");
    }
  } else if (options.implementation == "wmma_tiled_padded") {
    if constexpr (std::is_same<T, __half>::value) {
      launch_wmma_tiled_padded(a, b, c, m, n, k, stream);
    } else {
      throw std::logic_error("padded tiled WMMA launch reached an unsupported dtype");
    }
  } else if (options.implementation == "ptx_mma") {
    if constexpr (std::is_same<T, __half>::value) {
      gemm_lab::launch_ptx_mma_reference(a, b, c, m, n, k, stream);
      GEMM_LAB_CUDA_CHECK(cudaGetLastError());
    } else {
      throw std::logic_error("inline PTX launch reached an unsupported dtype");
    }
  } else if (options.implementation == "ptx_mma_tiled") {
    if constexpr (std::is_same<T, __half>::value) {
      gemm_lab::launch_ptx_mma_tiled(a, b, c, m, n, k, stream);
      GEMM_LAB_CUDA_CHECK(cudaGetLastError());
    } else {
      throw std::logic_error("tiled inline PTX launch reached an unsupported dtype");
    }
  } else if (options.implementation == "ptx_mma_pipelined") {
    if constexpr (std::is_same<T, __half>::value || std::is_same<T, __nv_bfloat16>::value) {
      gemm_lab::launch_ptx_mma_pipelined(a, b, c, m, n, k, stream);
      GEMM_LAB_CUDA_CHECK(cudaGetLastError());
    } else {
      throw std::logic_error("pipelined inline PTX launch reached an unsupported dtype");
    }
  } else if (options.implementation == "ptx_mma_small") {
    if constexpr (std::is_same<T, __half>::value || std::is_same<T, __nv_bfloat16>::value) {
      gemm_lab::launch_ptx_mma_pipelined_small(a, b, c, m, n, k, stream);
      GEMM_LAB_CUDA_CHECK(cudaGetLastError());
    } else {
      throw std::logic_error("small-tile inline PTX launch reached an unsupported dtype");
    }
  } else if (options.implementation == "ptx_mma_splitk") {
    if constexpr (std::is_same<T, __half>::value || std::is_same<T, __nv_bfloat16>::value) {
      if (splitk_workspace == nullptr && splitk_splits > 1) {
        throw std::logic_error("Split-K implementation requires a preallocated workspace");
      }
      gemm_lab::launch_ptx_mma_splitk(
          a, b, c, m, n, k, splitk_workspace, splitk_splits, stream);
      GEMM_LAB_CUDA_CHECK(cudaGetLastError());
    } else {
      throw std::logic_error("Split-K inline PTX launch reached an unsupported dtype");
    }
  } else if (options.implementation == "ptx_mma_small_8warp") {
    if constexpr (std::is_same<T, __half>::value || std::is_same<T, __nv_bfloat16>::value) {
      gemm_lab::launch_ptx_mma_small_8warp(a, b, c, m, n, k, stream);
      GEMM_LAB_CUDA_CHECK(cudaGetLastError());
    } else {
      throw std::logic_error("8-warp inline PTX launch reached an unsupported dtype");
    }
  } else if (options.implementation == "ptx_mma_small_unpadded") {
    if constexpr (std::is_same<T, __half>::value || std::is_same<T, __nv_bfloat16>::value) {
      gemm_lab::launch_ptx_mma_small_unpadded(a, b, c, m, n, k, stream);
      GEMM_LAB_CUDA_CHECK(cudaGetLastError());
    } else {
      throw std::logic_error("unpadded inline PTX launch reached an unsupported dtype");
    }
  } else {
    launch_naive(a, b, c, m, n, k, stream);
  }
}

double quantile(std::vector<float> values, double probability) {
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  const double position = probability * static_cast<double>(values.size() - 1);
  const size_t lower = static_cast<size_t>(std::floor(position));
  const size_t upper = static_cast<size_t>(std::ceil(position));
  const double fraction = position - static_cast<double>(lower);
  return static_cast<double>(values[lower]) * (1.0 - fraction) +
         static_cast<double>(values[upper]) * fraction;
}

TimingMetrics summarize_timings(const std::vector<float>& milliseconds, const Options& options) {
  TimingMetrics metrics;
  metrics.p05_ms = quantile(milliseconds, 0.05);
  metrics.p50_ms = quantile(milliseconds, 0.50);
  metrics.p95_ms = quantile(milliseconds, 0.95);
  metrics.mean_ms = std::accumulate(milliseconds.begin(), milliseconds.end(), 0.0) /
                    static_cast<double>(milliseconds.size());

  double squared_error = 0.0;
  std::vector<float> absolute_deviations;
  absolute_deviations.reserve(milliseconds.size());
  for (float value : milliseconds) {
    const double delta = static_cast<double>(value) - metrics.mean_ms;
    squared_error += delta * delta;
    absolute_deviations.push_back(static_cast<float>(std::abs(static_cast<double>(value) - metrics.p50_ms)));
  }
  metrics.stddev_ms = std::sqrt(squared_error / static_cast<double>(milliseconds.size()));
  metrics.mad_ms = quantile(absolute_deviations, 0.50);
  metrics.cv = metrics.mean_ms > 0.0 ? metrics.stddev_ms / metrics.mean_ms : 0.0;

  const long double operations = 2.0L * static_cast<long double>(options.m) *
                                 static_cast<long double>(options.n) *
                                 static_cast<long double>(options.k);
  metrics.tflops_p50 = static_cast<double>(operations / (metrics.p50_ms * 1.0e9L));
  return metrics;
}

template <typename T>
std::pair<double, double> tolerances() {
  if (std::is_same<T, float>::value) {
    return {1.0e-4, 1.0e-4};
  }
  if (std::is_same<T, __half>::value) {
    return {2.0e-2, 2.0e-2};
  }
  return {8.0e-2, 5.0e-2};
}

template <typename T>
ErrorMetrics evaluate_pairs(
    const std::vector<double>& actual,
    const std::vector<double>& expected,
    const std::string& reference) {
  if (actual.size() != expected.size() || actual.empty()) {
    throw std::logic_error("correctness vectors must have the same non-zero length");
  }

  ErrorMetrics metrics;
  std::tie(metrics.atol, metrics.rtol) = tolerances<T>();
  metrics.reference = reference;
  metrics.elements_checked = static_cast<int64_t>(actual.size());

  double squared_error = 0.0;
  double squared_reference = 0.0;
  bool all_close = true;
  constexpr double relative_epsilon = 1.0e-12;
  for (size_t index = 0; index < actual.size(); ++index) {
    const double candidate = actual[index];
    const double baseline = expected[index];
    if (std::isnan(candidate)) {
      ++metrics.nan_count;
      all_close = false;
      continue;
    }
    if (std::isinf(candidate)) {
      ++metrics.inf_count;
      all_close = false;
      continue;
    }
    const double absolute_error = std::abs(candidate - baseline);
    const double relative_error = absolute_error / std::max(std::abs(baseline), relative_epsilon);
    metrics.max_abs = std::max(metrics.max_abs, absolute_error);
    metrics.max_relative = std::max(metrics.max_relative, relative_error);
    squared_error += absolute_error * absolute_error;
    squared_reference += baseline * baseline;
    if (absolute_error > metrics.atol + metrics.rtol * std::abs(baseline)) {
      all_close = false;
    }
  }
  metrics.normalized_l2 = std::sqrt(squared_error) /
                          std::max(std::sqrt(squared_reference), relative_epsilon);
  metrics.passed = all_close && metrics.nan_count == 0 && metrics.inf_count == 0;
  return metrics;
}

template <typename T>
ErrorMetrics check_cublas_spots(
    const Options& options,
    const std::vector<T>& a,
    const std::vector<T>& b,
    const std::vector<T>& c) {
  const int64_t total = options.m * options.n;
  const int64_t requested = std::min<int64_t>(options.correctness_samples, total);
  std::vector<double> actual;
  std::vector<double> expected;
  actual.reserve(static_cast<size_t>(requested));
  expected.reserve(static_cast<size_t>(requested));

  uint64_t state = static_cast<uint64_t>(options.seed) + 0x9e3779b97f4a7c15ULL;
  for (int64_t sample = 0; sample < requested; ++sample) {
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    const int64_t output_index = static_cast<int64_t>((state * 2685821657736338717ULL) % static_cast<uint64_t>(total));
    const int64_t row = output_index / options.n;
    const int64_t column = output_index % options.n;
    long double accumulator = 0.0L;
    for (int64_t inner = 0; inner < options.k; ++inner) {
      accumulator += static_cast<long double>(to_float(a[static_cast<size_t>(row * options.k + inner)])) *
                     static_cast<long double>(to_float(b[static_cast<size_t>(inner * options.n + column)]));
    }
    actual.push_back(static_cast<double>(to_float(c[static_cast<size_t>(output_index)])));
    expected.push_back(static_cast<double>(accumulator));
  }
  return evaluate_pairs<T>(actual, expected, "cpu_fp64_spot");
}

template <typename T>
ErrorMetrics check_against_cublas(
    const std::vector<T>& candidate,
    const std::vector<T>& reference) {
  if (candidate.size() != reference.size()) {
    throw std::logic_error("candidate and reference sizes differ");
  }
  std::vector<double> actual(candidate.size());
  std::vector<double> expected(reference.size());
  for (size_t index = 0; index < candidate.size(); ++index) {
    actual[index] = static_cast<double>(to_float(candidate[index]));
    expected[index] = static_cast<double>(to_float(reference[index]));
  }
  return evaluate_pairs<T>(actual, expected, "cublas_full");
}

void evict_l2(DeviceBuffer<uint32_t>& eviction, cudaStream_t stream) {
  constexpr int threads = 256;
  const size_t blocks = (eviction.size() + threads - 1) / threads;
  evict_l2_kernel<<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(eviction.get(), eviction.size());
  GEMM_LAB_CUDA_CHECK(cudaGetLastError());
}

template <typename T>
int run_typed(const Options& options, const std::string& canonical_dtype) {
  const size_t a_elements = static_cast<size_t>(options.m) * static_cast<size_t>(options.k);
  const size_t b_elements = static_cast<size_t>(options.k) * static_cast<size_t>(options.n);
  const size_t c_elements = static_cast<size_t>(options.m) * static_cast<size_t>(options.n);

  std::vector<T> host_a = make_input<T>(a_elements, options.seed, 1);
  std::vector<T> host_b = make_input<T>(b_elements, options.seed, 2);
  std::vector<T> host_c(c_elements);

  CudaStream stream;
  CublasHandle handle(stream.get());
  if (std::is_same<T, float>::value) {
    GEMM_LAB_CUBLAS_CHECK(cublasSetMathMode(handle.get(), CUBLAS_PEDANTIC_MATH));
  } else {
    GEMM_LAB_CUBLAS_CHECK(cublasSetMathMode(handle.get(), CUBLAS_DEFAULT_MATH));
  }

  DeviceBuffer<T> device_a(a_elements);
  DeviceBuffer<T> device_b(b_elements);
  DeviceBuffer<T> device_c(c_elements);
  GEMM_LAB_CUDA_CHECK(cudaMemcpyAsync(
      device_a.get(), host_a.data(), device_a.bytes(), cudaMemcpyHostToDevice, stream.get()));
  GEMM_LAB_CUDA_CHECK(cudaMemcpyAsync(
      device_b.get(), host_b.data(), device_b.bytes(), cudaMemcpyHostToDevice, stream.get()));

  int l2_bytes = 0;
  GEMM_LAB_CUDA_CHECK(cudaDeviceGetAttribute(&l2_bytes, cudaDevAttrL2CacheSize, 0));
  const size_t eviction_bytes = std::max<size_t>(2ULL * static_cast<size_t>(l2_bytes), 32ULL * 1024ULL * 1024ULL);
  DeviceBuffer<uint32_t> eviction((eviction_bytes + sizeof(uint32_t) - 1) / sizeof(uint32_t));
  constexpr size_t max_lt_workspace_bytes = 32ULL * 1024ULL * 1024ULL;
  // Split-K needs splits*m*n FP32 partials. Allocated once here so no
  // allocation happens inside the timed region.
  int splitk_splits = 1;
  DeviceBuffer<float> splitk_workspace;
  if (options.implementation == "ptx_mma_splitk") {
    // An explicit override exists so the cost model's choices can be tested
    // against forced values rather than trusted.
    splitk_splits = options.split_k_override > 0
                        ? options.split_k_override
                        : gemm_lab::ptx_mma_split_count(
                              static_cast<int>(options.m), static_cast<int>(options.n),
                              static_cast<int>(options.k), 64, 64, 32);
    if (splitk_splits > 1) {
      splitk_workspace = DeviceBuffer<float>(
          static_cast<size_t>(splitk_splits) * static_cast<size_t>(options.m) *
          static_cast<size_t>(options.n));
    }
  }
  std::unique_ptr<CublasLtPlan<T>> lt_plan;
  if (options.implementation == "cublaslt") {
    lt_plan = std::make_unique<CublasLtPlan<T>>(
        static_cast<int>(options.m),
        static_cast<int>(options.n),
        static_cast<int>(options.k),
        max_lt_workspace_bytes);
  }

  // Correctness is evaluated before timing. cuBLAS itself is spot-checked against
  // FP64 CPU dot products; custom kernels are checked exhaustively against cuBLAS.
  launch_implementation(
      options, handle.get(), device_a.get(), device_b.get(), device_c.get(), stream.get(), lt_plan.get(), splitk_workspace.get(), splitk_splits);
  GEMM_LAB_CUDA_CHECK(cudaStreamSynchronize(stream.get()));
  GEMM_LAB_CUDA_CHECK(cudaMemcpy(
      host_c.data(), device_c.get(), device_c.bytes(), cudaMemcpyDeviceToHost));

  ErrorMetrics correctness;
  if (options.implementation == "cublas" || options.implementation == "cublaslt") {
    correctness = check_cublas_spots(options, host_a, host_b, host_c);
  } else {
    DeviceBuffer<T> device_reference(c_elements);
    launch_cublas(
        handle.get(),
        device_a.get(),
        device_b.get(),
        device_reference.get(),
        static_cast<int>(options.m),
        static_cast<int>(options.n),
        static_cast<int>(options.k));
    GEMM_LAB_CUDA_CHECK(cudaStreamSynchronize(stream.get()));
    std::vector<T> host_reference(c_elements);
    GEMM_LAB_CUDA_CHECK(cudaMemcpy(
        host_reference.data(), device_reference.get(), device_reference.bytes(), cudaMemcpyDeviceToHost));
    correctness = check_against_cublas(host_c, host_reference);
  }

  if (!correctness.passed) {
    throw std::runtime_error("correctness gate failed: refusing to time invalid output");
  }
  int actual_warmup = 0;
  const auto warmup_started = std::chrono::steady_clock::now();
  while (actual_warmup < options.warmup ||
         std::chrono::duration<double>(std::chrono::steady_clock::now() - warmup_started).count() <
             options.warmup_seconds) {
    launch_implementation(
        options, handle.get(), device_a.get(), device_b.get(), device_c.get(), stream.get(), lt_plan.get(), splitk_workspace.get(), splitk_splits);
    ++actual_warmup;
    GEMM_LAB_CUDA_CHECK(cudaStreamSynchronize(stream.get()));
  }

  CudaEvent start;
  CudaEvent stop;
  int sample_inner_repetitions = 1;
  if (options.cache_mode == "steady" && options.min_sample_ms > 0.0) {
    GEMM_LAB_CUDA_CHECK(cudaEventRecord(start.get(), stream.get()));
    launch_implementation(
        options, handle.get(), device_a.get(), device_b.get(), device_c.get(), stream.get(), lt_plan.get(), splitk_workspace.get(), splitk_splits);
    GEMM_LAB_CUDA_CHECK(cudaEventRecord(stop.get(), stream.get()));
    GEMM_LAB_CUDA_CHECK(cudaEventSynchronize(stop.get()));
    float probe_ms = 0.0f;
    GEMM_LAB_CUDA_CHECK(cudaEventElapsedTime(&probe_ms, start.get(), stop.get()));
    if (probe_ms > 0.0f) {
      sample_inner_repetitions = std::min(
          options.max_inner_repetitions,
          std::max(1, static_cast<int>(std::ceil(options.min_sample_ms / probe_ms))));
    }
  }
  std::vector<float> milliseconds;
  milliseconds.reserve(static_cast<size_t>(options.samples));
  const auto measurement_started = std::chrono::steady_clock::now();
  while (static_cast<int>(milliseconds.size()) < options.samples ||
         std::chrono::duration<double>(std::chrono::steady_clock::now() - measurement_started).count() <
             options.measurement_seconds) {
    if (options.cache_mode == "cold") {
      evict_l2(eviction, stream.get());
      GEMM_LAB_CUDA_CHECK(cudaStreamSynchronize(stream.get()));
    }
    GEMM_LAB_CUDA_CHECK(cudaEventRecord(start.get(), stream.get()));
    for (int inner = 0; inner < sample_inner_repetitions; ++inner) {
      launch_implementation(
          options, handle.get(), device_a.get(), device_b.get(), device_c.get(), stream.get(), lt_plan.get(), splitk_workspace.get(), splitk_splits);
    }
    GEMM_LAB_CUDA_CHECK(cudaEventRecord(stop.get(), stream.get()));
    GEMM_LAB_CUDA_CHECK(cudaEventSynchronize(stop.get()));
    float elapsed = 0.0f;
    GEMM_LAB_CUDA_CHECK(cudaEventElapsedTime(&elapsed, start.get(), stop.get()));
    milliseconds.push_back(elapsed / static_cast<float>(sample_inner_repetitions));
  }
  const TimingMetrics timing = summarize_timings(milliseconds, options);

  cudaDeviceProp properties{};
  GEMM_LAB_CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
  int runtime_version = 0;
  int driver_version = 0;
  GEMM_LAB_CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));
  GEMM_LAB_CUDA_CHECK(cudaDriverGetVersion(&driver_version));

  const char* newline = options.pretty ? "\n" : "";
  const char* indent1 = options.pretty ? "  " : "";
  const char* indent2 = options.pretty ? "    " : "";
  std::cout << std::setprecision(10);
  std::cout << '{' << newline
            << indent1 << "\"schema_version\":1," << newline
            << indent1 << "\"implementation\":{\"name\":\"" << options.implementation
            << "\",\"version\":\"native-v1\"}," << newline
            << indent1 << "\"workload\":{\"m\":" << options.m << ",\"n\":" << options.n
            << ",\"k\":" << options.k << ",\"input_dtype\":\"" << canonical_dtype
            << "\",\"accumulator_dtype\":\"float32\",\"output_dtype\":\"" << canonical_dtype
            << "\"}," << newline
            << indent1 << "\"environment\":{" << newline
            << indent2 << "\"gpu_name\":\"" << properties.name << "\","
            << "\"compute_capability\":\"" << properties.major << '.' << properties.minor << "\","
            << "\"l2_bytes\":" << l2_bytes << ','
            << "\"cuda_runtime\":" << runtime_version << ','
            << "\"cuda_driver\":" << driver_version << newline
            << indent1 << "}," << newline
            << indent1 << "\"correctness\":{" 
            << "\"passed\":" << (correctness.passed ? "true" : "false") << ','
            << "\"reference\":\"" << correctness.reference << "\","
            << "\"atol\":" << correctness.atol << ','
            << "\"rtol\":" << correctness.rtol << ','
            << "\"max_abs_error\":" << correctness.max_abs << ','
            << "\"max_relative_error\":" << correctness.max_relative << ','
            << "\"normalized_l2_error\":" << correctness.normalized_l2 << ','
            << "\"nan_count\":" << correctness.nan_count << ','
            << "\"inf_count\":" << correctness.inf_count << ','
            << "\"elements_checked\":" << correctness.elements_checked << "}," << newline
            << indent1 << "\"benchmark\":{" 
            << "\"cache_mode\":\"" << options.cache_mode << "\","
            << "\"launch_mode\":\"eager\",\"timing_method\":\"cuda_event\","
            << "\"warmup_iterations\":" << actual_warmup << ','
            << "\"warmup_min_seconds\":" << options.warmup_seconds << ','
            << "\"samples\":" << milliseconds.size() << ','
            << "\"sample_inner_repetitions\":" << sample_inner_repetitions << ','
            << "\"measurement_min_seconds\":" << options.measurement_seconds << ','
            << "\"workspace_bytes\":"
            << (lt_plan ? lt_plan->workspace_bytes() : splitk_workspace.bytes()) << ','
            << "\"split_k\":" << splitk_splits << ','
            << "\"latency_ms\":{" 
            << "\"p05\":" << timing.p05_ms << ','
            << "\"p50\":" << timing.p50_ms << ','
            << "\"p95\":" << timing.p95_ms << ','
            << "\"mean\":" << timing.mean_ms << ','
            << "\"stddev\":" << timing.stddev_ms << ','
            << "\"mad\":" << timing.mad_ms << ','
            << "\"cv\":" << timing.cv << "},"
            << "\"tflops_p50\":" << timing.tflops_p50 << ",\"raw_ms\":[";
  for (size_t i = 0; i < milliseconds.size(); ++i) {
    std::cout << (i ? "," : "") << milliseconds[i];
  }
  std::cout << "]}" << newline << '}' << '\n';

  return correctness.passed ? 0 : 2;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    if (options.dtype == "fp16") {
      return run_typed<__half>(options, "float16");
    }
    if (options.dtype == "bf16") {
      return run_typed<__nv_bfloat16>(options, "bfloat16");
    }
    return run_typed<float>(options, "float32");
  } catch (const std::exception& error) {
    std::cerr << "gemm_bench: " << error.what() << '\n';
    return 1;
  }
}
