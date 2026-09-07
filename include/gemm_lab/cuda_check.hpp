#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <sstream>
#include <stdexcept>
#include <string>

namespace gemm_lab {

inline const char* cublas_status_string(cublasStatus_t status) {
  switch (status) {
    case CUBLAS_STATUS_SUCCESS:
      return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:
      return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:
      return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:
      return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "CUBLAS_STATUS_INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED:
      return "CUBLAS_STATUS_NOT_SUPPORTED";
    case CUBLAS_STATUS_LICENSE_ERROR:
      return "CUBLAS_STATUS_LICENSE_ERROR";
    default:
      return "CUBLAS_STATUS_UNKNOWN";
  }
}

inline void check_cuda(cudaError_t status, const char* expression, const char* file, int line) {
  if (status == cudaSuccess) {
    return;
  }
  std::ostringstream message;
  message << file << ':' << line << ": " << expression << " failed: "
          << cudaGetErrorName(status) << " (" << cudaGetErrorString(status) << ')';
  throw std::runtime_error(message.str());
}

inline void check_cublas(cublasStatus_t status, const char* expression, const char* file, int line) {
  if (status == CUBLAS_STATUS_SUCCESS) {
    return;
  }
  std::ostringstream message;
  message << file << ':' << line << ": " << expression << " failed: "
          << cublas_status_string(status);
  throw std::runtime_error(message.str());
}

}  // namespace gemm_lab

#define GEMM_LAB_CUDA_CHECK(expression) \
  ::gemm_lab::check_cuda((expression), #expression, __FILE__, __LINE__)

#define GEMM_LAB_CUBLAS_CHECK(expression) \
  ::gemm_lab::check_cublas((expression), #expression, __FILE__, __LINE__)
