cmake_minimum_required(VERSION 3.25)

project(llm_gemm_qualification_lab VERSION 0.1.0 LANGUAGES CXX CUDA)

include(CTest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 89)
endif()

find_package(CUDAToolkit REQUIRED)

add_executable(gemm_bench csrc/gemm_bench.cu)
target_include_directories(gemm_bench PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(gemm_bench PRIVATE CUDA::cublas CUDA::cublasLt CUDA::cudart)
target_compile_options(gemm_bench PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;--extended-lambda;-Xcompiler=-Wall,-Wextra>
)
set_target_properties(gemm_bench PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
  CUDA_RESOLVE_DEVICE_SYMBOLS OFF
)

install(TARGETS gemm_bench RUNTIME DESTINATION bin)

if(BUILD_TESTING)
  add_test(NAME gemm_bench_help COMMAND gemm_bench --help)

  # gemm_bench validates against full cuBLAS output before timing and exits 2 on
  # a correctness failure, so a bare invocation is a usable correctness test.
  # These shapes are the ones that broke the tiled kernels during development:
  # odd k (no legal 16-byte load), odd n (unaligned __half2 epilogue store),
  # cross-tile remainders, and a single-row M.
  set(GEMM_LAB_TENSOR_CORE_IMPLS wmma wmma_tiled wmma_tiled_padded ptx_mma
                                 ptx_mma_tiled ptx_mma_pipelined ptx_mma_small
                                 ptx_mma_splitk ptx_mma_small_8warp)
  set(GEMM_LAB_EDGE_SHAPES
      "17x31x13"        # every dimension odd and below one MMA tile
      "129x257x65"      # one element past a tile boundary in all three
      "127x129x131"     # odd n forces the scalar epilogue path
      "1x4096x4096"     # single row, full-width k
      "509x4099x4103"   # non-aligned production-like shape
      "256x256x256")    # exact multiple of the 128x128 tile
  foreach(impl IN LISTS GEMM_LAB_TENSOR_CORE_IMPLS)
    foreach(shape IN LISTS GEMM_LAB_EDGE_SHAPES)
      string(REPLACE "x" ";" dims "${shape}")
      list(GET dims 0 shape_m)
      list(GET dims 1 shape_n)
      list(GET dims 2 shape_k)
      add_test(
        NAME "correctness_${impl}_${shape}"
        COMMAND gemm_bench --m ${shape_m} --n ${shape_n} --k ${shape_k}
                --dtype fp16 --impl ${impl}
                --warmup 0 --samples 1 --measurement-seconds 0)
    endforeach()
  endforeach()

  # The pipelined instantiations are templated on the element type and also
  # support BF16, which uses a different MMA opcode and packed-store conversion.
  set(GEMM_LAB_BF16_IMPLS ptx_mma_pipelined ptx_mma_small ptx_mma_small_unpadded
                          ptx_mma_splitk ptx_mma_small_8warp)
  foreach(impl IN LISTS GEMM_LAB_BF16_IMPLS)
    foreach(shape IN LISTS GEMM_LAB_EDGE_SHAPES)
      string(REPLACE "x" ";" dims "${shape}")
      list(GET dims 0 shape_m)
      list(GET dims 1 shape_n)
      list(GET dims 2 shape_k)
      add_test(
        NAME "correctness_${impl}_bf16_${shape}"
        COMMAND gemm_bench --m ${shape_m} --n ${shape_n} --k ${shape_k}
                --dtype bf16 --impl ${impl}
                --warmup 0 --samples 1 --measurement-seconds 0)
    endforeach()
  endforeach()

  # The earlier ladder rungs are FP16 only; assert the CLI rejects BF16 rather
  # than silently accepting it.
  add_test(NAME "rejects_bf16_ptx_mma"
           COMMAND gemm_bench --m 64 --n 64 --k 64 --dtype bf16 --impl ptx_mma
                   --warmup 0 --samples 1 --measurement-seconds 0)
  set_tests_properties("rejects_bf16_ptx_mma" PROPERTIES WILL_FAIL TRUE)
endif()
