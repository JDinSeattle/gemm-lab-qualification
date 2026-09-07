# Qualification contract

The v1 decision workload is small-batch linear projection on one RTX 4090, FP16 input/output and FP32 accumulation. The manifest pins four measured shapes and five edge shapes. Benchmarks use deterministic, rounded FP16 inputs; CPU FP64 checks compute from those same represented values. TF32 and reduced-precision FP16 reduction are disabled in the Python harness.

| Boundary | Behavior |
|---|---|
| Odd M/N/K, partial tiles | Masked Triton/native kernels; compare full output |
| Noncontiguous input views | Explicit PyTorch fallback; retain strides and poison unused storage |
| M/N/K equal to zero | Defined PyTorch fallback, including zero-filled empty reduction |
| NaN, infinity, FP16 result overflow | Input qualification rejects these; not propagation tests |
| Error | `abs(candidate - FP64) <= 0.003 + 0.005 * abs(FP64)` |
| Fused bias/SiLU | FP32 epilogue with FP16 output; separate tolerance 0.01 + 0.01 * abs(reference) |
| Output safety | Guard values before and after preallocated Triton outputs; Compute Sanitizer separately |
| CUTLASS 4.6.1 | Aligned contiguous K/N multiples of 8; tail case explicitly unsupported |
| Native extra coverage | Inherited 86 CTest cases include BF16 and FP16 kernels; broader than v1 decision workload |

The CPU scalar dot tests and mutation tests establish properties of the oracle and detector. They do not establish GPU correctness. The GPU check is a separate required phase; no missing-device skip is converted into success.

The native harness compares custom output with cuBLAS over the full matrix; cuBLAS/cuBLASLt use independent CPU FP64 spot checks. Its inherited tolerance is 0.02 absolute/relative, distinct from the tighter Python corpus. The two suites must not be reported as identical numeric qualifications.

## Timing and integration

Correctness and sanitizer processes finish before benchmark processes begin. Every raw sample is retained. Python implementations are shuffled within each round; native implementations run in three shuffled rounds of separate processes. Event spans surround a sequence of launches, and can include host dispatch gaps. Synchronized wall time includes dispatch. Nsight exports provide actual kernel durations, including the additional Split-K reduction.

Kernel buffers are preallocated where named. Fragment timings compare preloaded `linear -> bias -> SiLU`. A separate host-to-host pipeline includes allocation, transfer, output copy and a host checksum. The benchmark does not time the oracle. Correctness gates bind the source manifest, package versions, GPU UUID and driver. A failed or stale gate prevents timing.

Clocks remain unlocked on a display GPU. Thermal state and concurrent display activity limit fine-grained conclusions; p05/p50/p95 and MAD are reported. Measurements close to this spread are inconclusive. No cross-architecture optimum or production-serving claim follows from this corpus.

## Provenance

Native and Triton kernels originate from the user's [GEMM Lab](https://github.com/JDinSeattle/llm-gemm-qualification-lab), pinned in `manifests/provenance.json`. Previous performance data was not imported. CUTLASS is pinned at `e05f953a5b3d38adc240df2ff928e0421c2abba3`; its [profiler documentation](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/profiler.html) explains its separate profiling workflow. This lab adds the qualification corpus, strict evidence gates, raw native timing export, and current-run integration comparisons.
