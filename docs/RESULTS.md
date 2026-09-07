# Measured qualification report

Run: `verified-final-20260907`. Execution status: **passed**.

[All commands and exit codes](../results/verified-final-20260907/execution.json) · [CPU test log](../results/verified-final-20260907/cpu-tests.log) · [GPU correctness](../results/verified-final-20260907/gpu-check.json) · [Raw timing](../results/verified-final-20260907/benchmark.json)

Source digest: `1e37807295f680d6e02c0d8359d265a737ec8f262a8d648cffcbe59e82708a40`.

Hardware: RTX 4090 (sm_89), driver 595.84, native toolkit CUDA 13.2. Clocks unchanged; display GPU. Results apply to this run and workload only.

CPU: 41 tests. All command return codes are retained; missing prerequisites fail the reproduction driver.

GPU Python corpus: 76 full-output comparisons, output guards and fused epilogue checks. Native CTest: 86 cases.

| Workload | PyTorch matmul µs | Triton matmul µs | PyTorch fragment µs | Fused Triton fragment µs |
|---|---:|---:|---:|---:|
| decode-m8 | 20.17 | 30.92 | 23.96 | 32.56 |
| decode-m32 | 23.76 | 48.33 | 58.26 | 48.46 |
| tail | 19.87 | 43.42 | 50.88 | 44.23 |
| square | 15.46 | 42.18 | 43.73 | 43.32 |

Event-span medians above include eager host dispatch gaps. Fragment = linear + bias + SiLU, preloaded weights. This run does not qualify a Triton speedup.

| Workload | cuBLASLt µs | Unsplit PTX µs | Split-K µs | cuBLASLt / Split-K |
|---|---:|---:|---:|---:|
| decode-m8 | 22.43 | 51.00 | 23.65 | 0.948× |
| decode-m32 | 22.70 | 51.01 | 24.58 | 0.924× |
| tail | 5.15 | 6.18 | 6.21 | 0.829× |
| square | 6.14 | 8.14 | 8.15 | 0.754× |

Native medians are the median of three shuffled process-round medians. Split-K improves decode relative to the unsplit tile; it does not establish superiority to cuBLASLt. The tail dispatch selects one split, so it measures the existing fallback rather than a forced split experiment.

| Host-to-host pipeline | PyTorch wall µs | Candidate wall µs |
|---|---:|---:|
| decode-m8 | 3380.40 | 3578.68 |
| decode-m32 | 27524.22 | 20811.70 |
| tail | 138.70 | 183.07 |
| square | 5738.95 | 5936.64 |

| CUTLASS aligned workload | CuTe event µs | PyTorch event µs |
|---|---:|---:|
| decode-m8 | 109.98 | 22.32 |
| decode-m32 | 109.98 | 22.53 |
| square | 17.20 | 6.55 |

[Native raw events](../results/verified-final-20260907/native-benchmark.json) · [CuTe results](../results/verified-final-20260907/cute-benchmark.json) · [CUDA trace: decode kernels](../results/verified-final-20260907/profile-decode_cuda_gpu_kern_sum.csv) · [CUDA trace: tail kernels](../results/verified-final-20260907/profile-tail_cuda_gpu_kern_sum.csv)

Nsight Compute hardware counters require interactive administrator authentication on this host. That attempt is retained in exploratory logs. Nsight Systems provides actual kernel durations and transfer activity; occupancy, cache-hit and bandwidth-bottleneck hypotheses are not declared confirmed without counters.

## Safety and reproducibility

- [gpu-memcheck log](../results/verified-final-20260907/gpu-memcheck.log): exit 0.
- [native-memcheck log](../results/verified-final-20260907/native-memcheck.log): exit 0.
- [cute-memcheck log](../results/verified-final-20260907/cute-memcheck.log): exit 0.

The reproduction driver fails on missing GPU, failed checks, stale gates, sanitizer errors or subprocess failure. CPU CI is labeled separately. All performance comparisons retain failures and unsupported cases.
