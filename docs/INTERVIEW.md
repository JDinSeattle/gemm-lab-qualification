# Interview and résumé evidence

Suggested defensible résumé bullet, after reading the measured report:

> Built a reproducible GPU GEMM qualification suite spanning native CUDA/PTX, Triton, CUTLASS and cuBLASLt; validated boundary/layout contracts with independent FP64 references, 86 native CUDA tests and Compute Sanitizer, and separated kernel timing from linear-layer and host-to-host latency.

Do not write “beats cuBLAS” or reuse the older GEMM Lab's speedup numbers. If describing Split-K, name the comparison: the same inherited unsplit PTX tile on the measured decode shapes. Explain why the current result still favors a vendor-library baseline.

Be prepared to open these artifacts live:

1. `manifests/workloads.json`: explain the workload and numerical domain before showing timing.
2. `qualification.py`: show how the FP64 oracle and strided fallback differ from candidate execution.
3. `tests/test_contract.py`: run a mutation detector and explain what it cannot prove.
4. `docs/RESULTS.md`: explain event spans, host dispatch gaps, p95/MAD, and negative integration results.
5. Native source and `native_suite.py`: explain tile tails, Split-K scratch traffic and randomized process rounds.

The qualification harness, manifests, gating and current-run investigation are new here. The original kernels are explicitly reused and credited. Hardware-counter access remains restricted; do not describe an unmeasured occupancy or bandwidth hypothesis as established fact.
