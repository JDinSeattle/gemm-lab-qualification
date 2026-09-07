# Registered performance questions

Written before this qualification run's timing and counter collection.

1. **Decode grid underfill.** At M=8, N=4096, K=4096 the 64x64 candidate exposes few CTAs. A deterministic Split-K implementation should increase available parallel work, at the cost of a second reduction kernel and scratch traffic. Refutation: Split-K is no faster than the same unsplit tile in shuffled repeats, or counters show adequate utilization already. Compare `ptx_mma_small` and `ptx_mma_splitk`, then compare the selected candidate with cuBLASLt. A relative kernel win does not establish a serving win.
2. **Tail overhead.** At M=17, N=257, K=131 most tile lanes are padding and vector alignment is unavailable. Split-K's reduction/launch cost should outweigh extra parallel work. Refutation: it consistently wins outside the latency spread. Keep this losing case in the results.
3. **Integration.** A linear projection plus bias and SiLU can benefit from a fused epilogue, but host dispatch, transfers and allocation can erase a matmul gain. Measure preloaded fragments and the separate host-to-host pipeline. No architecture-wide threshold will be inferred from four shapes.

Baseline kernels are inherited from the pinned GEMM Lab commit in the provenance manifest. This project qualifies their boundary behavior and conclusions; it does not claim their prior authorship as new work. The native adaptation adds raw event samples and refuses timing after a failed numerical gate.
