# Decision from the current run

Keep the vendor path as the performance reference and default for this corpus. The current custom paths demonstrate correctness and explain implementation tradeoffs; they do not qualify a general dispatch replacement.

The native decode comparison is repeatable across three process rounds: the 64x64 unsplit tile is about 51 µs, Split-K about 24 µs, and cuBLASLt about 22–23 µs. Thus the Split-K mechanism helps the custom kernel but still loses the relevant library comparison. On the tail case the inherited dispatcher selects one split; the trace confirms no custom reduction launch. This observation supports the existence of a fallback, not a forced-Split-K performance hypothesis.

Python timings are visibly noisy for several cases. For example, the final decode-m32 fragment has p05–p95 around 40–80 µs for PyTorch and 32–65 µs for fused Triton. Its lower Triton median is insufficient to claim a stable application win. The complete raw arrays remain available. Host-to-host costs include transfers, CPU scheduling and result consumption and must not be described as isolated kernel latency.

The actual trace shows the custom Split-K matrix kernel and a distinct reduction kernel. Hardware counters could not be collected without interactive administrator authentication. Static grid size is a useful hypothesis, but this run cannot determine an occupancy, cache or bandwidth bottleneck from kernel names and timings alone. A future counter-enabled run and a quiescent host are the next useful experiments; neither result is fabricated here.

The first CUTLASS memcheck failed due to CUDA Python 13.3 API requests exceeding the 13.2 driver's capability. Matching 13.2 bindings removes the diagnostics and passes memcheck; both failed and successful runs are retained. This illustrates why numeric success alone is insufficient environment qualification.
