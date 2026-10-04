# GEMM Lab Qualification

A single-GPU numerical qualification lab for FP16 GEMM and bias+SiLU fragments, with independent CPU references, explicit unsupported-layout fallbacks, paired timing, and isolated experiment ownership.

## Confirmed qualification results

These are the user-confirmed results from a separate cloud test run, recorded in the experience bank. The device, workload, timing round, and counting boundaries below remain part of each result. They are distinct from the CPU checks performed in this checkout; cloud-hosted testing does not imply production deployment.

- Qualified FP16 GEMM and bias+SiLU linear fragments on a single RTX 4090 24 GB (sm_89, CUDA 13.2, Python 3.13, torch 2.13.0, Triton 3.7.1, CuTe DSL 4.6.1) across M=1/8/32/128 with N=K=4096, seed=7 and uniform inputs in [-0.5,0.5], running correctness and sanitizer checks before 20 warmups and 31 randomized AB/BA CUDA-event pairs with data resident on GPU; the 76 GPU comparison records and 86 native CUDA tests are retained together with the slower paths and fallbacks.

- Built an independent numerical gate at M=32, N=K=4096: pre-quantized FP16 inputs, a full CPU FP64 output reference, 64 dots cross-checked with independent scalar accumulation, a tolerance of abs <= 0.02 + 0.002 x |reference|, and poison-padding plus output-canary checks; 8 additional boundary probes (zero M, zero K, tail N, tail K and 4 strided inputs) stayed separate from the historical 76 records.

- Reported no cuBLASLt win at M=32, N=K=4096: p50 was 18.0 μs for cuBLASLt, 20.7 μs for Triton and 18.5 μs for CuTe, and the same-round reference/candidate values were 0.870 and 0.973; fused bias/SiLU is only compared against an equal-semantics GEMM+bias+SiLU baseline, never a bare GEMM timing.

- Kept unsupported layouts explicit: the aligned CuTe path accepts only selected K/N alignment while strided and empty inputs take a reported explicit fallback, and the sm_89 path does not use TMA or WGMMA.

- Hardened experiment execution: a hung child that spawned a cooperative worker was terminated as a whole process group after timeout, leaving only a failure receipt; atomic run-directory admission let only one of two publishers claim a run ID; and an flock-based GPU lock covered the entire qualification stage, while the 16 shared process/file regression cases are 16 behaviors across four projects - each project reports 16 passed, but that is not 64 independent items and the 57 CPU tests already include this project's 16.

## Implementation and reproduction

| Contract | Implementation |
|---|---|
| CPU FP64 and scalar-dot reference | [qualification.py](qualification.py) |
| Native and Triton operators | [csrc](csrc) |
| Pinned CuTe adapter | [cute_adapter.py](cute_adapter.py) |
| Bounded execution and evidence admission | [execution.py](execution.py) |

Run each experiment into a fresh output directory to preserve earlier evidence.

```bash
python3 -m pip install -r requirements.lock
python3 scripts/run.py --cpu-only --output results/my-cpu-run
# Requires the pinned sm_89 GPU/native/CuTe environment:
python3 scripts/run.py --output results/my-gpu-run
```

Regression entry points: [tests/test_contract.py](tests/test_contract.py), [tests/test_execution.py](tests/test_execution.py), [tests/test_evidence.py](tests/test_evidence.py).

## Scope and evidence

Historical 76 GPU records stay separate from the eight new boundary probes and the external shape-specific timing. The local scalar cross-check validates up to 64 output dots without GPU dispatch.

- No cuBLASLt win is established; the 18.0, 20.7 and 18.5 μs p50 values are for one shape (M=32, N=K=4096) and must not be generalized or cherry-picked as a leaderboard result.

- CuTe only accepts selected K/N alignment; strided and empty inputs use an explicit fallback and the dispatch path is reported, and the sm_89 path does not use TMA or WGMMA.

- The FP64 oracle is CPU FP64 with an independent scalar dot cross-check, chosen so the candidate and the oracle do not share one GPU dispatch defect; it is not evidence of whole-model accuracy.

- The 16 process/file tests exist across four projects: each project can report 16 passed, but that is 16 behaviors, not 64 independent items, and the 57 CPU tests already include this project's 16.

- Split-K is not automatically faster at small batches: partial-sum write-back, merge and extra launch are included in the timing at the same output precision; with K=4096 the extra merge can eat the gain, and any merge that changes summation order requires the full numerical gate again.

- The qualification overlaps the separate LLM GEMM lab: inherited kernels keep their attribution, only one entry per job should be selected, and the shared foundation must not be counted as two independent optimizations.

- This is a local single-GPU lab on synthetic inputs: the 8 new boundary probes are not mixed into the historical 76 records, and no cloud, production or whole-model performance claim is made.

The [previous README](README.historical.md) preserves earlier setup details, design discussion, and historical measurements. Its older counts, splits, versions, and timing cohorts must not be mixed with the confirmed round above. [Result provenance](docs/experience-bank-results.json) retains the confirmed bullet text; [checkout validation](docs/checkout-validation.md) records what was actually rerun here.
