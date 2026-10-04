# GPU qualification portfolio: dated engineering review

Reviewed 2026-09-07 (America/Los_Angeles; upstream metadata uses UTC). This is a
small purposive sample, not a labor-market census or a guarantee of eligibility.

## Context and evidence boundary

The workspace contains four independent `main` repositories, initially clean:
A `0962ddf58862b2f94a6b4096ec94fcd82b4f72a4`,
B `5f940f4c39714a27d587d077c95ca3609756110e`,
C `be3577d040e1c096471d975aaab46f8de1fa3f31`,
D `112f086f80e4c7f9f3301da07e37d50491bd33e8`.
Each has one initial implementation commit, not a long history of maintenance.
No applicable first-party AGENTS.md was found. Third-party `.deps` are not edited.

Read current code, contracts, tests, manifests, CI, retained raw results, the
prior creation conversation, and `ml-infra/04-kernels-libraries.md`. Local memory
storage was readable but contained no relevant project memory. The prior
conversation supplies context only; its results were checked against files.

Current authoritative banks are in
`/home/postedism/Desktop/Find work/jd-resume-tailor/data/`;
`Desktop/Find work/experience bank` does not exist. The supplied Downloads
Generator uses obsolete macOS paths. Its suggestions to invent deployment,
metrics, or passing tests conflict with the current request and are not used.

The current profile describes a Seattle-based CS master's student finishing in
May 2027, with an earlier AI master's and a software-engineering role beginning
December 2025. Early-career roles are the primary comparator; graduate-year and
work-authorization eligibility must be checked for an actual application.
Existing banks cover Go workflows/fencing, device control, transactional Java,
streaming/lakehouse systems, and a separate GEMM serving qualification lab.
Adding another queue, API service, or Kubernetes scaffold here would overlap
those projects. These four labs add independent numerical/semantic evidence.

## Official job sample

| Official requisition | Seniority/location and date evidence | Observed requirements and use here |
|---|---|---|
| [Amazon SDE 2026 US, 3177934](https://www.amazon.jobs/en/jobs/3177934/software-development-engineer-2026-us) | Explicit SDE-I; US including Seattle. Official page retrieved Sep 7, 2026. Publication date not exposed; third-party relative dates were not promoted to an official posting date. | General-purpose language, algorithms/design; project experience, debugging, distributed fault tolerance, CI/CD and operational monitoring. Primary early-career comparator. |
| [NVIDIA Physical Design Infrastructure, JR2021823](https://nvidia.wd5.myworkdayjobs.com/en-US/NVIDIAExternalCareerSite/job/Software-Engineer--Physical-Design-Infrastructure---New-College-Grad-2026_JR2021823) | New College Grad, BS/MS or equivalent, Santa Clara. Official-domain indexed content reviewed Sep 7; includes applications accepted at least until Jul 27, 2026. Dynamic direct page returned no readable body. Exact publication and current vacancy status not confirmed. | Linux/Python, task coordination, regression analysis and job telemetry. Strong fit for bounded execution and actionable failure records. Perl, EDA and Elasticsearch are team-specific preferences. |
| [NVIDIA AI/ML Infra GPU Clusters, JR2021591](https://nvidia.wd5.myworkdayjobs.com/en-US/NVIDIAExternalCareerSite/job/US-CA-Santa-Clara/AI-and-ML-Infra-Software-Engineer--GPU-Clusters---New-College-Grad-2026_JR2021591) | Graduate MS/PhD or equivalent; Santa Clara/Redmond. Same indexed-content limitation; Jul 27, 2026 acceptance lower bound. | Resource utilization, measurable researcher efficiency, Python/Go/Bash. Its distributed training, HPC storage, Slurm and fast-network requirements exceed this single-GPU lab; a stretch comparator. |
| [Amazon Data Center Builder Tools, 10513782](https://www.amazon.jobs/en/jobs/10513782/software-development-engineer-data-center-builder-tools) | Denver; **3+ years** plus design experience, despite a separate 1+ years line. Official body viewed Sep 7 shows Sep 4 deadline; search excerpt showed Sep 9. | Automation, release gates, reliability and operational debugging. Excluded from the primary early-career sample; dates conflict and availability is unconfirmed. |
| [Amazon Agent Platforms, 10530111](https://www.amazon.jobs/en/jobs/10530111/software-development-engineer-agent-platforms-services) | Seattle/New York; **3+ years**, not entry level. Official page retrieved Sep 7; exact official publication date unavailable. | Evaluation/regression gates and reliable observable infrastructure. Supplemental trajectory only; does not justify adding agents to a kernel project. |

Repeated across the small sample: testing, debugging, maintainable systems,
resource/reliability awareness and measurement. Cloud-native deployment is
explicit in Amazon SDE-I; HPC fabric/storage is specific to the NVIDIA cluster
role. No sample establishes that every backend job requires CUDA, Triton, or a
particular orchestration framework. These are July–September/current 2026
signals where dated evidence exists, not five verified recent publication dates.

## Official ecosystem check and upgrade decision

GitHub repository metadata and recent releases were fetched Sep 7. All seven
repositories below were unarchived and had September 2026 activity. Search
snippets lagged PyTorch's latest release; the official release/API was used.

| Component | Observed release and date | Decision, benefit, compatibility and cost |
|---|---|---|
| [PyTorch](https://github.com/pytorch/pytorch/releases/tag/v2.14.0) | 2.14.0, Sep 2, 2026; current lab 2.13.0 | NVGEMM/Inductor and distributed changes exist; this lab uses eager references and explicit kernels. Keep 2.13 for a controlled runner change. A future upgrade must pair Torch/Triton, rerun numerical gates and rebuild FlashInfer JIT; release notes contain backwards-incompatible migration examples. |
| [Triton](https://github.com/triton-lang/triton/releases/tag/v3.8.0) | 3.8.0, Aug 28; current 3.7.1 | New profiling helpers and cache/unload work are potentially useful. Compiler/runtime changes can alter generated kernels. No demonstrated benefit for process cleanup; defer separate numerical and timing qualification. |
| [CUTLASS stable maintenance](https://github.com/NVIDIA/cutlass/releases/tag/v4.7.1) | 4.7.1 and 4.6.3, Aug 26 | Decorator-leak and interop fixes merit a later bounded RSS/compile study. A's separate CuTe adapter checks the 4.6.1 source SHA and wheel; changing only one would violate its contract. Keep the pair for this update; do not claim long-running CuTe memory stability. |
| [CUTLASS dev](https://github.com/NVIDIA/cutlass/releases/tag/v4.8.0dev) | Aug 27, title/tag explicitly `dev`, despite GitHub `prerelease=false` | Rubin-specific support is not evidence for RTX 4090. Do not choose this as a stable upgrade merely because GitHub labels it Latest. |
| [CCCL](https://github.com/NVIDIA/cccl/releases/tag/v3.4.2) | 3.4.2, Aug 5 | B intentionally compares two immutable bad/patched revisions. Replacing them with latest would destroy the controlled defect experiment. The [3.0 migration guide](https://nvidia.github.io/cccl/unstable/cccl/3.0_migration_guide.html) documents removal of internal CUB APIs; public histogram calls remain the boundary. |
| [FlashInfer](https://github.com/flashinfer-ai/flashinfer/releases/tag/v0.6.18.post1) | Sep 5; current pin matches | Retain explicit FA2/no-split-KV contract. [Installation compatibility](https://docs.flashinfer.ai/installation.html) is tied to Torch/CUDA and compiled/JIT packages. New Rubin/low-precision paths in 0.6.18 do not qualify additional backends on this host. |
| [cuDF/pylibcudf](https://github.com/NVIDIA/cudf/releases/tag/v26.08.01) | Aug 25; current 26.8.1 matches | Keep direct Arrow interop and exact null/count contracts. 26.08 release notes include pandas API removals, so high-level cuDF migration assumptions must not be applied to this direct pylibcudf path. |
| [Arrow](https://github.com/apache/arrow/releases/tag/apache-arrow-25.0.1) | Aug 10; current 25.0.1 matches | Keep the independent Arrow baseline. No new dependency needed for Linux process/file primitives. |

## Prioritized implementation and acceptance

1. **P0: bounded qualification lifecycle.** The old `subprocess.run(timeout=...)`
   only killed the immediate child. A local CPU reproducer proved a grandchild
   continued writing after timeout. Use new sessions, group TERM/KILL and
   cancellation-aware polling; also reject wrappers whose descendants outlive
   their successful exit. Acceptance: real children, TERM-ignoring descendants,
   SIGINT/SIGTERM and failure short-circuit tests.
2. **P0: exclusive runs and complete evidence.** Replace check-then-write output
   admission with atomic directory creation; serialize cooperating same-user GPU
   runs across all four labs; record before launch; unique atomic JSON staging.
   Acceptance: process races, lock contention/reuse, concurrent writers,
   malformed numbers and injected publication failure.
3. **P1: real regression and bank synchronization.** Run CPU contracts, native
   CTest, GPU numerical/semantic corpora, sanitizer, and existing independent
   timing/profile phases. Preserve failures. New timing is an execution check,
   not evidence of an algorithmic speedup. Back up and merge both banks under
   their shared writer lock, validating all IDs and source hashes.

The Linux process-group API follows the [Python subprocess documentation](https://docs.python.org/3.13/library/subprocess.html).
SIGKILL of the supervisor, deliberate `setsid()` escape, uninterruptible device
waits, other users and noncooperating programs remain outside this boundary.
The shared display GPU has unlocked clocks; neither production isolation nor a
cross-architecture performance result is claimed.
