# Qualification runner lifecycle

Implemented 2026-09-07 in `execution.py`, `evidence.save`, and `scripts/run.py`.
The runtime and its 16 behavior tests are identical vendored copies in the four
standalone labs; they are not 64 distinct fault scenarios.

Use a **new, nonexistent** output directory for each attempt:

```bash
python scripts/run.py --cpu-only --output /tmp/my-new-cpu-run
python scripts/run.py --samples 31 --step-timeout 900 --termination-grace 1
```

Each stage runs in a new POSIX session/process group. Timeout or SIGINT/SIGTERM
terminates its group, waits a bounded grace period, then sends SIGKILL even when
the direct child has already exited. A wrapper leaving descendants behind is a
failed stage (125), including when that wrapper returns zero. Timeout is 124;
launch failure is 127. These codes are recorded in the receipt; the Python CLI
itself exits nonzero on failure. No subsequent benchmark runs after a failed
stage. Signals are latched, avoiding exceptions between fork and process-handle
assignment. The direct child is reaped; dead grandchildren may briefly remain
zombies until the OS reaps them.

`execution.json` schema 2 retains existing status/steps/environment fields and
adds run ID, timestamps, supervisor PID, per-stage PID/PGID, timeout, log path,
child exit and cleanup actions. Stage `running` is persisted **before** launch;
terminal states are `passed`, `cpu_passed`, `failed`, or `interrupted`. Abrupt
supervisor SIGKILL may leave `running`; that is incomplete evidence, never a
pass. JSON writes serialize finite values before publication, use unique
same-directory staging files, fsync and rename, then fsync the directory.
Readers see a complete old/new JSON file; concurrent generic saves are last
writer wins. Output directory admission prevents competing run owners.

The default `/tmp/gpu-qualification-<uid>.lock` serializes **all GPUs** for the
same user across cooperating lab runners. It is an advisory flock held through
cleanup, acquired before work and never unlinked. `--resource-lock` can point to
a site-managed shared lock; every cooperating runner must use the same path.
Contention fails immediately and leaves a failed receipt without executing a
stage. CPU-only runs do not take the GPU lock. Output directory reservation uses
atomic mkdir and rejects even an existing empty directory. Failed directories
remain available for inspection; retry with a new name.

## Limits and tradeoffs

- Linux local filesystems and trusted application-owned paths. This is not a
  cgroup, job scheduler, distributed lock, privilege or hostile-path boundary.
- Supervisor SIGKILL/host crash, `setsid()` escape, uninterruptible kernel waits
  and noncooperating users/programs require external supervision/isolation.
- Direct `qualification.py` invocations bypass the outer resource lock and
  process lifecycle policy; use `scripts/run.py` for qualification runs.
- Serialization sacrifices concurrent use of different GPUs for a simple
  conservative contract on the currently qualified single-GPU host.
- Stages have bounded time; output log sizes are not capped. Disk exhaustion is
  surfaced as failure, not converted into a successful result.
- Locks, fsync and supervision are outside timed kernel regions. This change
  makes no kernel, ETL, latency, availability or production-scale improvement
  claim. Old reports belong to their original source/driver/package snapshots.
- The initial sandboxed B ASan/UBSan run encountered LeakSanitizer's ptrace
  restriction; the failure is retained separately from unsandboxed validation.

See `tests/test_execution.py` for real process/failure tests and the portfolio
market review for the dependency-retention decision. Existing CI's pytest
invocation automatically includes these tests; local execution is not a new
hosted GitHub Actions result.
