"""Bounded Linux qualification execution; no CUDA or third-party imports.

The same file is vendored in the four standalone labs. A process group is a
cooperative boundary, not a cgroup: setsid(), SIGKILL of the supervisor and
uninterruptible kernel waits require an external service manager.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import math
import os
from pathlib import Path
import signal
import subprocess
import time
import uuid

from evidence import save


class StepFailed(RuntimeError):
    def __init__(self, step):
        self.returncode = step['returncode']
        super().__init__(f"{step['step']}: {step['status']} (exit {self.returncode})")


@contextmanager
def exclusive_lock(path):
    """Fail fast on contention. Never unlink a lock inode while it may be held."""
    fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f'qualification resource busy: {path}') from exc
        yield
    finally:
        os.close(fd)


def _signal_group(pid, signum):
    try:
        os.killpg(pid, signum)
        return True
    except ProcessLookupError:
        return False


def _cleanup(proc, grace):
    """TERM then KILL the whole group even if the direct child already exited."""
    term = _signal_group(proc.pid, signal.SIGTERM)
    deadline = time.monotonic() + grace
    while term and time.monotonic() < deadline:
        proc.poll()
        if not _signal_group(proc.pid, 0):
            break
        time.sleep(min(.02, max(0, deadline - time.monotonic())))
    killed = _signal_group(proc.pid, signal.SIGKILL)
    # A D-state process may remain: never turn that uncertainty into a pass.
    reaped = True
    try:
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        reaped = False
    return {'term_sent': term, 'kill_sent': killed, 'direct_child_reaped': reaped}


class Execution:
    def __init__(self, out, project, environment, *, cpu_only=False,
                 step_timeout=900, grace=1, lock_path=None):
        if any(not math.isfinite(x) or x <= 0 for x in (step_timeout, grace)):
            raise ValueError('timeouts must be finite and positive')
        self.out = Path(out)
        self.environment = environment
        self.cpu_only = cpu_only
        self.timeout = step_timeout
        self.grace = grace
        self.lock_path = Path(lock_path or f'/tmp/gpu-qualification-{os.getuid()}.lock')
        self.received_signal = None
        self.handlers = {}
        self.lock = None
        self.record = {
            'schema_version': 2, 'run_id': str(uuid.uuid4()), 'project': project,
            'status': 'running', 'steps': [], 'started_at': self._utc(),
            'environment': {}, 'supervisor_pid': os.getpid(),
            'policy': {'step_timeout_s': step_timeout, 'termination_grace_s': grace,
                       'resource_lock': None if cpu_only else str(self.lock_path),
                       'scope': 'same-user cooperating local Linux processes; all GPUs serialized'},
        }

    @staticmethod
    def _utc():
        return datetime.now(timezone.utc).isoformat()

    def _write(self):
        save(self.out / 'execution.json', self.record)

    def _receive(self, signum, frame):
        # Do not raise between Popen's fork and assigning its process handle.
        if self.received_signal is None:
            self.received_signal = signum

    def __enter__(self):
        self.out.parent.mkdir(parents=True, exist_ok=True)
        self.out.mkdir()  # atomic reservation, including empty/preexisting dirs
        try:
            for signum in (signal.SIGINT, signal.SIGTERM):
                self.handlers[signum] = signal.signal(signum, self._receive)
            self._write()
            if not self.cpu_only:
                self.lock = exclusive_lock(self.lock_path)
                self.lock.__enter__()
            self.record['environment'] = self.environment()
            self._write()
            return self
        except BaseException as exc:
            self.__exit__(type(exc), exc, exc.__traceback__)
            raise

    def run(self, label, argv, *, cwd, timeout=None):
        limit = self.timeout if timeout is None else min(timeout, self.timeout)
        if not math.isfinite(limit) or limit <= 0:
            raise ValueError('step timeout must be finite and positive')
        if not label or Path(label).name != label or label in {'.', '..'}:
            raise ValueError('step label must be a plain filename')
        if self.received_signal is not None:
            raise InterruptedError(f'interrupted by signal {self.received_signal}')
        step = {'step': label, 'argv': list(argv), 'cwd': str(cwd),
                'status': 'running', 'returncode': None, 'started_at': self._utc(),
                'timeout_s': limit, 'log': label + '.log'}
        started = time.monotonic()
        proc = None
        with (self.out / step['log']).open('x') as log:
            self.record['steps'].append(step)
            self._write()  # visible before launching; abrupt death cannot look complete
            print(f"[{self.record['project']}] {label}", flush=True)
            try:
                proc = subprocess.Popen(argv, cwd=cwd, stdout=log,
                                        stderr=subprocess.STDOUT, start_new_session=True)
                step['pid'] = step['pgid'] = proc.pid
                self._write()
                deadline = started + limit
                while True:
                    code = proc.poll()
                    if self.received_signal is not None:
                        step.update(status='interrupted', returncode=128 + self.received_signal)
                        break
                    if code is not None:
                        # A wrapper exiting successfully must not leave workers running.
                        orphaned = _signal_group(proc.pid, 0)
                        step.update(status='failed' if code or orphaned else 'passed',
                                    returncode=code or (125 if orphaned else 0))
                        if orphaned:
                            step['error'] = 'descendants outlived direct child'
                        break
                    if time.monotonic() >= deadline:
                        step.update(status='timed_out', returncode=124)
                        break
                    time.sleep(min(.05, max(0, deadline - time.monotonic())))
            except OSError as exc:
                step.update(status='failed', returncode=127, error=str(exc))
                log.write(str(exc) + '\n')
            except BaseException as exc:
                step.update(status='failed', returncode=1, error=repr(exc))
                raise
            finally:
                if proc is not None:
                    step['cleanup'] = _cleanup(proc, self.grace)
                    step['child_returncode'] = proc.returncode
                step['elapsed_s'] = time.monotonic() - started
                step['finished_at'] = self._utc()
                self._write()
        if step['returncode']:
            raise StepFailed(step)

    def __exit__(self, kind, error, traceback):
        try:
            self.record['status'] = ('interrupted' if self.received_signal is not None else
                                     'failed' if error is not None else
                                     'cpu_passed' if self.cpu_only else 'passed')
            if error is not None:
                self.record['error'] = repr(error)
            if self.received_signal is not None:
                self.record['signal'] = self.received_signal
            self.record['finished_at'] = self._utc()
            self._write()
        finally:
            if self.lock is not None:
                self.lock.__exit__(kind, error, traceback)
            for signum, handler in self.handlers.items():
                signal.signal(signum, handler)
        if error is None and self.received_signal is not None:
            raise InterruptedError(f'interrupted by signal {self.received_signal}')
        return False
