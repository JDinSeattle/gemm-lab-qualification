"""Real Linux processes exercise failure boundaries, without using a GPU."""
import concurrent.futures
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

import evidence
from execution import Execution, StepFailed, exclusive_lock


def environment():
    return {'mode': 'local_cpu_fault_injection'}


def new_run(path, **kwargs):
    return Execution(path, 'test', environment, cpu_only=True, grace=.1, **kwargs)


def command(code):
    return [sys.executable, '-c', code]


def wait_for(predicate, seconds=5):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.01)
    raise AssertionError('condition did not become true')


def stopped(pid):
    try:
        # Zombies are dead and cannot retain a GPU context or execute writes.
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[0] == 'Z'
    except FileNotFoundError:
        return True


def tree_code(pidfile, marker, parent_exits=False):
    worker = (
        "import os,signal,time;from pathlib import Path;"
        "signal.signal(signal.SIGTERM,signal.SIG_IGN);"
        f"Path({str(pidfile)!r}).write_text(str(os.getpid()));"
        f"time.sleep(2);Path({str(marker)!r}).write_text('leaked');time.sleep(20)"
    )
    return (
        "import subprocess,sys,time;from pathlib import Path;"
        f"subprocess.Popen([sys.executable,'-c',{worker!r}]);"
        f"\nwhile not Path({str(pidfile)!r}).exists():time.sleep(.01)\n"
        + ('' if parent_exits else 'time.sleep(20)')
    )


def test_running_receipt_precedes_child_and_success(tmp_path):
    out = tmp_path / 'run'
    code = (f"import json;d=json.load(open({str(out / 'execution.json')!r}));"
            "assert d['status']=='running';assert d['steps'][-1]['status']=='running';print('seen')")
    with new_run(out) as run:
        run.run('success', command(code), cwd=tmp_path)
    data = json.loads((out / 'execution.json').read_text())
    assert data['status'] == 'cpu_passed'
    assert data['steps'][0]['returncode'] == 0
    assert data['steps'][0]['cleanup']['direct_child_reaped']
    assert (out / 'success.log').read_text().strip() == 'seen'


@pytest.mark.parametrize('kind', ['nonzero', 'missing'])
def test_failed_stage_stops_following_work(tmp_path, kind):
    out = tmp_path / 'run'
    argv = command('raise SystemExit(7)') if kind == 'nonzero' else ['/no/such/kernel-tool']
    with pytest.raises(StepFailed):
        with new_run(out) as run:
            run.run('failed', argv, cwd=tmp_path)
            run.run('forbidden', command('pass'), cwd=tmp_path)
    data = json.loads((out / 'execution.json').read_text())
    assert data['status'] == 'failed'
    assert len(data['steps']) == 1
    assert data['steps'][0]['returncode'] == (7 if kind == 'nonzero' else 127)
    assert not (out / 'forbidden.log').exists()


@pytest.mark.parametrize('parent_exits', [False, True])
def test_timeout_and_successful_wrapper_cannot_leave_workers(tmp_path, parent_exits):
    out = tmp_path / 'run'
    pidfile, marker = tmp_path / 'pid', tmp_path / 'leaked'
    try:
        with pytest.raises(StepFailed):
            with new_run(out, step_timeout=.6) as run:
                run.run('tree', command(tree_code(pidfile, marker, parent_exits)), cwd=tmp_path)
        wait_for(lambda: stopped(int(pidfile.read_text())))
        step = json.loads((out / 'execution.json').read_text())['steps'][0]
        assert step['returncode'] == (125 if parent_exits else 124)
        assert step['cleanup']['kill_sent']
        assert not marker.exists()
    finally:
        if pidfile.exists() and not stopped(int(pidfile.read_text())):
            os.kill(int(pidfile.read_text()), signal.SIGKILL)


@pytest.mark.parametrize('signum', [signal.SIGTERM, signal.SIGINT])
def test_signal_cleans_descendants_and_records_interruption(tmp_path, signum):
    out = tmp_path / 'run'
    pidfile, marker = tmp_path / 'pid', tmp_path / 'leaked'
    code = (
        "from execution import Execution;"
        f"\nwith Execution({str(out)!r},'signal',lambda:{{}},cpu_only=True,grace=.1) as run:\n"
        f" run.run('tree',{command(tree_code(pidfile, marker))!r},cwd={str(tmp_path)!r})\n"
    )
    proc = subprocess.Popen(command(code), cwd=evidence.ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait_for(pidfile.exists)
        proc.send_signal(signum)
        assert proc.wait(timeout=5) != 0
        wait_for(lambda: stopped(int(pidfile.read_text())))
        data = json.loads((out / 'execution.json').read_text())
        assert data['status'] == 'interrupted' and data['signal'] == signum
        assert data['steps'][0]['returncode'] == 128 + signum
        assert not marker.exists()
    finally:
        if proc.poll() is None:
            proc.kill(); proc.wait()
        if pidfile.exists() and not stopped(int(pidfile.read_text())):
            os.kill(int(pidfile.read_text()), signal.SIGKILL)


def test_output_reservation_race_and_prior_bytes(tmp_path):
    out = tmp_path / 'run'
    code = ("from execution import Execution;"
            f"\nwith Execution({str(out)!r},'race',lambda:{{}},cpu_only=True):pass")
    procs = [subprocess.Popen(command(code), cwd=evidence.ROOT,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) for _ in range(6)]
    codes = [p.wait(timeout=5) for p in procs]
    assert codes.count(0) == 1
    original = (out / 'execution.json').read_bytes()
    with pytest.raises(FileExistsError):
        with new_run(out):
            pytest.fail('existing output accepted')
    assert (out / 'execution.json').read_bytes() == original


def test_existing_empty_directory_is_also_rejected(tmp_path):
    with pytest.raises(FileExistsError):
        with new_run(tmp_path):
            pytest.fail('existing output accepted')


def test_resource_lock_is_process_shared_and_reusable(tmp_path):
    lock = tmp_path / 'gpu.lock'
    out = tmp_path / 'blocked'
    code = ("from execution import Execution;"
            f"\nwith Execution({str(out)!r},'contender',lambda:{{}},lock_path={str(lock)!r}):pass")
    with exclusive_lock(lock):
        proc = subprocess.run(command(code), cwd=evidence.ROOT, capture_output=True, timeout=5)
        assert proc.returncode != 0
        assert b'resource busy' in proc.stderr
        data = json.loads((out / 'execution.json').read_text())
        assert data['status'] == 'failed' and data['steps'] == []
    inode = lock.stat().st_ino
    with exclusive_lock(lock):
        assert lock.stat().st_ino == inode


@pytest.mark.parametrize('value', [0, -1, float('inf'), float('nan')])
def test_invalid_timeout_does_not_reserve_output(tmp_path, value):
    out = tmp_path / 'run'
    with pytest.raises(ValueError):
        new_run(out, step_timeout=value)
    assert not out.exists()


def test_concurrent_atomic_writes_remain_complete(tmp_path):
    target = tmp_path / 'record.json'
    def write(i):
        evidence.save(target, {'writer': i, 'payload': str(i) * 10000})
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(write, range(24)))
    data = json.loads(target.read_text())
    assert data['payload'] == str(data['writer']) * 10000
    assert list(tmp_path.iterdir()) == [target]


def test_serialization_and_replace_failure_preserve_previous_record(tmp_path, monkeypatch):
    target = tmp_path / 'record.json'
    evidence.save(target, {'status': 'passed'})
    before = target.read_bytes()
    with pytest.raises(ValueError):
        evidence.save(target, {'bad': float('nan')})
    def fail(*args):
        raise OSError('injected rename failure')
    monkeypatch.setattr(evidence.os, 'replace', fail)
    with pytest.raises(OSError):
        evidence.save(target, {'status': 'new'})
    assert target.read_bytes() == before
    assert list(tmp_path.iterdir()) == [target]
