"""Small standalone evidence writer; intentionally has no GPU imports."""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent

def command(args):
    try:
        p = subprocess.run(args, text=True, capture_output=True, timeout=30)
        return {'argv':args, 'returncode':p.returncode, 'stdout':p.stdout, 'stderr':p.stderr}
    except (OSError, subprocess.TimeoutExpired) as e:
        return {'argv':args, 'error':str(e)}

def digest():
    files = {}
    for p in sorted(ROOT.rglob('*')):
        rel=p.relative_to(ROOT)
        if any(x in {'results','.git','.deps','.venv','build','__pycache__','.pytest_cache'} for x in rel.parts):
            continue
        if p.is_file() and (p.suffix in {'.py','.cu','.cuh','.hpp','.json','.sh','.lock'} or p.name=='CMakeLists.txt'):
            files[str(rel)] = hashlib.sha256(p.read_bytes()).hexdigest()
    return hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest(),files

def environment():
    packages={}
    for name in ['numpy','torch','triton','pytest','flashinfer-python','pylibcudf-cu13','pyarrow','cupy-cuda13x','rmm-cu13','nvidia-cutlass-dsl','cuda-bindings','cuda-python','nvidia-cuda-runtime']:
        try: packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: pass
    sha,files=digest()
    return {'utc':datetime.now(timezone.utc).isoformat(),'python':sys.version,'platform':platform.platform(),
            'packages':packages,'source_digest':sha,'source_files':files,
            'git':command(['git','rev-parse','HEAD']),
            'gpu':command(['nvidia-smi','--query-gpu=name,uuid,driver_version,compute_cap,pstate,temperature.gpu,clocks.sm,clocks.mem,power.draw,power.limit,memory.total,memory.used','--format=csv']),
            'nvcc':command(['nvcc','--version']),
            'measurement_policy':{'clock_changes':False,'display_gpu':True,'scope':'one GPU; unlocked clocks; no cross-architecture claim'}}

def save(path, data):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,sort_keys=True,allow_nan=False)+'\n')
    os.replace(tmp,path)

def stats(values):
    if not values or any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError('expected nonempty finite nonnegative measurements')
    s=sorted(values); med=statistics.median(s)
    return {'n':len(s),'p50_ms':med,'p05_ms':s[int(.05*(len(s)-1))],
            'p95_ms':s[int(.95*(len(s)-1))],
            'mad_ms':statistics.median(abs(v-med) for v in s),'raw_ms':values}

def require_gate(path):
    gate=json.loads(Path(path).read_text())
    if gate.get('status')!='passed' or gate['environment']['source_digest']!=digest()[0]:
        raise RuntimeError('missing, failed, or stale correctness gate')
    now=environment()
    before=gate['environment']
    if before['packages']!=now['packages']:
        raise RuntimeError('correctness and benchmark package versions differ')
    # UUID and driver must match; clocks and temperature are allowed to differ.
    def identity(e):
        lines=e['gpu'].get('stdout','').splitlines()
        return [','.join(line.split(',')[:3]) for line in lines[1:]]
    if not identity(now) or identity(before)!=identity(now):
        raise RuntimeError('correctness and benchmark GPU identities differ')
    return gate

def torch_measure(functions, samples=31, warmup=5, inner=10, seed=1729):
    import random,time,torch
    if samples < 3 or warmup < 0 or inner < 1: raise ValueError('invalid benchmark parameters')
    for f in functions.values():
        for _ in range(warmup): f()
    torch.cuda.synchronize()
    rng=random.Random(seed); raw={k:{'event':[],'wall':[]} for k in functions}; order=[]
    for round_id in range(samples):
        keys=list(functions);rng.shuffle(keys);order.append(keys)
        for key in keys:
            start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize(); t=time.perf_counter_ns();start.record()
            for _ in range(inner): functions[key]()
            end.record();end.synchronize()
            raw[key]['wall'].append((time.perf_counter_ns()-t)/1e6/inner)
            raw[key]['event'].append(start.elapsed_time(end)/inner)
    return {'order':order,'inner':inner,'warmup':warmup,'seed':seed,
            'measurements':{k:{'cuda_event':stats(v['event']),'synchronized_wall':stats(v['wall'])} for k,v in raw.items()}}
