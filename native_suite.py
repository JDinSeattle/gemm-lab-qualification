import argparse,hashlib,json,random,subprocess
from evidence import ROOT,save,environment,require_gate
from qualification import MANIFEST

def run(gate):
    require_gate(gate);binary=ROOT/'build/gemm_bench';rows=[];rng=random.Random(1729)
    tasks=[(s,impl) for s in MANIFEST['benchmark'] for impl in ['cublaslt','ptx_mma_small','ptx_mma_splitk']]
    for round_id in range(3):
        rng.shuffle(tasks)
        for shape,impl in tasks:
            args=[str(binary),'--m',str(shape['m']),'--n',str(shape['n']),'--k',str(shape['k']),
                  '--dtype','fp16','--impl',impl,'--warmup','10','--samples','31','--measurement-seconds','0','--seed','1729']
            p=subprocess.run(args,text=True,capture_output=True,timeout=120)
            if p.returncode:raise RuntimeError(p.stderr)
            result=json.loads(p.stdout)
            if not result['correctness']['passed']:raise AssertionError('native correctness failed')
            rows.append({'round':round_id,'case':shape['id'],'argv':args,'record':result,'stderr':p.stderr})
    return {'status':'measured','environment':environment(),'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),
            'seed':1729,'results':rows,'protocol':'3 shuffled rounds; each child validates before warmup/timing; no sanitizer/profiler in timed process'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gate',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    save(a.output,run(a.gate))
