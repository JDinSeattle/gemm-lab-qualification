"""Auditable reproduction driver. Every child has a log and recorded exit status."""
from __future__ import annotations
import argparse,json,os,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from evidence import ROOT,save,environment

def main():
    p=argparse.ArgumentParser();p.add_argument('--cpu-only',action='store_true');p.add_argument('--output')
    p.add_argument('--samples',type=int,default=31);args=p.parse_args()
    if args.samples<3:raise ValueError('at least three timing samples required')
    out=Path(args.output).resolve() if args.output else ROOT/'results'/datetime.now(timezone.utc).strftime('run-%Y%m%dT%H%M%SZ')
    out.mkdir(parents=True,exist_ok=True)
    if (out/'execution.json').exists():raise FileExistsError('use a new evidence directory; prior executions are immutable')
    project=json.loads((ROOT/'manifests/project.json').read_text())['project'];steps=[];py=sys.executable
    def run(label,command,timeout=900):
        print(f'[{project}] {label}',flush=True);started=time.monotonic()
        with (out/(label+'.log')).open('w') as log:
            try:result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=timeout);code=result.returncode
            except subprocess.TimeoutExpired:code=124
            except OSError as exc:log.write(str(exc));code=127
        steps.append({'step':label,'argv':command,'returncode':code,'elapsed_s':time.monotonic()-started})
        save(out/'execution.json',{'status':'running' if code==0 else 'failed','steps':steps,'environment':environment()})
        if code:raise RuntimeError(f'{label} failed with exit {code}; inspect {out/(label+".log")}')
    try:
        run('cpu-tests',[py,'-m','pytest','-q','--junitxml='+str(out/'cpu.xml')])
        if project=='B':
            (ROOT/'build').mkdir(exist_ok=True)
            run('cpu-build',['g++','-std=c++17','-O1','-g','-fsanitize=address,undefined','-fno-omit-frame-pointer','-Wall','-Wextra','-Werror','-Iinclude','tests/oracle_test.cpp','-o','build/oracle_sanitized'])
            run('cpu-sanitizer',['./build/oracle_sanitized'])
        if args.cpu_only:
            save(out/'execution.json',{'status':'cpu_passed','steps':steps,'environment':environment()});return
        if project=='A':
            run('native-configure',['cmake','-S','.','-B','build','-G','Ninja','-DCMAKE_BUILD_TYPE=Release','-DCMAKE_CUDA_ARCHITECTURES='+os.getenv('CUDA_ARCH','89')])
            run('native-build',['cmake','--build','build','-j','4'])
            run('native-tests',['ctest','--test-dir','build','--output-on-failure','--output-junit',str(out/'native.xml')])
        if project=='B':run('cccl-build',['bash','scripts/setup.sh'])
        run('gpu-check',[py,'qualification.py','check','--output',str(out/'gpu-check.json')])
        sanitizer=['compute-sanitizer','--tool','memcheck','--error-exitcode','99','--target-processes','all']
        if project=='B':
            rev=json.loads((ROOT/'manifests/versions.json').read_text())['fixed']
            run('gpu-memcheck',sanitizer+['build/'+rev,'check'])
            run('gpu-racecheck',['compute-sanitizer','--tool','racecheck','--error-exitcode','99','build/'+rev,'check'])
        else:run('gpu-memcheck',sanitizer+[py,'qualification.py','check','--output',str(out/'sanitizer-check.json')])
        if project=='A':
            run('native-memcheck',sanitizer+['build/gemm_bench','--m','129','--n','257','--k','65','--dtype','fp16','--impl','ptx_mma_splitk','--warmup','0','--samples','1','--measurement-seconds','0'])
        command=[py,'qualification.py','bench','--gate',str(out/'gpu-check.json'),'--output',str(out/'benchmark.json')]
        if project!='B':command+=['--samples',str(args.samples)]
        run('benchmark',command)
        if project=='A':
            run('native-benchmark',[py,'native_suite.py','--gate',str(out/'gpu-check.json'),'--output',str(out/'native-benchmark.json')])
            run('cute-source',['bash','scripts/setup_cute.sh'])
            cute=os.getenv('CUTE_PYTHON','/tmp/gemm-qualification-cute-venv/bin/python')
            if not Path(cute).exists():run('cute-venv',['uv','venv','--python','3.13',str(Path(cute).parent.parent)])
            if 'CUTE_PYTHON' not in os.environ:run('cute-dependencies',['uv','pip','sync','--python',cute,'requirements-cute.lock'])
            run('cute-check',[cute,'cute_qualification.py','check','--output',str(out/'cute-check.json')])
            run('cute-memcheck',sanitizer+[cute,'cute_qualification.py','check','--output',str(out/'cute-sanitizer-check.json')])
            run('cute-benchmark',[cute,'cute_qualification.py','bench','--gate',str(out/'cute-check.json'),'--output',str(out/'cute-benchmark.json'),'--samples',str(args.samples)])
            for case,shape in [('decode',(8,4096,4096)),('tail',(17,257,131))]:
                run('profile-'+case,['nsys','profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none','--force-overwrite=true','-o',str(out/('profile-'+case)),
                    'build/gemm_bench','--m',str(shape[0]),'--n',str(shape[1]),'--k',str(shape[2]),'--dtype','fp16','--impl','ptx_mma_splitk','--warmup','0','--samples','1','--measurement-seconds','0','--min-sample-ms','0'])
                run('profile-'+case+'-stats',['nsys','stats','--report','cuda_gpu_kern_sum,cuda_gpu_mem_time_sum','--format','csv','--output',str(out/('profile-'+case)),str(out/('profile-'+case+'.nsys-rep'))])
        if project=='C':
            run('dispatch-trace',[py,'profile.py','--output',str(out/'dispatch-trace.json')])
            run('failure-reduction',[py,'reduce.py','--output',str(out/'minimal-mutant.json')])
        if project=='D':
            run('profile',['nsys','profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none','--force-overwrite=true','-o',str(out/'profile'),py,'qualification.py','profile','--output',str(out/'profile.json')])
            run('profile-stats',['nsys','stats','--report','cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,nvtx_sum','--format','csv','--output',str(out/'profile'),str(out/'profile.nsys-rep')])
        save(out/'execution.json',{'status':'passed','steps':steps,'environment':environment()})
        print(str(out),flush=True)
    except BaseException:
        save(out/'execution.json',{'status':'failed','steps':steps,'environment':environment()});raise
if __name__=='__main__':main()
