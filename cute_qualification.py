import argparse,json
import importlib.metadata
from qualification import MANIFEST,inputs,reference,compare
from cute_adapter import CuteMatmulPlan,CUTLASS_COMMIT,CUTLASS_VERSION,cutlass_source_root
from evidence import environment,save,require_gate,torch_measure,command

def source_gate():
    if importlib.metadata.version('nvidia-cutlass-dsl')!=CUTLASS_VERSION:raise RuntimeError('CUTLASS DSL version mismatch')
    head=command(['git','-C',str(cutlass_source_root()),'rev-parse','HEAD'])
    if head.get('stdout','').strip()!=CUTLASS_COMMIT:raise RuntimeError('CUTLASS source revision mismatch')

def check():
    source_gate();rows=[]
    for shape in MANIFEST['benchmark']:
        a,b=inputs(shape)
        if shape['k']%8 or shape['n']%8:
            rows.append({'case':shape['id'],'status':'unsupported','reason':'K/N must be divisible by 8'});continue
        plan=CuteMatmulPlan(a.cuda(),b.cuda())
        rows.append({'case':shape['id'],'status':'passed','error':compare(plan(),reference(a,b))})
    return {'status':'passed','environment':environment(),'cutlass_commit':CUTLASS_COMMIT,'cutlass_version':CUTLASS_VERSION,'records':rows}

def bench(gate,samples):
    require_gate(gate);source_gate();rows=[]
    import torch
    for shape in MANIFEST['benchmark']:
        if shape['k']%8 or shape['n']%8:continue
        a,b=inputs(shape);da,db=a.cuda(),b.cuda();plan=CuteMatmulPlan(da,db);out=torch.empty_like(plan.output)
        rows.append({'case':shape['id'],'timing':torch_measure({'cute':plan,'torch':lambda:torch.mm(da,db,out=out)},samples=samples)})
    return {'status':'measured','environment':environment(),'results':rows,'cutlass_commit':CUTLASS_COMMIT,'cutlass_version':CUTLASS_VERSION}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['check','bench']);p.add_argument('--output',required=True);p.add_argument('--gate');p.add_argument('--samples',type=int,default=31);a=p.parse_args()
    try:save(a.output,check() if a.phase=='check' else bench(a.gate,a.samples))
    except Exception as exc:
        save(a.output,{'status':'failed','environment':environment(),'error':repr(exc)});raise
