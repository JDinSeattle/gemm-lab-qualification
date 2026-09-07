from __future__ import annotations
import argparse,json,math,time
from pathlib import Path
import torch
from evidence import ROOT,environment,save,require_gate,torch_measure

MANIFEST=json.loads((ROOT/'manifests/workloads.json').read_text())

def inputs(shape,pattern='random',layout='contiguous'):
    m,n,k=(shape[x] for x in ('m','n','k'))
    g=torch.Generator().manual_seed(MANIFEST['seed'])
    a=(torch.randn((m,k),generator=g)*.2).half()
    b=(torch.randn((k,n),generator=g)*.2).half()
    if pattern=='zeros': a.zero_();b.zero_()
    elif pattern=='cancellation':
        a.fill_(256);b.fill_(.125);b[1::2].neg_()
    elif pattern!='random':raise ValueError(pattern)
    if layout=='strided':
        aa=torch.full((m,k*2+2),float('nan'),dtype=a.dtype);aa[:,1:1+2*k:2]=a;a=aa[:,1:1+2*k:2]
        bb=torch.full((k,n*2+2),float('nan'),dtype=b.dtype);bb[:,1:1+2*n:2]=b;b=bb[:,1:1+2*n:2]
    elif layout!='contiguous':raise ValueError(layout)
    return a,b

def reference(a,b):
    if a.ndim!=2 or b.ndim!=2 or a.shape[1]!=b.shape[0]:raise ValueError('shape contract')
    if a.dtype!=torch.float16 or b.dtype!=a.dtype:raise ValueError('v1 is FP16 only')
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():raise ValueError('finite input contract')
    out=a.double()@b.double()
    if (out.abs()>65504).any():raise ValueError('FP16 output overflow excluded')
    return out

def compare(actual,expected,atol=None,rtol=None):
    atol=MANIFEST['atol'] if atol is None else atol;rtol=MANIFEST['rtol'] if rtol is None else rtol
    actual=actual.detach().cpu().double();expected=expected.cpu().double()
    if actual.shape!=expected.shape:raise AssertionError('output shape mismatch')
    error=(actual-expected).abs();bound=atol+rtol*expected.abs()
    if not torch.isfinite(actual).all() or not (error<=bound).all():
        raise AssertionError(f'numerical gate: max absolute error {error.max().item() if error.numel() else 0}')
    return {'max_abs':error.max().item() if error.numel() else 0,
            'max_scaled_error':(error/bound).max().item() if error.numel() else 0,
            'rms':error.square().mean().sqrt().item() if error.numel() else 0}

def path(a,b):
    return 'torch-fallback' if 0 in (*a.shape,*b.shape) or not a.is_contiguous() or not b.is_contiguous() else 'triton'

def candidate(a,b):
    if path(a,b)=='torch-fallback':return a@b
    from kernels import triton_matmul
    return triton_matmul(a,b)

def gpu_inputs(a,b):
    # Tensor.to() compacts some sliced views. Build the view again on CUDA.
    if a.is_contiguous() and b.is_contiguous():return a.cuda(),b.cuda()
    def view(x):
        storage=torch.full((x.shape[0],x.shape[1]*2+2),float('nan'),device='cuda',dtype=x.dtype)
        storage[:,1:1+2*x.shape[1]:2]=x.cuda();return storage[:,1:1+2*x.shape[1]:2]
    return view(a),view(b)

def check():
    if not torch.cuda.is_available():raise RuntimeError('real CUDA device required')
    from kernels import triton_matmul_into,triton_bias_silu
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
    records=[]
    for shape in MANIFEST['edges']+MANIFEST['benchmark']:
        patterns=MANIFEST['edge_patterns'] if shape in MANIFEST['edges'] else ['random']
        for pattern in patterns:
            for layout in MANIFEST['layouts']:
                a,b=inputs(shape,pattern,layout);expected=reference(a,b);da,db=gpu_inputs(a,b)
                for implementation,fn in [('torch',lambda: da@db),('candidate',lambda: candidate(da,db))]:
                    err=compare(fn(),expected)
                    records.append({'case':shape['id'],'pattern':pattern,'layout':layout,'implementation':implementation,
                                    'dispatch':path(da,db) if implementation=='candidate' else 'torch','error':err,
                                    'strides':[list(da.stride()),list(db.stride())]})
                if path(da,db)=='triton':
                    guard=torch.full((shape['m']*shape['n']+32,),123,device='cuda',dtype=da.dtype)
                    out=guard[16:-16].view(shape['m'],shape['n'])
                    triton_matmul_into(da,db,out);compare(out,expected)
                    assert (guard[:16]==123).all() and (guard[-16:]==123).all()
                    bias=torch.linspace(-.2,.2,shape['n'],dtype=torch.float16,device='cuda')
                    # FP64 mathematical epilogue; fused FP32 epilogue may differ from staged FP16.
                    ref=torch.nn.functional.silu(expected+bias.cpu().double())
                    compare(triton_bias_silu(da,db,bias),ref,atol=.01,rtol=.01)
    return {'status':'passed','environment':environment(),'records':records,'guards':'passed','fused_epilogue':'passed'}

def benchmark(gate,samples):
    require_gate(gate)
    from kernels import TritonMatmulPlan,TritonBiasSiluPlan
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
    results=[]
    for shape in MANIFEST['benchmark']:
        a,b=inputs(shape);da,db=a.cuda(),b.cuda();bias=torch.linspace(-.2,.2,shape['n'],device='cuda',dtype=da.dtype)
        output=torch.empty((shape['m'],shape['n']),device='cuda',dtype=da.dtype)
        plan=TritonMatmulPlan(da,db);fused=TritonBiasSiluPlan(da,db,bias)
        def baseline_fragment():return torch.nn.functional.silu(da@db+bias)
        timing=torch_measure({'torch-preallocated':lambda:torch.mm(da,db,out=output),'triton-preallocated':plan,
                              'torch-fragment':baseline_fragment,'triton-fragment':fused},samples=samples)
        # Includes allocation, H2D, op, D2H and host result consumption. No oracle in this timed region.
        def etl(fn):
            aa,bb=a.cuda(),b.cuda();result=fn(aa,bb).cpu();return float(result.sum(dtype=torch.float64))
        e2e=torch_measure({'torch-e2e':lambda:etl(lambda x,y:x@y),'candidate-e2e':lambda:etl(candidate)},samples=samples,inner=1)
        transfer=torch_measure({'h2d':lambda:(a.cuda(),b.cuda()),'allocation':lambda:torch.empty_like(output),
                                'd2h':lambda:output.cpu()},samples=samples,inner=1)
        results.append({'case':shape,'timing':timing,'end_to_end':e2e,'transfer_and_allocation':transfer})
    return {'status':'measured','environment':environment(),'correctness_gate':str(gate),'results':results,
            'timing_scope':'CUDA events enclose the launch sequence; synchronized wall includes host dispatch; no CUDA graph replay',
            'integration':'linear projection + bias + SiLU; preloaded weights; separate host-to-host pipeline'}

def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['check','bench']);p.add_argument('--output',required=True)
    p.add_argument('--gate');p.add_argument('--samples',type=int,default=31);args=p.parse_args()
    try:
        result=check() if args.phase=='check' else benchmark(args.gate,args.samples)
        save(args.output,result)
    except Exception as exc:
        save(args.output,{'status':'failed','environment':environment(),'error':repr(exc)})
        raise
if __name__=='__main__':main()
