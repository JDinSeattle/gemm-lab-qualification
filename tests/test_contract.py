import pytest
import torch
from qualification import inputs,reference,compare,path,MANIFEST
from evidence import stats

@pytest.mark.parametrize('shape',MANIFEST['edges'])
@pytest.mark.parametrize('pattern',MANIFEST['edge_patterns'])
@pytest.mark.parametrize('layout',MANIFEST['layouts'])
def test_full_reference(shape,pattern,layout):
    a,b=inputs(shape,pattern,layout);r=reference(a,b)
    # Scalar dot is independent of BLAS and catches transposition and empty-K mistakes.
    for i,j in [(0,0),(shape['m']-1,shape['n']-1)]:
        if shape['m'] and shape['n']:
            assert abs(r[i,j].item()-sum(float(a[i,k])*float(b[k,j]) for k in range(shape['k'])))<1e-9

@pytest.mark.parametrize('mutation',['transpose','tail','nan','scale'])
def test_detector_mutations(mutation):
    a,b=inputs({'m':17,'n':31,'k':13});r=reference(a,b);bad=r.clone()
    if mutation=='transpose':bad=bad.T
    if mutation=='tail':bad[-1,-1]+=1
    if mutation=='nan':bad[0,0]=float('nan')
    if mutation=='scale':bad*=2
    with pytest.raises(AssertionError):compare(bad,r)

@pytest.mark.parametrize('value',[float('nan'),float('inf'),-float('inf'),65504.])
def test_invalid_numerics(value):
    a=torch.full((1,1),value,dtype=torch.float16);b=torch.full((1,1),2.,dtype=torch.float16)
    with pytest.raises(ValueError):reference(a,b)

def test_statistics_and_fallback():
    assert stats([1,2,3])['p50_ms']==2
    with pytest.raises(ValueError):stats([])
    a,b=inputs({'m':3,'n':4,'k':5},layout='strided');assert path(a,b)=='torch-fallback'
