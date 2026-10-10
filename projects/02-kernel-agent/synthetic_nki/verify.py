"""Trusted locally generated snippets only; independent NumPy numerical checker."""
import ast
import numpy as np
import nki
import nkibench
from failure_selection import classify_failure

def reference(spec,args):
    a=args[0];kind=spec['kind']
    if kind=='slice':return a[1:3,1:4].copy()
    if kind in ('copy','partition','psum'):return a.copy()
    if kind=='multiply':return a*1.5
    if kind=='add':return a+.25
    if kind=='binary':return a*args[1]
    if kind=='sum':return np.sum(a,axis=1,keepdims=True)
    if kind=='max':return np.max(a,axis=1,keepdims=True)
    return a.T@args[1]+.25

def validate(source,spec,trials=3):
    result=dict(status=None,passed=False,cases=[],sdk_version=nki.__version__,hardware_target='trn2',error=None)
    try:
        compile(source,'<synthetic>','exec');tree=ast.parse(source)
        imports=[n for n in ast.walk(tree) if isinstance(n,(ast.Import,ast.ImportFrom))]
        if any(isinstance(n,ast.ImportFrom) or any(not x.name.startswith('nki') for x in n.names) for n in imports):raise ValueError('Only NKI imports permitted in synthetic kernels')
        function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='kernel')
        if not any(ast.unparse(d)=='nki.jit' for d in function.decorator_list):raise ValueError('Missing @nki.jit')
        result['status']='STATIC_VERIFIED'
        namespace={};exec(compile(source,'<synthetic>','exec'),namespace)
        for offset in range(trials):
            rng=np.random.default_rng(spec['seed']+offset)
            args=[rng.normal(size=shape).astype(dtype) for shape,dtype in zip(spec['shapes'],spec['dtypes'])]
            before=[a.copy() for a in args];want=reference(spec,args)
            try:
                got,counted=nkibench.simulate_and_count(namespace['kernel'],args)
                if any(not np.array_equal(a,b) for a,b in zip(args,before)):raise AssertionError('Input mutation detected')
                hazards=[w for w in counted.get('warnings',[]) if 'incorrect results on hardware' in w]
                if hazards:raise AssertionError('Hardware correctness hazard: '+hazards[0])
                if np.asarray(got).shape!=want.shape:raise AssertionError(f'Numerical output shape mismatch: {np.asarray(got).shape} versus {want.shape}')
                try:np.testing.assert_allclose(got,want,rtol=2e-5,atol=2e-5,equal_nan=False)
                except AssertionError as error:raise AssertionError('NUMERICAL MISMATCH: '+str(error)) from error
                result['cases'].append(dict(seed=spec['seed']+offset,passed=True,numerically_correct=True,inputs_untouched=True))
            except Exception as error:
                result['cases'].append(dict(seed=spec['seed']+offset,passed=False,error=f'{type(error).__name__}: {error}'))
                raise
        result.update(status='SIMULATOR_VERIFIED',passed=True)
    except Exception as error:result['error']=f'{type(error).__name__}: {error}'
    result['failure_category']=classify_failure(result['error'] or '').failure_category
    return result
