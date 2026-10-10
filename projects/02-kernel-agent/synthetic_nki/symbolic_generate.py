"""Instantiate original task dimensions through bounded host-side expressions."""
import json
from pathlib import Path
import sympy as sp
from synthetic_nki.generate import templates
from synthetic_nki.mutations import inject
from synthetic_nki.verify import validate
from symbolic_shapes import analyze


def build(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False);records=[];rejected=[]
    K,M,N=sp.symbols('K M N',integer=True,positive=True)
    # New concrete tiny layout, distinct from benchmark tests; no device imports.
    values={K:3,M:2,N:4}
    spec=next(s for s in templates() if s['kind']=='matmul')
    spec=dict(spec,source=spec['source'].replace('(4,2)','(3,2)').replace('(4,3)','(3,4)').replace('(2,3)','(2,4)'),shapes=[tuple(int(d.subs(values)) for d in shape) for shape in ((K,M),(K,N))])
    correct=spec['source'];broken=correct.replace('psum=nl.ndarray((2,4)','psum=nl.ndarray((1,2)',1)
    shapes=dict(zip(('a','b'),spec['shapes']))
    clean_symbolic=analyze(correct,shapes);bad_symbolic=analyze(broken,shapes)
    clean=validate(correct,spec);bad=validate(broken,spec);repair=validate(correct,spec)
    accepted=clean['passed'] and repair['passed'] and not bad['passed'] and bad['failure_category']=='INVALID_TENSOR_DIMENSIONS' and any(v['kind']=='matmul_dimensions' for v in bad_symbolic['violations'])
    record=dict(example_id='sympy-matmul-offset-3x2x4',symbolic_constraints=['stationary=(K,M)','moving=(K,N)','dst=(M,N)','K<=128','M<=128','N<=512'],instantiated_dimensions={str(k):int(v) for k,v in values.items()},intended_mutation='Wrong PSUM result shape (1,2)',correct_kernel=correct,broken_kernel=broken,actual_simulator_feedback=bad['error'],symbolic_diagnostic=bad_symbolic,verified_correction=repair,verification_results=dict(clean=clean,broken=bad,repaired=repair),sdk_version=clean['sdk_version'],hardware_target=clean['hardware_target'],clean_symbolic=clean_symbolic,accepted=accepted,verification_status='SIMULATOR_VERIFIED' if accepted else None,device_verified=False,source_provenance='Project-authored tiny offset matmul; general AWS invariants only; no benchmark reference source')
    with (output/'record.json').open('x') as file:json.dump(record,file,indent=2)
    if not accepted:raise RuntimeError('Symbolic instantiation or actual mutation verification failed; artifact preserved')
    return record
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();r=build(a.output);print(r['example_id'],r['accepted'])
