"""Small independently specified primitive compositions; never benchmark answers.

AWS tutorial/API concepts are attributed in PROVENANCE. Implementations are
project-authored, offset computations on different shapes and operation tasks.
CPU simulation and independent NumPy verification are the only claimed gates.
"""
import hashlib
import inspect
import json
from pathlib import Path
import nki
import nki.isa as isa
import nki.language as nl
from failure_selection import code_fingerprint
from shape_repair import plan_repair
from synthetic_nki.verify import validate

PROVENANCE=[
 'https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/guides/tutorials/average_pool2d.html',
 'https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/generated/nki.isa.tensor_scalar.html',
 'https://awsdocs-neuron.readthedocs-hosted.com/en/v2.31.0/nki/guides/tutorials/transpose2d.html',
 'https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/generated/nki.isa.nc_transpose.html',
 'https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/generated/nki.isa.nc_matmul.html',
]
HEADER='import nki\nimport nki.language as nl\nimport nki.isa as nisa\n@nki.jit\n'

def specs():
    ap=dict(kind='primitive_ap_mean_offset',family='access_pattern_mean_offset',shapes=[(2,12)],dtypes=['float32'],seed=20001,description='Group each row into triples of four elements, compute each group mean, then add .375.',source=HEADER+'''def kernel(a):
    t=nl.ndarray((2,12),a.dtype,buffer=nl.sbuf)
    nisa.dma_copy(dst=t,src=a)
    view=t.ap([[12,2],[4,3],[1,4]])
    y=nl.sum(view,axis=2)
    z=nl.ndarray(y.shape,a.dtype,buffer=nl.sbuf)
    nisa.tensor_scalar(dst=z,data=y,op0=nl.multiply,operand0=.25,op1=nl.add,operand1=.375)
    out=nl.ndarray((2,3),a.dtype,buffer=nl.shared_hbm)
    nisa.dma_copy(dst=out,src=z)
    return out
''')
    free=dict(kind='primitive_free_permutation_offset',family='free_permutation_offset',shapes=[(3,8)],dtypes=['float32'],seed=20002,description='Permute the two inner free axes in each independent 2 by 4 row and add .125.',source=HEADER+'''def kernel(a):
    t=nl.ndarray((3,8),a.dtype,buffer=nl.sbuf)
    u=nl.ndarray((3,8),a.dtype,buffer=nl.sbuf)
    nisa.dma_copy(dst=t,src=a)
    source=t.ap([[8,3],[4,2],[1,4]])
    destination=u.ap([[8,3],[1,2],[2,4]])
    nisa.tensor_copy(dst=destination,src=source)
    nisa.tensor_scalar(dst=u,data=u,op0=nl.add,operand0=.125)
    out=nl.ndarray((3,8),a.dtype,buffer=nl.shared_hbm)
    nisa.dma_copy(dst=out,src=u)
    return out
''')
    pf=dict(kind='primitive_partition_transpose_offset',family='partition_transpose_offset',shapes=[(5,7)],dtypes=['float32'],seed=20003,description='Exchange partition/free axes for a small rectangular tile and multiply by .75.',source=HEADER+'''def kernel(a):
    t=nl.ndarray((5,7),a.dtype,buffer=nl.sbuf)
    nisa.dma_copy(dst=t,src=a)
    p=nl.ndarray((7,5),nl.float32,buffer=nl.psum)
    nisa.nc_transpose(dst=p,data=t,engine=nisa.engine.tensor)
    u=nl.ndarray((7,5),a.dtype,buffer=nl.sbuf)
    nisa.tensor_copy(dst=u,src=p)
    nisa.tensor_scalar(dst=u,data=u,op0=nl.multiply,operand0=.75)
    out=nl.ndarray((7,5),a.dtype,buffer=nl.shared_hbm)
    nisa.dma_copy(dst=out,src=u)
    return out
''')
    k=dict(kind='primitive_k_accumulate_offset',family='k_accumulate_offset',shapes=[(12,5),(12,7)],dtypes=['float32','float32'],seed=20004,description='Contract two rectangular inputs across three disjoint four-element blocks and add .125 to every result.',source=HEADER+'''def kernel(a,b):
    p=nl.ndarray((5,7),nl.float32,buffer=nl.psum)
    for step in range(3):
        left=nl.ndarray((4,5),a.dtype,buffer=nl.sbuf)
        right=nl.ndarray((4,7),b.dtype,buffer=nl.sbuf)
        nisa.dma_copy(dst=left,src=a[step*4:(step+1)*4,:])
        nisa.dma_copy(dst=right,src=b[step*4:(step+1)*4,:])
        nisa.nc_matmul(dst=p,stationary=left,moving=right,accumulate=(step>0))
    u=nl.ndarray((5,7),nl.float32,buffer=nl.sbuf)
    nisa.tensor_copy(dst=u,src=p)
    nisa.tensor_scalar(dst=u,data=u,op0=nl.add,operand0=.125)
    out=nl.ndarray((5,7),nl.float32,buffer=nl.shared_hbm)
    nisa.dma_copy(dst=out,src=u)
    return out
''')
    return [ap,free,pf,k]


def build(base,output):
    records=[json.loads(line) for split in ('train','heldout') for line in (Path(base)/(split+'.jsonl')).read_text().splitlines()]
    rules=[('nisa.tensor_scalar(dst=z','nl.tensor_scalar(dst=z','INVALID_API_FUNCTION','scalar_instruction_namespace'),('[4,3]','[4,4]','OUT_OF_BOUNDS','access_pattern_bounds'),('operand0=.25','operand0=.5','NUMERICAL_MISMATCH','incorrect_normalization')]
    mutations={
     'primitive_ap_mean_offset':rules,
     'primitive_free_permutation_offset':[('[1,2],[2,4]','[4,2],[1,4]','NUMERICAL_MISMATCH','missing_free_permutation')],
     'primitive_partition_transpose_offset':[('buffer=nl.psum','buffer=nl.sbuf','INVALID_BUFFER_PLACEMENT','transpose_output_buffer')],
     'primitive_k_accumulate_offset':[('accumulate=(step>0)','accumulate=False','NUMERICAL_MISMATCH','missing_contraction_accumulation')],
    }
    rejected=[];accepted=[];clean_results=[]
    for s in specs():
        clean=validate(s['source'],s);clean_results.append(dict(kind=s['kind'],verification=clean))
        if not clean['passed']:raise ValueError(clean_results[-1])
        for old,new,category,cause in mutations[s['kind']]:
            broken=s['source'].replace(old,new,1);failure=validate(broken,s);repaired=validate(s['source'],s)
            if broken==s['source'] or failure['passed'] or failure['failure_category']!=category or not repaired['passed']:
                rejected.append(dict(kind=s['kind'],cause=cause,intended_category=category,actual=failure));continue
            fp=code_fingerprint(s['source'])
            example_id='primitive-'+hashlib.sha256(broken.encode()).hexdigest()[:16]
            if any(code_fingerprint(r['broken_kernel'])==code_fingerprint(broken) for r in records):
                rejected.append(dict(kind=s['kind'],cause=cause,reason='duplicate broken AST'));continue
            record=dict(example_id=example_id,operation_family=s['family'],seed=s['seed'],sdk_version=nki.__version__,hardware_target='trn2',input_shapes=s['shapes'],input_dtypes=s['dtypes'],task_description=s['description'],correct_kernel=s['source'],broken_kernel=broken,observed_error=failure['error'],failure_category=category,root_cause=cause,repair_strategy=plan_repair(broken,failure['error'])['guidance'],corrected_kernel=s['source'],verification_status='SIMULATOR_VERIFIED',verification_results=dict(clean=clean,broken=failure,repaired=repaired),source_provenance=['Project-authored independent offset/scaling tasks, concepts attributed to AWS']+PROVENANCE,license='Project-authored; no external implementation copied',structural_hash=fp,split='train')
            records.append(record);accepted.append(example_id)
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    for split in ('train','heldout'):
        with (out/(split+'.jsonl')).open('x') as file:
            for record in records:
                if record['split']==split:file.write(json.dumps(record)+'\n')
    summary=dict(verified_pairs=len(records),independent_clean_kernels=len({r['structural_hash'] for r in records}),added_verified_pairs=len(accepted),accepted=accepted,rejected=rejected,clean_primitive_results=clean_results,train_records=sum(r['split']=='train' for r in records),heldout_records=sum(r['split']=='heldout' for r in records),sdk_version=nki.__version__,device_verified=0,signatures={name:str(inspect.signature(getattr(isa,name))) for name in ('dma_copy','tensor_scalar','tensor_copy','nc_transpose','nc_matmul')},access_pattern_signature=str(inspect.signature(nl.NkiTensor.ap)))
    (out/'summary.json').write_text(json.dumps(summary,indent=2));return summary

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--base',default='synthetic_nki/data_v6');p.add_argument('--output',required=True);a=p.parse_args();print(json.dumps(build(a.base,a.output),indent=2))
