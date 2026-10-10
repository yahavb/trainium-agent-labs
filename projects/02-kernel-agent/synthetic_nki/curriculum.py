"""Four small non-benchmark tasks for observed reduction/layout/tiling failures."""
import hashlib
import json
from pathlib import Path
import nki
from failure_selection import code_fingerprint
from synthetic_nki.verify import validate
from shape_repair import plan_repair


def specs():
    prefix='import nki\nimport nki.language as nl\nimport nki.isa as nisa\n@nki.jit\ndef kernel(a):\n'
    def task(kind,body,shapes,args='a'):
        source=prefix.replace('kernel(a)','kernel('+args+')')+'\n'.join('    '+line for line in body.splitlines())+'\n'
        return dict(kind=kind,family=kind,source=source,shapes=shapes,dtypes=['float32']*len(shapes),seed=9100,description=kind+'; tiny independent operation with additive offset')
    transpose=task('transpose_offset','t=nl.ndarray((2,3),a.dtype,buffer=nl.sbuf)\nnisa.dma_copy(dst=t,src=a)\ny=nl.ndarray((3,2),a.dtype,buffer=nl.sbuf)\nnisa.nc_transpose(dst=y,data=t,engine=nisa.engine.vector)\nnisa.tensor_scalar(dst=y,data=y,op0=nl.add,operand0=.25)\nout=nl.ndarray((3,2),a.dtype,buffer=nl.shared_hbm)\nnisa.dma_copy(dst=out,src=y)\nreturn out',[(2,3)])
    grouped=task('grouped_sum_offset','t=nl.ndarray((2,2,3),a.dtype,buffer=nl.sbuf)\nnisa.dma_copy(dst=t,src=a)\ny=nl.sum(t,axis=2,keepdims=False)\nnisa.tensor_scalar(dst=y,data=y,op0=nl.add,operand0=-.25)\nout=nl.ndarray((2,2),a.dtype,buffer=nl.shared_hbm)\nnisa.dma_copy(dst=out,src=y)\nreturn out',[(2,6)])
    rows=task('row_tiles_offset','out=nl.ndarray((4,3),a.dtype,buffer=nl.shared_hbm)\nfor i in range(2):\n    left=nl.ndarray((4,2),a.dtype,buffer=nl.sbuf)\n    right=nl.ndarray((4,3),b.dtype,buffer=nl.sbuf)\n    nisa.dma_copy(dst=left,src=a[:,i*2:(i+1)*2])\n    nisa.dma_copy(dst=right,src=b)\n    p=nl.ndarray((2,3),nl.float32,buffer=nl.psum)\n    nisa.nc_matmul(dst=p,stationary=left,moving=right,accumulate=False)\n    y=nl.ndarray((2,3),a.dtype,buffer=nl.sbuf)\n    nisa.tensor_copy(dst=y,src=p)\n    nisa.tensor_scalar(dst=y,data=y,op0=nl.add,operand0=.25)\n    nisa.dma_copy(dst=out[i*2:(i+1)*2,:],src=y)\nreturn out',[(4,4),(4,3)],'a,b')
    ragged=task('ragged_k_offset','p=nl.ndarray((2,3),nl.float32,buffer=nl.psum)\nfor i in range(3):\n    start=i*3\n    end=min(7,start+3)\n    length=end-start\n    left=nl.ndarray((length,2),a.dtype,buffer=nl.sbuf)\n    right=nl.ndarray((length,3),b.dtype,buffer=nl.sbuf)\n    nisa.dma_copy(dst=left,src=a[start:end,:])\n    nisa.dma_copy(dst=right,src=b[start:end,:])\n    nisa.nc_matmul(dst=p,stationary=left,moving=right,accumulate=(i>0))\ny=nl.ndarray((2,3),a.dtype,buffer=nl.sbuf)\nnisa.tensor_copy(dst=y,src=p)\nnisa.tensor_scalar(dst=y,data=y,op0=nl.add,operand0=.25)\nout=nl.ndarray((2,3),a.dtype,buffer=nl.shared_hbm)\nnisa.dma_copy(dst=out,src=y)\nreturn out',[(7,2),(7,3)],'a,b')
    rules=[('buffer=nl.sbuf)\n    nisa.nc_transpose','buffer=nl.shared_hbm)\n    nisa.nc_transpose','INVALID_BUFFER_PLACEMENT','transpose_regions'),('axis=2,keepdims=False','axis=(1,2),keepdims=False','INVALID_TENSOR_DIMENSIONS','reduction_rank'),('range(2)','range(1)','NUMERICAL_MISMATCH','output_coverage'),('end=min(7,start+3)','end=start+3','OUT_OF_BOUNDS','boundary_slice_tail')]
    return list(zip((transpose,grouped,rows,ragged),rules))


def build(base,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    records=[json.loads(line) for split in ('train','heldout') for line in (Path(base)/(split+'.jsonl')).read_text().splitlines()];accepted=0;rejected=[]
    for spec,(old,new,category,cause) in specs():
        clean=validate(spec['source'],spec);broken=spec['source'].replace(old,new,1);failure=validate(broken,spec);repair=validate(spec['source'],spec)
        if broken==spec['source'] or not clean['passed'] or failure['passed'] or failure['failure_category']!=category or not repair['passed']:
            rejected.append(dict(kind=spec['kind'],clean=clean,broken=failure));continue
        fp=code_fingerprint(spec['source'])
        if any(r['structural_hash']==fp for r in records):rejected.append(dict(kind=spec['kind'],reason='duplicate clean structure'));continue
        records.append(dict(example_id='curr-'+hashlib.sha256(broken.encode()).hexdigest()[:16],operation_family=spec['family'],seed=spec['seed'],sdk_version=nki.__version__,hardware_target='trn2',input_shapes=spec['shapes'],input_dtypes=spec['dtypes'],task_description=spec['description'],correct_kernel=spec['source'],broken_kernel=broken,observed_error=failure['error'],failure_category=category,root_cause=cause,repair_strategy=plan_repair(broken,failure['error'])['guidance'],corrected_kernel=spec['source'],verification_status='SIMULATOR_VERIFIED',verification_results=dict(clean=clean,broken=failure,repaired=repair),source_provenance=['Project-authored independent offset operation; no benchmark reference source'],license='Licensing unspecified; no external implementation copied',structural_hash=fp,split='train'))
        accepted+=1
    for split in ('train','heldout'):
        with (output/(split+'.jsonl')).open('x') as file:
            for r in records:
                if r['split']==split:file.write(json.dumps(r)+'\n')
    result=dict(added_verified_pairs=accepted,rejected=rejected,verified_pairs=len(records),independent_clean_kernels=len({r['structural_hash'] for r in records}),train_records=sum(r['split']=='train' for r in records),heldout_records=sum(r['split']=='heldout' for r in records),device_verified=0)
    with (output/'summary.json').open('x') as file:json.dump(result,file,indent=2)
    print(json.dumps(result,indent=2));return result
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--base',default='synthetic_nki/data_v2');p.add_argument('--output',default='synthetic_nki/data_v3');a=p.parse_args();build(a.base,a.output)
