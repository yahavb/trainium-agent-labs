"""Generate 12 independent small tasks; validate clean, mutated, and restored code."""
import argparse
import hashlib
import json
from pathlib import Path
import nki
from failure_selection import code_fingerprint
from shape_repair import plan_repair
from synthetic_nki.verify import validate
from synthetic_nki.mutations import inject

PREFIX='import nki\nimport nki.language as nl\nimport nki.isa as nisa\n\n@nki.jit\ndef kernel(a'+'):\n'

def templates(seed=2026):
    specs=[]
    def add(kind,family,body,shapes,description,arguments='a'):
        source=PREFIX.replace('kernel(a)','kernel('+arguments+')')+'\n'.join('    '+line for line in body.splitlines())+'\n'
        specs.append(dict(kind=kind,family=family,source=source,shapes=shapes,dtypes=['float32']*len(shapes),seed=seed+len(specs)*17,description=description))
    load='tile=nl.ndarray((2,3),a.dtype,buffer=nl.sbuf)\nnisa.dma_copy(dst=tile,src=a)'
    store='out=nl.ndarray((2,3),a.dtype,buffer=nl.shared_hbm)\nnisa.dma_copy(dst=out,src=tile)\nreturn out'
    add('copy','dma_copy',load+'\n'+store,[(2,3)],'Copy a tiny 2D tensor unchanged.')
    add('slice','dma_slices',load.replace('src=a','src=a[1:3,1:4]')+'\n'+store,[(4,5)],'Copy a bounded 2x3 subrectangle.')
    add('multiply','scalar_arithmetic',load+'\nnisa.tensor_scalar(dst=tile,data=tile,op0=nl.multiply,operand0=1.5)\n'+store,[(2,3)],'Scale each element by 1.5.')
    add('add','scalar_arithmetic',load+'\nnisa.tensor_scalar(dst=tile,data=tile,op0=nl.add,operand0=.25)\n'+store,[(2,3)],'Add a quarter to each element.')
    add('binary','elementwise_arithmetic',load+'\nother=nl.ndarray((2,3),b.dtype,buffer=nl.sbuf)\nnisa.dma_copy(dst=other,src=b)\nnisa.tensor_tensor(dst=tile,data1=tile,data2=other,op=nl.multiply)\n'+store,[(2,3),(2,3)],'Multiply corresponding elements of two tiny tensors.',arguments='a,b')
    for kind in ('sum','max'):
        add(kind,'reductions',load+f'\nresult=nl.{kind}(tile,axis=1,keepdims=True)\nout=nl.ndarray((2,1),a.dtype,buffer=nl.shared_hbm)\nnisa.dma_copy(dst=out,src=result)\nreturn out',[(2,3)],'Reduce the trailing axis, preserving a singleton free axis.')
    mm='left=nl.ndarray((4,2),a.dtype,buffer=nl.sbuf)\nright=nl.ndarray((4,3),b.dtype,buffer=nl.sbuf)\npsum=nl.ndarray((2,3),nl.float32,buffer=nl.psum)'
    result='tile=nl.ndarray((2,3),nl.float32,buffer=nl.sbuf)\nnisa.tensor_copy(dst=tile,src=psum)\nnisa.tensor_scalar(dst=tile,data=tile,op0=nl.add,operand0=.25)\n'+store
    add('matmul','matmul_offset',mm+'\nnisa.dma_copy(dst=left,src=a)\nnisa.dma_copy(dst=right,src=b)\nnisa.nc_matmul(dst=psum,stationary=left,moving=right,accumulate=False)\n'+result,[(4,2),(4,3)],'Compute a tiny transposed-left matrix product plus a quarter.',arguments='a,b')
    add('accum','contraction_accumulation',mm+'\nfor i in range(2):\n    nisa.dma_copy(dst=left,src=a[i*4:(i+1)*4,:])\n    nisa.dma_copy(dst=right,src=b[i*4:(i+1)*4,:])\n    nisa.nc_matmul(dst=psum,stationary=left,moving=right,accumulate=(i>0))\n'+result,[(8,2),(8,3)],'Combine two disjoint contraction contributions, then add a quarter.',arguments='a,b')
    add('partition','partition_copy','out=nl.ndarray(a.shape,a.dtype,buffer=nl.shared_hbm)\nfor i in range(2):\n    tile=nl.ndarray((128,3),a.dtype,buffer=nl.sbuf)\n    nisa.dma_copy(dst=tile,src=a[i*128:(i+1)*128,:])\n    nisa.dma_copy(dst=out[i*128:(i+1)*128,:],src=tile)\nreturn out',[(256,3)],'Copy two independent legal partition tiles.')
    column_body='out=nl.ndarray((2,6),a.dtype,buffer=nl.shared_hbm)\nfor i in range(2):\n'+ '\n'.join('    '+line for line in (mm+'\nnisa.dma_copy(dst=left,src=a)\nnisa.dma_copy(dst=right,src=b[:,i*3:(i+1)*3])\nnisa.nc_matmul(dst=psum,stationary=left,moving=right,accumulate=False)\ntile=nl.ndarray((2,3),nl.float32,buffer=nl.sbuf)\nnisa.tensor_copy(dst=tile,src=psum)\nnisa.tensor_scalar(dst=tile,data=tile,op0=nl.add,operand0=.25)\nnisa.dma_copy(dst=out[:,i*3:(i+1)*3],src=tile)').splitlines())+'\nreturn out'
    add('columns','output_tiles',column_body,[(4,2),(4,6)],'Fill two disjoint column tiles of an offset matrix product.',arguments='a,b')
    add('psum','onchip_regions',load+'\nacc=nl.ndarray((2,3),nl.float32,buffer=nl.psum)\nnisa.tensor_copy(dst=acc,src=tile)\nnisa.tensor_copy(dst=tile,src=acc)\n'+store,[(2,3)],'Round-trip a tile through PSUM using on-chip copies only.')
    return specs

def build(output,seed=2026):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    records=[];rejected=[];seen=set()
    for spec in templates(seed):
        clean=validate(spec['source'],spec)
        if not clean['passed']:rejected.append(dict(kind=spec['kind'],stage='clean',result=clean));continue
        fp=code_fingerprint(spec['source'])
        if fp in seen:rejected.append(dict(kind=spec['kind'],stage='duplicate'));continue
        seen.add(fp)
        broken,expected,cause=inject(spec['source'],spec);failed=validate(broken,spec)
        repaired=validate(spec['source'],spec)
        if failed['passed'] or failed['failure_category']!=expected or not repaired['passed']:
            rejected.append(dict(kind=spec['kind'],stage='mutation',expected=expected,result=failed));continue
        record=dict(example_id='syn-'+hashlib.sha256(spec['source'].encode()).hexdigest()[:12],operation_family=spec['family'],seed=spec['seed'],sdk_version=nki.__version__,hardware_target='trn2',input_shapes=spec['shapes'],input_dtypes=spec['dtypes'],task_description=spec['description'],correct_kernel=spec['source'],broken_kernel=broken,observed_error=failed['error'],failure_category=failed['failure_category'],root_cause=cause,repair_strategy=plan_repair(broken,failed['error'])['guidance'],corrected_kernel=spec['source'],verification_status='SIMULATOR_VERIFIED',verification_results=dict(clean=clean,broken=failed,repaired=repaired),source_provenance=['Original independent tiny task templates; no benchmark reference kernels','https://awsdocs-neuron.readthedocs-hosted.com/en/v2.32.0/nki/'],license='Project-authored; licensing unspecified; no external implementation copied',structural_hash=fp,split='heldout' if spec['family']=='elementwise_arithmetic' else 'train')
        records.append(record)
    for split in ('train','heldout'):
        with (output/(split+'.jsonl')).open('x') as file:
            for record in records:
                if record['split']==split:file.write(json.dumps(record)+'\n')
    import collections
    summary=dict(generated_clean_tasks=len(templates(seed)),verified_clean_tasks=len(seen),verified_repair_pairs=len(records),train_records=sum(r['split']=='train' for r in records),heldout_records=sum(r['split']=='heldout' for r in records),failure_categories=dict(collections.Counter(r['failure_category'] for r in records)),rejected=rejected,seed=seed,verification_status='SIMULATOR_VERIFIED',device_verified=0,heldout_policy='Entire elementwise_arithmetic family excluded from retrieval',heldout_repaired_passes=sum(r['verification_results']['repaired']['passed'] for r in records if r['split']=='heldout'))
    with (output/'summary.json').open('x') as file:json.dump(summary,file,indent=2)
    return summary

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='synthetic_nki/data');parser.add_argument('--seed',type=int,default=2026)
    args=parser.parse_args();print(json.dumps(build(args.output,args.seed),indent=2))
if __name__=='__main__':main()
