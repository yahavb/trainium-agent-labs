"""Independent grouped-mean-plus-bias task: shape and mathematical bug injection."""
import hashlib
import json
from pathlib import Path
import nki
from failure_selection import code_fingerprint
from shape_repair import plan_repair
from synthetic_nki.verify import validate


def spec():
    source='''import nki
import nki.language as nl
import nki.isa as nisa
@nki.jit
def kernel(a):
    t=nl.ndarray((3,2,3),a.dtype,buffer=nl.sbuf)
    nisa.dma_copy(dst=t,src=a)
    y=nl.sum(t,axis=2,keepdims=False)
    nisa.tensor_scalar(dst=y,data=y,op0=nl.multiply,operand0=1.0/3)
    nisa.tensor_scalar(dst=y,data=y,op0=nl.add,operand0=.125)
    out=nl.ndarray((3,2),a.dtype,buffer=nl.shared_hbm)
    nisa.dma_copy(dst=out,src=y)
    return out
'''
    return dict(kind='grouped_mean_offset',family='grouped_mean_offset',source=source,shapes=[(3,6)],dtypes=['float32'],seed=10301,description='Group each row into triples; compute each group mean and add one eighth. This is not 2D spatial pooling.')


def build(base,output):
    s=spec();clean=validate(s['source'],s)
    if not clean['passed']:raise ValueError(clean)
    records=[json.loads(x) for split in ('train','heldout') for x in (Path(base)/(split+'.jsonl')).read_text().splitlines()]
    rules=[('nisa.tensor_scalar(dst=y,data=y,op0=nl.multiply,operand0=1.0/3)','y=y/3','UNKNOWN','python_tensor_division'),('(3,2,3)','(3,2,3,3)','DMA_SHAPE_MISMATCH','group_axis_element_conservation'),('operand0=1.0/3','operand0=.5','NUMERICAL_MISMATCH','incorrect_normalization'),('y=nl.sum','y=nl.max','NUMERICAL_MISMATCH','incorrect_reduction_operation')]
    rejected=[];accepted=[]
    for old,new,category,cause in rules:
        broken=s['source'].replace(old,new,1);failure=validate(broken,s);repair=validate(s['source'],s)
        if failure['passed'] or failure['failure_category']!=category or not repair['passed']:
            rejected.append(dict(cause=cause,expected=category,actual=failure));continue
        record=dict(example_id='semantic-'+hashlib.sha256(broken.encode()).hexdigest()[:16],operation_family=s['family'],seed=s['seed'],sdk_version=nki.__version__,hardware_target='trn2',input_shapes=s['shapes'],input_dtypes=s['dtypes'],task_description=s['description'],correct_kernel=s['source'],broken_kernel=broken,observed_error=failure['error'],failure_category=category,root_cause=cause,repair_strategy=plan_repair(broken,failure['error'])['guidance'],corrected_kernel=s['source'],verification_status='SIMULATOR_VERIFIED',verification_results=dict(clean=clean,broken=failure,repaired=repair),source_provenance=['Project-authored independent grouped mean plus bias, independently checked by NumPy; no official benchmark implementation used'],license='Project-authored; licensing unspecified',structural_hash=code_fingerprint(s['source']),split='train')
        records.append(record);accepted.append(record['example_id'])
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    for split in ('train','heldout'):
        with (out/(split+'.jsonl')).open('x') as f:
            for row in records:
                if row['split']==split:f.write(json.dumps(row)+'\n')
    summary=dict(verified_pairs=len(records),independent_clean_kernels=len({r['structural_hash'] for r in records}),added_verified_pairs=len(accepted),rejected=rejected,train_records=sum(r['split']=='train' for r in records),heldout_records=sum(r['split']=='heldout' for r in records),device_verified=0,active_training_uses_previous_frozen_corpus=True)
    (out/'summary.json').write_text(json.dumps(summary,indent=2));return summary

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--base',default='synthetic_nki/data_v4');p.add_argument('--output',required=True);a=p.parse_args();print(json.dumps(build(a.base,a.output),indent=2))
