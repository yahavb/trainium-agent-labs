"""Extend the preserved 12-task corpus to 24 independently executed bug pairs."""
import hashlib
import json
from pathlib import Path
from synthetic_nki.generate import templates
from synthetic_nki.verify import validate
from shape_repair import plan_repair


def build_pairs(base,output):
    base=Path(base);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    records=[json.loads(line) for split in ('train','heldout') for line in (base/(split+'.jsonl')).read_text().splitlines()]
    specs={s['source']:s for s in templates()};accepted=[];rejected=[];seen=set()
    for original in records:
        record=dict(original);record['example_id']='pair-'+hashlib.sha256(record['broken_kernel'].encode()).hexdigest()[:16]
        accepted.append(record);seen.add(record['broken_kernel'])
        spec=specs[record['correct_kernel']]
        # A single call-site mutation, with actual unsupported API feedback.
        broken=record['correct_kernel'].replace('nisa.dma_copy','nisa.nonexistent_dma',1)
        failure=validate(broken,spec);restored=validate(record['correct_kernel'],spec)
        if failure['passed'] or failure['failure_category']!='INVALID_API_FUNCTION' or not restored['passed'] or broken in seen:
            rejected.append(dict(kind=spec['kind'],actual=failure));continue
        seen.add(broken)
        new=dict(record,example_id='pair-'+hashlib.sha256(broken.encode()).hexdigest()[:16],broken_kernel=broken,observed_error=failure['error'],failure_category=failure['failure_category'],root_cause='unsupported_dma_api',repair_strategy=plan_repair(broken,failure['error'])['guidance'],verification_results=dict(clean=record['verification_results']['clean'],broken=failure,repaired=restored))
        accepted.append(new)
    for split in ('train','heldout'):
        with (output/(split+'.jsonl')).open('x') as file:
            for record in accepted:
                if record['split']==split:file.write(json.dumps(record)+'\n')
    import collections
    summary=dict(independent_clean_kernels=len(records),accepted_mutations=len(accepted),verified_repair_pairs=len(accepted),rejected_mutations=len(rejected),rejected=rejected,train_records=sum(r['split']=='train' for r in accepted),heldout_records=sum(r['split']=='heldout' for r in accepted),failure_categories=dict(collections.Counter(r['failure_category'] for r in accepted)),device_verified=0,verification_status='SIMULATOR_VERIFIED',source_dataset=str(base),heldout_policy='Entire elementwise_arithmetic family; two bug variants of one independent task, neither indexed.')
    with (output/'summary.json').open('x') as file:json.dump(summary,file,indent=2)
    return summary

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--base',default='synthetic_nki/data');parser.add_argument('--output',default='synthetic_nki/data_v2');args=parser.parse_args()
    print(json.dumps(build_pairs(args.base,args.output),indent=2))
