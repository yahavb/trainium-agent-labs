"""Preserve generated successes and per-run partial bests; private checker replay."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import tempfile
import agent
from experiment_metrics import CASE_RESULTS,GRADE_DIRECTORY


def archive(roots,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False);index=[];strategies=collections.defaultdict(lambda:dict(attempts=0,solved=0))
    for root in map(Path,roots):
        manifest=json.loads((root/'manifest.json').read_text())
        for path in sorted(root.glob('level*/*/attempts.jsonl')):
            rows=[json.loads(line) for line in path.read_text().splitlines()]
            for row in rows:
                if row['level']==3 and row['round']==0:
                    key=row.get('prompt_strategy','unknown');strategies[key]['attempts']+=1;strategies[key]['solved']+=row['reward']>=1-1e-9
            winners=[r for r in rows if r['reward']>=1-1e-9]
            if not winners and rows:winners=[max(rows,key=lambda r:r['reward'])]
            for row in winners:
                digest=hashlib.sha256(row['code'].encode()).hexdigest();key=f"{root.name}-{path.parent.parent.name}-{path.parent.name}-r{row['round']}-c{row['candidate_index']}"
                directory=output/key;directory.mkdir();(directory/'kernel.py').write_text(row['code'])
                record=dict(level=row['level'],reward=row['reward'],source_sha256=digest,cold_start=row['round']==0,round=row['round'],candidate_index=row['candidate_index'],prompt_strategy=row.get('prompt_strategy'),repair_parent_hash=row.get('repair_parent_hash'),shape_results=row['shape_results'],verification_status='SIMULATOR_VERIFIED' if row['reward']>=1-1e-9 else 'PARTIAL_OR_FAILED',device_verified=False,experiment_path=str(path),model_configuration={k:manifest[k] for k in ('model','context','temperature','top_p','thinking','sdk_version')},prompt_tokens=row.get('prompt_tokens'),completion_tokens=row.get('completion_tokens'),total_tokens=row.get('total_tokens'))
                if row['reward']>=1-1e-9:
                    (directory/'private-replay').mkdir()
                    cases=[];token=CASE_RESULTS.set(cases);directory_token=GRADE_DIRECTORY.set(str(directory/'private-replay'))
                    try:score,parts,feedback=agent.grade(row['code'],row['level'])
                    finally:CASE_RESULTS.reset(token);GRADE_DIRECTORY.reset(directory_token)
                    record['independent_replay']=dict(reward=score,parts=parts,feedback=feedback,shape_results=cases)
                    if score<1-1e-9:raise RuntimeError('Previously solved kernel failed private replay; evidence preserved')
                with (directory/'metadata.json').open('x') as f:json.dump(record,f,indent=2)
                index.append(dict(directory=str(directory),**record))
    result=dict(kernels=index,first_round_level3_strategies=dict(strategies),fully_verified_levels=sorted({r['level'] for r in index if r['reward']>=1-1e-9}),limitations='Round-zero solves do not demonstrate repairs. Separate cold-start requests are not a balanced repeated reliability study; simulator only.')
    with (output/'index.json').open('x') as f:json.dump(result,f,indent=2)
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('roots',nargs='+');p.add_argument('--output',required=True);a=p.parse_args();r=archive(a.roots,a.output);print('Archived kernels',len(r['kernels']),'verified levels',r['fully_verified_levels'])
