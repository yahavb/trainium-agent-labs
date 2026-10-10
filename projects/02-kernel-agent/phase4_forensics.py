"""Read-only historical analysis and private CPU replay; no model generation."""
import ast
import collections
import hashlib
import json
from pathlib import Path
import tempfile
import agent
from experiment_metrics import CASE_RESULTS, GRADE_DIRECTORY
from failure_selection import classify_failure, code_fingerprint

PROJECT=Path(__file__).resolve().parent

def replay(source, level, directory):
    directory.mkdir(parents=True, exist_ok=False)
    cases=[]
    ct=CASE_RESULTS.set(cases);dt=GRADE_DIRECTORY.set(str(directory))
    try: reward,parts,feedback=agent.grade(source,level)
    finally: CASE_RESULTS.reset(ct);GRADE_DIRECTORY.reset(dt)
    return dict(reward=reward,parts=parts,feedback=feedback,shape_results=cases)

def main():
    root=Path(tempfile.mkdtemp(prefix='phase4-forensics-',dir=PROJECT/'runs'))
    original=Path('/workspace/projects/02-kernel-agent')
    rows=[json.loads(line) for line in (original/'attempts.jsonl').read_text().splitlines()]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [*original.glob('*.py'),original/'attempts.jsonl',original/'run.log']}
    (root/'protected_hashes.json').write_text(json.dumps(hashes,indent=2))
    # Baseline lacks run IDs: a transition back to Level 1/round 0 starts a new repetition.
    run=-1;previous=None
    for i,row in enumerate(rows):
        key=(row['level'],row['round'])
        if key==(1,0) and previous!=key:run+=1
        row['inferred_repeat']=run;row['record_index']=i;previous=key
    (root/'baseline_snapshot.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    pilot=[]
    for path in sorted((PROJECT/'runs/controlled-20261010T173044-tvi8dlbf').glob('level*/*/attempts.jsonl')):
        for line in path.read_text().splitlines():
            row=json.loads(line);row['artifact']=str(path);pilot.append(row)
    findings=[]
    for level in range(1,5):
        records=[r for r in rows if r['level']==level]
        unique={r['code']:r for r in records}
        source_dir=root/f'level{level}';source_dir.mkdir()
        for i,(source,row) in enumerate(unique.items()):
            (source_dir/f'historical-{i}.py').write_text(source)
            ast.parse(source) # Parse every distinct historical source, no execution here.
        best=max(records,key=lambda row:row['reward'])
        (source_dir/'best.py').write_text(best['code'])
        actual=replay(best['code'],level,source_dir/'private-grade')
        # Also replay every distinct highest-reward Level 4 candidate.
        best_replays=[]
        if level==4:
            for i,row in enumerate(r for r in unique.values() if r['reward']==best['reward']):
                (source_dir/f'tied-best-{i}.py').write_text(row['code'])
                best_replays.append(replay(row['code'],level,source_dir/f'tied-grade-{i}'))
        transitions=[]
        for repeat in sorted({r['inferred_repeat'] for r in records}):
            batches=collections.defaultdict(list)
            for row in records:
                if row['inferred_repeat']==repeat:batches[row['round']].append(row)
            selected=[max(batch,key=lambda r:r['reward']) for _,batch in sorted(batches.items())]
            for prior,current in zip(selected,selected[1:]):
                transitions.append(dict(repeat=repeat,round=current['round'],ast_changed=code_fingerprint(prior['code'])!=code_fingerprint(current['code']),prior_category=classify_failure(prior['feedback']).failure_category,new_category=classify_failure(current['feedback']).failure_category,reward_change=current['reward']-prior['reward']))
        findings.append(dict(level=level,records=len(records),unique_sources=len(unique),best_reward=best['reward'],best_record=best['record_index'],best_source=str(source_dir/'best.py'),replay=actual,tied_best_replays=best_replays,categories=dict(collections.Counter(classify_failure(r['feedback']).failure_category for r in records)),transitions=transitions))
    pilot_findings=[]
    for artifact in sorted({r['artifact'] for r in pilot}):
        rr=[r for r in pilot if r['artifact']==artifact]
        for r in rr:
            try:ast.parse(r['code'])
            except SyntaxError:pass
        pilot_findings.append(dict(artifact=artifact,records=len(rr),truncated=sum(r.get('truncated') is True for r in rr),finish_reasons=dict(collections.Counter(r.get('finish_reason') for r in rr)),shape_pass_max=max(sum(c.get('verified') is True for c in r.get('shape_results',[])) for r in rr),categories=dict(collections.Counter(r['failure_category'] for r in rr))))
    (root/'findings.json').write_text(json.dumps(dict(baseline_repeats=run+1,levels=findings,pilot=pilot_findings),indent=2))
    print(root,flush=True)
    for item in findings:print(item['level'],item['best_reward'],item['replay'],flush=True)
if __name__=='__main__':main()
