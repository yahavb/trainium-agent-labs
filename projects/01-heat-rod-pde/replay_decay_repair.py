"""Counterfactual replay of completed recorded runs; never estimates live latency."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import agent
import decay_repair
from improved_agent import grade_candidate


def replay(root):
    runs, excluded, cache = [], [], {}
    for path in sorted(root.rglob('attempts.jsonl')):
        summaries = [parent/'summary.json' for parent in path.parents
                     if parent.is_relative_to(root) and (parent/'summary.json').exists()]
        if not summaries:
            excluded.append(dict(path=str(path.relative_to(root)), reason='no batch summary'))
            continue
        summary=json.loads(summaries[0].read_text())
        if not summary.get('batch_complete'):
            excluded.append(dict(path=str(path.relative_to(root)),reason='batch incomplete'))
            continue
        rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        outcomes=[]
        for row in rows:
            key=(row['problem'],row['seed'],row['answer'])
            if key not in cache:
                level, sub=map(int,row['problem'].removeprefix('level').split('.'))
                problem=agent.LEVELS[level].make(sub,row['seed'])
                before=grade_candidate(problem,row['answer'])
                eligible=(before['parts'].get('equation') is False and all(
                    before['parts'].get(k) is True for k in ('left_bc','right_bc','start_shape')))
                started=time.perf_counter()
                proposal=decay_repair.repair(row['answer'],problem['k']) if eligible else None
                after=grade_candidate(problem,proposal['answer']) if proposal and proposal['applied'] else before
                accepted=proposal is not None and proposal['applied'] and after['reward']==1.0
                cache[key]=dict(model_original_reward=before['original_reward'],
                    before_original_solved=before['reward']==1.0, eligible=eligible,
                    accepted=accepted, after_original_solved=after['reward']==1.0 if accepted else before['reward']==1.0,
                    proposal=proposal, input_grade=before, output_grade=after if accepted else None,
                    repair_and_check_seconds=time.perf_counter()-started)
            outcomes.append(dict(round=row['round'],sample=row['sample'],answer=row['answer'],**cache[key]))
        runs.append(dict(path=str(path.relative_to(root)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            candidates=len(rows),before_solved=any(r['before_original_solved'] for r in outcomes),
            after_solved=any(r['after_original_solved'] for r in outcomes),
            eligible=sum(r['eligible'] for r in outcomes),accepted=sum(r['accepted'] for r in outcomes),
            outcomes=outcomes))
    return dict(kind='counterfactual_recorded_candidate_replay', grading_policy='original_checker',
        note='Uses existing model outputs only. No new inference, no live solve-rate or latency estimate. '
             'Later recorded rounds might not have occurred if a repaired earlier answer had succeeded.',
        completed_recorded_runs=len(runs),before_solved_runs=sum(r['before_solved'] for r in runs),
        after_solved_runs=sum(r['after_solved'] for r in runs),
        candidates=sum(r['candidates'] for r in runs),unique_candidates=len(cache),
        eligible_candidates=sum(r['eligible'] for r in runs),accepted_repairs=sum(r['accepted'] for r in runs),
        excluded=excluded,runs=runs)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    result=replay(a.input_root.resolve())
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='runs'},indent=2))


if __name__=='__main__':
    main()
