"""Read completed raw logs; distinguish verified progress from error transitions."""
import argparse
import collections
import json
from pathlib import Path
from failure_selection import classify_failure

def analyze(root):
    findings=[]
    for path in sorted(Path(root).glob('level*/*/attempts.jsonl')):
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        if not rows:continue
        selected=[r for r in rows if r.get('selected')]
        transitions=[]
        for before,after in zip(selected,selected[1:]):
            previous=sum(c.get('verified') is True for c in before['shape_results'])
            now=sum(c.get('verified') is True for c in after['shape_results'])
            transitions.append(dict(round=after['round'],source_changed=before['ast_source_hash']!=after['ast_source_hash'],prior_failure=before['failure_category'],new_failure=after['failure_category'],prior_signature_no_longer_first=before['normalized_signature']!=after['normalized_signature'],verified_shapes_delta=now-previous))
        counts=collections.Counter(r['failure_category'] for r in rows if r['reward']<1-1e-9)
        summary=path.parent/'summary.json'
        recorded=json.loads(summary.read_text()) if summary.exists() else {}
        result=dict(artifact=str(path),level=rows[0]['level'],arm=path.parent.name,best_reward=max(r['reward'] for r in rows),solved=any(r['reward']>=1-1e-9 for r in rows),max_verified_shapes=max(sum(c.get('verified') is True for c in r['shape_results']) for r in rows),max_numerical_shapes=max(sum(c.get('numerically_correct') is True for c in r['shape_results']) for r in rows),candidate_count=len(rows),round_count=len({r['round'] for r in rows}),categories=dict(counts),truncated=sum(r.get('truncated') is True for r in rows),selected_transitions=transitions,verified_progress_repairs=sum(t['verified_shapes_delta']>0 for t in transitions),simulation_seconds=sum(c['simulation_seconds'] for r in rows for c in r['shape_results'] if c.get('evaluated')) if all(type(c.get('simulation_seconds')) in (int,float) for r in rows for c in r['shape_results'] if c.get('evaluated')) else None,wall_seconds=recorded.get('wall_seconds'),checker_seconds=sum(r['checker_seconds'] for r in rows),mean_ast_unique=sum(next(r for r in rows if r['round']==rnd)['diversity']['unique_ast_sources'] for rnd in {r['round'] for r in rows})/len({r['round'] for r in rows}),synthetic_retrieval_candidates=sum(bool((r.get('synthetic_context') or {}).get('example_ids')) for r in rows))
        for key in ('prompt_tokens','completion_tokens','total_tokens'):result[key]=sum(r[key] for r in rows) if all(type(r.get(key)) is int for r in rows) else None
        findings.append(result)
    return findings

def report(roots,title):
    lines=['# '+title,'','These are sequential cold-start single-repeat experiments. Reward 1.00 requires the unchanged official checker. A disappearing first error is not proof that its constraint was fixed; successful repairs below require increased verified-shape coverage. CPU numerics do not establish physical-device correctness or speed.','','| Run | Level | Arm | Best reward | Fully solved | Shapes verified | Candidates | AST unique/round | Prompt tokens | Completion tokens | Total tokens | Wall seconds |','|---|---|---|---|---|---|---|---|---|---|---|---|']
    data=[]
    for root in roots:
        found=analyze(root);data.extend(found)
        for r in found:lines.append(f"| {Path(root).name} | {r['level']} | {r['arm']} | {r['best_reward']:.3f} | {int(r['solved'])}/1 | {r['max_verified_shapes']} | {r['candidate_count']} | {r['mean_ast_unique']:.2f} | {r['prompt_tokens']} | {r['completion_tokens']} | {r['total_tokens']} | {r['wall_seconds']} |")
    lines+=['','## Failure and repair evidence','']
    for r in data:lines += [f"- {r['level']} / {r['arm']}: {r['categories']}; truncated={r['truncated']}; verified-progress repairs={r['verified_progress_repairs']}; synthetic retrieval candidates={r['synthetic_retrieval_candidates']}; checker time={r['checker_seconds']:.3f}s. Raw log: `{r['artifact']}`.", '  Selected source transitions: '+json.dumps(r['selected_transitions'])]
    return '\n'.join(lines)+'\n',data

def main():
    parser=argparse.ArgumentParser();parser.add_argument('roots',nargs='+');parser.add_argument('--report',required=True);args=parser.parse_args()
    text,data=report(args.roots,'Synthetic NKI experiments')
    Path(args.report).write_text(text)
    output=Path(args.roots[-1])/'detailed-analysis.json'
    with output.open('x') as file:json.dump(data,file,indent=2)
if __name__=='__main__':main()
