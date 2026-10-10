"""Snapshot and compare real agents sequentially under matched per-case budgets."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time

import agent
import validation

PROJECT = Path(__file__).resolve().parent
DEFAULT_CASES = [[0, sub, 0] for sub in (1, 2, 3)] + [[1, sub, 0] for sub in (1, 2, 3)] + [[1, 3, s] for s in (1, 2)]


def summarize_attempts(paths, problem):
    """Both variants receive the same post-hoc validation; do not mix score layers."""
    rows = [json.loads(line) for path in paths for line in path.read_text().splitlines() if line.strip()]
    original_solved = model_solved = validated_solved = False
    model_requests = output_tokens = truncated = accepted_repairs = 0
    validations = []
    for row in rows:
        executed = row.get('executed_answer', row['answer'])
        grade = validation.grade(problem, executed)
        model_grade = row.get('model_grade') or validation.grade(problem, row['answer'])
        model_solved |= model_grade.get('original_reward') == 1.0
        original_solved |= grade.get('original_reward') == 1.0
        validated_solved |= grade['reward'] == 1.0
        if grade.get('original_reward') == 1.0:
            validations.append(grade.get('validation'))
        for turn in row.get('trace', []):
            if turn.get('type', 'model') == 'model':
                model_requests += 1
                count = (turn.get('usage') or {}).get('completion_tokens')
                output_tokens = None if count is None or output_tokens is None else output_tokens + count
                truncated += turn.get('finish_reason') == 'length'
            accepted_repairs += turn.get('type') == 'decay_repair' and turn.get('accepted', False)
    return dict(candidates=len(rows), rounds=max((r['round'] for r in rows), default=-1)+1,
                model_original_solved=model_solved, executed_original_solved=original_solved,
                validated_solved=validated_solved, validations=validations,
                model_requests=model_requests, completion_tokens=output_tokens,
                truncated_replies=truncated, accepted_repairs=accepted_repairs)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variants', nargs='+', choices=('baseline', 'improved', 'decay'), default=['improved', 'decay'])
    p.add_argument('--cases', type=Path, help='JSON array of [level, subproblem, problem seed]')
    p.add_argument('--repeats', type=int, default=1, help='repeat the same case list without changing problem seeds')
    p.add_argument('--samples', type=int, default=2)
    p.add_argument('--rounds', type=int, default=3)
    p.add_argument('--max-tokens', type=int, default=1200)
    p.add_argument('--tool-steps', type=int, default=1)
    p.add_argument('--case-timeout', type=float, default=600)
    p.add_argument('--output-root', type=Path, default=PROJECT/'comparison-runs')
    a = p.parse_args(argv)
    if min(a.repeats, a.samples, a.rounds, a.max_tokens, a.case_timeout) <= 0 or a.tool_steps < 0:
        p.error('budgets must be positive; tool-steps may be zero')
    if len(set(a.variants)) != len(a.variants):
        p.error('variants must be unique')
    cases = json.loads(a.cases.read_text()) if a.cases else DEFAULT_CASES
    if (not isinstance(cases, list) or not cases or any(
        not isinstance(c, list) or len(c) != 3 or any(type(n) is not int for n in c)
        or c[0] not in (0, 1) or c[1] not in (1, 2, 3) for c in cases)):
        p.error('cases must contain [level 0/1, subproblem 1/2/3, integer seed]')
    if not os.environ.get('HEATROD_BASE_URL'):
        p.error('HEATROD_BASE_URL is required; this comparison has no offline mode')
    a.output_root.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ-'), dir=a.output_root.resolve()))
    source = root/'source'
    source.mkdir()
    hashes = {}
    for path in PROJECT.glob('*.py'):
        shutil.copy2(path, source/path.name)
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    settings = vars(a).copy()
    settings.update(cases=cases, model=os.getenv('HEATROD_MODEL', 'Qwen/Qwen3-8B'),
                    seat=os.getenv('HOSTNAME'), offline=False,
                    retries=0, workers=a.samples, source_sha256=hashes)
    settings = {key: str(value) if isinstance(value, Path) else value for key,value in settings.items()}
    results = []
    expected = len(cases)*a.repeats*len(a.variants)
    def save():
        (root/'comparison.json').write_text(json.dumps(dict(settings=settings, expected_runs=expected,
            batch_complete=len(results)==expected and all(r['status'] not in ('interrupted','process_error') for r in results),
            results=results), indent=2)+'\n')
    save()
    print(f'Results: {root}', flush=True)
    for repeat in range(a.repeats):
        for index, (level, sub, seed) in enumerate(cases):
            order = a.variants if (repeat+index)%2 == 0 else list(reversed(a.variants))
            for variant in order:
                case = root/f'repeat-{repeat+1:02d}-{variant}-level{level}.{sub}-seed{seed}'
                case.mkdir()
                common = ['--level',str(level),'--sub',str(sub),'--seed',str(seed),
                          '--samples',str(a.samples),'--rounds',str(a.rounds),
                          '--max-tokens',str(a.max_tokens),'--tool-steps',str(a.tool_steps)]
                if variant == 'baseline':
                    command = [sys.executable,'-u',str(source/'agent.py'),'--log',str(case/'attempts.jsonl')]+common
                else:
                    command = [sys.executable,'-u',str(source/'improved_agent.py'),
                               '--output',str(case),'--workers',str(a.samples),'--retries','0',
                               '--timeout',str(a.case_timeout)]+common+(['--decay-repair'] if variant=='decay' else [])
                (case/'command.json').write_text(json.dumps(command, indent=2)+'\n')
                entry = dict(variant=variant,level=level,sub=sub,seed=seed,repeat=repeat+1,artifact=str(case.relative_to(root)))
                print(f'START {case.name}',flush=True)
                started=time.perf_counter()
                interrupted=False
                with (case/'console.log').open('w') as out:
                    child=subprocess.Popen(command,cwd=source,stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT)
                    try:
                        code=child.wait(timeout=a.case_timeout)
                        entry.update(status='complete' if code==0 else 'process_error',exit_code=code)
                    except subprocess.TimeoutExpired:
                        child.kill();child.wait()
                        entry['status']='budget_timeout'
                    except KeyboardInterrupt:
                        child.terminate()
                        try: child.wait(timeout=5)
                        except subprocess.TimeoutExpired: child.kill();child.wait()
                        entry['status']='interrupted'
                        interrupted=True
                entry['process_wall_seconds']=time.perf_counter()-started
                paths=list(case.rglob('attempts.jsonl'))
                if paths:
                    entry.update(summarize_attempts(paths,agent.LEVELS[level].make(sub,seed)))
                if variant!='baseline':
                    summaries=list(case.glob('*/summary.json'))
                    if summaries:
                        result=json.loads(summaries[0].read_text())['results'][0]
                        entry['agent_status']=result['status']
                        if entry['status']=='complete' and result['status']=='infrastructure_error':
                            entry['status']='infrastructure_error'
                entry['wall_seconds']=time.perf_counter()-started
                results.append(entry);save()
                print(f"DONE {case.name}: {entry['status']}, validated={entry.get('validated_solved',False)}, {entry['wall_seconds']:.1f}s",flush=True)
                if interrupted: return 130
    print(f'COMPLETE {root}',flush=True)
    return 0


if __name__=='__main__':
    def stop(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop)
    sys.exit(main())
