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
from attempt_records import read_attempts, request_metrics
from checker_runtime import grade_candidate

PROJECT = Path(__file__).resolve().parent
DEFAULT_CASES = [[0, sub, 0] for sub in (1, 2, 3)] + [[1, sub, 0] for sub in (1, 2, 3)] + [[1, 3, s] for s in (1, 2)]


def summarize_attempts(paths, problem):
    """Use the same original checker for both variants and both answer layers."""
    rows, log_errors = read_attempts(paths)
    original_solved = model_solved = False
    best_reward = None
    accepted_repairs = 0
    evaluation_errors = []
    for index, row in enumerate(rows):
        accepted_repairs += sum(turn.get('type') == 'decay_repair' and turn.get('accepted', False)
                                for turn in row.get('trace', []))
        if row.get('error'):
            continue  # A failed model request is not a scored mathematical answer.
        executed = row.get('executed_answer', row['answer'])
        grade = grade_candidate(problem, executed)
        model_grade = grade if executed == row['answer'] else grade_candidate(problem, row['answer'])
        layers = [('model_and_executed', grade)] if executed == row['answer'] else [
            ('model', model_grade), ('executed', grade)]
        for layer, result in layers:
            if result['evaluation_error']:
                evaluation_errors.append(dict(candidate=index, layer=layer,
                                              **result['evaluation_error']))
        model_solved |= model_grade['reward'] == 1.0
        original_solved |= grade['reward'] == 1.0
        if grade['reward'] is not None:
            best_reward = grade['reward'] if best_reward is None else max(best_reward, grade['reward'])
    return dict(candidates=len(rows), rounds=max((r['round'] for r in rows), default=-1)+1,
                model_original_solved=model_solved, executed_original_solved=original_solved,
                best_original_reward=best_reward,
                **request_metrics(rows), accepted_repairs=accepted_repairs,
                artifacts_complete=bool(rows) and not log_errors, log_errors=log_errors,
                evaluation_failures=len(evaluation_errors), evaluation_errors=evaluation_errors)


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
                    retries=0, workers=a.samples, grading_policy='original_checker', source_sha256=hashes)
    settings = {key: str(value) if isinstance(value, Path) else value for key,value in settings.items()}
    results = []
    expected = len(cases)*a.repeats*len(a.variants)
    def batch_complete():
        return len(results) == expected and all(
            r['status'] in ('complete', 'budget_timeout') and r.get('artifacts_complete', False)
            for r in results)

    def save():
        temporary = root/'comparison.json.tmp'
        temporary.write_text(json.dumps(dict(settings=settings, expected_runs=expected,
            batch_complete=batch_complete(),
            results=results), indent=2)+'\n')
        temporary.replace(root/'comparison.json')
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
                entry.update(summarize_attempts(paths,agent.LEVELS[level].make(sub,seed)))
                if variant!='baseline':
                    summaries=list(case.glob('*/summary.json'))
                    if summaries:
                        try:
                            result=json.loads(summaries[0].read_text())['results'][0]
                            entry['agent_status']=result['status']
                            if (entry['status'] in ('complete', 'process_error') and
                                    result['status'] in ('infrastructure_error', 'evaluation_error')):
                                entry['status']=result['status']
                        except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
                            entry['artifacts_complete']=False
                            entry['log_errors'].append(dict(path=str(summaries[0]), line=None,
                                                            error=f'{type(exc).__name__}: {exc}'))
                    elif entry['status']=='complete':
                        entry['artifacts_complete']=False
                        entry['log_errors'].append(dict(path=None, line=None, error='Missing summary.json'))
                if entry['status']=='complete' and not entry['artifacts_complete']:
                    entry['status']='artifact_error'
                entry['wall_seconds']=time.perf_counter()-started
                results.append(entry);save()
                print(f"DONE {case.name}: {entry['status']}, original_solved={entry.get('executed_original_solved')}, {entry['wall_seconds']:.1f}s",flush=True)
                if interrupted: return 130
                if entry['status'] in ('infrastructure_error', 'process_error', 'artifact_error', 'evaluation_error'):
                    print(f"STOPPED {entry['status']}. Partial results: {root}", flush=True)
                    return 1
    print(f"{'COMPLETE' if batch_complete() else 'INCOMPLETE'} {root}",flush=True)
    return 0 if batch_complete() else 1


if __name__=='__main__':
    def stop(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop)
    sys.exit(main())
