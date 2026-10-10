"""Sequential real-model comparison; never uses the offline generator."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import sys
import time
import agent
from compare_agents import summarize_attempts


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--concise', action='store_true', help='Measure experimental concise mode rather than the default agent')
    parser.add_argument('--timeout', type=int, default=180)
    parser.add_argument('--output-root', type=Path, default=Path('benchmark-results'))
    args = parser.parse_args(argv)
    root = args.output_root / datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    root.mkdir(parents=True)
    results = []
    cases = ([(0, sub, 0) for sub in (1, 2, 3)] + [(1, sub, 0) for sub in (1, 2, 3)]
             + [(1, 3, seed) for seed in (1, 2)])
    for level, sub, seed in cases:
        for variant in ('baseline', 'improved'):
            case = root / f'{variant}-level{level}.{sub}-seed{seed}'
            case.mkdir()
            common = ['--level', str(level), '--sub', str(sub), '--seed', str(seed),
                      '--samples', '2', '--rounds', '3', '--max-tokens', '512', '--tool-steps', '1']
            command = ([sys.executable, 'agent.py', '--log', str(case / 'attempts.jsonl')] if variant == 'baseline'
                       else [sys.executable, 'improved_agent.py', '--output', str(case), '--workers', '2']
                       + (['--concise'] if args.concise else [])) + common
            print(f'START {case.name}', flush=True)
            started = time.perf_counter()
            entry = {'variant': variant, 'level': level, 'sub': sub, 'seed': seed,
                     'samples': 2, 'max_rounds': 3, 'tool_steps': 1}
            with (case / 'console.log').open('w') as console:
                try:
                    process = subprocess.run(command, stdout=console, stderr=subprocess.STDOUT, timeout=args.timeout)
                    entry['status'] = 'unsolved' if process.returncode == 0 else 'infrastructure_error'
                    entry['exit_code'] = process.returncode
                except subprocess.TimeoutExpired:
                    entry['status'] = 'budget_timeout'
            entry['seconds'] = time.perf_counter() - started
            entry.update(summarize_attempts(list(case.rglob('attempts.jsonl')),
                                           agent.LEVELS[level].make(sub, seed)))
            if variant == 'improved':
                summaries = list(case.glob('*/summary.json'))
                if summaries:
                    try:
                        summary = json.loads(summaries[0].read_text())['results'][0]
                        entry['agent_status'] = summary['status']
                        # A saved summary must not hide a later timeout/process failure.
                        entry.update({key: value for key, value in summary.items() if key != 'status'})
                        if (entry['status'] == 'unsolved' or
                                (entry['status'] == 'infrastructure_error' and
                                 summary['status'] in ('infrastructure_error', 'evaluation_error'))):
                            entry['status'] = summary['status']
                    except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
                        entry['artifacts_complete'] = False
                        entry['log_errors'].append(dict(path=str(summaries[0]), line=None,
                                                       error=f'{type(exc).__name__}: {exc}'))
                elif entry['status'] == 'unsolved':
                    entry['artifacts_complete'] = False
                    entry['log_errors'].append(dict(path=None, line=None, error='Missing summary.json'))
            elif entry['status'] == 'unsolved':
                entry['reward'] = entry['best_original_reward']
                if entry['executed_original_solved']:
                    entry['status'] = 'solved'
            if entry['status'] in ('solved', 'unsolved') and not entry['artifacts_complete']:
                entry['status'] = 'artifact_error'
            results.append(entry)
            (root / 'comparison.json').write_text(json.dumps({'settings': {'model': 'Qwen/Qwen3-8B',
                'seat': 85, 'samples': 2, 'rounds': 3, 'max_tokens': 512, 'tool_steps': 1,
                'timeout_per_case_seconds': args.timeout, 'concise': args.concise,
                'offline': False, 'grading_policy': 'original_checker'},
                'expected_runs': len(cases)*2,
                'batch_complete': len(results)==len(cases)*2 and all(
                    r['status'] in ('solved', 'unsolved', 'budget_timeout') and r['artifacts_complete']
                    for r in results), 'results': results}, indent=2))
            print(f"DONE {case.name}: {entry['status']} ({entry['seconds']:.1f}s)", flush=True)
            if entry['status'] in ('infrastructure_error', 'evaluation_error', 'artifact_error'):
                print(f"STOPPED {entry['status']}. Partial results: {root}", flush=True)
                return 1
    complete = all(r['artifacts_complete'] for r in results)
    print(f"{'COMPLETE' if complete else 'INCOMPLETE'} {root}", flush=True)
    return 0 if complete else 1


if __name__ == '__main__':
    sys.exit(main())
