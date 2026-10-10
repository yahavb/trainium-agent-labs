"""Sequential real-model comparison; never uses the offline generator."""
import argparse
import datetime
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import agent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--concise', action='store_true', help='Measure experimental concise mode rather than the default agent')
    parser.add_argument('--timeout', type=int, default=180)
    args = parser.parse_args()
    root = Path('benchmark-results') / datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
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
            if variant == 'improved':
                summaries = list(case.glob('*/summary.json'))
                if summaries:
                    entry.update(json.loads(summaries[0].read_text())['results'][0])
            elif entry['status'] == 'unsolved':
                rows = [json.loads(line) for line in (case / 'attempts.jsonl').read_text().splitlines()]
                entry['reward'] = max((r['reward'] for r in rows), default=0)
                entry['rounds'] = max((r['round'] for r in rows), default=-1) + 1
                solved = next((r for r in rows if r['reward'] == 1.0), None)
                if solved:
                    entry['status'] = 'solved'
            results.append(entry)
            (root / 'comparison.json').write_text(json.dumps({'settings': {'model': 'Qwen/Qwen3-8B',
                'seat': 85, 'samples': 2, 'rounds': 3, 'max_tokens': 512, 'tool_steps': 1,
                'timeout_per_case_seconds': args.timeout, 'concise': args.concise,
                'offline': False, 'grading_policy': 'original_checker'}, 'results': results}, indent=2))
            print(f"DONE {case.name}: {entry['status']} ({entry['seconds']:.1f}s)", flush=True)
    print(f'COMPLETE {root}', flush=True)


if __name__ == '__main__':
    main()
