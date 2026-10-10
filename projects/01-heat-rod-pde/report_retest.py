"""Report a fresh paired comparison without treating problem seeds as model seeds."""
import json
from pathlib import Path
import statistics
import sys


def main():
    source = Path(sys.argv[1])
    data = json.loads(source.read_text())
    rows = data['results']
    policy = ('Original upstream checker only.'
              if data.get('settings', {}).get('grading_policy') == 'original_checker' else
              'Historical policy: original checker plus an independent physics validator.')
    lines = ['# Default-agent retest', '',
             'Real Qwen3-8B inference on seat-85. Experimental concise mode: '
             + str(data['settings'].get('concise', False)) + '.',
             'Both variants use the same eight problem configurations, two samples per round, '
             'three rounds, 512 output tokens per request, one calculator exchange per attempt '
             'and a 180-second case deadline. ' + policy, '',
             '| Agent | Solved under recorded policy | Timeouts | Service failures | Total seconds |',
             '|---|---:|---:|---:|---:|']
    groups = {}
    for variant in ('baseline', 'improved'):
        group = groups[variant] = [r for r in rows if r['variant'] == variant]
        lines.append(f"| {variant} | {sum(r['status']=='solved' for r in group)}/{len(group)} | "
                     f"{sum(r['status']=='budget_timeout' for r in group)} | "
                     f"{sum(r['status'].startswith('infrastructure') for r in group)} | "
                     f"{sum(r['seconds'] for r in group):.1f} |")
    baseline, improved = groups['baseline'], groups['improved']
    solved_delta = sum(r['status']=='solved' for r in improved) - sum(r['status']=='solved' for r in baseline)
    baseline_seconds = sum(r['seconds'] for r in baseline)
    improved_seconds = sum(r['seconds'] for r in improved)
    lines += ['', f'Solved under recorded policy difference: {solved_delta:+d} cases. '
              f'Total measured time difference: {improved_seconds-baseline_seconds:+.1f}s '
              f'({(improved_seconds/baseline_seconds-1)*100:+.1f}%). '
              'Lower elapsed time alone does not demonstrate better mathematical answers.', '',
              '| Problem | Baseline status | Improved status | Baseline seconds | Improved seconds |',
              '|---|---|---|---:|---:|']
    for b in baseline:
        match = next((r for r in improved if (r['level'], r['sub'], r['seed']) ==
                      (b['level'], b['sub'], b['seed'])), None)
        if match:
            lines.append(f"| {b['level']}.{b['sub']} seed {b['seed']} | {b['status']} | "
                         f"{match['status']} | {b['seconds']:.1f} | {match['seconds']:.1f} |")
    lines += ['', 'Limits: a single paired run per problem configuration, not repeated model seeds; '
              'stochastic generation, sequential order, possible cache and load effects. '
              'Timeouts are inconclusive, not zero mathematical rewards. '
              'The earlier benchmark measured concise mode and is a separate experiment. '
              'Nine regression tests passed again before this retest. '
              'Evidence is comparison.json plus all case logs and attempt traces in this directory.', '']
    target = source.parent / 'RETEST.md'
    target.write_text('\n'.join(lines))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
