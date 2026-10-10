#!/usr/bin/env python3
"""Paired baseline/solver trials with shared endpoint and sampling settings."""
from collections import Counter
from contextlib import contextmanager
import json
from pathlib import Path
import statistics
import time
import uuid
import agent
import nkibench
from candidate_store import candidate_hash
from neuron_solver import parser, solve
from nki_knowledge import environment
from repair_engine import diagnose

@contextmanager
def baseline_counters():
    original_ask, original_simulate, original_grade = agent.ask, nkibench.simulate_and_count, agent.grade
    counts = dict(model_calls=0, simulator_calls=0, error_categories=Counter(), response_hashes=[])
    def ask(*args, **kwargs):
        # CPython threads serialize these short list/counter operations under the GIL.
        counts['model_calls'] += 1
        reply = original_ask(*args, **kwargs)
        counts['response_hashes'].append(candidate_hash(agent.extract_code(reply)))
        return reply
    def simulate(*args, **kwargs):
        counts['simulator_calls'] += 1
        return original_simulate(*args, **kwargs)
    def grade(source, level):
        reward, parts, feedback = original_grade(source, level)
        if not parts['correct']:
            ds = diagnose(source, {'cases': [{'label': 'baseline', 'passed': False, 'feedback': feedback}]})
            counts['error_categories'].update(d['category'] for d in ds)
        return reward, parts, feedback
    agent.ask, nkibench.simulate_and_count, agent.grade = ask, simulate, grade
    try:
        yield counts
    finally:
        agent.ask, nkibench.simulate_and_count, agent.grade = original_ask, original_simulate, original_grade


def baseline(a, level, directory):
    directory.mkdir(parents=True)
    start = time.perf_counter()
    with baseline_counters() as counts, (directory/'attempts.jsonl').open('w') as log:
        reward, rounds = agent.solve(a, level, log)
    records = [json.loads(line) for line in (directory/'attempts.jsonl').read_text().splitlines()]
    result = dict(level=level, mode='original-baseline-reference-in-prompts', solved=reward >= 1.-1e-9,
                  best_reward=reward, mean_reward=statistics.mean(r['reward'] for r in records),
                  rounds=rounds, rounds_until_success=rounds if reward >= 1.-1e-9 else None,
                  wall_seconds=time.perf_counter()-start, repair_attempts=None, repair_success_rate=None,
                  **counts)
    (directory/'result.json').write_text(json.dumps(result, indent=2))
    return result


def aggregate(trials):
    report = {}
    for method in ('baseline', 'solver'):
        for lv in sorted({t['level'] for t in trials}):
            selected = [t for t in trials if t['method'] == method and t['level'] == lv]
            if not selected:
                continue
            successful_rounds = [t['rounds_until_success'] for t in selected if t['solved']]
            sequences = [tuple(t['response_hashes']) for t in selected]
            report[f'{method}/level-{lv}'] = dict(
                runs=len(selected), solve_rate=statistics.mean(t['solved'] for t in selected),
                mean_reward=statistics.mean(t['mean_reward'] for t in selected),
                best_reward=max(t['best_reward'] for t in selected),
                mean_rounds_until_success=statistics.mean(successful_rounds) if successful_rounds else None,
                model_calls=sum(t['model_calls'] for t in selected),
                simulator_calls=sum(t['simulator_calls'] for t in selected),
                simulator_calls_incomplete=any(t.get('simulator_calls_incomplete', False) for t in selected),
                error_categories=dict(sum((Counter(t['error_categories']) for t in selected), Counter())),
                wall_seconds=sum(t['wall_seconds'] for t in selected),
                identical_trial_outputs=len(selected)>1 and len(set(sequences))==1,
                independence_note='Repeated trials may not be independent; identical output sequences are flagged. Server-side greedy decoding cannot be inferred from requested temperature.')
    return report


def main():
    p = parser()
    p.description = __doc__
    p.add_argument('--repeat', type=int, default=5)
    p.add_argument('--terse', type=int, default=0, choices=(0,1,2))
    p.add_argument('--give-up-after', type=int, default=4)
    a = p.parse_args()
    if a.doctor:
        print(json.dumps(environment(), indent=2))
        return
    if a.oracle_debug:
        p.error('Oracle mode is excluded from evaluation')
    if not environment()['nki_available']:
        p.error('Evaluation needs the Linux NKI SDK and a running endpoint')
    from urllib.parse import urlparse
    u = urlparse(a.base or '')
    if u.scheme not in ('http','https') or not u.netloc:
        p.error('Set --base or KERNEL_AGENT_BASE_URL')
    if min(a.repeat,a.rounds,a.samples,a.context,a.max_tokens,a.timeout)<=0:
        p.error('budgets must be positive')
    a.offline = False
    root = Path(a.output)/('evaluation-'+uuid.uuid4().hex[:12])
    root.mkdir(parents=True)
    settings = {k:v for k,v in vars(a).items() if k not in ('base','oracle_debug')}
    settings.update(environment=environment(), sampling={'temperature': .6,'top_p': .95},
                    fairness_limitation='Original baseline includes reference source in generation prompts; solver does not. Results are a historical implementation comparison, not a contamination-free performance claim.',
                    timeout_limitation='Solver candidate checks use subprocess timeout; original baseline is unchanged and has no timeout.',
                    context_accounting='Shared legacy character/4 estimate, not exact model tokenization.')
    (root/'config.json').write_text(json.dumps(settings, indent=2))
    trials = []
    for repeat in range(a.repeat):
        for lv in (a.levels or ([a.level] if a.level else [1,2,3,4])):
            # Alternate order to reduce temporal endpoint bias.
            for method in (('baseline','solver') if repeat%2==0 else ('solver','baseline')):
                directory=root/f'{method}-{repeat+1}-level-{lv}'
                result = baseline(a, lv, directory) if method=='baseline' else solve(a, lv, directory)
                result.update(method=method, trial=repeat+1)
                trials.append(result)
                with (root/'trials.jsonl').open('a') as f:
                    f.write(json.dumps(result)+'\n')
                (root/'summary.json').write_text(json.dumps(aggregate(trials), indent=2))
    print(f'Evaluation: {root}/summary.json')

if __name__ == '__main__':
    main()
