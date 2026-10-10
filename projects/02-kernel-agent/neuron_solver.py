#!/usr/bin/env python3
"""NeuronSolver: generate, grade, diagnose and repair without reference prompts."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import uuid
import agent
import nkibench
from candidate_store import CandidateStore, candidate_hash
from nki_knowledge import environment, retrieve
from repair_engine import diagnose, repair_prompt

SEMANTICS = {
    1: ('in_tensor, pool_size', 'Input [C,H,W]; output [C,H//p,W//p]. Each output is the average of its nonoverlapping p by p spatial window; discard incomplete windows.'),
    2: ('in_tensor, shape2D', 'Input/output [P,F1*F2], shape2D=(F1,F2). Within each partition, output[p,j*F1+i]=input[p,i*F2+j]. Partition axis stays unchanged.'),
    8: ('q, k, v', 'All inputs [seq,dim]; output [seq,dim] = softmax(q times k-transpose / sqrt(dim)) times v. Softmax is row-wise and numerically stable.'),
}

def specification(level):
    spec = nkibench.LEVELS[level]
    signature, semantics = SEMANTICS.get(level, ('lhsT, rhs', 'lhsT [K,M], rhs [K,N]; output [M,N]. output[m,n]=sum over k of lhsT[k,m]*rhs[k,n].'))
    return (f'Write an NKI kernel @nki.jit def {spec["entry"]}({signature}).\n{semantics}\n'
            f'Official shapes: {json.dumps(spec["shapes"])}\nTraffic ceiling (multiple of minimum bytes): {spec["max_waste"]}.\n'
            'Import nki, nki.language as nl, nki.isa as nisa. Allocate a fresh shared_hbm output; return it.\n'
            + retrieve(['api', 'dimensions', 'matmul_layout' if 3 <= level <= 7 else 'reshape'])
            + '\nReply with one complete Python code block.')

def evaluate_candidate(source, level, timeout=120, probes=False):
    with tempfile.TemporaryDirectory() as tmp:
        request, output = Path(tmp) / 'request.json', Path(tmp) / 'result.json'
        request.write_text(json.dumps(dict(source=source, level=level, probes=probes)))
        try:
            run = subprocess.run([sys.executable, str(Path(__file__).with_name('solver_checker.py')), str(request), str(output)], capture_output=True, text=True, timeout=timeout)
            if run.returncode == 0 and output.exists():
                return json.loads(output.read_text())
            feedback = 'Checker process failed: ' + run.stderr[-2000:]
        except subprocess.TimeoutExpired:
            feedback = f'Checker timeout after {timeout}s'
        return dict(reward=0., solved=False, cases=[dict(label='checker', passed=False, feedback=feedback)], passed_cases=[], simulator_calls=0, simulator_calls_incomplete=True)

def solve(a, level, directory, generate=None, checker=None):
    generate = generate or (lambda prompt, count: agent.ask_parallel(a, prompt, count))
    checker = checker or (lambda source, lv: evaluate_candidate(source, lv, a.timeout, a.probes))
    store = CandidateStore(directory)
    start = time.perf_counter()
    prompt = specification(level)
    history, categories = [], Counter()
    best_source, best_result = '', None
    model_calls = simulator_calls = duplicates = repairs = repair_improvements = 0
    rewards, response_hashes = [], []
    simulator_calls_incomplete = False
    for rnd in range(a.rounds):
        parent_reward = (best_result or {}).get('reward', 0)
        try:
            replies = generate(prompt, a.samples)
        except (Exception, SystemExit) as exc:
            store.write('interrupted.json', dict(level=level, round=rnd+1, reason=str(exc), model_calls=model_calls, environment=environment()))
            raise
        if not replies:
            replies = ['']
        model_calls += len(replies)
        for reply in replies:
            source = agent.extract_code(reply)
            digest = candidate_hash(source)
            response_hashes.append(digest)
            duplicate = digest in store.seen
            if duplicate:
                duplicates += 1
                result = store.seen[digest]
            else:
                result = checker(source, level)
                simulator_calls += result.get('simulator_calls', 0)
                simulator_calls_incomplete |= result.get('simulator_calls_incomplete', False)
            ds = diagnose(source, result)
            categories.update(d['category'] for d in ds)
            passed = set(result.get('passed_cases', []))
            regression = sorted(set((best_result or {}).get('passed_cases', [])) - passed)
            if rnd:
                repairs += 1
                repair_improvements += int(not duplicate and not regression and result['reward'] > parent_reward)
            store.record(source, result, round=rnd + 1, duplicate=duplicate, diagnostics=ds, regression=regression, prompt=prompt)
            rewards.append(result['reward'])
            history.append(dict(hash=digest, categories=[d['category'] for d in ds], reward=result['reward'], regressions=regression))
            if best_result is None or (not regression and result['reward'] > best_result['reward']):
                best_source, best_result = source, result
            if result['solved']:
                break
        print(f'level {level} round {rnd+1}: best={best_result["reward"]:.3f} duplicates={duplicates}', flush=True)
        if best_result['solved']:
            break
        # Best working logic is retained; fallback changes the strategy and ledger breaks cycles.
        prompt = repair_prompt(specification(level), best_source, best_result, history, fallback=(duplicates > 0 or max(categories.values(), default=0) >= 3))
        # If source/diagnostics won't fit, restart from concise spec and failure ledger.
        if len(prompt) // 4 + a.max_tokens + 64 > a.context:
            prompt = specification(level) + '\nUse a simpler approach. Previous failures: ' + json.dumps(history[-3:])
    summary = dict(level=level, solved=best_result['solved'], best_reward=best_result['reward'], mean_reward=sum(rewards)/len(rewards), rounds=rnd+1, rounds_until_success=rnd+1 if best_result['solved'] else None, model_calls=model_calls, simulator_calls=simulator_calls, simulator_calls_incomplete=simulator_calls_incomplete, duplicates=duplicates, error_categories=dict(categories), repair_attempts=repairs, repair_improvements=repair_improvements, repair_success_rate=repair_improvements/repairs if repairs else None, wall_seconds=time.perf_counter()-start, response_hashes=response_hashes, environment=environment(), mode='autonomous')
    store.write('result.json', summary)
    return summary

def parser():
    p = argparse.ArgumentParser()
    group = p.add_mutually_exclusive_group()
    group.add_argument('--level', type=int, choices=range(1,9))
    group.add_argument('--levels', type=int, nargs='+', choices=range(1,9))
    p.add_argument('--rounds', type=int, default=8)
    p.add_argument('--samples', type=int, default=4)
    p.add_argument('--context', type=int, default=8192)
    p.add_argument('--max-tokens', type=int, default=2500)
    p.add_argument('--base', default=os.environ.get('KERNEL_AGENT_BASE_URL') or os.environ.get('GPTOSS_BASE_URL'))
    p.add_argument('--model', default=agent.MODEL)
    p.add_argument('--think', action='store_true')
    p.add_argument('--timeout', type=float, default=120)
    p.add_argument('--probes', action='store_true')
    p.add_argument('--output', default='solver_runs')
    p.add_argument('--doctor', action='store_true')
    p.add_argument('--oracle-debug', type=Path, help='Grade a developer-supplied kernel; excluded from autonomous results')
    return p

def main():
    p = parser()
    a = p.parse_args()
    if a.doctor:
        print(json.dumps(environment(), indent=2))
        return
    if min(a.rounds, a.samples, a.max_tokens, a.context, a.timeout) <= 0:
        p.error('budgets must be positive')
    if not environment()['nki_available']:
        p.error('NKI SDK missing. Run local unit tests here; run the solver in the Linux Neuron environment.')
    if a.oracle_debug:
        print(json.dumps(dict(mode='oracle-debug; excluded from benchmark', result=evaluate_candidate(a.oracle_debug.read_text(), a.level or 1, a.timeout, a.probes)), indent=2))
        return
    from urllib.parse import urlparse
    u = urlparse(a.base or '')
    if u.scheme not in ('http', 'https') or not u.netloc:
        p.error('Set KERNEL_AGENT_BASE_URL or --base to an OpenAI-compatible /v1 endpoint')
    root = Path(a.output) / (time.strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:8])
    root.mkdir(parents=True)
    # Never persist endpoint credentials.
    config = {k:v for k,v in vars(a).items() if k not in ('base', 'oracle_debug')}
    (root/'config.json').write_text(json.dumps(config, indent=2))
    results = [solve(a, lv, root/f'level-{lv}') for lv in (a.levels or [a.level or 1])]
    (root/'results.json').write_text(json.dumps(results, indent=2))
    print(f'Results: {root}')

if __name__ == '__main__':
    main()
