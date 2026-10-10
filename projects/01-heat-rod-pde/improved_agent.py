"""Resilient feedback agent, retaining upstream tasks and directional checker."""
import argparse
import concurrent.futures as cf
import datetime
import json
import os
from pathlib import Path
import time
import uuid
import httpx
import agent as upstream
import pdecheck
import tool_calc
import validation


class ModelError(RuntimeError):
    pass


def ask(a, messages, trace, seed):
    body = dict(model=a.model, messages=messages, temperature=.6, top_p=.95,
                max_tokens=a.max_tokens,
                chat_template_kwargs={'enable_thinking': False})
    for retry in range(a.retries + 1):
        started = time.perf_counter()
        try:
            response = httpx.post(a.base.rstrip('/') + '/chat/completions', json=body,
                                  timeout=a.timeout)
            if response.status_code != 200:
                trace.append({'type': 'http_error', 'status': response.status_code,
                              'retry': retry, 'seconds': time.perf_counter() - started})
                if response.status_code not in (429, 500, 502, 503, 504):
                    raise ModelError(f'Non-retryable HTTP {response.status_code}')
                if retry == a.retries:
                    raise ModelError(f'HTTP {response.status_code} after {retry + 1} requests')
            else:
                choice = response.json()['choices'][0]
                text = choice['message'].get('content') or ''
                trace.append({'type': 'model', 'answer': text,
                              'messages': json.loads(json.dumps(messages)),
                              'finish_reason': choice.get('finish_reason'),
                              'usage': response.json().get('usage'),
                              'seconds': time.perf_counter() - started, 'retry': retry})
                if not text.strip():
                    raise ModelError('Model returned empty content')
                return text
        except httpx.TransportError as exc:
            trace.append({'type': 'transport_error', 'error': type(exc).__name__, 'retry': retry})
            if retry == a.retries:
                raise ModelError(f'{type(exc).__name__} after {retry + 1} requests') from exc
        time.sleep(min(2 ** retry, 8))
    raise ModelError('Request failed')


def attempt(a, prompt, seed):
    trace, used = [], 0
    messages = [{'role': 'user', 'content': prompt + ('' if a.no_tools else '\n\n' + tool_calc.INSTRUCTIONS)}]
    if getattr(a, 'concise', False):
        messages.insert(0, {'role': 'system', 'content':
                 'You solve PDEs using a checker and calculator. Keep output concise. ' +
                 ('If calculations are needed, output only COMPUTE: lines and stop. ' if not a.no_tools else '') +
                 'Otherwise output only the final u(x, t) = <expression> line in Python syntax. '
                 'Do not write explanations, LaTeX, or symbolic sums.'})
    try:
        for step in range(a.tool_steps + 1):
            reply = ask(a, messages, trace, seed + step)
            requests = [] if a.no_tools else tool_calc.requests_in(reply)
            if not requests or step == a.tool_steps:
                return {'answer': reply, 'tool_calls': used, 'trace': trace, 'error': None}
            values = tool_calc.answer_block(requests)
            used += len(requests)
            trace.append({'type': 'calculator', 'requests': requests, 'results': values})
            messages.extend([{'role': 'assistant', 'content': reply},
                             {'role': 'user', 'content': values + '\nUse these results. Give u(x, t) = <expression>.'}])
    except (ModelError, ValueError, KeyError, IndexError) as exc:
        return {'answer': '', 'tool_calls': used, 'trace': trace, 'error': str(exc)}


def repair_prompt(problem, best, history):
    passed = ', '.join(k for k, value in best['parts'].items() if value) or 'none'
    failed = ', '.join(k for k, value in best['parts'].items() if not value) or 'answer format'
    ledger = '\n'.join(f"- {h['expr']}: {h['feedback'][:600]}" for h in history[-3:])
    return (pdecheck.prompt_of(problem) +
            f"\n\nBest verified attempt so far: u(x, t) = {best['expr']}\n" +
            f"Already correct: {passed}. Still failing: {failed}.\n" +
            f"Checker: {best['feedback']}\nRecent distinct failed attempts:\n{ledger}\n" +
            'Preserve the correct conditions. Do not repeat a failed expression. ' +
            'Use the calculator for coefficients instead of guessing. Match each decay ' +
            'rate to the frequency in the same wave. End with the complete expression.')


def solve(problem, a, log, run_id):
    prompt = pdecheck.prompt_of(problem)
    best, history, calls, errors = None, [], 0, 0
    started = time.perf_counter()
    for rnd in range(a.rounds):
        with cf.ThreadPoolExecutor(max_workers=a.workers) as pool:
            answers = list(pool.map(lambda i: attempt(a, prompt, a.seed * 100000 + rnd * 100 + i),
                                    range(a.samples)))
        valid = []
        for sample, response in enumerate(answers):
            calls += sum(t['type'] in ('model', 'http_error', 'transport_error') for t in response['trace'])
            if response['error']:
                errors += 1
                graded = None
            else:
                graded = validation.grade(problem, response['answer'])
                if response['trace'] and response['trace'][-1].get('finish_reason') == 'length' and graded['reward'] < 1.0:
                    graded['feedback'] = ('Your output was truncated before a usable solution. '
                                          'Skip the derivation; output only calculator requests or the final expression. ' + graded['feedback'])
                valid.append(graded)
            log.write(json.dumps({'run_id': run_id, 'problem': problem['name'], 'seed': a.seed,
                                  'round': rnd, 'sample': sample, 'prompt': prompt,
                                  **response, 'grade': graded}, allow_nan=False) + '\n')
            log.flush()
        if not valid:
            return {'problem': problem['name'], 'seed': a.seed, 'status': 'infrastructure_error',
                    'reward': best['reward'] if best else None, 'rounds': rnd + 1,
                    'requests': calls, 'errors': errors, 'seconds': time.perf_counter() - started}
        candidate = max(valid, key=lambda g: (g['reward'], -(g['start_error'] if g['start_error'] is not None else 1e99)))
        if best is None or (candidate['reward'], -(candidate['start_error'] if candidate['start_error'] is not None else 1e99)) > (best['reward'], -(best['start_error'] if best['start_error'] is not None else 1e99)):
            best = candidate
        print(f"{problem['name']} round {rnd}: rewards {[g['reward'] for g in valid]}, errors {len(answers)-len(valid)}", flush=True)
        if best['reward'] == 1.0:
            return {'problem': problem['name'], 'seed': a.seed, 'status': 'solved', 'reward': 1.0,
                    'rounds': rnd + 1, 'requests': calls, 'errors': errors,
                    'seconds': time.perf_counter() - started, 'answer': best['expr'],
                    'validation': best['validation']}
        if candidate['expr'] and all(h['expr'] != candidate['expr'] for h in history):
            history.append(candidate)
        prompt = repair_prompt(problem, best, history)
    return {'problem': problem['name'], 'seed': a.seed, 'status': 'unsolved', 'reward': best['reward'],
            'rounds': a.rounds, 'requests': calls, 'errors': errors,
            'seconds': time.perf_counter() - started, 'answer': best['expr']}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--level', type=int, choices=(0, 1), default=1)
    p.add_argument('--sub', type=int, choices=(1, 2, 3))
    p.add_argument('--all', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--repeat', type=int, default=1)
    p.add_argument('--samples', type=int, default=4)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--rounds', type=int, default=4)
    p.add_argument('--retries', type=int, default=2)
    p.add_argument('--timeout', type=float, default=180)
    p.add_argument('--max-tokens', type=int, default=1200)
    p.add_argument('--tool-steps', type=int, default=2)
    p.add_argument('--no-tools', action='store_true')
    p.add_argument('--concise', action='store_true', help='experimental concise output contract; disabled by default')
    p.add_argument('--model', default=os.getenv('HEATROD_MODEL', 'Qwen/Qwen3-8B'))
    p.add_argument('--base', default=os.getenv('HEATROD_BASE_URL', 'http://localhost:8000/v1'))
    p.add_argument('--output', default='results')
    a = p.parse_args()
    if min(a.samples, a.workers, a.rounds, a.repeat, a.max_tokens) < 1 or min(a.retries, a.tool_steps) < 0:
        p.error('Counts must be positive; retries/tool-steps may be zero')
    run_id = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8]
    output = Path(a.output) / run_id
    output.mkdir(parents=True)
    settings = vars(a).copy()
    results = []
    first_seed = a.seed
    with (output / 'attempts.jsonl').open('w') as log:
        for repeat in range(a.repeat):
            a.seed = first_seed + repeat
            for sub in ((a.sub,) if a.sub and not a.all else upstream.LEVELS[a.level].SUBS):
                results.append(solve(upstream.LEVELS[a.level].make(sub, a.seed), a, log, run_id))
                (output / 'summary.json').write_text(json.dumps({'run_id': run_id, 'settings': settings,
                                                               'results': results}, indent=2, allow_nan=False))
    print(json.dumps(results, indent=2), flush=True)
    print(f'Artifacts: {output}', flush=True)


if __name__ == '__main__':
    main()
