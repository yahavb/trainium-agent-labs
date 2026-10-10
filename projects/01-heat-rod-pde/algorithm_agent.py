"""Budgeted PDE reasoning controller; only public problem fields enter this agent.

Feature switches support isolated experiments. The mathematical verifier is unchanged.
"""
import argparse
import concurrent.futures as cf
import datetime
import json
import os
from pathlib import Path
import threading
import time
import uuid
import sys

import httpx
import sympy as sp

import agent as original
import improved_agent as previous
import pdecheck
import algorithm_tools as research_math
import tool_calc
from checker_runtime import grade_candidate


def finite_waves(f):
    """Inspect the stated initial condition, without solving or reading an oracle."""
    x = pdecheck.x
    for term in sp.Add.make_args(sp.expand(f)):
        _, wave = term.as_independent(x, as_Add=False)
        if wave.func not in (sp.sin, sp.cos):
            return False
        argument = wave.args[0]
        if sp.diff(argument, x, 2) != 0 or argument.subs(x, 0) != 0:
            return False
    return True


def public_problem(raw):
    p = {key: raw[key] for key in ('name', 'level', 'sub', 'seed', 'L', 'k', 'left', 'right', 'f', 'tol')}
    # Compatibility sentinel for the original checker's hint selection, never an answer.
    p['exact'] = 0 if finite_waves(p['f']) else None
    L = p['L']
    if p['left'] == 'dirichlet' and p['right'] == 'neumann':
        p['lam'] = lambda n: (2*n-1)*sp.pi/(2*L)
        p['basis'] = lambda n: sp.sin(p['lam'](n)*pdecheck.x)
    elif p['left'] == 'dirichlet' and p['right'] == 'dirichlet':
        p['lam'] = lambda n: n*sp.pi/L
        p['basis'] = lambda n: sp.sin(p['lam'](n)*pdecheck.x)
    else:
        raise ValueError('Unsupported boundary configuration')
    return p


def problem_statement(p):
    L, k, f = map(sp.sstr, (p['L'], p['k'], p['f']))
    left = 'u(0,t)=0' if p['left']=='dirichlet' else 'u_x(0,t)=0'
    right = f'u({L},t)=0' if p['right']=='dirichlet' else f'u_x({L},t)=0'
    return (f'Solve u_t = ({k})*u_xx for 0<x<{L}, t>0.\n'
            f'Boundary conditions: {left}; {right}.\nInitial shape: u(x,0)=({f}).\n'
            f'Relative initial-shape tolerance: {float(p["tol"]):g}. '
            'The PDE and both boundary conditions must also pass independently.')


def workflow(p, sample, coefficients_needed=True, reuse_coefficients=False):
    L, k, f = map(sp.sstr, (p['L'], p['k'], p['f']))
    if p['right'] == 'neumann':
        basis = f'sin((2*n-1)*pi*x/(2*({L})))'
        condition = f'The right end is insulated: u_x({L},t)=0, so use half-integer sine frequencies, not integer ones.'
    else:
        basis = f'sin(n*pi*x/({L}))'
        condition = 'Both ends have zero value; use integer sine frequencies.'
    mode = ('Read the actual initial waves first. Preserve each stated amplitude and spatial frequency.'
            if sample % 2 == 0 else
            'Check the boundary derivative of each initial wave first, then check its decay exponent.')
    text = (f'METHOD CHECKLIST: {condition}\n'
            f'The stated initial shape is ({f}), L={L}, k={k}. {mode}\n'
            'For every wave sin(mu*x), its time factor is exp(-k*mu**2*t). '
            'Use the identical mu in the wave and its exponent; do not replace it with the term number.\n')
    if finite_waves(p['f']):
        text += ('This initial condition is already a finite collection of waves. '
                 'Keep those waves and their amplitudes. No Fourier integrals are needed if they satisfy the boundaries. '
                 'Write the complete numerical expression directly; a long derivation is unnecessary.\n')
    elif coefficients_needed and reuse_coefficients:
        text += ('Previous calculator results are available in the feedback. First check normalization '
                 'and match each recorded coefficient to its own wave. Reuse those values. '
                 'Only request additional calculations for specifically missing modes or an incorrect prior setup. '
                 'If the needed values are already present, assemble the corrected final expression directly.\n')
    elif coefficients_needed:
        text += (f'For this non-wave shape, compute coefficients by (2/({L}))*Integral(({f})*{basis}, (x,0,{L})). '
                 'Include the factor 2/L INSIDE every calculator request. Substitute concrete integer mode indices; '
                 'initially batch the first six mode coefficients in one exchange, including zero coefficients. '
                 'If the checker later requests more modes, choose those missing modes instead of repeating this initial range. '
                 'Then assemble only the nonzero modes with their matching decay rates. '
                 f'The stated relative initial-shape tolerance is {float(p["tol"]):g}; '
                 'the checker will decide whether enough modes were kept. Never guess the required coefficients.\n')
    else:
        text += ('The checker already accepted the initial shape. Use the existing coefficients and repair only the failed conditions. '
                 'Do not recompute already accepted coefficients.\n')
    if coefficients_needed and not finite_waves(p['f']) and not reuse_coefficients:
        return text + ('STAGE 1 OF 2: This response is for calculator requests only. '
                       'Write one COMPUTE: <arithmetic expression containing Integral(...)> line per chosen numeric mode. '
                       'No assignments such as c_1=, no prose inside COMPUTE, no symbolic n, no COMPUTER label, and no final solution yet. '
                       'Stop after your batch; the next response will assemble the solution from returned values.')
    return text + 'Use at most two short planning sentences. End with one complete u(x, t) = <expression> line.'


def targeted_feedback(p, best, recent, cache_memory):
    parts = best.get('parts') or {}
    messages = []
    if not best.get('expr'):
        messages.append('FORMAT: the last response had no usable expression. Finish the expression within the output budget; do not restart a long derivation.')
    if parts and (not parts.get('left_bc') or not parts.get('right_bc')):
        messages.append('BOUNDARY: select waves satisfying both endpoint conditions. For an insulated right end the derivative, not the value, must vanish.')
    if parts and not parts.get('equation'):
        if parts.get('start_shape'):
            messages.append('DECAY: keep amplitudes that already match the initial condition.')
        messages.append('DECAY: for EACH selected spatial wave sin(mu*x), use exp(-k*(mu)**2*t); include its full denominator in mu. Nonzero modes may skip indices.')
    if parts and not parts.get('start_shape'):
        if 'missing' in best.get('feedback', ''):
            messages.append('TRUNCATION/COEFFICIENT: compute the specifically missing modes; add terms only after checking their projection coefficients.')
        if parts.get('equation') and parts.get('left_bc') and parts.get('right_bc'):
            messages.append('COEFFICIENT: keep verified frequencies and decay.')
        messages.append('COEFFICIENT: the coefficient is the projection integral multiplied by 2/L, not the raw integral. Check that normalization and signs; do not guess scale factors.')
    passed = ', '.join(k for k, v in parts.items() if v) or 'none'
    ledger = '\n'.join('- ' + h[:350] for h in recent[-2:])
    binding = research_math.coefficient_feedback(p, best.get('expr') or '', cache_memory.get('records', []))
    memory_text = cache_memory.get('text', '')
    return (f'Best candidate (passed: {passed}): u(x,t) = {best.get("expr")}\n'
            + ' '.join(messages) + '\nChecker evidence: ' + best.get('feedback', '')[:1300]
            + '\nDo not repeat these rejected expressions:\n' + ledger
            + ('\nCalculator-to-wave audit:\n' + binding if binding else '')
            + ('\nPrevious calculator results (exactly as computed):\n' + memory_text[-2200:] if memory_text else ''))


class BudgetError(RuntimeError):
    pass


class ServiceError(RuntimeError):
    pass


class Context:
    def __init__(self, a):
        self.a = a
        self.features = set(filter(None, a.features.split(',')))
        self.started = time.monotonic()
        self.end = self.started + a.deadline
        self.lock = threading.Lock()
        self.requests = self.prompt_tokens = self.completion_tokens = 0
        self.events = []
        self.calc = research_math.CalculatorCache(enabled='cache' in self.features, timeout=3.0)
        self.memory = {}
        self.calculations = {}
        self.unknown_usage = 0

    def record(self, event):
        with self.lock:
            self.events.append(event)
            if getattr(self.a, 'telemetry', None):
                with open(self.a.telemetry, 'a') as f:
                    f.write(json.dumps(event, allow_nan=False) + '\n')

    def ask(self, messages, trace, stage, attempt_id=None):
        remaining = self.end - time.monotonic()
        reserve = 4 if 'adaptive' in self.features else 0.5
        if remaining <= reserve + 1:
            raise BudgetError('Case deadline reached before request')
        cap = self.a.max_tokens
        if 'adaptive' in self.features:
            # Stay within the original per-request cap; shrink only near the deadline.
            cap = min(cap, max(48, int((remaining-reserve)*11)))
        with self.lock:
            if self.requests >= self.a.samples*self.a.rounds*(1+self.a.tool_steps):
                raise BudgetError('Original model request budget exhausted')
            self.requests += 1
        body = dict(model=self.a.model, messages=messages, temperature=.6, top_p=.95,
                    max_tokens=cap, chat_template_kwargs={'enable_thinking': False})
        started = time.monotonic()
        request_id = uuid.uuid4().hex
        self.record(dict(type='request_start', request_id=request_id, attempt_id=attempt_id,
                         stage=stage, messages=messages, max_tokens=cap))
        try:
            response = httpx.post(self.a.base.rstrip('/')+'/chat/completions', json=body,
                                  timeout=min(self.a.timeout, remaining-reserve))
            response.raise_for_status()
            payload = response.json()
            choice = payload['choices'][0]
            reply = choice['message'].get('content') or ''
            usage = payload.get('usage') or {}
            event = dict(type='model', request_id=request_id, attempt_id=attempt_id, stage=stage, messages=messages, answer=reply,
                         finish_reason=choice.get('finish_reason'), usage=usage,
                         max_tokens=cap, seconds=time.monotonic()-started)
            with self.lock:
                self.prompt_tokens += usage.get('prompt_tokens') or 0
                self.completion_tokens += usage.get('completion_tokens') or 0
                self.unknown_usage += not all(type(usage.get(k)) is int for k in ('prompt_tokens','completion_tokens'))
            trace.append(event)
            self.record(event)
            if not reply.strip():
                raise ServiceError('Empty model response')
            return reply, choice.get('finish_reason')
        except httpx.TimeoutException as exc:
            with self.lock:
                self.unknown_usage += 1
            event = dict(type='deadline', request_id=request_id, attempt_id=attempt_id,
                         error=type(exc).__name__, seconds=time.monotonic()-started)
            trace.append(event)
            self.record(event)
            if isinstance(exc, httpx.ConnectTimeout):
                raise ServiceError(str(exc)) from exc
            raise BudgetError(str(exc)) from exc
        except (httpx.HTTPError, KeyError, ValueError, IndexError, TypeError) as exc:
            with self.lock:
                self.unknown_usage += 1
            self.record(dict(type='service_error', request_id=request_id, attempt_id=attempt_id, error=str(exc)))
            raise ServiceError(str(exc)) from exc


def run_attempt(p, a, ctx, prompt, sample, round_index=0):
    trace = []
    messages = [{'role': 'user', 'content': prompt + '\n\n' + tool_calc.INSTRUCTIONS}]
    if 'feedback' in ctx.features:
        best = getattr(ctx, 'best', None)
        coefficients_needed = not (best and best.get('parts', {}).get('start_shape'))
        # Remove the unrelated concrete calculator example from the new workflow.
        messages = [{'role': 'user', 'content': prompt + '\n\n' + workflow(p, sample+round_index, coefficients_needed,
                                                                         reuse_coefficients=bool(ctx.calculations))
                     + '\nCalculator protocol: COMPUTE: <SymPy arithmetic or Integral expression>. '
                     'Write each distinct request once, then stop and wait for the results.'}]
    if 'final' in ctx.features:
        messages[0]['content'] += ('\nFinal format: an explicit arithmetic expression in x,t only. '
            'Expand any finite series explicitly with justified indices and coefficients; never leave a free n, Integral, or unknown parameter.')
    evaluations = hits = tool_requests = 0
    last = ''
    structure = None
    repaired = False
    try:
        for step in range(a.tool_steps+1):
            reply, finish = ctx.ask(messages, trace, 'initial' if step == 0 else 'continuation',
                                    attempt_id=f'{round_index}:{sample}')
            last = reply
            structure = research_math.assess_final(reply, finish) if 'final' in ctx.features else None
            requests = tool_calc.requests_in(reply)
            if requests and step < a.tool_steps:
                values, records = ctx.calc.answer_block(requests, deadline=ctx.end-(4 if 'adaptive' in ctx.features else 1),
                    observer=lambda event: ctx.record(dict(event, attempt_id=f'{round_index}:{sample}')))
                tool_requests += len(requests)
                evaluations += sum(bool(r.get('executed')) for r in records)
                hits += sum(bool(r.get('cached')) for r in records)
                event = dict(type='calculator', attempt_id=f'{round_index}:{sample}', requests=requests, results=values, records=records)
                trace.append(event)
                ctx.record(event)
                with ctx.lock:
                    for r in records:
                        if r.get('status') == 'ok':
                            ctx.memory[r.get('key', r['expression'])] = r['expression'] + ' = ' + r['result']
                            ctx.calculations[r.get('key', r['expression'])] = r
                assistant = reply
                if 'cache' in ctx.features or 'adaptive' in ctx.features:
                    assistant = '\n'.join('COMPUTE: '+r for r in dict.fromkeys(requests))
                instruction = 'Use these results. Give u(x, t) = <expression>.'
                if 'feedback' in ctx.features:
                    instruction += (f' Apply k={sp.sstr(p["k"])} times the square of each ACTUAL wave frequency, including its denominator. '
                                    'The computed values already contain any factors included in your requests. '
                                    'Match EACH returned coefficient to the wave in ITS integral, including differing amplitudes. '
                                    'Never copy the first coefficient to the other waves. '
                                    'Do not calculate them again. Write every nonzero mode in one final expression.')
                    messages = [{'role':'user','content':problem_statement(p)},
                                {'role':'assistant','content':assistant},
                                {'role':'user','content':values+'\nSTAGE 2 OF 2: '+instruction+
                                 '\nUse the wave frequencies from the integrals you actually requested. '
                                 'Expand the finite collection explicitly. Output only u(x,t) = <expression> in Python syntax.'}]
                else:
                    messages = messages + [{'role':'assistant', 'content':assistant}, {'role':'user', 'content':values+'\n'+instruction}]
                continue
            if structure and structure['valid']:
                break
            # Advisory syntax diagnostics never veto an answer accepted by the official checker.
            if structure and grade_candidate(p, reply)['reward'] == 1.0:
                break
            if structure and step < a.tool_steps and not requests and not repaired:
                issue_text = '; '.join(i['message'] for i in structure['issues'])
                repair = ('Repair only the reported final-answer defects: '+issue_text+
                          '. Preserve mathematically justified coefficients and wave frequencies. '
                          'Never delete or guess an index. Expand a chosen finite sum explicitly, substituting all indices. '
                          'If a truncation is needed, choose its limits from the stated initial-shape tolerance and existing coefficient calculations. '
                          'Output the complete u(x,t) = expression. This is your final model call for this attempt.')
                messages = [messages[0], {'role':'assistant','content':reply[-2400:]}, {'role':'user','content':repair}]
                trace.append(dict(type='targeted_repair', issues=structure['issues'], costs_one_reserved_call=True))
                ctx.record(dict(trace[-1], attempt_id=f'{round_index}:{sample}'))
                repaired = True
                continue
            break
        return dict(answer=last, trace=trace, structure=structure, tool_calls=tool_requests,
                    tool_executions=evaluations, cache_hits=hits, error=None)
    except (BudgetError, ServiceError) as exc:
        return dict(answer=last, trace=trace, structure=structure, tool_calls=tool_requests,
                    tool_executions=evaluations, cache_hits=hits, error=str(exc),
                    error_category='budget_timeout' if isinstance(exc, BudgetError) else 'infrastructure_error')


def solve(raw, a, log, run_id):
    p = public_problem(raw)
    ctx = Context(a)
    base = problem_statement(p) if 'feedback' in ctx.features else pdecheck.prompt_of(p)
    prompt = base
    best, history, recent = None, [], []
    seen = set()
    duplicates = invalid = executions = hits = requests_tool = repairs = 0
    rounds = 0
    status = 'unsolved'
    for rnd in range(a.rounds):
        if time.monotonic() >= ctx.end - 2:
            status = 'budget_timeout'
            break
        rounds = rnd+1
        ctx.best = best
        pool = cf.ThreadPoolExecutor(max_workers=a.workers)
        pending = {pool.submit(run_attempt,p,a,ctx,prompt,sample,rnd):sample for sample in range(a.samples)}
        answers = []
        viable = []
        for future in cf.as_completed(pending):
            sample, result = pending[future], future.result()
            answers.append(result)
            executions += result['tool_executions']
            hits += result['cache_hits']
            requests_tool += result['tool_calls']
            repairs += sum(event['type']=='targeted_repair' for event in result['trace'])
            grade = grade_candidate(raw,result['answer']) if not result.get('error') else None
            duplicate = False
            if grade:
                if not grade.get('parts'):
                    invalid += 1
                try:
                    key = research_math.canonical_expression(grade['expr']) if grade.get('expr') else result['answer'].strip()
                except (ValueError, TypeError, SyntaxError):
                    key = result['answer'].strip()
                duplicate = key in seen
                duplicates += duplicate
                seen.add(key)
                if grade['reward'] is not None:
                    viable.append(grade)
                if grade.get('expr') and grade['reward'] is not None and grade['reward'] < 1:
                    recent.append(grade['expr'])
            log.write(json.dumps(dict(run_id=run_id,problem=p['name'],seed=a.seed,round=rnd,sample=sample,
                                      prompt=prompt,**result,grade=grade,duplicate=duplicate),allow_nan=False)+'\n')
            log.flush()
        pool.shutdown(wait=True)
        if any(r.get('error_category') == 'infrastructure_error' for r in answers):
            status = 'infrastructure_error'
            break
        if viable:
            rank = lambda g: (g['reward'], -(g.get('start_error') if g.get('start_error') is not None else 1e99))
            candidate = max(viable,key=rank)
            if best is None or rank(candidate)>rank(best):
                best = candidate
            print(f"{p['name']} round {rnd}: rewards {[g['reward'] for g in viable]}",flush=True)
            if best['reward'] == 1.0:
                status = 'solved'
                break
            if candidate.get('expr') and all(h['expr']!=candidate['expr'] for h in history):
                history.append(candidate)
            if 'feedback' in ctx.features:
                prompt = base+'\n\n'+targeted_feedback(p,best,list(dict.fromkeys(recent)),
                    dict(text='\n'.join(ctx.memory.values()), records=list(ctx.calculations.values())))
            else:
                prompt = previous.repair_prompt(p,best,history)
        if any(r.get('error_category') == 'budget_timeout' for r in answers):
            status = 'budget_timeout'
            break
    if status == 'unsolved' and best is None:
        status = 'evaluation_error'
    return dict(problem=p['name'],seed=a.seed,status=status,reward=best['reward'] if best else None,
                original_reward=best['reward'] if best else None, grading_policy='original_checker',
                additional_validation_status='not_run',
                answer=best.get('expr') if best else None,
                rounds=rounds,requests=ctx.requests,
                prompt_tokens=None if ctx.unknown_usage else ctx.prompt_tokens,
                completion_tokens=None if ctx.unknown_usage else ctx.completion_tokens,
                tool_calls=requests_tool,tool_executions=executions,cache_hits=hits,
                validation_failures=invalid,duplicate_candidates=duplicates,repair_requests=repairs,
                features=sorted(ctx.features),
                seconds=time.monotonic()-ctx.started)


def main():
    parser = argparse.ArgumentParser()
    for name, default in [('level',1),('sub',1),('seed',0),('samples',2),('rounds',3),('max-tokens',512),('tool-steps',1),('workers',2)]:
        parser.add_argument('--'+name,type=int,default=default)
    parser.add_argument('--features',default='cache,final,feedback,adaptive')
    parser.add_argument('--deadline',type=float,default=177)
    parser.add_argument('--timeout',type=float,default=180)
    parser.add_argument('--model',default=os.getenv('HEATROD_MODEL','Qwen/Qwen3-8B'))
    parser.add_argument('--base',default=os.getenv('HEATROD_BASE_URL','http://localhost:8000/v1'))
    parser.add_argument('--output',default='algorithm-results')
    a=parser.parse_args()
    if a.level not in (0,1) or a.sub not in (1,2,3) or min(a.samples,a.rounds,a.max_tokens,a.workers)<1 or a.tool_steps<0:
        parser.error('Invalid problem or budget')
    unknown=set(filter(None,a.features.split(',')))-{'cache','final','feedback','adaptive'}
    if unknown:
        parser.error('Unknown features: '+str(unknown))
    run_id=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8]
    output=Path(a.output)/run_id
    output.mkdir(parents=True)
    a.telemetry=str(output/'telemetry.jsonl')
    with (output/'attempts.jsonl').open('w') as log:
        result=solve(original.LEVELS[a.level].make(a.sub,a.seed),a,log,run_id)
    (output/'summary.json').write_text(json.dumps(dict(run_id=run_id,settings=vars(a),results=[result]),indent=2,allow_nan=False))
    print(json.dumps(result,indent=2),flush=True)
    return 1 if result['status'] in ('infrastructure_error','evaluation_error') else 0


if __name__=='__main__':
    sys.exit(main())
