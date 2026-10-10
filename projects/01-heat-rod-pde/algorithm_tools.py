"""Conservative per-problem caching and advisory final-answer diagnostics."""
import ast
from concurrent.futures import Future, TimeoutError
import io
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
import tokenize

import sympy as sp
import pdecheck
import tool_calc


def canonical_expression(source):
    """Ignore parentheses/whitespace, but preserve numeric spelling and float precision."""
    source = source.strip().rstrip(';').strip()
    tree = ast.parse(source, mode='eval')
    # Python's AST rounds long decimal literals. Their source tokens must remain in the key.
    numbers = tuple(token.string for token in tokenize.generate_tokens(io.StringIO(source).readline)
                    if token.type == tokenize.NUMBER)
    return repr((ast.dump(tree, include_attributes=False), numbers))


def evaluate_with_timeout(source, timeout=3):
    """Run the original calculator unchanged, including its float semantics."""
    started = time.monotonic()
    try:
        result = subprocess.run([sys.executable, str(Path(tool_calc.__file__).resolve()), source],
                                capture_output=True, text=True, timeout=max(.01, timeout))
        value = result.stdout.strip()
        error = result.returncode != 0 or value.startswith((
            'could not', 'unsupported', 'rejected:', 'the expression contains',
            'expression too long', 'nothing to compute'))
        return dict(status='error' if error else 'ok',
                    result=value or 'Calculator process failed',
                    seconds=time.monotonic()-started)
    except subprocess.TimeoutExpired:
        return dict(status='timeout', result='Calculation exceeded its time budget; simplify the request',
                    seconds=time.monotonic()-started)


class CalculatorCache:
    """Single executing owner for equivalent concurrent requests in one problem."""
    def __init__(self, enabled=True, timeout=3, evaluator=None):
        self.enabled, self.timeout, self.evaluator = enabled, timeout, evaluator
        self._lock, self._entries = threading.Lock(), {}

    def compute(self, source, deadline=None):
        started = time.monotonic()
        remaining = self.timeout if deadline is None else min(self.timeout, deadline-started)
        if remaining <= .01:
            return dict(expression=source, key=None, status='budget_timeout',
                        result='No calculator time remains', executed=False, cached=False, seconds=0)
        try:
            key = canonical_expression(source)
        except (ValueError, SyntaxError, tokenize.TokenError):
            key = 'literal:' + source.strip()
        owner, future = True, None
        if self.enabled:
            with self._lock:
                future = self._entries.get(key)
                if future is None:
                    future = Future()
                    self._entries[key] = future
                else:
                    owner = False
        if owner:
            try:
                value = self.evaluator(source) if self.evaluator else evaluate_with_timeout(source, remaining)
            except Exception as exc:
                value = dict(status='error', result=f'Calculator error: {type(exc).__name__}')
            if future is not None:
                future.set_result(value)
                if value['status'] in ('timeout', 'budget_timeout'):
                    with self._lock:
                        self._entries.pop(key, None)  # A short remaining deadline is not a permanent failure.
        else:
            try:
                value = future.result(timeout=remaining)
            except TimeoutError:
                value = dict(status='budget_timeout', result='Deadline while waiting for identical calculation')
        return dict(value, expression=source, key=key, executed=owner, cached=not owner,
                    seconds=time.monotonic()-started)

    def answer_block(self, requests, deadline=None, observer=None):
        records = []
        for source in requests[:8]:
            if observer:
                observer(dict(type='calculator_request', expression=source))
            record = self.compute(source, deadline)
            records.append(record)
            if observer:
                observer(dict(type='calculator_result', record=record))
        seen, shown = set(), []
        for record in records:
            if not self.enabled or record['key'] not in seen:
                shown.append(record)
            seen.add(record['key'])
        return '\n'.join(f"COMPUTE: {r['expression']}\n  = {r['result']}" for r in shown), records


def assess_final(answer, finish_reason=None):
    """Diagnostics only: never change checker scores or silently alter an expression."""
    expression = pdecheck.extract(answer) if '=' in answer else answer.strip()
    issues = []
    if re.search(r'^\s*COMPUTE:', answer, re.M):
        issues.append(dict(category='unfinished_compute', message='Finish the requested calculation and assemble the solution'))
    if not expression:
        issues.append(dict(category='no_answer', message='Supply one complete u(x,t) = expression line'))
    else:
        if re.search(r'\b(Sum|Integral|integrate)\s*\(', expression):
            issues.append(dict(category='unfinished_symbolic', message='Expand a finite set of explicitly chosen modes and evaluate each coefficient; do not leave Sum or Integral'))
        try:
            parsed = pdecheck.parse(expression)
            if not isinstance(parsed, sp.Expr):
                raise ValueError('Use one scalar expression, not a list or tuple')
            extra = parsed.free_symbols - {pdecheck.x, pdecheck.t}
            if extra:
                issues.append(dict(category='free_parameter', message='Substitute justified values for '+str(sorted(map(str,extra)))+'; never delete an index'))
        except Exception as exc:
            issues.append(dict(category='syntax', message='Return a complete supported scalar expression: '+type(exc).__name__))
    if finish_reason == 'length' and issues:
        issues.append(dict(category='truncated', message='Output was cut off; use the remaining call for the complete expression only'))
    return dict(valid=not issues, answer=answer, expression=expression, issues=issues,
                repairs=[], truncated=finish_reason=='length')


def coefficient_feedback(problem, answer, records):
    """Bind only model-requested normalized projections to their own output modes."""
    notes = []
    try:
        candidate = pdecheck.parse(pdecheck.extract(answer) if '=' in answer else answer)
        initial = sp.expand(candidate.subs(pdecheck.t, 0))
    except Exception:
        return ''
    from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
    glob = {'__builtins__': {}, **{name:getattr(sp,name) for name in ('Integer','Float','Rational','Symbol')}}
    for record in records:
        if record.get('status') != 'ok':
            continue
        try:
            requested = parse_expr(record['expression'], local_dict=dict(tool_calc._ALLOWED),
                                   global_dict=glob, transformations=standard_transformations+(implicit_multiplication_application,))
            integrals = list(requested.atoms(sp.Integral))
            if len(integrals) != 1:
                continue
            integral = integrals[0]
            if integral.limits != ((tool_calc.x, 0, problem['L']),):
                continue
            # Only use an actual normalized projection chosen by the model.
            for wave in integral.function.atoms(sp.sin, sp.cos):
                if not wave.has(tool_calc.x):
                    continue
                remainder = sp.simplify(requested / integral *
                                        integral.function / (problem['f'] * wave))
                if sp.simplify(remainder - 2/problem['L']) != 0:
                    continue
                exact = record['result'].split('  (=')[0].strip()
                value = pdecheck.parse(exact)
                got = initial.coeff(wave)
                if sp.simplify(got-value) != 0:
                    notes.append(f'Your calculator already returned coefficient {exact} for {sp.sstr(wave)}; '
                                 f'the submitted t=0 coefficient is {sp.sstr(got)}. Keep each result with its own wave.')
        except Exception:
            continue  # Ambiguous tool requests do not become coefficient claims.
    return '\n'.join(dict.fromkeys(notes))[:1800]
