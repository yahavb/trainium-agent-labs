"""Grading-only boundary. Candidate execution belongs on an isolated trusted pod.

A subprocess enforces timeouts, not a security sandbox. No reference source is returned.
"""
import ast
import json
import sys
import tempfile
import traceback
from pathlib import Path
import numpy as np
import nkibench as bench


def grade(source, level, probes=False):
    spec = bench.LEVELS[level]
    result = dict(reward=0., solved=False, cases=[], passed_cases=[], simulator_calls=0, probe_results=[])
    def fail(message):
        result['cases'].append(dict(label='load', passed=False, feedback=message))
        return result
    try:
        compile(source, '<candidate>', 'exec')
    except SyntaxError as e:
        return fail(f'SyntaxError on line {e.lineno}: {e.msg}')
    result['reward'] = .1
    # The model has no filesystem tools. Also reject importing grading/oracle helpers.
    allowed = {'nki', 'nki.language', 'nki.isa', 'nki.typing', 'numpy', 'math'}
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imports = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            imports = [node.module or '']
        else:
            continue
        if any(name not in allowed for name in imports):
            return fail(f'Rule violation on line {node.lineno}: imports must be NKI, numpy constants, or math; grading helpers are unavailable to kernels')
    if any(isinstance(n, ast.Name) and n.id in {'open', 'exec', 'eval', '__import__', 'compile'} for n in ast.walk(ast.parse(source))):
        return fail('Rule violation: dynamic execution and file reads are excluded from autonomous kernels')
    violations = bench.check_rules(source, level)
    if violations:
        return fail('Rule violations: ' + ' '.join(violations))
    result['reward'] = .3
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'candidate.py'
        path.write_text(source)
        try:
            kernel = bench.load_kernel(str(path), spec['entry'])
        except Exception as e:
            return fail(f'{type(e).__name__}: {e}')
        ran = False
        for case in spec['shapes']:
            args, _ = bench.make_inputs(case, level, seed=0)
            before = [x.copy() if isinstance(x, np.ndarray) else x for x in args]
            want = spec['ref'](*args)
            result['simulator_calls'] += 1
            try:
                got, counted = bench.simulate_and_count(kernel, args)
                ran = True
                error = (bench.check_inputs_untouched(before, args) or bench.describe_mismatch(got, want)
                         or bench.check_traffic_bar(level, counted, args, want))
                hazards = [w for w in counted.get('warnings', []) if 'incorrect results on hardware' in w.lower()]
                error = error or ('Hardware correctness warning: ' + hazards[0] if hazards else None)
            except Exception as e:
                error = f'{type(e).__name__}: {e}'
                # Only candidate traceback locations; no grader/reference source.
                locations = [f'line {t.lineno}' for t in traceback.extract_tb(e.__traceback__) if t.filename == str(path)]
                error += ' ' + ', '.join(locations)
            label = bench.label(case, level)
            result['cases'].append(dict(label=label, passed=error is None, feedback=error or 'passed'))
            if error is None:
                result['passed_cases'].append(label)
        result['reward'] += .2 * ran + .5 * len(result['passed_cases']) / len(spec['shapes'])
        result['solved'] = len(result['passed_cases']) == len(spec['shapes'])
        if probes and ran:
            for pattern in ('indices', 'impulse', 'identity'):
                args, _ = bench.make_inputs(spec['shapes'][0], level, seed=0)
                for x in args:
                    if not isinstance(x, np.ndarray):
                        continue
                    x.fill(0)
                    if pattern == 'indices':
                        x[:] = (np.arange(x.size).reshape(x.shape) % 17) / 17
                    elif pattern == 'impulse':
                        x.flat[min(x.size - 1, 129)] = 1
                    elif x.ndim == 2:
                        np.fill_diagonal(x, 1)
                    else:
                        x.fill(1)
                before = [x.copy() if isinstance(x, np.ndarray) else x for x in args]
                want = spec['ref'](*args)  # Attention stays nonlinear; actual signature retained.
                result['simulator_calls'] += 1
                try:
                    got, _ = bench.simulate_and_count(kernel, args)
                    error = bench.check_inputs_untouched(before, args) or bench.describe_mismatch(got, want)
                except Exception as e:
                    error = f'{type(e).__name__}: {e}'
                result['probe_results'].append(dict(pattern=pattern, feedback=error or 'passed'))
    return result

if __name__ == '__main__':
    request = json.loads(Path(sys.argv[1]).read_text())
    result = grade(**request)
    Path(sys.argv[2]).write_text(json.dumps(result))
