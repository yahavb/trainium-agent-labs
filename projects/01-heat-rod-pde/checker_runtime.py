"""Process boundary for untrusted symbolic work; JSON carries public data only."""
import json
import math
import os
from pathlib import Path
import subprocess
import sys

WALL_SECONDS = 8
WORKER = Path(__file__).with_name('checker_worker.py')


def diagnostic(category, condition, evidence, correction, *, location=None, severity='error'):
    return dict(category=category, severity=severity, condition=condition,
                evidence=evidence, location=location, correction=correction)


def failure(message, category='NUMERICAL_VALIDATION_INCONCLUSIVE', evidence=None):
    uncertain = category == 'NUMERICAL_VALIDATION_INCONCLUSIVE'
    return dict(accepted=False, parts={}, errors={}, start_error=None,
                status='inconclusive' if uncertain else 'invalid', error=message,
                feedback=message, diagnostics=[diagnostic(category, 'expression' if not uncertain else 'evaluation',
                    evidence or {}, message, severity='warning' if uncertain else 'error')])


def no_score(validation, expression=None):
    # Numeric reward is required by existing Agent ranking. scored=False means
    # this fallback zero is not a measured mathematical score.
    return dict(reward=0.0, parts={}, expr=expression, start_error=None,
                feedback=validation['feedback'], original_reward=None, scored=False,
                status=validation['status'], diagnostics=validation['diagnostics'])


def assemble(original, validation):
    from pdecheck import WEIGHTS
    result = dict(original)
    result.update(original_reward=original.get('original_reward', original['reward']),
                  original_parts=original['parts'], original_feedback=original['feedback'],
                  validation=validation, diagnostics=validation['diagnostics'],
                  status=validation['status'], feedback=validation['feedback'])
    verified_reward = round(sum(WEIGHTS[n] for n, ok in validation['parts'].items() if ok), 3)
    result['verified_reward'] = verified_reward
    # Keep the old grade policy: supplementary validation can veto baseline 1.0,
    # but cannot promote a failed baseline or change its partial score.
    if original['reward'] == 1.0:
        result['reward'] = 1.0 if validation['accepted'] else verified_reward
        result['parts'] = validation['parts']
        if not validation['accepted'] and result['reward'] == 1.0:
            result['reward'] = 0.0
            result['scored'] = False
    result['solved'] = bool(original['reward'] == 1.0 and validation['accepted'])
    if validation['accepted'] and not result['solved']:
        result['status'] = 'original_rejected' if original.get('scored') else 'inconclusive'
        result['feedback'] = original['feedback'] + ' Supplementary sampling passed; the baseline failure remains decisive.'
    return result


def clean_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    return value


def run_check(mode, problem, answer, *, n_points=24, validation_seed=9173):
    original = None
    validation = None
    try:
        if not isinstance(answer, str) or len(answer) > 48000:
            raise ValueError('Missing answer text or reply exceeds the checker input budget.')
        if not 1 <= int(n_points) <= 4096:
            raise ValueError('n_points must be between 1 and 4096.')
        # No callbacks, hidden solutions, oracle flags or arbitrary objects.
        public = {key: str(problem[key]) for key in ('L', 'k', 'f')}
        if any(len(value) > 12000 for value in public.values()):
            raise ValueError('Public problem expression exceeds the input budget.')
        public.update(left=problem['left'], right=problem['right'],
                      tol=float(problem.get('tol', 1e-6)), seed=int(problem.get('seed', 0)),
                      name=str(problem.get('name', 'unnamed'))[:256],
                      allow_symbolic_sum=bool(problem.get('allow_symbolic_sum', False)))
        payload = json.dumps(dict(mode=mode, problem=public, answer=answer,
                                  n_points=int(n_points), validation_seed=int(validation_seed)), allow_nan=False)
        env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
        try:
            process = subprocess.run([sys.executable, str(WORKER)], input=payload,
                                     text=True, encoding='utf-8', capture_output=True,
                                     timeout=WALL_SECONDS, env=env, cwd=str(WORKER.parent))
            output = process.stdout
            if process.returncode:
                validation = failure('Checker worker failed; no correctness conclusion.',
                                     evidence={'returncode': process.returncode})
        except subprocess.TimeoutExpired as exc:
            output = exc.stdout or ''
            if isinstance(output, bytes):
                output = output.decode('utf-8', errors='replace')
            validation = failure('Checker exceeded its wall-time budget; simplify the expression and resubmit.',
                                 evidence={'timeout_seconds': WALL_SECONDS})
        for line in output.splitlines():
            try:
                record = json.loads(line)
            except (ValueError, TypeError):
                continue
            if record.get('kind') == 'original':
                original = record['result']
            if record.get('kind') == 'validation' and validation is None:
                validation = record['result']
        if validation is None and mode != 'original':
            validation = failure('Checker returned no complete validation result.')
    except Exception as exc:
        validation = failure('Checker could not evaluate the supplied input.',
                             evidence={'exception': type(exc).__name__, 'detail': str(exc)[:200]})
    if mode == 'verify':
        return clean_json(validation)
    original = original or no_score(validation or failure('Checker returned no original score.'))
    if mode == 'original':
        return clean_json(original)
    return clean_json(assemble(original, validation))
