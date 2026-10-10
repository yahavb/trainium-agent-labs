"""Bounded, evidence-based supplementary checks. No answer oracle is accessed.

Thresholds match the existing checker. Extra samples/uncertainty handling belong
to validation, never to the separately reported original numerical score.
"""
import hashlib
import math

import numpy as np
import sympy as sp

from checker_enhanced_runtime import diagnostic
from checker_syntax import x, t

RESIDUAL_TOL = 1e-6
MAX_MODES = 8


def values(expr, X, T):
    """Compile only the AST-built SymPy tree; never feed raw text to lambdify."""
    with np.errstate(all='ignore'):
        out = np.asarray(sp.lambdify((x, t), expr, 'numpy', cse=True)(X, T))
    if np.iscomplexobj(out):
        if np.any(out.imag != 0):
            raise ValueError('Non-real temperature or residual at a sample.')
        out = out.real
    out = np.broadcast_to(np.asarray(out, dtype=float), np.broadcast(X, T).shape)
    if not np.all(np.isfinite(out)):
        raise ValueError('Non-finite temperature or residual at a sample.')
    return out


def small_reduce(expr):
    # Avoid unrestricted simplify, integrate, or doit. A process deadline also
    # bounds differentiation, lambdification, and this limited cancellation.
    if sp.count_ops(expr) <= 160:
        return sp.factor_terms(sp.expand_mul(expr))
    return expr


def high_precision(expr, at_x, at_t):
    val = expr.evalf(50, subs={x: sp.Float(str(at_x), 50), t: sp.Float(str(at_t), 50)})
    if val.is_real is not True or val.is_finite is not True:
        raise ValueError('High-precision evaluation was not finite and real.')
    result = float(val)
    if not math.isfinite(result):
        raise ValueError('High-precision result exceeds the numeric range.')
    return result


def condition(expr, X, T, scale, threshold=RESIDUAL_TOL):
    reduced = small_reduce(expr)
    sampled = values(reduced, X, T)
    i = int(np.argmax(np.abs(sampled)))
    at_x, at_t = float(np.asarray(X).flat[i]), float(np.asarray(T).flat[i])
    error = float(abs(sampled.flat[i]) / scale)
    result = dict(status='pass' if error < threshold else 'fail', error=error,
                  threshold=threshold, value=float(sampled.flat[i]),
                  location={'x': at_x, 't': at_t}, symbolic_zero=bool(reduced == 0))
    # Confirm apparent failures/near-threshold decisions at the worst sample.
    # Conflicting precision is uncertainty, never a guessed mathematical error.
    if error >= threshold * .1:
        hp = high_precision(reduced, at_x, at_t)
        result['high_precision_value'] = hp
        hp_error = abs(hp) / scale
        if (hp_error < threshold) != (error < threshold) or abs(hp_error-error) > max(threshold*.1, error*.02):
            result['status'] = 'inconclusive'
    return result


def basis(p, n):
    """Derive modes from public BCs; do not consume generator callbacks."""
    L = p['L']
    pair = (p['left'], p['right'])
    if pair == ('dirichlet', 'dirichlet'):
        return sp.sin(n*sp.pi*x/L)
    if pair == ('dirichlet', 'neumann'):
        return sp.sin((2*n-1)*sp.pi*x/(2*L))
    if pair == ('neumann', 'dirichlet'):
        return sp.cos((2*n-1)*sp.pi*x/(2*L))
    return sp.cos((n-1)*sp.pi*x/L)  # NN includes its constant mode, norm L.


def max_cycles(expressions, L):
    cycles = 0.0
    for expression in expressions:
        for wave in expression.atoms(sp.sin, sp.cos):
            rate = sp.diff(wave.args[0], x)
            if not rate.free_symbols:
                cycles = max(cycles, abs(float(rate))*L/(2*math.pi))
    return cycles


def initial_condition(p, u, rng):
    L, tol = float(p['L']), float(p['tol'])
    grids = [np.linspace(0, L, 1601)]
    for size in (1601, 3201):
        # Each interior point is jittered within its own cell: sorted, bounded,
        # and not commensurate with either historical uniform grid.
        g = np.linspace(0, L, size)
        g[1:-1] += rng.uniform(-.4, .4, size-2)*(L/(size-1))
        grids.append(g)
    profiles, errors = [], []
    for grid in grids:
        f = values(p['f'], grid, grid*0)
        got = values(u, grid, grid*0)
        delta = got-f
        with np.errstate(all='ignore'):
            denominator = float(np.sqrt(np.trapezoid(f*f, grid))) or 1.0
            error = float(np.sqrt(np.trapezoid(delta*delta, grid)))/denominator
        if not math.isfinite(error):
            raise ValueError('Initial-profile quadrature overflowed.')
        profiles.append((grid, f, got))
        errors.append(error)
    i = int(np.argmax(np.abs(profiles[-1][2]-profiles[-1][1])))
    stable = abs(errors[1]-errors[2]) <= max(1e-10, .02*max(tol, errors[1], errors[2]))
    same_side = (errors[1] < tol) == (errors[2] < tol)
    state = 'pass' if max(errors) < tol else 'fail'
    if not stable or not same_side:
        state = 'inconclusive'
    cycles = max_cycles((u.subs(t, 0), p['f']), L)
    if state == 'pass' and cycles > (len(grids[-1])-1)/16:
        # Frequency beyond conservative quadrature resolution; don't certify it
        # merely because both finite grids appear favorable.
        if small_reduce(u.subs(t, 0)-p['f']) != 0:
            state = 'inconclusive'
    result = dict(status=state, error=max(errors), threshold=tol, grid_errors=errors,
                  quadrature_stable=stable, grid_sizes=[len(g) for g in grids],
                  largest_linear_frequency_cycles=cycles,
                  location={'x': float(profiles[-1][0][i]), 't': 0.0},
                  local_deviation=float(profiles[-1][2][i]-profiles[-1][1][i]))
    return result, profiles


def coefficient_diagnostics(p, profiles, conditions):
    if conditions['start_shape']['status'] != 'fail':
        return []
    records = []
    L = float(p['L'])
    # Compare projections on two distinct grids. Directions are diagnostic only;
    # they do not alter acceptance or print a target coefficient vector.
    for n in range(1, MAX_MODES+1):
        wave = basis(p, n)
        estimates = []
        for grid, f, got in profiles[-2:]:
            b = values(wave, grid, grid*0)
            norm = float(np.trapezoid(b*b, grid))
            estimates.append((float(np.trapezoid(got*b, grid))/norm,
                              float(np.trapezoid(f*b, grid))/norm))
        (g0, w0), (got, want) = estimates
        magnitude = max(abs(got), abs(want), 1e-8)
        numerical = max(abs(got-g0), abs(want-w0))
        threshold = max(.02*magnitude, 8*numerical, 1e-8)
        if abs(got-want) <= threshold:
            records.append(('matched', n, wave, 0.0))
            continue
        if abs(got) <= max(8*numerical, 1e-8):
            label = 'missing'
        elif abs(want) <= max(8*numerical, 1e-8):
            label = 'spurious'
        elif got*want < 0:
            label = 'wrong sign'
        else:
            label = 'too large' if abs(got) > abs(want) else 'too small'
        records.append((label, n, wave, abs(got-want)))
    mismatches = [item for item in records if item[0] not in ('matched', 'missing')]
    missing = [item for item in records if item[0] == 'missing']
    messages = []
    for label, n, wave, error in sorted(mismatches, key=lambda item: item[3], reverse=True)[:3]:
        messages.append(diagnostic('FOURIER_COEFFICIENT_ERROR', 'start_shape',
            {'mode': n, 'basis': str(wave), 'direction': label, 'projection_error': error},
            f'The initial projection onto {wave} is {label}. Recompute its coefficient as integral(f*b)/integral(b*b); preserve each mode\'s own decay.'))
    physics_pass = all(conditions[key]['status'] == 'pass' for key in ('equation', 'left_bc', 'right_bc'))
    if missing and not mismatches and physics_pass:
        messages.append(diagnostic('TRUNCATION_ERROR', 'start_shape',
            {'missing_inspected_modes': [item[1] for item in missing], 'inspected_modes': MAX_MODES,
             'scope': 'projection evidence; not a proof that all retained coefficients are correct'},
            'PDE and boundaries pass, but required initial-profile modes are absent. Compute additional coefficients until the stated relative L2 tolerance is met; an infinite sum is not required.'))
    elif missing:
        messages.append(diagnostic('FOURIER_COEFFICIENT_ERROR', 'start_shape',
            {'missing_inspected_modes': [item[1] for item in missing]},
            'Required initial-profile projections are missing. Correct existing coefficients and mode choices before assuming that adding higher terms is sufficient.'))
    return messages


def decay_diagnostics(p, u, conditions):
    if conditions['equation']['status'] != 'fail' or sp.count_ops(u) > 400:
        return []
    result = []
    # Diagnose only explicit separated terms with one linear sine/cosine and
    # one exponential. More general equivalent forms retain residual feedback.
    terms = sp.Add.make_args(sp.expand_mul(u))
    if len(terms) > 64:
        return []
    for term in terms:
        waves = list(term.atoms(sp.sin, sp.cos))
        exponentials = list(term.atoms(sp.exp))
        if len(waves) != 1 or len(exponentials) != 1:
            continue
        wave, exponential = waves[0], exponentials[0]
        arg = wave.args[0]
        frequency = sp.diff(arg, x)
        rate = -sp.diff(exponential.args[0], t)
        if frequency.free_symbols or rate.free_symbols or arg.has(t) or exponential.args[0].has(x):
            continue
        amplitude = small_reduce(term/(wave*exponential))
        if amplitude.free_symbols or amplitude == 0:
            continue
        expected = p['k']*frequency**2
        got, want = float(rate), float(expected)
        if not (math.isfinite(got) and math.isfinite(want)):
            continue
        if abs(got-want) > 1e-8*max(abs(want), 1):
            result.append(diagnostic('DECAY_RATE_ERROR', 'equation',
                {'spatial_factor': str(wave), 'candidate_rate': got, 'k_times_frequency_squared': want,
                 'scope': 'explicit term; cancellation with other terms is not excluded'},
                f'The displayed {wave} term uses a decay rate inconsistent with k*frequency**2. Reassemble paired spatial and temporal factors; the full PDE residual also fails.'))
        if len(result) == 3:
            break
    return result


def basis_diagnostics(p, u, conditions):
    if all(conditions[side+'_bc']['status'] != 'fail' for side in ('left', 'right')):
        return []
    result = []
    for wave in sorted(u.atoms(sp.sin, sp.cos), key=str)[:16]:
        if wave.has(t) or sp.diff(wave.args[0], x).free_symbols:
            continue
        sides = []
        for side, at in (('left', 0), ('right', p['L'])):
            expr = wave if p[side] == 'dirichlet' else sp.diff(wave, x)*p['L']
            deviation = small_reduce(expr.subs(x, at))
            if conditions[side+'_bc']['status'] == 'fail' and abs(float(deviation)) > RESIDUAL_TOL:
                sides.append(side)
        if sides:
            result.append(diagnostic('SPATIAL_BASIS_MISMATCH', 'boundary_basis',
                {'factor': str(wave), 'sides': sides, 'scope': 'individual factor, not a proof of a unique root cause'},
                f'The factor {wave} alone violates a failing boundary. Recheck sine/cosine choice, phase and wave number. Other terms can cancel individual boundary defects; judge the assembled solution.', severity='warning'))
        if len(result) == 2:
            break
    return result


def verify_expression(p, u, validation_seed):
    L, k = float(p['L']), p['k']
    digest = hashlib.sha256(f"{p['name']}:{p['seed']}:{validation_seed}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], 'big'))
    # Preserve all earlier supplementary samples, then add crossed space/time
    # points so every spatial slice is examined close to t=0.
    X = np.r_[rng.uniform(0, L, 257), np.linspace(0, L, 129)[1:-1]]
    T = np.r_[rng.uniform(0, .1*L*L, 257), np.geomspace(1e-9*L*L, .1*L*L, 127)]
    time_scale = L*L/float(k)
    spatial = np.r_[rng.uniform(0, L, 65), np.linspace(0, L, 67)[1:-1]]
    times = np.geomspace(1e-12*time_scale, .1*time_scale, 12)
    xx, tt = np.meshgrid(spatial, times)
    X, T = np.r_[X, xx.ravel()], np.r_[T, tt.ravel()]
    values(u, X, T)  # A zero residual alone must not certify undefined temperatures.
    grid = np.linspace(0, L, 1601)
    f = values(p['f'], grid, grid*0)
    scale = float(np.max(np.abs(f))) or 1.0
    ux = sp.diff(u, x)
    expressions = {'equation': sp.diff(u, t)-k*sp.diff(u, x, 2),
                   'left_bc': u if p['left'] == 'dirichlet' else ux*L,
                   'right_bc': u if p['right'] == 'dirichlet' else ux*L}
    conditions = {}
    for key, expr in expressions.items():
        at_x = X if key == 'equation' else np.full_like(T, 0.0 if key == 'left_bc' else L)
        try:
            # Boundary temperature itself must also remain well defined for Neumann.
            values(u, at_x, T)
            conditions[key] = condition(expr, at_x, T, scale)
        except Exception as exc:
            conditions[key] = dict(status='inconclusive', error=None, exception=type(exc).__name__)
    profiles = None
    try:
        conditions['start_shape'], profiles = initial_condition(p, u, rng)
    except Exception as exc:
        conditions['start_shape'] = dict(status='inconclusive', error=None, exception=type(exc).__name__)
    categories = dict(equation='PDE_RESIDUAL_ERROR', left_bc='LEFT_BOUNDARY_ERROR',
                      right_bc='RIGHT_BOUNDARY_ERROR', start_shape='INITIAL_CONDITION_ERROR')
    corrections = dict(equation='Recheck u_t - k*u_xx, using the actual k and L. Pair each spatial frequency with its own k*frequency**2 decay.',
                       start_shape='Recheck coefficients against the public f(x). Meet the stated relative L2 tolerance; the largest local deviation is diagnostic, not an extra acceptance threshold.')
    for side in ('left', 'right'):
        what = 'temperature u' if p[side] == 'dirichlet' else 'slope u_x'
        corrections[side+'_bc'] = f'Correct the {side} {p[side]} boundary: {what} must be zero. Choose modes compatible with both stated boundaries, including the actual rod length.'
    diagnostics = []
    for key, item in conditions.items():
        if item['status'] == 'pass':
            continue
        uncertain = item['status'] == 'inconclusive'
        diagnostics.append(diagnostic('NUMERICAL_VALIDATION_INCONCLUSIVE' if uncertain else categories[key],
            key, item, 'Evaluation was not reliable enough to decide this condition. Simplify the expression or increase verification resolution in a separately reviewed run.' if uncertain else corrections[key],
            location=item.get('location'), severity='warning' if uncertain else 'error'))
    # Diagnostics are optional. Their failure cannot turn a completed mathematical
    # result into a success or invent a cause for an observed condition failure.
    try:
        diagnostics.extend(basis_diagnostics(p, u, conditions))
        diagnostics.extend(decay_diagnostics(p, u, conditions))
        if profiles:
            diagnostics.extend(coefficient_diagnostics(p, profiles, conditions))
    except Exception as exc:
        diagnostics.append(diagnostic('DIAGNOSTIC_UNAVAILABLE', 'fourier_analysis',
            {'exception': type(exc).__name__}, 'Use the condition-level evidence; no modal root cause was established.', severity='info'))
    parts = {key: item['status'] == 'pass' for key, item in conditions.items()}
    accepted = all(parts.values())
    status = 'valid' if accepted else ('inconclusive' if any(item['status'] == 'inconclusive' for item in conditions.values()) else 'invalid')
    if accepted:
        diagnostics.append(diagnostic('VALID_SOLUTION', 'all', {'criteria': 'existing tolerances on bounded verification samples'},
                                      'All required sampled checks pass. This is numerical validation, not a general symbolic proof.', severity='info'))
    preserved = ', '.join(key for key, ok in parts.items() if ok)
    feedback = ('Preserve the passing conditions: '+preserved+'. ' if preserved and not accepted else '')
    feedback += ' '.join(item['category']+': '+item['correction'] +
                         (f" Evidence: normalized error={item['evidence']['error']:.4g}, limit={item['evidence']['threshold']:.4g}."
                          if item['evidence'].get('error') is not None and 'threshold' in item['evidence'] else '')
                         for item in diagnostics[:8])
    return dict(accepted=accepted, status=status, parts=parts, conditions=conditions,
                errors={key: conditions[key]['error'] for key in expressions},
                start_error=conditions['start_shape']['error'], diagnostics=diagnostics,
                feedback=feedback, validation_seed=validation_seed, points=len(X),
                shape_points=3201, protocol='enhanced-checker-v1')
