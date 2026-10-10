"""Derive time factors from the model's own eigenwaves, never from a known answer.

This tool changes only exponential decay. It cannot choose waves, compute Fourier
coefficients, repair missing modes, or recover a truncated answer.
"""
import sympy as sp
import pdecheck
from validation import checked_expression


def repair(answer, conductivity):
    """Return a proposal and an audit trail, or decline unsupported expressions."""
    try:
        u = pdecheck.parse(checked_expression(answer))
        x, t = pdecheck.x, pdecheck.t
        k = sp.sympify(conductivity)
        if k.free_symbols or k.is_positive is not True:
            return {'applied': False, 'reason': 'conductivity must be a positive constant'}
        terms = sp.Add.make_args(sp.expand_mul(u))
        if len(terms) > 32:
            return {'applied': False, 'reason': 'too many terms'}
        proposals, changes = [], []
        for term in terms:
            spatial, temporal = term.as_independent(t, as_Add=False)
            if temporal != 1:
                if (temporal.func != sp.exp or temporal.args[0].free_symbols - {t}
                        or sp.diff(temporal.args[0], t, 2) != 0
                        or temporal.args[0].subs(t, 0) != 0):
                    return {'applied': False, 'reason': 'unsupported time factor'}
            waves = spatial.atoms(sp.sin, sp.cos)
            if len(waves) != 1:
                return {'applied': False, 'reason': 'each term must contain one eigenwave'}
            wave = next(iter(waves))
            amplitude = sp.simplify(spatial / wave)
            mu = sp.diff(wave.args[0], x)
            if (amplitude.free_symbols or mu.free_symbols or mu.is_zero is not False
                    or sp.simplify(sp.diff(wave.args[0], x, 2)) != 0):
                return {'applied': False, 'reason': 'unsupported spatial wave'}
            ratio = sp.simplify(sp.diff(spatial, x, 2) / spatial)
            if ratio.free_symbols or ratio.is_negative is not True:
                return {'applied': False, 'reason': 'spatial factor is not a decaying eigenwave'}
            new_temporal = sp.exp(k * ratio * t)
            proposals.append(spatial * new_temporal)
            if sp.simplify(temporal - new_temporal) != 0:
                changes.append({'spatial': sp.sstr(spatial),
                                'old_time_factor': sp.sstr(temporal),
                                'new_time_factor': sp.sstr(new_temporal),
                                'derived_decay_rate': sp.sstr(-k * ratio)})
        if not changes:
            return {'applied': False, 'reason': 'decay already consistent'}
        proposed = sp.Add(*proposals)
        if sp.simplify(proposed.subs(t, 0) - u.subs(t, 0)) != 0:
            return {'applied': False, 'reason': 'initial expression was not preserved'}
        return {'applied': True, 'answer': 'u(x, t) = ' + sp.sstr(proposed),
                'changes': changes, 'initial_expression_preserved': True}
    except Exception as exc:
        return {'applied': False, 'reason': f'unsupported expression: {type(exc).__name__}'}
