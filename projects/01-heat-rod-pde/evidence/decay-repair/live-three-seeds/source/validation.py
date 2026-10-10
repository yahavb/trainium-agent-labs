"""Held-out physics checks. Does not inspect exact answers or series_answer."""
import ast
import hashlib
import math
import re
import numpy as np
import sympy as sp
import pdecheck


def checked_expression(answer):
    expression = pdecheck.extract(answer) if '=' in answer else answer.strip()
    if not expression or len(expression) > 12000:
        raise ValueError('Missing or oversized answer expression')
    tree = ast.parse(expression, mode='eval')
    names = {'x', 't', 'pi', 'E', 'exp', 'sin', 'cos', 'sqrt', 'sinh', 'cosh', 'tan'}
    nodes = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name,
             ast.Load, ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div,
             ast.Pow, ast.UAdd, ast.USub)
    if sum(1 for _ in ast.walk(tree)) > 2500:
        raise ValueError('Expression is too complex')
    for node in ast.walk(tree):
        if not isinstance(node, nodes):
            raise ValueError('Only arithmetic expressions and allowed functions are accepted')
        if isinstance(node, ast.Name) and node.id not in names:
            raise ValueError('Unknown symbol: ' + node.id)
        if isinstance(node, ast.Call) and (not isinstance(node.func, ast.Name)
                                          or node.func.id not in names - {'x', 't', 'pi', 'E'}
                                          or node.keywords):
            raise ValueError('Unsupported function call')
        if isinstance(node, ast.Constant) and (type(node.value) not in (int, float)
                                               or abs(node.value) > 1e6):
            raise ValueError('Unsupported numeric constant')
    return expression


def verify(problem, answer, validation_seed=9173):
    """Separate random points, near-zero times, and a refined initial-shape grid."""
    try:
        expression = checked_expression(answer)
        u = pdecheck.parse(expression)
        x, t = pdecheck.x, pdecheck.t
        L, k = float(problem['L']), problem['k']
        digest = hashlib.sha256(f"{problem['name']}:{problem['seed']}:{validation_seed}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], 'big'))
        X = np.r_[rng.uniform(0, L, 257), np.linspace(0, L, 129)[1:-1]]
        T = np.r_[rng.uniform(0, .1 * L * L, 257),
                  np.geomspace(1e-9 * L * L, .1 * L * L, 127)]
        grid = np.linspace(0, L, 1601)
        f = pdecheck._num(problem['f'], grid, grid * 0)
        shape = pdecheck._num(u, grid, grid * 0)
        scale = float(np.max(np.abs(f))) or 1.0
        residual = pdecheck._num(sp.diff(u, t) - k * sp.diff(u, x, 2), X, T)
        ux = sp.diff(u, x)
        def boundary(side, at):
            expr = u if problem[side] == 'dirichlet' else ux * L
            return pdecheck._num(expr, at + T * 0, T)
        def error(values):
            return float(np.max(np.abs(values)) / scale) if np.all(np.isfinite(values)) else None
        errors = {'equation': error(residual), 'left_bc': error(boundary('left', 0)),
                  'right_bc': error(boundary('right', L))}
        denom = float(np.sqrt(np.trapezoid(f * f, grid))) or 1.0
        start_error = (float(np.sqrt(np.trapezoid((shape - f) ** 2, grid))) / denom
                       if np.all(np.isfinite(shape)) else None)
        parts = {name: value is not None and value < 1e-6 for name, value in errors.items()}
        parts['start_shape'] = start_error is not None and start_error < problem['tol']
        return {'accepted': all(parts.values()), 'parts': parts, 'errors': errors,
                'start_error': start_error, 'validation_seed': validation_seed,
                'points': len(X), 'shape_points': len(grid)}
    except Exception as exc:
        return {'accepted': False, 'parts': {}, 'error': f'{type(exc).__name__}: {exc}'}


def grade(problem, answer):
    try:
        checked_expression(answer)
        result = pdecheck.check(problem, answer)
    except Exception as exc:
        return {'reward': 0.0, 'parts': {}, 'expr': None, 'start_error': None,
                'feedback': f'Invalid answer: {exc}', 'validation': {'accepted': False},
                'original_reward': None, 'original_parts': {},
                'validation_status': 'syntax_rejected'}
    # Preserve the original checker outcome before a held-out check can lower it.
    # This lets reports compare agents using the same scoring rule.
    result['original_reward'] = result['reward']
    result['original_parts'] = dict(result['parts'])
    result['validation_status'] = 'not_run'
    if result['reward'] == 1.0:
        result['validation'] = verify(problem, answer)
        result['validation_status'] = ('passed' if result['validation']['accepted'] else 'failed')
        if not result['validation']['accepted']:
            parts = result['validation']['parts']
            result['reward'] = round(sum(pdecheck.WEIGHTS[n] for n, ok in parts.items() if ok), 3)
            result['feedback'] = ('Independent validation failed: ' +
                                  ', '.join(n for n, ok in parts.items() if not ok) +
                                  '. Recheck early-time behavior and the initial-shape approximation.')
    if result.get('start_error') is not None and not math.isfinite(result['start_error']):
        result['start_error'] = None
    return result
