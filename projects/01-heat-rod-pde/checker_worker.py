"""Optional worker. Run via checker_enhanced_runtime; never evaluates Python input."""
import json
import sys


def main():
    # Portable wall deadline is in the parent. POSIX additionally caps CPU/memory.
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (7, 7))
        resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
    except (ImportError, ValueError, OSError):
        pass
    from checker_enhanced_runtime import clean_json, failure, no_score
    from checker_syntax import ExpressionError, expression_source, parse_expression, x, t
    from checker_scoring import _check_original
    import math

    def emit(kind, result):
        print(json.dumps({'kind': kind, 'result': clean_json(result)}, allow_nan=False), flush=True)

    request = json.loads(sys.stdin.read(100000))
    mode = request['mode']
    try:
        p = request['problem']
        for key in ('L', 'k', 'f'):
            p[key] = parse_expression(p[key])
        if p['L'].free_symbols or p['k'].free_symbols or p['f'].free_symbols - {x}:
            raise ValueError('Unsupported free symbols in the public PDE definition.')
        if any(not math.isfinite(float(p[key])) or float(p[key]) <= 0 for key in ('L', 'k', 'tol')):
            raise ValueError('Length, conductivity, and initial tolerance must be positive and finite.')
        if any(p[side] not in ('dirichlet', 'neumann') for side in ('left', 'right')):
            raise ValueError('Only the project homogeneous Dirichlet/Neumann boundary schema is supported.')
        source = expression_source(request['answer'])
        u = parse_expression(source, allow_sum=p['allow_symbolic_sum'])
        unknown = u.free_symbols - {x, t}
        if unknown:
            raise ExpressionError('UNBOUND_SYMBOL', 'Evaluate coefficients and expand or bind each index; do not delete an unbound index.',
                                  symbols=sorted(map(str, unknown)))
    except Exception as exc:
        category = exc.category if isinstance(exc, ExpressionError) else 'NUMERICAL_VALIDATION_INCONCLUSIVE'
        bad = failure(str(exc), category, getattr(exc, 'evidence', {}))
        if mode != 'verify':
            emit('original', no_score(bad))
        if mode != 'original':
            emit('validation', bad)
        return
    if mode != 'verify':
        try:
            from checker_math import basis
            import sympy as sp
            # Restore legacy modal-feedback helpers from public BCs only.
            p['basis'] = lambda n: basis(p, n)
            if p['left'] != p['right']:
                p['lam'] = lambda n: (2*n-1)*sp.pi/(2*p['L'])
            elif p['left'] == 'neumann':
                p['lam'] = lambda n: (n-1)*sp.pi/p['L']
            else:
                p['lam'] = lambda n: n*sp.pi/p['L']
            # Sentinel used only for historical feedback. No hidden field is read.
            p['exact'] = None
            original = _check_original(p, source, request['n_points'], _expression=u)
            original.update(original_reward=original['reward'] if original['parts'] else None, scored=bool(original['parts']),
                            score_protocol='c3c16f6-numerical', status='original_scored')
        except Exception as exc:
            original = no_score(failure('Original scoring could not complete.', evidence={'exception': type(exc).__name__}))
        emit('original', original)  # Preserve score even if enhancement times out.
    if mode != 'original':
        core_emitted = False

        def checkpoint(core):
            nonlocal core_emitted
            emit('validation_core', core)
            core_emitted = True

        try:
            from checker_math import verify_expression
            result = verify_expression(p, u, request['validation_seed'],
                                       on_core=checkpoint)
        except Exception as exc:
            result = failure('Numerical or symbolic validation could not complete.',
                             evidence={'exception': type(exc).__name__, 'detail': str(exc)[:160]})
            if core_emitted:
                emit('diagnostic_failure', result)
                return
        emit('validation', result)


if __name__ == '__main__':
    main()
