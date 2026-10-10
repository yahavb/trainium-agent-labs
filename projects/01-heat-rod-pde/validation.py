"""Opt-in enhanced checks; does not replace master's public Checker interfaces."""
from checker_enhanced_runtime import run_check
from checker_syntax import ExpressionError, expression_source, parse_expression, x, t


def checked_expression(answer):
    """Compatibility helper with structural limits. Use grade for isolation."""
    source = expression_source(answer)
    expression = parse_expression(source)
    unknown = expression.free_symbols - {x, t}
    if unknown:
        raise ExpressionError('UNBOUND_SYMBOL', 'Evaluate coefficients and expand or bind every index.',
                              symbols=sorted(map(str, unknown)))
    return source


def verify(problem, answer, validation_seed=9173):
    return run_check('verify', problem, answer, validation_seed=validation_seed)


def grade(problem, answer):
    return run_check('grade', problem, answer)
