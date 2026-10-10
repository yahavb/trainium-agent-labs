"""Bounded mathematical syntax construction, without eval/parse_expr on model text.

The full checking entrypoints run this parser in an isolated, time-limited process.
This standalone helper has structural limits, not a process-level time guarantee.
"""
import ast
import io
import math
import re
import tokenize
from decimal import Decimal, InvalidOperation

import sympy as sp

x, t = sp.symbols('x t', real=True)
MAX_SOURCE = 12000
MAX_NODES = 1800
MAX_DEPTH = 48
MAX_SUM_TERMS = 64
FUNCTIONS = {name: getattr(sp, name) for name in
             ('exp', 'sin', 'cos', 'sqrt', 'sinh', 'cosh', 'tan')}
CONSTRUCTORS = {'Integer', 'Float', 'Rational', 'Symbol'}
SPECIAL = {'Sum', 'Integral', 'Derivative'}
CONSTANTS = {'x': x, 't': t, 'pi': sp.pi, 'E': sp.E}
NUMBER_TEXT = re.compile(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\Z')


class ExpressionError(ValueError):
    def __init__(self, category, message, **evidence):
        super().__init__(message)
        self.category, self.evidence = category, evidence


def expression_source(answer):
    text = re.sub(r'<think>.*?</think>', '', answer or '', flags=re.S)
    hits = re.findall(r'u\s*\(\s*x\s*,\s*t\s*\)\s*=\s*(.+)', text)
    source = hits[-1].strip().strip('`$ ').rstrip('.') if hits else text.strip()
    if not source:
        raise ExpressionError('INCOMPLETE_EXPRESSION', 'Provide a complete u(x,t) = expression line.')
    if not hits and re.search(r'^\s*COMPUTE:', text, re.M):
        raise ExpressionError('INCOMPLETE_EXPRESSION', 'The response requests a calculation but contains no assembled solution.')
    return source.replace('^', '**').replace('\\pi', 'pi').replace('π', 'pi')


def _tokens(source):
    if len(source) > MAX_SOURCE:
        raise ExpressionError('NUMERICAL_VALIDATION_INCONCLUSIVE', 'Expression exceeds the source-size budget.', limit=MAX_SOURCE)
    try:
        raw = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, IndentationError) as exc:
        raise ExpressionError('INCOMPLETE_EXPRESSION', 'Unfinished parentheses, literal or expression.') from exc
    allowed_ops = {'+', '-', '*', '/', '**', '(', ')', ','}
    result = []
    for token in raw:
        if token.type in (tokenize.ENDMARKER, tokenize.NEWLINE, tokenize.NL):
            continue
        if token.type not in (tokenize.NAME, tokenize.NUMBER, tokenize.OP, tokenize.STRING):
            raise ExpressionError('PARSE_ERROR', 'Only a mathematical scalar expression is supported.')
        if token.type == tokenize.OP and token.string not in allowed_ops:
            raise ExpressionError('PARSE_ERROR', 'Attributes, indexing, assignments and non-arithmetic operators are forbidden.')
        if token.type == tokenize.NAME and ('__' in token.string or token.string.startswith('_')):
            raise ExpressionError('PARSE_ERROR', 'Private Python names are forbidden.')
        # Preserve conventional implicit products such as 2 sin(pi x), x(t+1), and xt.
        if token.type == tokenize.NAME and len(token.string) > 1 and set(token.string) <= {'x', 't'}:
            for char in token.string:
                result.append((tokenize.NAME, char))
        else:
            result.append((token.type, token.string))
    output = []
    for index, (kind, value) in enumerate(result):
        if index:
            prev_kind, prev = result[index-1]
            ends = prev_kind in (tokenize.NAME, tokenize.NUMBER) or prev == ')'
            begins = kind in (tokenize.NAME, tokenize.NUMBER) or value == '('
            call = prev_kind == tokenize.NAME and value == '(' and prev in set(FUNCTIONS) | CONSTRUCTORS | SPECIAL
            if ends and begins and not call:
                output.append('*')
        output.append(value)
    return ' '.join(output)


def parse_expression(source, *, allow_sum=False):
    text = _tokens(source.strip())
    try:
        tree = ast.parse(text, mode='eval')
    except (SyntaxError, RecursionError) as exc:
        category = 'INCOMPLETE_EXPRESSION' if not text or text.rstrip().endswith(('+', '-', '*', '/', '(')) else 'PARSE_ERROR'
        raise ExpressionError(category, 'Return one complete arithmetic expression in supported Python/SymPy syntax.') from exc
    if sum(1 for _ in ast.walk(tree)) > MAX_NODES:
        raise ExpressionError('NUMERICAL_VALIDATION_INCONCLUSIVE', 'Expression exceeds the syntax-node budget.', limit=MAX_NODES)
    expansion_terms = 0

    def literal(node, depth=0):
        if depth > MAX_DEPTH:
            raise ExpressionError('NUMERICAL_VALIDATION_INCONCLUSIVE', 'Numeric literal nesting exceeds the evaluation budget.')
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            value = literal(node.operand, depth+1)
            if isinstance(value, str):
                value = value.lstrip('+')
                return (value[1:] if value.startswith('-') else '-'+value) if isinstance(node.op, ast.USub) else value
            return -value if isinstance(node.op, ast.USub) else value
        if not isinstance(node, ast.Constant) or type(node.value) not in (int, float, str):
            raise ExpressionError('PARSE_ERROR', 'Numeric constructors require literal arguments.')
        value = node.value
        if isinstance(value, str):
            if len(value) > 128 or not NUMBER_TEXT.fullmatch(value):
                raise ExpressionError('PARSE_ERROR', 'Only bounded numeric strings are allowed in numeric constructors.')
            try:
                decimal = Decimal(value)
                if not decimal.is_finite() or decimal.adjusted() > 12 or decimal.copy_abs() > Decimal('1e12') or (decimal != 0 and decimal.adjusted() < -10000):
                    raise ExpressionError('NUMERICAL_VALIDATION_INCONCLUSIVE', 'Numeric constructor literal exceeds the evaluation budget.')
            except InvalidOperation as exc:
                raise ExpressionError('PARSE_ERROR', 'Invalid numeric constructor literal.') from exc
        elif abs(value) > 10**12 or not math.isfinite(value):
            raise ExpressionError('NUMERICAL_VALIDATION_INCONCLUSIVE', 'Numeric literal exceeds the evaluation budget.')
        return value

    def build(node, depth=0):
        nonlocal expansion_terms
        if depth > MAX_DEPTH:
            raise ExpressionError('NUMERICAL_VALIDATION_INCONCLUSIVE', 'Expression nesting exceeds the evaluation budget.')
        child = lambda item: build(item, depth+1)
        if isinstance(node, ast.Name):
            if node.id in {'nan', 'NaN', 'oo', 'Infinity', 'inf', 'I'}:
                raise ExpressionError('NUMERICAL_VALIDATION_INCONCLUSIVE', 'Temperature expressions must be finite and real.', name=node.id)
            return CONSTANTS.get(node.id, sp.Symbol(node.id))
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            literal(node)
            token = ast.get_source_segment(text, node)
            return sp.Integer(token) if type(node.value) is int else sp.Float(token)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            value = child(node.operand)
            return sp.Mul(-1, value, evaluate=False) if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.BinOp):
            left, right = child(node.left), child(node.right)
            if isinstance(node.op, ast.Add): return sp.Add(left, right, evaluate=False)
            if isinstance(node.op, ast.Sub): return sp.Add(left, sp.Mul(-1, right, evaluate=False), evaluate=False)
            if isinstance(node.op, ast.Mult): return sp.Mul(left, right, evaluate=False)
            if isinstance(node.op, ast.Div): return sp.Mul(left, sp.Pow(right, -1, evaluate=False), evaluate=False)
            if isinstance(node.op, ast.Pow):
                if not right.free_symbols:
                    # Inspect the unevaluated tree before converting to a scalar;
                    # a numeric power tower is not a bounded literal exponent.
                    unsafe_powers = any(power.exp != -1 or not power.base.is_Number for power in right.atoms(sp.Pow))
                    if unsafe_powers or sp.count_ops(right) > 16:
                        raise ExpressionError('NUMERICAL_VALIDATION_INCONCLUSIVE', 'Numeric exponent expression exceeds the evaluation budget.')
                    exponent = float(right)
                    if not math.isfinite(exponent) or abs(exponent) > 10000:
                        raise ExpressionError('NUMERICAL_VALIDATION_INCONCLUSIVE', 'Numeric exponent exceeds the evaluation budget.')
                return sp.Pow(left, right, evaluate=False)
            raise ExpressionError('PARSE_ERROR', 'Unsupported arithmetic operator.')
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.keywords:
            raise ExpressionError('PARSE_ERROR', 'Only scalar arithmetic and named mathematical functions are supported.')
        name, args = node.func.id, node.args
        if name in FUNCTIONS:
            if len(args) != 1:
                raise ExpressionError('PARSE_ERROR', name+' needs one argument.')
            return FUNCTIONS[name](child(args[0]), evaluate=False)
        if name == 'Symbol':
            if len(args) != 1 or not isinstance(args[0], ast.Constant) or not isinstance(args[0].value, str):
                raise ExpressionError('PARSE_ERROR', 'Symbol needs one literal identifier.')
            value = args[0].value
            if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{0,63}', value) or '__' in value:
                raise ExpressionError('PARSE_ERROR', 'Unsupported symbol identifier.')
            return CONSTANTS.get(value, sp.Symbol(value))
        if name in {'Integer', 'Float', 'Rational'}:
            if len(args) not in ({1, 2} if name == 'Rational' else {1}):
                raise ExpressionError('PARSE_ERROR', 'Unsupported numeric constructor arity.')
            values = [literal(arg) for arg in args]
            if name == 'Integer' and not re.fullmatch(r'[+-]?\d+', str(values[0])):
                raise ExpressionError('PARSE_ERROR', 'Integer requires an integer literal.')
            return getattr(sp, name)(*values)
        if name in {'Integral', 'Derivative'}:
            raise ExpressionError('INCOMPLETE_EXPRESSION', 'Evaluate coefficients and derivatives before submitting the solution.', construct=name)
        if name == 'Sum':
            if len(args) != 2 or not isinstance(args[1], ast.Tuple) or len(args[1].elts) != 3:
                raise ExpressionError('INCOMPLETE_EXPRESSION', 'A Sum needs an explicit (index, lower, upper) range.')
            index, lower, upper = args[1].elts
            if not isinstance(index, ast.Name) or index.id in CONSTANTS:
                raise ExpressionError('UNBOUND_SYMBOL', 'Use a distinct bound index in the finite Sum.')
            lo, hi = literal(lower), literal(upper)
            if type(lo) is not int or type(hi) is not int:
                raise ExpressionError('NUMERICAL_VALIDATION_INCONCLUSIVE', 'Only finite integer sum bounds can be verified within this budget.')
            if hi < lo:
                raise ExpressionError('NUMERICAL_VALIDATION_INCONCLUSIVE', 'Reversed Sum bounds require explicit expansion to preserve their mathematical convention.')
            count = max(0, hi-lo+1)
            expansion_terms += count
            if expansion_terms > MAX_SUM_TERMS:
                raise ExpressionError('NUMERICAL_VALIDATION_INCONCLUSIVE', 'Finite Sum expansion exceeds the term budget.', limit=MAX_SUM_TERMS)
            if not allow_sum:
                raise ExpressionError('INCOMPLETE_EXPRESSION', 'This problem requires explicit terms; a bounded Sum is supported only when allow_symbolic_sum is enabled.')
            body = child(args[0])
            if count * sum(1 for _ in sp.preorder_traversal(body)) > MAX_NODES:
                raise ExpressionError('NUMERICAL_VALIDATION_INCONCLUSIVE', 'Expanded Sum exceeds the expression budget.')
            terms = [body.subs(sp.Symbol(index.id), sp.Integer(i)) for i in range(lo, hi+1)]
            return sp.Add(*terms)
        raise ExpressionError('PARSE_ERROR', 'Unsupported mathematical function: '+name)

    try:
        expression = build(tree.body)
    except ExpressionError:
        raise
    except Exception as exc:
        raise ExpressionError('PARSE_ERROR', 'Could not construct the mathematical expression.', exception=type(exc).__name__) from exc
    if not isinstance(expression, sp.Expr):
        raise ExpressionError('PARSE_ERROR', 'The answer must be one scalar expression.')
    if expression.has(sp.nan, sp.zoo, sp.oo, -sp.oo):
        raise ExpressionError('NUMERICAL_VALIDATION_INCONCLUSIVE', 'The expression contains an undefined or infinite value.')
    return expression
