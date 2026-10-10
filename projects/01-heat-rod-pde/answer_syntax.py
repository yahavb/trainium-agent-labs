"""Syntax limits for the optional decay tool; never a scoring gate."""
import ast
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

