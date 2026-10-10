"""Lower bounded model-proposed operation graphs to NKI, never execute model Python."""

import re

import numpy as np

from math_tasks import TASKS


OPERATIONS = ("mul", "add", "sub", "neg", "mul_neg", "sum", "serial_sum", "copy")


def lower(task, program):
    if task not in TASKS or not isinstance(program, dict):
        raise ValueError("Unknown task or invalid program")
    operations = program.get("operations")
    if not isinstance(operations, list) or not 1 <= len(operations) <= 16:
        raise ValueError("Supply 1..16 operations")
    shapes = ({"displacement": (64, 8), "velocity": (64, 8),
               "stiffness": (64, 1), "damping": (64, 1)} if task == "spring" else {"forces": (64, 8)})
    lines = ["import nki", "import nki.isa as nisa", "import nki.language as nl", "", "@nki.jit",
             "def calculate(" + ", ".join(TASKS[task]["input_names"]) + "):"]
    names = {name: "input_" + name for name in shapes}
    for name, shape in shapes.items():
        lines += [f"    {names[name]} = nl.ndarray({shape}, dtype=nl.float32, buffer=nl.sbuf)",
                  f"    nisa.dma_copy(dst={names[name]}, src={name})"]
    for node in operations:
        if not isinstance(node, dict) or set(node) - {"id", "op", "inputs", "engine"}:
            raise ValueError("Invalid operation fields")
        identifier, op = node.get("id"), node.get("op")
        operands, engine = node.get("inputs"), node.get("engine", "auto")
        if not isinstance(identifier, str) or not re.fullmatch(r"v[0-9]{1,2}", identifier) or identifier in names:
            raise ValueError("Operation IDs must be distinct v0..v99")
        if op not in OPERATIONS or engine not in ("auto", "vector"):
            raise ValueError("Unsupported operation or engine")
        arity = 2 if op in ("mul", "add", "sub", "mul_neg") else 1
        if not isinstance(operands, list) or len(operands) != arity or any(
                not isinstance(x, str) or x not in shapes for x in operands):
            raise ValueError("Operands must name previously defined values")
        shape = shapes[operands[0]]
        args = [names[x] for x in operands]
        if arity == 2 and shapes[operands[1]] not in (shape, (shape[0], 1)):
            raise ValueError("Unsupported broadcast layout")
        if op == "mul_neg" and shapes[operands[1]] != (shape[0], 1):
            raise ValueError("Fused scalar operand must have shape (64, 1)")
        if op in ("sum", "serial_sum"):
            shape = (shape[0], 1)
        shapes[identifier] = shape
        names[identifier] = identifier
        lines.append(f"    {identifier} = nl.ndarray({shape}, dtype=nl.float32, buffer=nl.sbuf)")
        placement = ", engine=nisa.engine.vector" if engine == "vector" else ""
        if op in ("mul", "add", "sub"):
            operator = {"mul": "multiply", "add": "add", "sub": "subtract"}[op]
            if shapes[operands[1]] == shapes[operands[0]]:
                call = f"nisa.tensor_tensor(dst={identifier}, data1={args[0]}, data2={args[1]}, op=nl.{operator}{placement})"
            else:
                call = f"nisa.tensor_scalar(dst={identifier}, data={args[0]}, op0=nl.{operator}, operand0={args[1]}{placement})"
            lines.append("    " + call)
        elif op == "mul_neg":
            lines.append(f"    nisa.tensor_scalar(dst={identifier}, data={args[0]}, op0=nl.multiply, "
                         f"operand0={args[1]}, op1=nl.multiply, operand1=-1.0{placement})")
        elif op == "neg":
            lines.append(f"    nisa.tensor_scalar(dst={identifier}, data={args[0]}, op0=nl.multiply, operand0=-1.0{placement})")
        elif op == "sum":
            lines.append(f"    nisa.tensor_reduce(dst={identifier}, data={args[0]}, op=nl.add, axis=(1,), keepdims=True)")
        elif op == "serial_sum":
            width = shapes[operands[0]][1]
            lines += [f"    nisa.tensor_copy(dst={identifier}, src={args[0]}[:, 0:1])",
                      f"    for i in nl.static_range(1, {width}):",
                      f"        nisa.tensor_tensor(dst={identifier}, data1={identifier}, data2={args[0]}[:, i:i + 1], op=nl.add{placement})"]
        else:
            lines.append(f"    nisa.tensor_copy(dst={identifier}, src={args[0]})")
    result = program.get("result")
    expected = (64, 8) if task == "spring" else (64, 1)
    if not isinstance(result, str) or result not in names or shapes[result] != expected:
        raise ValueError("Result must name a value with the task's output shape")
    lines += [f"    output = nl.ndarray({expected}, dtype=nl.float32, buffer=nl.shared_hbm)",
              f"    nisa.dma_copy(dst=output, src={names[result]})", "    return output"]
    return "\n".join(lines) + "\n"


def evaluate(task, program, inputs):
    lower(task, program)
    values = dict(zip(TASKS[task]["input_names"], inputs))
    for node in program["operations"]:
        op = node["op"]
        a = values[node["inputs"][0]]
        b = values[node["inputs"][1]] if len(node["inputs"]) == 2 else None
        if op == "mul": value = a * b
        elif op == "add": value = a + b
        elif op == "sub": value = a - b
        elif op == "neg": value = -a
        elif op == "mul_neg": value = -(a * b)
        elif op == "sum": value = a.sum(axis=1, keepdims=True, dtype=np.float32)
        elif op == "serial_sum":
            value = a[:, :1].copy()
            for i in range(1, a.shape[1]): value += a[:, i:i + 1]
        else: value = a.copy()
        values[node["id"]] = value
    return values[program["result"]].copy()
