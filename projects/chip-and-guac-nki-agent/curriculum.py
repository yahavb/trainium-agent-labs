"""Feedback-only adapter; canonical grade AST differs only in its candidate path."""

import ast
from contextlib import contextmanager
import copy
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import tempfile
import textwrap

LIMITS = {"K": 128, "M": 128, "N": 512}
PROTECTED = ("agent.py", "nkibench.py", "kernelbench.py",
             "reference_level1.py", "reference_level2.py",
             "reference_level3.py", "reference_level4.py")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_value(value):
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    if isinstance(value, set):
        return sorted(json_value(item) for item in value)
    return value


def write_json(path, data):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")


def process_guard():
    """Fail closed on active known controllers/graders; no environment reads."""
    if sys.platform != "linux":
        raise RuntimeError("Execution requires Linux /proc process checks")
    prohibited = {"agent.py", "offline_analyze.py", "curriculum_agent.py",
                  "parity_test.py", "nkibench.py", "kernelbench.py", "verify_kernel.py"}
    active = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            stat = (proc / "stat").read_text()
            if stat[stat.rfind(")") + 2:].split()[0] == "Z":
                continue
            args = (proc / "cmdline").read_bytes().decode(errors="replace").split("\0")
        except (FileNotFoundError, ProcessLookupError):
            continue
        # Only executable argument tokens, never shell command text.
        if any(Path(arg).name in prohibited for arg in args):
            active.append({"pid": int(proc.name), "scripts":
                           [Path(arg).name for arg in args if Path(arg).name in prohibited]})
    if active:
        raise RuntimeError(f"Active agent/grader detected: {active}")
    return {"active_agents_or_graders": active}


@contextmanager
def exclusive_execution():
    import fcntl
    with Path("/tmp/nki_offline_analyze.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another NKI experiment holds the grading lock") from exc
        process_guard()
        yield


def private_grade(agent, path):
    """Compile the installed function with exactly one verified AST substitution."""
    canonical = getattr(agent, "_canonical_grade", agent.grade)
    tree = ast.parse(textwrap.dedent(inspect.getsource(canonical)))
    original = ast.dump(tree, include_attributes=False)
    assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "path" for t in node.targets)]
    if len(assignments) != 1:
        raise RuntimeError("Canonical candidate path assignment changed")
    target = assignments[0]
    expected = ast.parse('path = f"/tmp/_agent_level{level}.py"').body[0].value
    if ast.dump(target.value) != ast.dump(expected):
        raise RuntimeError("Unexpected canonical candidate path expression")
    saved = target.value
    target.value = ast.copy_location(ast.Constant(str(path)), saved)
    compiled = compile(ast.fix_missing_locations(tree), inspect.getsourcefile(canonical), "exec")
    target.value = saved
    if ast.dump(tree, include_attributes=False) != original:
        raise RuntimeError("Non-path AST delta detected")
    namespace = dict(canonical.__globals__)
    exec(compiled, namespace)
    return namespace["grade"]


def select_feedback(result, level, failures, passed, bench, policy):
    """Reuse enriched canonical messages, without changing or rerunning a gate."""
    reward, parts, feedback = result
    details = {"canonical_feedback": feedback, "selected_shape": None,
               "changed": False, "failures": [list(f) for f in failures]}
    if policy not in ("official", "curriculum"):
        raise ValueError("Unknown feedback policy")
    if level != 4 or len(failures) < 2:
        return result, details
    cases = bench.LEVELS[level]["shapes"]
    labels = {bench.label(case, level): (i, case) for i, case in enumerate(cases)}
    first_label, first_message = failures[0]
    expected = f"{passed} of {len(cases)} shapes passed. On {first_label}: {first_message}"
    # Early returns (e.g. NkiMissing) must never be rewritten.
    if feedback != expected:
        return result, details
    ranked = []
    for label, message in failures:
        index, case = labels[label]
        exceeded = [dim for dim, limit in LIMITS.items() if case[dim] > limit]
        ranked.append((len(exceeded), index, label, message, exceeded))
    choice = min(ranked, key=lambda item: item[:2]) if policy == "curriculum" else ranked[0]
    count, index, label, message, exceeded = choice
    selected = f"{passed} of {len(cases)} shapes passed. On {label}: {message}"
    details.update(selected_shape=dict(cases[index]), selected_shape_index=index + 1,
                   exceeded_dimensions=exceeded, exceeded_count=count,
                   changed=selected != feedback)
    return (reward, parts, selected), details


def evaluate(agent, source, level, policy="curriculum"):
    """One canonical simulation pass, observed delegates, private candidate path."""
    bench = agent.nkibench
    shapes, originals, final = [], {}, {}
    current = None
    names = ("make_inputs", "simulate_and_count", "check_inputs_untouched",
             "describe_mismatch", "check_traffic_bar")
    with tempfile.TemporaryDirectory(prefix="nki-curriculum-") as folder:
        path = Path(folder) / "candidate.py"
        grade = private_grade(agent, path)

        def delegate(name, function):
            def wrapped(*args, **kwargs):
                nonlocal current
                if name == "make_inputs":
                    current = {"shape_index": len(shapes) + 1, "case": dict(args[0]),
                               "checks": {}, "simulation": "not reached", "counted": None,
                               "exception": None}
                    shapes.append(current)
                try:
                    result = function(*args, **kwargs)
                except BaseException as exc:
                    if name == "simulate_and_count":
                        current["simulation"] = "raised"
                        current["exception"] = {"type": type(exc).__name__, "message": str(exc)}
                        cursor = exc.__traceback__
                        while cursor:
                            if cursor.tb_frame.f_code is function.__code__:
                                counter = cursor.tb_frame.f_locals.get("counter")
                                if counter is not None:
                                    current["counted"] = json_value(copy.deepcopy(counter))
                            cursor = cursor.tb_next
                    raise
                if name == "simulate_and_count":
                    current["simulation"] = "returned"
                    current["counted"] = json_value(copy.deepcopy(result[1]))
                elif name in names[2:]:
                    gate = {"check_inputs_untouched": "mutation", "describe_mismatch": "numerical",
                            "check_traffic_bar": "traffic"}[name]
                    current["checks"][gate] = {"evaluated": True, "passed": not bool(result),
                                               "message": result}
                return result
            return wrapped

        def trace(frame, event, arg):
            if frame.f_code is grade.__code__:
                if event == "return":
                    final.update(failures=list(frame.f_locals.get("failures", [])),
                                 passed=frame.f_locals.get("passed", 0))
                return trace
            return None

        previous = sys.gettrace()
        if previous is not None:
            raise RuntimeError("An existing trace hook would be displaced")
        try:
            for name in names:
                originals[name] = getattr(bench, name)
                setattr(bench, name, delegate(name, originals[name]))
            sys.settrace(trace)
            canonical = grade(source, level)
        finally:
            sys.settrace(previous)
            for name, function in originals.items():
                setattr(bench, name, function)
        # Candidate path isolation is intentionally not a feedback intervention.
        logical_path = f"/tmp/_agent_level{level}.py"
        canonical = (canonical[0], canonical[1], canonical[2].replace(str(path), logical_path))
        failures = [(lbl, msg.replace(str(path), logical_path)) for lbl, msg in final.get("failures", [])]
        selected, details = select_feedback(canonical, level, failures, final.get("passed", 0), bench, policy)
        failure_labels = {label for label, _ in failures}
        for item in shapes:
            item["passed"] = (item["simulation"] == "returned"
                              and bench.label(item["case"], level) not in failure_labels)
        details.update(shapes=shapes, canonical_reward=canonical[0], canonical_parts=canonical[1],
                       candidate_path=str(path), ast_delta="candidate path assignment only")
        return selected, details
