"""CHECKS: compile (parse + lint), text scan (nkibench.check_rules), run (nki.simulate), correctness.

The reward is computed exactly as agent.grade computes it (same weights, same proration by shapes
passed), so the two systems' numbers compare. Two deliberate differences:
  - lint runs before the simulator, and a lint finding scores what a run failure scores (parses +
    rules = 0.3), so lint changes the message, not the reward;
  - levels 5-7: zero counted HBM bytes is a failure (NO HBM TRAFFIC COUNTED). nkibench passes it, which
    let `from nki.isa import dma_copy` slip under the traffic bar.

Checks run in worker processes: the byte counter swaps nisa.dma_copy module-wide while it simulates,
so two checks in one process would mix their counts.
"""

import multiprocessing as mp
import os
import sys
import tempfile
import threading
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import nkibench                      # noqa: E402
from agents2 import errors, lint     # noqa: E402

WEIGHTS = dict(parses=0.1, rules=0.2, runs=0.2, correct=0.5)   # agent.WEIGHTS


def _score(parts, passed=0, total=1):
    return round(sum(WEIGHTS[k] for k in ("parses", "rules", "runs") if parts[k])
                 + (WEIGHTS["correct"] if parts["correct"] else WEIGHTS["correct"] * passed / total), 6)


def _kernel_line(exc, path):
    """The line of the candidate file where the exception passed through, if it did."""
    frames = [f.lineno for f in traceback.extract_tb(exc.__traceback__) if f.filename == path]
    return frames[-1] if frames else None


def _corner(got, want, n=4):
    """A small top-left corner of got and expected, as nested lists, for the debugger."""
    def two_d(x):
        x = np.asarray(x, np.float64)
        return x.reshape(x.shape[0], -1) if x.ndim >= 2 else x.reshape(1, -1)
    g, w = two_d(got), two_d(want)
    if g.shape[0] < 1 or w.shape[0] < 1:
        return None
    r, c = min(n, g.shape[0], w.shape[0]), min(n, g.shape[1], w.shape[1])
    fmt = lambda a: [[float(f"{v:.4g}") for v in row[:c]] for row in a[:r]]
    return dict(got=fmt(g), want=fmt(w))


def run_checks(source, level):
    """Every check, in order, in this process. Returns a plain dict (it crosses a process boundary)."""
    spec = nkibench.LEVELS[level]
    total = len(spec["shapes"])
    parts = dict(parses=False, rules=False, runs=False, correct=False)
    r = dict(stage="empty", parts=parts, reward=0.0, passed=0, total=total, correct=False,
             kind="EMPTY", error="No code came back.", line=None, shape=None, corner=None,
             shapes=[], lint=[], warnings=[], seconds=0.0)
    t0 = time.perf_counter()
    if not source.strip():
        return r
    try:
        compile(source, "<candidate>", "exec")
    except SyntaxError as e:
        r.update(stage="parse", kind="PARSE", error=f"The code does not parse: {e.msg} on line "
                 f"{e.lineno}.", line=e.lineno)
        return r
    parts["parses"] = True

    violations = nkibench.check_rules(source, level)
    if violations:
        err = " ".join(violations)
        if any("no function named" in v for v in violations):
            import inspect
            args = ", ".join(inspect.signature(spec["ref"]).parameters)
            err += (f" Start the function with exactly this line:  def {spec['entry']}({args}):  "
                    f"and put @nki.jit on the line above it.")
        r.update(stage="rules", kind="RULES", error=err, reward=_score(parts))
        return r
    parts["rules"] = True

    found = lint.lint(source, traffic_level=bool(spec.get("max_waste")))
    r["lint"] = found
    if found:
        f = found[0]
        # ALIAS keeps its own kind: as TRAFFIC it went to the reviewer, which treats a kernel as
        # having computed the right values, and this one has not run.
        r.update(stage="lint", kind=f["kind"], error=f["msg"], line=f["line"], reward=_score(parts))
        return r

    fd, path = tempfile.mkstemp(prefix=f"_agent2_level{level}_", suffix=".py")
    with os.fdopen(fd, "w") as fh:
        fh.write(source)
    try:
        try:
            kernel = nkibench.load_kernel(path, spec["entry"])
        except Exception as e:
            if isinstance(e, ModuleNotFoundError) and lint._import("nki") is None:
                r.update(stage="env", kind="ENV", error="the Neuron SDK (nki) is not installed here")
                return r
            kind = "RULES" if "defines no" in str(e) else errors.classify_runtime(str(e))
            r.update(stage="load", kind=kind, error=f"{type(e).__name__}: {e}",
                     line=_kernel_line(e, path), reward=_score(parts))
            return r
        first = None
        for case in spec["shapes"]:
            lbl = nkibench.label(case, level)
            args, _ = nkibench.make_inputs(case, level)
            before = [x.copy() if isinstance(x, np.ndarray) else x for x in args]
            want = spec["ref"](*args)
            try:
                got, counted = nkibench.simulate_and_count(kernel, args)
            except nkibench.NkiMissing as e:
                r.update(stage="env", kind="ENV", error=str(e))
                return r
            except Exception as e:
                msg = f"{type(e).__name__}: {e}"[:800]
                r["shapes"].append(dict(label=lbl, ok=False, kind=errors.classify_runtime(msg)))
                if first is None:
                    first = dict(stage="run", kind=errors.classify_runtime(msg), error=msg,
                                 line=_kernel_line(e, path), shape=lbl)
                continue
            parts["runs"] = True
            floor = nkibench.minimum_hbm_bytes(args, want)
            m = (nkibench.check_inputs_untouched(before, args)
                 or nkibench.describe_mismatch(got, want))
            if not m and spec.get("max_waste") and not counted.get("bytes"):
                m = ("NO HBM TRAFFIC COUNTED: the kernel produced the right values, but no nisa.dma_copy "
                     "call was counted, so the traffic bar cannot be checked. Move data with "
                     "nisa.dma_copy through the module (import nki.isa as nisa).")
            if not m:
                m = nkibench.check_traffic_bar(level, counted, args, want)
            hazards = [w for w in counted.get("warnings", []) if "incorrect results on hardware" in w]
            if hazards and not m:
                m = ("CORRECT ON CPU BUT WRONG ON HARDWARE: " + hazards[0] + ". Fix that before "
                     "anything else -- the simulator agrees with the reference here and the device "
                     "would not.")
            r["warnings"] = sorted(set(r["warnings"]) | set(counted.get("warnings", [])))[:5]
            shape = dict(label=lbl, ok=not m, bytes=int(counted.get("bytes") or 0),
                         transfers=int(counted.get("transfers") or 0), floor=int(floor or 0),
                         ratio=round(counted["bytes"] / floor, 3) if floor and counted.get("bytes")
                         else None)
            if m:
                kind = errors.classify_mismatch(m)
                shape["kind"] = kind
                if first is None:
                    first = dict(stage="values" if kind != "TRAFFIC" else "traffic", kind=kind,
                                 error=m, line=None, shape=lbl,
                                 corner=_corner(got, want) if kind in ("VALUES", "PARTIAL") else None)
            else:
                r["passed"] += 1
            r["shapes"].append(shape)
        if first:
            r.update(first)
            r["reward"] = _score(parts, r["passed"], total)
            return r
        parts["correct"] = True
        r.update(stage="correct", kind="CORRECT", error=None, correct=True, reward=_score(parts))
        return r
    finally:
        os.unlink(path)
        r["seconds"] = round(time.perf_counter() - t0, 2)


def _warm():
    try:
        import nki  # noqa: F401
    except Exception:
        pass


class CheckPool:
    """A pool of worker processes for checks, with a timeout that survives a hung kernel.

    A timed-out check kills the whole pool and starts a new one; checks other threads had in flight on
    the old pool are re-sent rather than reported as timeouts of their own."""

    def __init__(self, workers=2, timeout=180):
        self.workers, self.timeout = workers, timeout
        self.ctx = mp.get_context("spawn")
        self.lock = threading.Lock()
        self.gen = 0
        self.pool = self.ctx.Pool(self.workers, initializer=_warm)

    def run(self, source, level):
        for _ in range(3):
            with self.lock:
                pool, gen = self.pool, self.gen
            res = pool.apply_async(run_checks, (source, level))
            deadline = time.monotonic() + self.timeout
            while True:
                try:
                    return res.get(timeout=1.0)
                except mp.TimeoutError:
                    if self.gen != gen:
                        break                       # someone restarted the pool: send it again
                    if time.monotonic() > deadline:
                        with self.lock:
                            if self.gen == gen:
                                self.pool.terminate()
                                self.pool = self.ctx.Pool(self.workers, initializer=_warm)
                                self.gen += 1
                        return self._timeout(level)
                except Exception as e:
                    if self.gen != gen:
                        break
                    raise RuntimeError(f"check worker failed: {type(e).__name__}: {e}") from e
        return self._timeout(level)

    def _timeout(self, level):
        spec = nkibench.LEVELS[level]
        parts = dict(parses=True, rules=True, runs=False, correct=False)
        return dict(stage="run", parts=parts, reward=_score(parts), passed=0, total=len(spec["shapes"]),
                    correct=False, kind="TIMEOUT", error=f"The check took over {self.timeout} s.",
                    line=None, shape=None, corner=None, shapes=[], lint=[], warnings=[],
                    seconds=float(self.timeout))

    def close(self):
        self.pool.terminate()


class InlineChecks:
    """Checks in this process, one at a time. For tests and for --check-workers 0."""

    def __init__(self):
        self.lock = threading.Lock()

    def run(self, source, level):
        with self.lock:
            return run_checks(source, level)

    def close(self):
        pass
