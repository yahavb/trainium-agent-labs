"""Differential testing: run an intermediate IR in the numpy interpreter and compare with an oracle
(eager torch or numpy). `Sched(check=make_checker(...))` calls this after every primitive."""

from __future__ import annotations

import numpy as np

from . import interp, ir
from .lower import infer_kinds


class Mismatch(Exception):
    pass


def hostile_inputs(shapes_by_arg: dict, seed=0, dtype="f32"):
    """Random inputs plus the usual hostile cases (a large-magnitude element)."""
    r = np.random.default_rng(seed)
    out = {}
    for name, shp in shapes_by_arg.items():
        x = r.standard_normal(shp).astype(np.float32)
        if x.size > 1:
            x.flat[0] = 1e3  # large-magnitude element
        out[name] = interp.quantize(dtype, x)
    if dtype != "f32":
        out["__dtypes__"] = {n: dtype for n in shapes_by_arg}
    return out


def torch_oracle(fn, arg_names=("lhsT", "rhs")):
    """Wrap a torch function as an oracle over numpy inputs (float32 storage); arguments are taken
    from the inputs dict in the order of arg_names."""
    import torch

    def run(inputs):
        dts = inputs.get("__dtypes__", {})
        tens = []
        for name in arg_names:
            t = torch.from_numpy(np.asarray(inputs[name], dtype=np.float32))
            if dts.get(name) == "bf16":
                t = t.to(torch.bfloat16)
            tens.append(t)
        return fn(*tens).to(torch.float32).numpy()

    return run


def _compare(got, want, rtol, atol, shp, reverse):
    if got.shape != want.shape or not np.allclose(got, want, rtol=rtol, atol=atol):
        bad = np.argwhere(~np.isclose(got, want, rtol=rtol, atol=atol)) if got.shape == want.shape else []
        where = tuple(bad[0]) if len(bad) else None
        tag = " (affine loops run in reverse)" if reverse else ""
        raise Mismatch(f"shape {shp}{tag}: first mismatch at {where}: "
                       f"got {got[where] if where else got.shape}, want {want[where] if where else want.shape}")


def make_checker(oracle, shapes, out="C", rtol=1e-4, atol=1e-4, dtype="f32", seeds=(0, 1), reverse_affine=True):
    """shapes: list of dicts {"K":..,"M":..,"N":..}. Raises Mismatch with a locating message.

    With reverse_affine the program is run a second time with every (inferred or marked) affine loop
    iterated in reverse order: an affine annotation on a loop that actually carries a dependence then
    fails the oracle, so the loop-kind analysis is itself tested, not trusted."""

    def check(proc: ir.Proc):
        for shp in shapes:
            for seed in seeds:
                inputs = hostile_inputs({"lhsT": (shp["K"], shp["M"]), "rhs": (shp["K"], shp["N"])}, seed, dtype)
                want = oracle(inputs)
                runs = [(proc, False)] + ([(infer_kinds(proc), True)] if reverse_affine else [])
                for prog, rev in runs:
                    got = interp.run(prog, inputs, reverse_affine=rev)[out]
                    _compare(got, want, rtol, atol, shp, rev)

    return check
