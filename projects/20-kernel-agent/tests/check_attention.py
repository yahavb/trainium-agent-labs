"""Independent SDK simulation cases beyond Level 8's three default inputs.

Run on a seat with the NKI SDK: python tests/check_attention.py candidate.py --json results.json
This checks correctness and simulated traffic, not Trainium latency or compilation.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import nkibench as nb
from lint import lint_kernel


def cases():
    for seq, dim in ((128, 64), (64, 128), (96, 32)):
        for seed in (17, 97, 2026):
            rng = np.random.default_rng(seed)
            args = tuple(rng.standard_normal((seq, dim)).astype(np.float32) for _ in range(3))
            yield f"random-{seq}x{dim}-seed{seed}", args, None
        q, k, v = args
        yield f"zero-q-{seq}x{dim}", (np.zeros_like(q), k, v), np.broadcast_to(v.mean(0), v.shape)
        constant = np.broadcast_to(np.linspace(-1, 1, dim, dtype=np.float32), v.shape).copy()
        yield f"constant-v-{seq}x{dim}", (q, k, constant), constant
        yield f"large-logits-{seq}x{dim}", (q * 10, k * 10, v), None
        permutation = rng.permutation(seq)
        yield f"permuted-kv-{seq}x{dim}", (q, k[permutation], v[permutation]), nb.ref_attention(q, k, v)
    for seq, dim in ((32, 48), (16, 16), (1, 8)):
        rng = np.random.default_rng(7331)
        args = tuple(rng.standard_normal((seq, dim)).astype(np.float32) for _ in range(3))
        yield f"extra-shape-{seq}x{dim}", args, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate")
    parser.add_argument("--json")
    options = parser.parse_args()
    source = Path(options.candidate).read_text()
    issues = nb.check_rules(source, 8) + lint_kernel(source)
    if issues:
        raise SystemExit("\n".join(issues))
    kernel = nb.load_kernel(options.candidate, "nki_attention_")
    records = []
    for name, args, invariant in cases():
        before = [x.copy() for x in args]
        want = nb.ref_attention(*args)
        try:
            got, counted = nb.simulate_and_count(kernel, args)
            error = nb.describe_mismatch(got, want)
            if invariant is not None:
                error = error or nb.describe_mismatch(got, invariant)
            if any(not np.array_equal(x, y) for x, y in zip(args, before)):
                error = "input mutated"
            hazards = [w for w in counted.get("warnings", []) if "incorrect results on hardware" in w]
            if hazards:
                error = "hardware hazard: " + hazards[0]
            scale = float(np.sqrt(np.mean(want.astype(np.float64) ** 2))) or 1.0
            normalized_error = float(np.max(np.abs(np.asarray(got) - want))) / scale
            floor = sum(x.nbytes for x in args) + want.nbytes
            record = dict(case=name, passed=error is None, error=error,
                          normalized_max_error=normalized_error, bytes=counted["bytes"], floor=floor,
                          warnings=counted.get("warnings", []))
        except Exception as exc:
            record = dict(case=name, passed=False, error=f"{type(exc).__name__}: {exc}")
        records.append(record)
        print(("PASS " if record["passed"] else "FAIL ") + name +
              (" " + record["error"] if record.get("error") else ""), flush=True)
    passed = sum(r["passed"] for r in records)
    print(f"{passed}/{len(records)} independent cases passed", flush=True)
    if options.json:
        Path(options.json).write_text(json.dumps(records, indent=2) + "\n")
    return 0 if passed == len(records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
