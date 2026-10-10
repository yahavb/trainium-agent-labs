"""Real-hardware correctness suite against the frozen FP32 reference."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import ml_dtypes
import nki

from reference import make_inputs, score_reference, compare, validate_inputs, ATOL, RTOL
from runtime import LoadedKernel
from kernel import fused_score, separate_score
from baseline import aws_separate_score


def cases(quick=False, large_only=False):
    shapes = [(3, 17)] if quick else [(1, 1), (3, 17), (128, 8192), (129, 8193), (7, 32769), (512, 32768)]
    if large_only:
        shapes = [(512, 151936)]
    for rows, vocab in shapes:
        yield f"random-{rows}x{vocab}", make_inputs(rows, vocab)
        if not quick:
            yield f"wide-{rows}x{vocab}", make_inputs(rows, vocab, std=10.0)
            x, y = make_inputs(rows, vocab)
            x.fill(0)
            yield f"uniform-{rows}x{vocab}", (x, y)
            x, y = make_inputs(rows, vocab, std=8, offset=8192)
            yield f"positive-offset-{rows}x{vocab}", (x, y)
            x, y = make_inputs(rows, vocab, std=8, offset=-8192)
            yield f"negative-offset-{rows}x{vocab}", (x, y)
            x, y = make_inputs(rows, vocab)
            x.fill(-100)
            x[np.arange(rows), y] = 100
            yield f"peaked-{rows}x{vocab}", (x, y)
    if not quick and not large_only:
        # Successive tiles raise the max: exercises the online u shift correction.
        x, y = make_inputs(7, 32769, std=0.1)
        for lo in range(0, 32769, 8192):
            x[:, lo:min(lo + 8192, 32769)] = (x[:, lo:min(lo + 8192, 32769)].astype(np.float32) + lo / 8192).astype(ml_dtypes.bfloat16)
        yield "increasing-tile-max", (x, y)


def invalid_inputs():
    x, y = make_inputs(3, 17)
    bad = [ (x.astype(np.float32), y), (x, y.astype(np.int64)),
            (x[:, ::2], y), (x, y[:2]), (x, np.array([-1, 0, 1], np.int32)),
            (x, np.array([17, 0, 1], np.int32)),
            (np.empty((0, 17), dtype=ml_dtypes.bfloat16), np.empty(0, np.int32)) ]
    for value in (np.nan, np.inf, 11000):
        z = x.copy()
        z[0, 0] = value
        bad.append((z, y))
    for a, b in bad:
        try:
            validate_inputs(a, b)
        except (ValueError, TypeError):
            continue
        raise AssertionError("invalid input accepted")
    return len(bad)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--large-only", action="store_true", help="Validate the optional 512x151936 workload")
    parser.add_argument("--simulate", action="store_true", help="CPU correctness only; never supplies benchmark timing")
    parser.add_argument("--row-tile", type=int, default=128)
    parser.add_argument("--vocab-tile", type=int, default=8192)
    parser.add_argument("--tuning", help="Validate the selected benchmark configurations on their corresponding shapes")
    parser.add_argument("--output", default="results/validation.json")
    args = parser.parse_args()
    results = []
    cache = {}
    selections = json.loads(Path(args.tuning).read_text())["selected"] if args.tuning else {}
    labels = {(128, 8192): "A", (512, 32768): "B", (129, 8193): "C", (512, 151936): "D"}
    for name, (logits, indices) in cases(args.quick, args.large_only):
        expected = score_reference(logits, indices)
        for kernel in (fused_score, separate_score, aws_separate_score):
            config = selections.get(labels.get(logits.shape), {}).get(kernel.__name__,
                         {"row_tile": args.row_tile, "vocab_tile": args.vocab_tile})
            row_tile, vocab_tile = config["row_tile"], config["vocab_tile"]
            key = (kernel.__name__, logits.shape, row_tile, vocab_tile)
            if args.simulate:
                actual = nki.simulate(kernel[2])(logits, indices, row_tile, vocab_tile)
            else:
                if key not in cache:
                    cache[key] = LoadedKernel(kernel, logits, indices, row_tile, vocab_tile)
                actual = cache[key].run(logits, indices)
            errors = compare(actual, expected, logits.shape[1])
            result = {"case": name, "kernel": kernel.__name__, "shape": list(logits.shape),
                      "config": config, "passed": True, "max_abs_error": errors}
            results.append(result)
            print(json.dumps(result), flush=True)
    execution = "CPU simulation" if args.simulate else "Trainium2 hardware"
    record = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "execution": execution,
              "atol": ATOL, "rtol": RTOL, "row_tile": args.row_tile, "vocab_tile": args.vocab_tile,
              "invalid_input_rejections": invalid_inputs(), "comparisons": results}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2) + "\n")
    print(f"PASS: {len(results)} {execution} comparisons and {record['invalid_input_rejections']} invalid input rejections", flush=True)


if __name__ == "__main__":
    main()
