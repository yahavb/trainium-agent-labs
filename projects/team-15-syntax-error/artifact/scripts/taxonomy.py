#!/usr/bin/env python3
"""
taxonomy.py -- the failure taxonomy, generated from the attempt logs. FAILURES.md is its output.

    python scripts/taxonomy.py runs/*/*.jsonl > FAILURES.md

Every attempt is classified by projects/02-kernel-agent/verdicts.classify(feedback) -- the same
function the agent uses to choose its doc retrieval -- so the taxonomy and the agent cannot drift.
Counts are per ATTEMPT (one model reply), split by level and by the run tag (= model + config).
Identical replies within a round (the local server decodes deterministically) are counted once,
so a bucket is not inflated 4x by duplicate samples.
"""

import collections
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "projects", "02-kernel-agent"))
import verdicts  # noqa: E402

# bucket -> (one-sentence definition, the instruction that fixes it)
DEFS = {
    "no_code": ("the reply contained no code block at all",
                "re-ask with a shorter prompt; never ask it to 'fix' empty code"),
    "syntax_error": ("the code block does not parse",
                     "send one complete python block; check the answer was not truncated"),
    "rule_wrong_entry_name": ("the entry point has the wrong name",
                              "mechanical: rename by AST (--mech), no model call needed"),
    "rule_missing_jit": ("the entry point lacks @nki.jit", "mechanical: add it by AST (--mech)"),
    "rule_partition_over_128": ("a tile literally allocated with partition dim > 128",
                                "loop the partition dim in chunks of min(128, remaining)"),
    "rule_framework_call": ("hands the whole op to numpy/torch or uses @/.T on an input",
                            "replace the framework call with the nl/nisa primitive"),
    "rule_other": ("another static rule violation", "fix exactly the rule named"),
    "invented_import": ("imports a module that does not exist (e.g. nki.nl)",
                        "mechanical: the only imports are nki, nki.language as nl, nki.isa as nisa"),
    "load_error": ("module imports but the entry point cannot be loaded",
                   "read the loader error; usually a top-level statement that raises"),
    "invented_api": ("calls an nl/nisa name that does not exist (nl.dot, nl.value ...)",
                     "list the closest real names and let it choose"),
    "invented_kwarg": ("passes a keyword argument the real function does not take",
                       "remove it and show the real signature"),
    "call_memory_region": ("calls nl.sbuf(...) as if it were a function",
                           "allocate with nl.ndarray(shape, dtype, buffer=nl.sbuf)"),
    "tile_1d": ("allocates a 1-D SBUF/PSUM tile", "give it a partition dim: (P, 1) or (1, N)"),
    "reshape_instead_of_slice": ("reshapes a tensor instead of slicing tiles out of it",
                                 "slice: a[k0:k0+k, m0:m0+m]; never reshape across partitions"),
    "partition_over_128": ("one tile for the whole tensor; partition dim > 128 at run time",
                           "remainder-safe loop over the partition dim, tile sized inside the loop"),
    "contraction_over_128": ("contracts K > 128 in one nc_matmul",
                             "split K, accumulate in ONE psum tile allocated outside the K loop"),
    "contraction_mismatch": ("nc_matmul operands disagree on K -- often nc_matmul used for a "
                             "non-matmul op", "same first dim for both operands, or do not use "
                             "nc_matmul for data movement"),
    "rule_device_comprehension": ("comprehension in the kernel: passes the simulator, rejected by the "
                                  "device compiler", "for loop + .append (--device-rules)"),
    "softmax_ops": ("[L8] softmax built from nl.sum / negation instead of maximum, subtract, reciprocal",
                    "name the exact op sequence (--v10)"),
    "api_misuse_lint": ("nisa/nl calls with wrong/missing keywords or swapped tensor_scalar roles "
                        "(caught statically, all at once)", "fix every listed call (--v9)"),
    "missing_positional": ("a required argument is missing", "pass every argument by keyword (--v9)"),
    "undefined_name": ("a name (often np) used inside the kernel is undefined", "define it / plain Python (--v9)"),
    "used_before_assigned": ("a tile read before it is allocated", "reorder (--v9)"),
    "kloop_outside": ("[L5-7] contraction loop k outside m/n: fresh PSUM per k, nothing accumulates",
                      "m, n, then k innermost; one psum acc per (m, n) (--v8)"),
    "psum_sbuf_conflated": ("[L8] one name for the PSUM tile and the SBUF tile",
                            "two tiles: transpose/matmul into psum, tensor_copy into a separate sbuf tile (--v8)"),
    "assumed_pretransposed": ("[L8] q/k/v fed to nc_matmul from HBM as if already transposed",
                              "dma_copy to SBUF, nc_transpose into psum, copy to SBUF (--v8)"),
    "tiles_never_loaded": ("[L5-7] tiles allocated for reuse but never dma_copy'd -> NaN",
                           "fill each tile with dma_copy right after allocating it (--v7)"),
    "exp_overflow": ("[L8] NaN because exp() overflowed (no row-max subtraction)",
                     "subtract the row max before exp (--v6)"),
    "traffic_reloads": ("[L5-7] correct, but operand tiles copied from HBM more than once",
                        "load every tile once into a[k][m] / b[k][n], then only matmul (--v5)"),
    "traffic_lhs_not_hoisted": ("[L5] correct, but lhsT reloaded on every n: over the traffic bar",
                                "load lhsT tiles once per m, before the n loop (--v4)"),
    "scalar_window_reduce": ("[L1] reduces each window to a scalar and assigns it element-wise",
                             "tensor_reduce into a (C,1) tile, tensor_copy into a column (--v3)"),
    "swapped_axes": ("slices an input with its axes in the wrong order (lhsT[m.., k..])",
                     "name the slice and give the right index order (--v2-verdicts)"),
    "partition_collapsed": ("loops over / single-indexes the partition axis, collapsing it",
                            "keep the partition axis whole with ':'; no loop over it (--v2-verdicts)"),
    "rule_fancy_indexing": ("[Stage A] boolean-mask or index-array indexing",
                            "slices only; conditions via np.where on the whole tile"),
    "stride_bug": ("wrong rows/cols form a regular stride: the loop steps wrong",
                   "visit every row/col exactly once"),
    "off_by_one": ("only the first/last row or column is wrong", "fix the loop bound / slice end"),
    "wrong_scale": ("every value off by one constant factor", "fix the normalisation"),
    "sign_flipped": ("got = -expected", "a subtraction is reversed"),
    "index_shift": ("values come from the neighbouring row/col", "an index is off by one"),
    "overflow": ("exp overflowed", "subtract the row max before exp"),
    "dma_shape_mismatch": ("dma_copy src and dst hold different numbers of elements",
                           "allocate the tile with exactly the slice's shape"),
    "assign_shape_mismatch": ("assigns a value into a slice of a different shape",
                              "index the destination to match the value"),
    "out_of_bounds": ("indexes past the end of a tensor (fixed tile size on a smaller dim)",
                      "derive bounds from the shape: min(limit, size)"),
    "wrong_buffer": ("an operand lives in the wrong buffer (e.g. matmul dst not in psum)",
                     "dst in nl.psum; stationary and moving in nl.sbuf"),
    "engine_cannot_reach_hbm": ("tensor_copy/compute engine writes straight to HBM",
                                "psum->sbuf with tensor_copy, then sbuf->hbm with dma_copy"),
    "arg_bound_twice": ("an argument passed both positionally and by keyword",
                        "pass every argument by keyword"),
    "python_operator_on_tile": ("uses + / += on tiles", "use nisa.tensor_tensor / tensor_scalar"),
    "numpy_method_on_tile": ("calls a numpy method (.mean, .sum) on a tile",
                             "use nisa.tensor_reduce then tensor_scalar"),
    "raised_other": ("the simulator raised an error with no translation yet",
                     "add a row to verdicts.RULES"),
    "wrong_output_shape": ("output has the wrong shape", "check the output-size arithmetic"),
    "non_finite_output": ("NaN/Inf in the output -- usually an uninitialised tile was read",
                          "write every tile before reading it"),
    "output_not_written": ("output mostly zeros: results never reached HBM",
                           "copy every tile psum->sbuf->hbm, at the right indices"),
    "partial_coverage": ("a block of the output was never written",
                         "check loop bounds and destination indices"),
    "ragged_edge": ("wrong only in the final partial tile", "handle the partial tile's size"),
    "wrong_arithmetic": ("numerically wrong across the output",
                         "operand layout or the core arithmetic, not an edge case"),
    "traffic_over_bar": ("correct but moves more HBM bytes than the level allows",
                         "hoist/ block loads so each byte crosses the bus fewer times"),
    "hardware_hazard": ("correct on CPU but the simulator warns it is wrong on hardware",
                        "fix the hazard named in the warning"),
    "mutates_input": ("writes into its input tensor", "allocate a fresh output in shared_hbm"),
    "cannot_simulate": ("the SDK was missing -- an environment failure, not the model's",
                        "run inside the seat pod"),
    "solved": ("correct on every shape (and under the traffic bar)", "--"),
    "other": ("unclassified feedback", "extend verdicts.classify"),
}


def root_cause(r):
    """Bucket by ROOT CAUSE read from the code, not by how that run's feedback worded it.

    The same swapped lhsT slice is 'out_of_bounds' (L3) or 'wrong_arithmetic' (L4, square tiles)
    under v1 feedback and 'swapped_axes' under v2. Counting by wording would make the taxonomy
    depend on which experiment produced the attempt, so failed attempts whose code shows a known
    root cause are re-bucketed by it. The original wording-bucket is kept in r['_wording']."""
    r["_wording"] = r["_bucket"]
    if r.get("stage") == "A" or r["_bucket"] == "solved":
        return r["_bucket"]
    code, lv = r.get("code") or "", r["level"]
    if verdicts._swapped_axes(code, lv):
        return "swapped_axes"
    if lv == 1 and r["_bucket"] in ("tile_1d", "partition_collapsed")             and verdicts._SCALAR_REDUCE.search(code):
        return "scalar_window_reduce"
    return r["_bucket"]


def load(paths):
    rows = []
    for p in paths:
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if line:
                r = json.loads(line)
                if "claim" in r:
                    continue
                r["_file"] = p
                rows.append(r)
    return rows


def dedupe(rows):
    """One row per distinct reply per (file, tag, run, level, round)."""
    seen, out = set(), []
    for r in rows:
        k = (r["_file"], r.get("tag"), r.get("run"), r["level"], r["round"], r.get("code", ""))
        if k not in seen:
            seen.add(k)
            out.append(r)
    return out


def snippet(code, n=10):
    lines = [l for l in (code or "").splitlines() if l.strip()]
    return "\n".join(lines[:n])


def main(paths):
    rows = dedupe(load(paths))
    for r in rows:
        # Stage-A rows carry the bucket their own grader (stagea.instruct) assigned
        r["_bucket"] = (r.get("bucket") if r.get("stage") == "A"
                        else verdicts.classify(r.get("feedback", "")))
        r["_bucket"] = root_cause(r)
        # Stage A (NumPy ladder) and Stage B (NKI ladder) are different ladders: never share a column
        r["_lv"] = ("A" if r.get("stage") == "A" else "L") + str(r["level"])
    total = len(rows)
    by_bucket = collections.Counter(r["_bucket"] for r in rows)
    by_bl = collections.Counter((r["_bucket"], r["_lv"]) for r in rows)
    # group runs by experiment (tag prefix before "_L"): base, skel, c1, c2, retr, A_qwen ...
    exp = lambda r: (r.get("tag") or "?").split("_L")[0]
    by_bt = collections.Counter((r["_bucket"], exp(r)) for r in rows)
    levels = sorted({r["_lv"] for r in rows}, key=lambda v: (v[0] != "L", int(v[1:])))
    tags = sorted({exp(r) for r in rows})
    example = {}
    for r in rows:
        example.setdefault(r["_bucket"], r)

    print("# FAILURES — the failure taxonomy\n")
    print(f"Generated by `scripts/taxonomy.py` from {len(paths)} attempt log(s): "
          f"**{total} distinct attempts** (identical replies within a round counted once). "
          f"Every attempt is bucketed by `verdicts.classify()`, the same function the agent uses, "
          f"then re-bucketed by ROOT CAUSE where the code shows one (`root_cause()`: a swapped "
          f"lhsT slice, or L1's scalar window reduction), so that the same mistake counts the same "
          f"whatever wording that run's feedback used. "
          f"All results **[sim]**.\n")
    print("## Counts by level\n")
    print("Columns L1-L8 are the NKI ladder (Stage B, [sim]); A1-A10 the NumPy ladder (Stage A, CPU).\n")
    print("| bucket | " + " | ".join(levels) + " | total | share |")
    print("|---|" + "---|" * (len(levels) + 2))
    for b, n in by_bucket.most_common():
        print(f"| `{b}` | " + " | ".join(str(by_bl.get((b, l), 0) or "") for l in levels)
              + f" | **{n}** | {n / total:.0%} |")
    print("\n## Counts by experiment (model + config; see RESULTS.md)\n")
    print("| bucket | " + " | ".join(tags) + " |")
    print("|---|" + "---|" * len(tags))
    for b, _ in by_bucket.most_common():
        print(f"| `{b}` | " + " | ".join(str(by_bt.get((b, t), 0) or "") for t in tags) + " |")
    print("\n## Buckets\n")
    for b, n in by_bucket.most_common():
        d, fix = DEFS.get(b, ("(undefined bucket)", "?"))
        r = example[b]
        print(f"### `{b}` — {n} attempts\n")
        print(f"**Definition:** {d}.  \n**Instruction that fixes it:** {fix}.  ")
        print(f"**Representative** (level {r['level']}, tag `{r.get('tag')}`, run {r.get('run')}, "
              f"round {r['round']}):\n")
        print("```python\n" + snippet(r.get("code")) + "\n```\n")
        print("Checker said: _" + (r.get("feedback") or "")[:300].replace("\n", " ")
              .replace("_", "\\_") + "_\n")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ps = [p for a in sys.argv[1:] for p in glob.glob(a)]
    if not ps:
        sys.exit("usage: taxonomy.py runs/*/*.jsonl")
    main(ps)
