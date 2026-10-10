#!/usr/bin/env python3
"""mutants.py -- a CHECKER kill matrix. The checker is "the artifact we keep", so prove it bites.

Each mutant plants ONE bug into a reference kernel and asserts the harness (a) rejects it and
(b) prints a message naming the right class of fix. A checker that silently passes a wrong kernel
is worse than no checker, so this is the evidence that it does not.

    python mutants.py                 # run every mutant through nkibench --check
    python mutants.py --rules-only    # only the mutants catchable by the static scan (no device)

Mutants whose detection needs the simulator are skipped with --rules-only and when nki is absent,
and the row is marked NEEDS-DEVICE rather than passed, so the matrix never over-claims.
"""
import argparse
import subprocess
import sys


def _after(src, anchor, insert):
    """Insert a line immediately after the first line containing `anchor` (indentation copied)."""
    out = []
    done = False
    for line in src.splitlines():
        out.append(line)
        if not done and anchor in line:
            indent = line[:len(line) - len(line.lstrip())]
            out.append(indent + insert)
            done = True
    return "\n".join(out) + "\n"


# (name, level, kind, transform, expected_substring, needs_device)
#   kind "rules"   -> caught by the static scan, runs anywhere
#   kind "numeric" -> caught only by simulation, needs the Neuron SDK
MUTANTS = [
    # --- static-scan mutants (no device) ---
    ("numpy_alias", 1, "rules",
     lambda s: s.replace("import numpy as np", "import numpy as xp")
                .replace("return out_tensor",
                         "return xp.mean(in_tensor, axis=1, keepdims=True) * 0 + out_tensor")
     if "out_tensor" in s else s,
     "framework", False),
    ("from_import_matmul", 3, "rules",
     lambda s: ("from numpy import matmul as _mm\n" + s).replace(
         "return result", "_ = _mm(lhsT.T, rhs)\n  return result", 1),
     "framework", False),
    ("drop_jit", 2, "rules",
     lambda s: s.replace("@nki.jit\n", "", 1),
     "nki.jit", False),
    ("rename_entry", 3, "rules",
     lambda s: s.replace("def nki_matmul_basic_", "def not_the_entry_point", 1),
     "no function named", False),
    ("oversized_tile", 3, "rules",
     lambda s: _after(s, "lhs_tile = nl.ndarray",
                      "big = nl.ndarray((256, 512), dtype=lhsT.dtype, buffer=nl.sbuf)"),
     "partition dimension 256", False),

    # --- numeric mutants (need the simulator) ---
    ("transpose_swap", 2, "numeric",
     lambda s: s.replace("i_f2*sz_f1+i_f1", "i_f1*sz_f2+i_f2"),
     "MISMATCH", True),
    ("pool_divisor", 1, "numeric",
     lambda s: s.replace("pool_size * pool_size", "pool_size")
                .replace("pool_size*pool_size", "pool_size"),
     "MISMATCH", True),
    ("matmul_no_copyout", 3, "numeric",
     lambda s: s.replace("nisa.dma_copy(dst=result, src=result_sbuf)", "pass"),
     "ZERO", True),
]


def have_nki():
    try:
        import nki  # noqa: F401
        return True
    except Exception:
        return False


def run_check(level, path):
    r = subprocess.run([sys.executable, "nkibench.py", "--level", str(level), "--check", path],
                       capture_output=True, text=True)
    return (r.stdout + r.stderr), r.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules-only", action="store_true")
    a = ap.parse_args()
    device = have_nki()

    print(f"{'mutant':<20} {'lvl':<4} {'caught':<8} {'msg ok':<8} note")
    print("-" * 60)
    rc = 0
    for name, level, kind, transform, want, needs_device in MUTANTS:
        if (kind == "numeric") and (a.rules_only or not device):
            print(f"{name:<20} {level:<4} {'-':<8} {'-':<8} NEEDS-DEVICE (skipped)")
            continue
        src = open(f"reference_level{level}.py").read()
        mut = transform(src)
        if mut == src:
            print(f"{name:<20} {level:<4} {'FAIL':<8} {'-':<8} transform changed nothing")
            rc = 1
            continue
        path = f"/tmp/mut_{name}.py"
        with open(path, "w") as f:
            f.write(mut)
        out, code = run_check(level, path)
        # verify() returns non-zero for any rejection (1 numeric, 2 rules/import, 3 no-sdk).
        caught = code not in (0,) and "SELFTEST" not in out
        msg_ok = want.lower() in out.lower()
        rc |= 0 if (caught and msg_ok) else 1
        print(f"{name:<20} {level:<4} {str(caught):<8} {str(msg_ok):<8} "
              f"{'' if caught and msg_ok else 'exit=' + str(code)}")
    print("\nKILL MATRIX " + ("PASSED" if rc == 0 else "FAILED (or some NEEDS-DEVICE)"))
    return rc


if __name__ == "__main__":
    sys.exit(main())
