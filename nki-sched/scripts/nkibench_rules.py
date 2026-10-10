"""Run nkibench's static rule scan (no Neuron SDK needed) on an emitted kernel.

nkibench lives in a different repository, so nothing in tests/ or examples/ depends on it; this is the
integration entry point. The full check (numerics + traffic) runs on a Trainium node via
scripts/device_check.sh.

    python scripts/nkibench_rules.py kernel.py 4 --nkibench /path/to/nkibench.py
    NKIBENCH=/path/to/nkibench.py python scripts/nkibench_rules.py kernel.py 4
"""

import argparse
import importlib.util
import os
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("kernel")
    ap.add_argument("level", type=int, help="nkibench level (4-7 for the matmul ladder)")
    ap.add_argument("--nkibench", default=os.environ.get("NKIBENCH"), help="path to nkibench.py (or $NKIBENCH)")
    a = ap.parse_args(argv)
    if not a.nkibench or not os.path.exists(a.nkibench):
        print("error: pass --nkibench /path/to/nkibench.py (or set $NKIBENCH)", file=sys.stderr)
        return 2
    spec = importlib.util.spec_from_file_location("nkibench", a.nkibench)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["nkibench"] = mod
    spec.loader.exec_module(mod)
    violations = mod.check_rules(open(a.kernel).read(), a.level)
    for v in violations:
        print(v)
    print("rules clean" if not violations else f"{len(violations)} rule violation(s)")
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
