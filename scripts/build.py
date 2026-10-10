"""Emit a kernel from a schedule function.

    python scripts/build.py examples/fast_mm.py:fast out.py [--dtype bf16] [--name mm] [--ir]
"""

import argparse
import importlib.util
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "examples"))

import _common as c  # noqa: E402
from nki_sched import NC_DEFAULT, Sched  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target", help="file.py:function  (function(s, hw))")
    ap.add_argument("out")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--name", default="mm")
    ap.add_argument("--ir", action="store_true")
    a = ap.parse_args()
    path, fn = a.target.split(":")
    spec = importlib.util.spec_from_file_location("sched_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    s = Sched(c.matmul_proc(dtype=a.dtype, name=a.name), NC_DEFAULT)
    getattr(mod, fn)(s, NC_DEFAULT)
    if a.ir:
        print(s.show())
    with open(a.out, "w") as f:
        f.write(s.source(itemsize=2 if a.dtype in ('bf16', 'f16') else 4))


if __name__ == "__main__":
    main()
