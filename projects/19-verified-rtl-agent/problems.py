"""VerilogEval spec-to-RTL problems, and the fixed dev / held-out split.

DESIGN.md 6.1. The dataset is NOT copied into this repo: setup.sh clones it at c498220, and the root
comes from VE_ROOT. Every problem is four facts read straight from its files -- the spec the model
sees, the reference and testbench the checker uses, and the port list both of them agree on.

    python problems.py --split          # writes eval/dev.txt and eval/heldout.txt
    python problems.py --show Prob050_kmap1
    python problems.py --stats
"""
from __future__ import annotations

import argparse
import datetime
import os
import random
import re
import sys
from dataclasses import dataclass, field

HERE = os.path.dirname(os.path.abspath(__file__))
VE_ROOT = os.environ.get("VE_ROOT", "/root/verilog-eval/dataset_spec-to-rtl")
SEED = 20261010

# Probed by hand on seat 93 before the split existed (STATUS.md, exploratory measurements). They
# have been looked at, so they can only ever be dev.
PROBED = ["Prob001_zero", "Prob021_mux256to1v", "Prob027_fadd", "Prob035_count1to10",
          "Prob050_kmap1", "Prob054_edgedetect", "Prob071_always_casez", "Prob086_lfsr5",
          "Prob109_fsm1", "Prob128_fsm_ps2", "Prob142_lemmings2", "Prob156_review2015_fancytimer"]
DEV_EXTRA = 8
HELDOUT_PER_KIND_PER_QUARTILE = 5


@dataclass(frozen=True)
class Problem:
    id: str            # e.g. "Prob035_count1to10"
    spec: str          # contents of <id>_prompt.txt
    ref_path: str      # <id>_ref.sv  (module RefModule)
    test_path: str     # <id>_test.sv (module tb)
    kind: str          # "seq" if RefModule has a clk input, else "comb"
    ports: list = field(default_factory=list)   # (direction, name, width) from RefModule

    @property
    def inputs(self):
        return [p for p in self.ports if p[0] == "input"]

    @property
    def outputs(self):
        return [p for p in self.ports if p[0] == "output"]

    def port_list(self):
        """The ports as a reader would write them: `input [7:0] in, output out`."""
        return ", ".join(f"{d} {f'[{w - 1}:0] ' if w > 1 else ''}{n}" for d, n, w in self.ports)


_HEADER = re.compile(r"module\s+RefModule\s*(?:#\s*\(.*?\))?\s*\((.*?)\)\s*;", re.S)
_PORT = re.compile(r"^(input|output|inout)\b\s*(?:wire|reg|logic|signed|\s)*"
                   r"(?:\[\s*(\d+)\s*:\s*(\d+)\s*\])?\s*([A-Za-z_]\w*)\s*$")


def parse_ports(src: str) -> list:
    """(direction, name, width) for every port of RefModule, in declaration order.

    Measured over all 156 references: one port per line, ANSI style, integer ranges only, so a
    regex is enough. Anything it cannot read is a loud error rather than a silently missing port.
    """
    m = _HEADER.search(src)
    if not m:
        raise ValueError("no `module RefModule (...);` header")
    ports = []
    for line in m.group(1).split("\n"):
        line = re.sub(r"//.*", "", line).strip().rstrip(",").strip()
        if not line:
            continue
        p = _PORT.match(line)
        if not p:
            raise ValueError(f"cannot read port declaration {line!r}")
        direction, hi, lo, name = p.groups()
        width = abs(int(hi) - int(lo)) + 1 if hi is not None else 1
        ports.append((direction, name, width))
    return ports


def load(problem_id: str, root: str = None) -> Problem:
    root = root or VE_ROOT
    base = os.path.join(root, problem_id)
    paths = {k: f"{base}_{k}" for k in ("prompt.txt", "ref.sv", "test.sv")}
    for p in paths.values():
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p} is missing -- is VE_ROOT ({root}) right? Run setup.sh")
    with open(paths["prompt.txt"]) as f:
        spec = f.read().strip() + "\n"
    with open(paths["ref.sv"]) as f:
        ports = parse_ports(f.read())
    kind = "seq" if any(d == "input" and n == "clk" for d, n, _ in ports) else "comb"
    return Problem(problem_id, spec, paths["ref.sv"], paths["test.sv"], kind, ports)


def all_ids(root: str = None) -> list:
    root = root or VE_ROOT
    return sorted(f[:-len("_prompt.txt")] for f in os.listdir(root) if f.endswith("_prompt.txt"))


def number(problem_id: str) -> int:
    return int(re.match(r"Prob(\d+)", problem_id).group(1))


def load_list(path: str, root: str = None) -> list:
    with open(path) as f:
        ids = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    return [load(i, root) for i in ids]


def make_split(seed: int = SEED, root: str = None) -> tuple:
    """(dev, heldout), deterministic in `seed`.

    Dev is the 12 probed problems plus 8 drawn at random. Held-out is drawn from what is left:
    sorted by problem number, cut into 4 quartiles (the numbering runs roughly easy to hard), then
    5 comb + 5 seq from each, so neither difficulty nor kind is left to luck.
    """
    ids = all_ids(root)
    missing = [p for p in PROBED if p not in ids]
    if missing:
        raise ValueError(f"probed problems not in the dataset: {missing}")
    kinds = {i: load(i, root).kind for i in ids}
    rng = random.Random(seed)

    pool = sorted((i for i in ids if i not in PROBED), key=number)
    extra = sorted(rng.sample(pool, DEV_EXTRA), key=number)
    dev = sorted(PROBED + extra, key=number)

    rest = [i for i in pool if i not in extra]
    q = len(rest) / 4
    heldout = []
    for k in range(4):
        quartile = rest[round(k * q):round((k + 1) * q)]
        comb = [i for i in quartile if kinds[i] == "comb"]
        seq = [i for i in quartile if kinds[i] == "seq"]
        n = HELDOUT_PER_KIND_PER_QUARTILE
        base_c, base_s = min(n, len(comb)), min(n, len(seq))
        # A quartile short of one kind is filled from the other, so every quartile gives 10.
        take_c = base_c + min(n - base_s, len(comb) - base_c)
        take_s = base_s + min(n - base_c, len(seq) - base_s)
        heldout += rng.sample(comb, take_c) + rng.sample(seq, take_s)
    heldout = sorted(heldout, key=number)
    assert not set(dev) & set(heldout)
    return dev, heldout


def write_list(path: str, ids: list, seed: int):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with open(path, "w") as f:
        f.write(f"# seed={seed} created={stamp}\n")
        f.write("".join(i + "\n" for i in ids))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", action="store_true", help="write eval/dev.txt and eval/heldout.txt")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--show", metavar="ID", help="print one problem as the checker sees it")
    ap.add_argument("--stats", action="store_true", help="count problems by kind")
    ap.add_argument("--root", default=VE_ROOT)
    a = ap.parse_args()

    if a.show:
        p = load(a.show, a.root)
        print(f"{p.id}  kind={p.kind}\nports: {p.port_list()}\n\n{p.spec}")
    if a.stats:
        ps = [load(i, a.root) for i in all_ids(a.root)]
        print(f"{len(ps)} problems: {sum(p.kind == 'seq' for p in ps)} seq, "
              f"{sum(p.kind == 'comb' for p in ps)} comb")
    if a.split:
        heldout_path = os.path.join(HERE, "eval", "heldout.txt")
        dev, heldout = make_split(a.seed, a.root)
        if os.path.exists(heldout_path):
            with open(heldout_path) as f:
                existing = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
            if existing != heldout:
                sys.exit(f"{heldout_path} already holds a different list, and earlier held-out runs used "
                         f"it. Refusing to overwrite.")
        write_list(os.path.join(HERE, "eval", "dev.txt"), dev, a.seed)
        write_list(heldout_path, heldout, a.seed)
        kinds = lambda ids: sum(load(i, a.root).kind == "seq" for i in ids)
        print(f"dev     {len(dev)} problems ({kinds(dev)} seq)  -> eval/dev.txt")
        print(f"heldout {len(heldout)} problems ({kinds(heldout)} seq)  -> eval/heldout.txt")
    if not (a.show or a.stats or a.split):
        ap.print_help()


if __name__ == "__main__":
    main()
