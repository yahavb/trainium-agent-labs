"""Five levels, stable physical IDs, balanced disjoint datasets."""
import hashlib
import importlib.util
import json
import random
from pathlib import Path

import sympy as sp

from .physics import STARTER, basis_for, x, t


def identity(record):
    physical = {key: record[key] for key in ("L", "k", "left", "right", "f")}
    return hashlib.sha256(json.dumps(physical, sort_keys=True).encode()).hexdigest()


def make(level, sub, seed=0, references=False):
    if level not in range(5) or sub not in (1, 2, 3):
        raise ValueError("level must be 0..4 and sub must be 1..3")
    if level < 2:
        spec = importlib.util.spec_from_file_location(f"controller_level{level}", STARTER / f"level{level}_heatrod.py")
        mod = importlib.util.module_from_spec(spec)
        # Starter level modules import pdecheck by name. Give them their own import scope
        # without replacing unrelated modules in sys.modules.
        source = (STARTER / f"level{level}_heatrod.py").read_text()
        source = source.replace("import pdecheck", "from controllerlab.physics import checker as pdecheck")
        source = source.replace("from pdecheck import x, t", "from controllerlab.physics import x, t")
        exec(compile(source, str(STARTER / f"level{level}_heatrod.py"), "exec"), mod.__dict__)
        p = mod.make(sub, seed)
        L, k, left, right, f, exact, tol = (p[key] for key in ("L", "k", "left", "right", "f", "exact", "tol"))
    else:
        rng = random.Random(10000*level + 100*seed + sub)
        L = sp.Rational(rng.choice([2, 3, 4, 5, 6, 7]), 2)
        k = sp.Rational(rng.choice([1, 2, 3, 5]), rng.choice([1, 2, 4]))
        amplitude = rng.choice([-3, -2, -1, 1, 2, 3])
        left, right = ("neumann", "dirichlet") if level == 2 else ("neumann", "neumann")
        exact, tol = None, 1e-6
        if level < 4:
            basis, lam = basis_for(left, right, L)
            if level == 3 and sub == 1:
                modes = [(1, amplitude)]
            else:
                count = 1 if sub == 1 or (level == 3 and sub == 2) else sub
                indices = rng.sample(range(2 if level == 3 else 1, 7), count)
                modes = [(j, rng.choice([-3, -1, 1, 2, 3])) for j in sorted(indices)]
                if level == 3 and sub == 3:
                    modes.insert(0, (1, amplitude))
            f = sum(a*basis(j) for j, a in modes)
            exact = sum(a*basis(j)*sp.exp(-k*lam(j)**2*t) for j, a in modes)
        else:
            left, right = "dirichlet", "dirichlet"
            tol = 0.005
            if sub == 1:
                skew = sp.Rational(rng.choice([1, 2, 3]), 3)
                f = amplitude*x*(L-x)*(1+skew*x/L)
            elif sub == 2:
                pivot = L*sp.Rational(rng.choice([1, 2, 3]), 4)
                f = amplitude*sp.Piecewise((x/pivot, x <= pivot), ((L-x)/(L-pivot), True))
                tol = 0.02
            else:
                right = "neumann"
                skew = sp.Rational(rng.choice([1, 2, 3]), 3)
                f = amplitude*x*(2*L-x)*(1+skew*(x-L)**2/L**2)
    record = dict(level=level, sub=sub, seed=seed, L=sp.sstr(L), k=sp.sstr(k),
                  left=left, right=right, f=sp.sstr(sp.expand(f)), tol=tol)
    record["id"] = identity(record)
    if references and exact is not None:
        record["reference"] = sp.sstr(exact)
    return record


def signature(records):
    return hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()


def generate(directory, seed=42, train=60, validation=15, test=15):
    sizes = dict(train=train, validation=validation, test=test)
    if any(size < 5 or size % 5 for size in sizes.values()):
        raise ValueError("split sizes must be positive multiples of five")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if any((directory / f"{split}.jsonl").exists() for split in sizes):
        raise ValueError("dataset already exists; choose a new output directory")
    seen, splits, serial = set(), {}, seed
    for split, size in sizes.items():
        records = []
        for index in range(size):
            level, sub = index % 5, (index // 5) % 3 + 1
            for _ in range(10000):
                record = make(level, sub, serial)
                serial += 1
                if record["id"] not in seen:
                    break
            else:
                raise ValueError("unique problem pool exhausted")
            seen.add(record["id"])
            record["split"] = split
            records.append(record)
        splits[split] = records
    manifest = dict(version=1, seed=seed, splits={})
    for split, records in splits.items():
        (directory / f"{split}.jsonl").write_text("".join(json.dumps(r, sort_keys=True)+"\n" for r in records))
        manifest["splits"][split] = dict(count=len(records), signature=signature(records))
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    return manifest


def load(path, split):
    path = Path(path)
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not records or any(r.get("split") != split or r["id"] != identity(r) or "reference" in r for r in records):
        raise ValueError("invalid dataset content or split")
    if len({r["id"] for r in records}) != len(records):
        raise ValueError("duplicate physical problems")
    manifest = json.loads((path.parent / "manifest.json").read_text())
    if manifest["version"] != 1 or manifest["splits"][split]["signature"] != signature(records):
        raise ValueError("dataset manifest mismatch")
    return records
