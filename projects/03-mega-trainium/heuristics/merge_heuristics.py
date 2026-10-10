#!/usr/bin/env python3
"""Merge the per-source heuristic extractions into one rule database and generate the derived files.

    python3 merge_heuristics.py SRC_DIR OUT_DIR

Inputs:  SRC_DIR/from_*.json  (arrays of records in the shared schema) and SRC_DIR/nki_constants.json
Outputs: OUT_DIR/rules.json          merged, de-duplicated records with a `tier` and `source_file`
         OUT_DIR/APPENDIX.md         every rule grouped by category (generated; cited)
         OUT_DIR/cards/<category>.md one retrievable card per category (tier 1)
         OUT_DIR/nki_constants.json  copied through

De-duplication: exact id collisions keep the record with the longer `why`; near-duplicates (same
normalised rule text) are merged by concatenating sources. Nothing is dropped silently: a
`merged_from` field lists what was folded in.
"""
import collections, json, os, re, shutil, sys

SRC, OUT = sys.argv[1], sys.argv[2]
os.makedirs(os.path.join(OUT, "cards"), exist_ok=True)

CATEGORY_ORDER = [
    "op-selection", "dma", "memory-sbuf-psum", "tiling", "matmul", "engine-mapping", "scheduling-pipelining",
    "lnc-sharding", "collectives", "numerics-dtype", "correctness-pitfall", "api-constraints", "compiler-flags",
    "profiling-diagnosis",
]
CATEGORY_TITLE = {
    "op-selection": "Where custom kernels pay off (op selection)",
    "dma": "DMA: payload size, descriptors, coalescing",
    "memory-sbuf-psum": "SBUF and PSUM budgeting",
    "tiling": "Tiling and layout",
    "matmul": "Tensor Engine matmul",
    "engine-mapping": "Which engine runs what",
    "scheduling-pipelining": "Scheduling, multi-buffering, pipelining",
    "lnc-sharding": "LNC=2 sharding across the two physical cores",
    "collectives": "Collectives",
    "numerics-dtype": "Numerics and dtypes",
    "correctness-pitfall": "Correctness pitfalls",
    "api-constraints": "API constraints (NKI 0.6.0)",
    "compiler-flags": "Compiler flags and options",
    "profiling-diagnosis": "Reading a profile: diagnosis rules",
}


def norm(text):
    return re.sub(r"[^a-z0-9 ]", "", (text or "").lower()).strip()


records = []
for f in sorted(os.listdir(SRC)):
    if f.startswith("from_") and f.endswith(".json"):
        for r in json.load(open(os.path.join(SRC, f))):
            r = dict(r)
            r["source_file"] = f
            r.setdefault("numbers", {})
            records.append(r)
print(f"loaded {len(records)} records from {SRC}")

# 1. exact id collisions
by_id = collections.OrderedDict()
for r in records:
    if r["id"] in by_id:
        keep, other = by_id[r["id"]], r
        if len(other.get("why") or "") > len(keep.get("why") or ""):
            keep, other = other, keep
        keep["source"] = f"{keep['source']} ; {other['source']}"
        keep.setdefault("merged_from", []).append(f"{other['source_file']}:{other['id']}")
        by_id[r["id"]] = keep
    else:
        by_id[r["id"]] = r
# 2. near-duplicate rule text
by_rule = collections.OrderedDict()
for r in by_id.values():
    k = norm(r["rule"])[:120]
    if k in by_rule:
        keep = by_rule[k]
        keep["source"] = f"{keep['source']} ; {r['source']}"
        keep.setdefault("merged_from", []).append(f"{r['source_file']}:{r['id']}")
    else:
        by_rule[k] = r
merged = list(by_rule.values())
print(f"after de-duplication: {len(merged)} records ({len(records) - len(merged)} folded)")

# tiers: 0 = always-on card (hand-picked ids in TIER0 below), 1 = category cards, 2 = appendix only
TIER0 = {
    # the rules the agent must never violate, chosen for a batch=1 decode / memory-bound regime
    "mm-operands-sbuf-dst-psum", "mm-contraction-on-partition-dim", "psum-accumulate-over-k",
    "output-tensors-shared-hbm", "layout-partition-dim-first-no-block-dims", "mig-neuronxcc-namespace-forbidden",
    "dma-free-dim-at-least-2kib-contiguous", "dma-descriptor-at-least-4kib", "use-full-128-partitions",
    "load-once-reuse-in-sbuf", "keep-intermediates-in-sbuf", "hoist-parameter-loads-out-of-inner-loops",
    "memory-vs-compute-bound-via-intensity", "write-k-cache-transposed-for-decode",
    "dma-use-all-128-partitions", "mm-fast-load-stationary-vector-moving", "dge-hwdge-600ns-per-instruction",
}
for r in merged:
    r["tier"] = 0 if r["id"] in TIER0 else 1
    r["keywords"] = sorted(set(re.findall(r"[a-z_][a-z0-9_\.]{3,}", norm(r["rule"] + " " + (r.get("trigger") or "") + " " + r["id"].replace("-", " ")))))[:40]

json.dump(merged, open(os.path.join(OUT, "rules.json"), "w"), indent=1)
if os.path.exists(os.path.join(SRC, "nki_constants.json")):
    shutil.copy(os.path.join(SRC, "nki_constants.json"), os.path.join(OUT, "nki_constants.json"))


def fmt(r, with_numbers=True):
    out = [f"- **{r['id']}** — {r['rule']}"]
    if r.get("why"):
        out.append(f"  - why: {r['why']}")
    if r.get("trigger"):
        out.append(f"  - when: {r['trigger']}")
    if r.get("fix"):
        out.append(f"  - fix: {r['fix']}")
    if with_numbers and r.get("numbers"):
        out.append("  - numbers: " + "; ".join(f"{k} = {v}" for k, v in r["numbers"].items()))
    if r.get("check_sketch"):
        out.append(f"  - check ({r.get('detectable')}): {r['check_sketch']}")
    out.append(f"  - source: {r['source']}  [{r.get('confidence')}, {r.get('generation')}]")
    return "\n".join(out)


bycat = collections.defaultdict(list)
for r in merged:
    bycat[r["category"]].append(r)

with open(os.path.join(OUT, "APPENDIX.md"), "w") as f:
    f.write("# NKI heuristics — full rule appendix (generated by merge_heuristics.py)\n\n")
    f.write(f"{len(merged)} rules from {len(set(r['source_file'] for r in merged))} extraction passes. ")
    f.write("Each rule cites its source; `stated` means the source says it explicitly, `inferred` means derived.\n\n")
    for c in CATEGORY_ORDER:
        rs = bycat.get(c, [])
        if not rs:
            continue
        f.write(f"\n## {CATEGORY_TITLE.get(c, c)} ({len(rs)})\n\n")
        for r in rs:
            f.write(fmt(r) + "\n")

for c in CATEGORY_ORDER:
    rs = bycat.get(c, [])
    if not rs:
        continue
    with open(os.path.join(OUT, "cards", f"{c}.md"), "w") as f:
        f.write(f"# {CATEGORY_TITLE.get(c, c)}\n\n")
        for r in rs:
            line = f"- {r['rule']}"
            if r.get("fix"):
                line += f" FIX: {r['fix']}"
            f.write(line.replace("\n", " ") + "\n")

hist = collections.Counter(r["category"] for r in merged)
print("by category:", dict(sorted(hist.items(), key=lambda kv: -kv[1])))
print("detectable:", dict(collections.Counter(r["detectable"] for r in merged)))
print("tier0 present:", sorted(r["id"] for r in merged if r["tier"] == 0))
