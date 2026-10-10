#!/usr/bin/env python3
"""
memory.py — the optimization memory (the AccelOpt idea, scaled to a workshop).

AccelOpt curates insights from past slow->fast kernel pairs and injects them into later
generations. The workshop equivalent: as runs accumulate, distill WHAT WORKED and WHAT
KEPT FAILING into one compact card, and carry it forward -- across rounds within a run
and across runs on the same machine. Stored next to the traces (runs/memory.json), so
sync.sh brings it back with everything else and the dashboard can show it growing.

What memory deliberately is NOT: answers. The reference is already in every prompt; an
insight may say "the level-8 wall was one-pass variance cancellation; two-pass cleared
it" but never contain kernel code.

Write policy (conservative, measured in spirit):
  * a solved level adds one insight per NEW taxonomy that appeared on the way
  * a failure mode seen >= REPEAT_INSIGHT times on a level adds a "the wall" insight
  * insights dedupe by (level, kind, text-hash); repeat counts rise instead

Read policy: the card carries at most CARD_LINES insights, closest level first.
"""

import hashlib
import json
import os
import time

REPEAT_INSIGHT = 3
CARD_LINES = 5

# Distilled, per taxonomy, one line each -- the strategy, not the answer. These are the
# seed insights a fresh install ships with; run-derived insights accumulate beside them.
SEED = {
    "ragged-edge": "clamp tile bounds from the slice's real shape, never from the constant 128",
    "non-finite": "subtract the row max before exp; initialise every output element you read",
    "partial-coverage": "each tile writes ITS OWN output slice; check loop bounds against out.shape",
    "core-arithmetic": "verify the formula on a 3x3 array with a SCRATCH line before writing the loop",
    "wrong-shape": "recompute the output size formula from the input shape first",
    "no-tile-loop": "wrap work in for-loops stepping 128 rows / 512 cols",
    "whole-array-op": "slice a tile first, then do arithmetic on the slice",
    "banned-call": "the ban is on the WHOLE input; the same library call on a tile slice is legal",
    "nondeterministic": "allocate fresh output per call; no globals",
    "modified-input": "write to a new output array; the arguments belong to the caller",
}


class Memory:
    def __init__(self, path):
        self.path = path
        self.items = []
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                self.items = json.load(open(self.path)).get("insights", [])
            except Exception:
                self.items = []

    def save(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        json.dump(dict(insights=self.items[-200:]), open(self.path, "w"), indent=1)

    @staticmethod
    def _key(level, kind, text):
        h = hashlib.sha1(f"{level}|{kind}|{text}".encode()).hexdigest()[:10]
        return h

    def observe(self, level, taxonomy_counts, solved):
        """Record one level's outcome. taxonomy_counts: {label: n} for the attempt."""
        for tax, n in taxonomy_counts.items():
            if not n:
                continue
            text = SEED.get(tax)
            if not text:
                continue
            if solved and n >= 1:
                self._add(level, "cleared-with", text)
            elif n >= REPEAT_INSIGHT:
                self._add(level, "the-wall", text)

    def _add(self, level, kind, text):
        key = self._key(level, kind, text)
        for it in self.items:
            if it["key"] == key:
                it["count"] += 1
                it["last"] = time.time()
                return
        self.items.append(dict(key=key, level=level, kind=kind, text=text,
                               count=1, last=time.time()))

    def card(self, level=None, max_lines=CARD_LINES):
        """The compact injection card. Level-matched insights first, then global."""
        scored = []
        for it in self.items:
            proximity = 0 if it["level"] == level else 1
            scored.append((proximity, -it["count"], it))
        scored.sort(key=lambda t: (t[0], t[1]))
        lines = [f"- ({it['kind']}, level {it['level']}, x{it['count']}) {it['text']}"
                 for _p, _c, it in scored[:max_lines]]
        return "\n".join(lines)
