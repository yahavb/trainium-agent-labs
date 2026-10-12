#!/usr/bin/env python3
"""The kernel-transfer track: the organisers' NKI kernel agent, with one feedback switch.

DESIGN.md 6.9. The question: does the checker's message fix what the model gets wrong when the
model has never seen the language (NKI), the way it does for a language it knows (Verilog)?

    --feedback located   the organisers' agent exactly as shipped: enrich() turns errors into
                         instructions (K-located; the 2026-10-10 baseline is this setting)
    --feedback raw       enrich() switched off: the simulator's error, verbatim (K-raw)
    --feedback docs      located, plus the real signature and docstring of every NKI call the
                         failing kernel uses, read from the installed nki package (K-docs)

Everything else -- prompts, ledger, stop rules, scoring, logging -- is the organisers' code, imported
from ../../02-kernel-agent, so the three settings differ in the feedback text and nothing else.

    cd /tmp/kt && python3 /workspace/projects/19-verified-rtl-agent/kernel-transfer/kernel_agent.py \\
        --feedback raw --all --rounds 8 --samples 1 --context 8192 --repeat 3 \\
        --log /workspace/projects/19-verified-rtl-agent/kernel-transfer/runs/K-raw.jsonl

Run it from a scratch directory: the NKI simulator writes cache folders into the working directory.
"""
import argparse
import inspect
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
KERNEL_AGENT_DIR = os.path.normpath(os.path.join(HERE, "..", "..", "02-kernel-agent"))
sys.path.insert(0, KERNEL_AGENT_DIR)

import agent as K  # noqa: E402  (the organisers' projects/02-kernel-agent/agent.py)

DOCS_BUDGET = 2400          # characters, about 600 tokens: the repair prompt must still fit 8192
DOC_LINES = 3               # first lines of each docstring
NKI_CALL = re.compile(r"\b(nl|nisa)\.([A-Za-z_]\w*)")


def docs_for(source: str) -> str:
    """Signature and first docstring lines for each NKI call in the kernel, in order of first use.

    Names that do not exist are skipped: enrich() already answers those with the closest real names.
    """
    mods = {}
    try:
        import nki.isa as nisa
        import nki.language as nl
        mods = {"nl": nl, "nisa": nisa}
    except ImportError:
        return ""
    seen, lines, used = set(), [], 0
    for prefix, name in NKI_CALL.findall(source or ""):
        key = f"{prefix}.{name}"
        if key in seen:
            continue
        seen.add(key)
        obj = getattr(mods[prefix], name, None)
        if obj is None or not callable(obj):
            continue
        try:
            sig = str(inspect.signature(obj))
        except (TypeError, ValueError):
            sig = "(...)"
        doc = " ".join((inspect.getdoc(obj) or "").splitlines()[:DOC_LINES]).strip()
        entry = f"- {key}{sig}" + (f": {doc}" if doc else "")
        if used + len(entry) > DOCS_BUDGET:
            break
        lines.append(entry)
        used += len(entry)
    if not lines:
        return ""
    return "Reference for the NKI calls in your kernel, from the installed nki package:\n" + "\n".join(lines)


def install(mode: str):
    """Patch exactly one thing per mode; 'located' patches nothing."""
    if mode == "raw":
        K.enrich = lambda error_text: error_text
    elif mode == "docs":
        original_grade = K.grade

        def grade_with_docs(source, level):
            reward, parts, feedback = original_grade(source, level)
            if not parts.get("correct"):
                extra = docs_for(source)
                if extra:
                    feedback = f"{feedback}\n\n{extra}"
            return reward, parts, feedback

        K.grade = grade_with_docs


def selftest() -> int:
    fails = 0
    src = ("import nki.language as nl\nimport nki.isa as nisa\n"
           "t = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.sbuf)\n"
           "nisa.dma_copy(dst=t, src=a)\nnl.dot(t, t)\nnisa.dma_copy(dst=o, src=t)\n")
    d = docs_for(src)
    ok = ("nl.ndarray" in d and "nisa.dma_copy" in d and "nl.dot" not in d
          and d.count("nisa.dma_copy") == 1 and len(d) <= DOCS_BUDGET + 200)
    fails += not ok
    print(f"  {'ok  ' if ok else 'FAIL'}  docs_for: real calls documented once, invented ones skipped ({len(d)} chars)")
    before = K.enrich
    install("raw")
    ok = K.enrich("raised X: 'MemoryRegion' object is not callable") == "raised X: 'MemoryRegion' object is not callable"
    fails += not ok
    print(f"  {'ok  ' if ok else 'FAIL'}  raw: enrich() returns the error unchanged")
    K.enrich = before
    ok = "nl.ndarray(shape, dtype=nl.float32, buffer=nl.sbuf)" in K.enrich("'MemoryRegion' object is not callable")
    fails += not ok
    print(f"  {'ok  ' if ok else 'FAIL'}  located: the organisers' enrich() is untouched")
    print("ALL OK" if not fails else f"{fails} FAILED")
    return 1 if fails else 0


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--feedback", choices=("located", "raw", "docs"), default="located")
    ap.add_argument("--kt-selftest", action="store_true")
    mine, rest = ap.parse_known_args()
    if mine.kt_selftest:
        sys.exit(selftest())
    install(mine.feedback)
    print(f"kernel-transfer: feedback={mine.feedback} (organisers' agent from {KERNEL_AGENT_DIR})", flush=True)
    sys.argv = [sys.argv[0]] + rest
    K.main()


if __name__ == "__main__":
    main()
