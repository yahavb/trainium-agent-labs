#!/usr/bin/env python3
"""Dev-only experiment: does the repair prompt's wording cause the model to copy its broken code?

Seen in runs/dev/C-1.jsonl: 14 of 20 S3 repairs returned code byte-identical to the code they were
asked to fix, so the loop rescued nothing. The S3 wording ends "Change exactly what the checker names
and keep everything else identical." This script swaps only that S3 text for a rewrite request and
otherwise runs agent.py unchanged:

    python3 dev_experiments/s3_rewrite.py --run C --rep 1 --problems eval/dev.txt \\
        --only Prob001_zero,... --out runs/dev/C-s3rewrite-1.jsonl

Dev problems only (DESIGN 10): nothing here may touch eval/heldout.txt.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import prompts  # noqa: E402

TAIL_OLD = "Change exactly what the checker names and keep everything else identical."
TAIL_NEW = ("Rewrite the complete module TopModule so that what the checker names is fixed. "
            "Your module must differ from the code above.")

# Since DESIGN 1.3.0 (D-013) prompts.py ships TAIL_NEW itself, so this script is then a plain agent.py.
if TAIL_OLD in prompts.S3:
    prompts.S3 = prompts.S3.replace(TAIL_OLD, TAIL_NEW)
    prompts.S3_NO_SPEC = prompts.S3_NO_SPEC.replace(TAIL_OLD, TAIL_NEW)
assert TAIL_NEW in prompts.S3 and TAIL_NEW in prompts.S3_NO_SPEC, "prompts.py changed; update this script"

if "heldout" in " ".join(sys.argv):
    sys.exit("dev only: this experiment must not run on the held-out set")

import agent  # noqa: E402

if __name__ == "__main__":
    agent.main()
