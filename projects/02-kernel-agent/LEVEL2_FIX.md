# Level 2: transpose repair

Branch: `fix/kernel-agent-level2`, based on `master` at `8f1ca41`.

In seat 21's first baseline pass, 28 attempts scored 0.30 or 0.50 and none
passed all shapes. Candidates allocated 128-row tiles for smaller inputs,
loaded only part of the flattened free axis, or used `nl.min(128, P)` for an
integer bound. The last call raises an `int`/`dtype` error; the old feedback
incorrectly told the model to use more nl/nisa functions.

The change gives level 2 a transpose-specific API card, keeps the partition
axis intact, describes shape-derived tiles and rank-preserving column copies,
and distinguishes Python integer bounds from tensor reductions. Repair prompts
permit correcting allocations and index mappings. Other levels keep their
prompts and feedback; scoring and numeric tolerances are unchanged.

Validation performed: Python compilation and the checker self-test passed.
The pod's NKI simulator reproduced the `nl.min(128, P)` integer/dtype failure;
replacing that call with Python `min` in the diagnostic kernel passed all four
configured shapes. The shipped reference also passed 4/4. Non-target initial
and repair prompts, representative error messages and scoring weights were
compared against `master` and remained unchanged.

Run this branch in a separate checkout after other agent runs have stopped:

```bash
python nkibench.py --selftest
python nkibench.py --level 2 --check reference_level2.py
nohup python -u agent.py --level 2 --rounds 8 --samples 4 --context 8192 --repeat 5 \
  --log attempts-level2-fix.jsonl > run-level2-fix.log 2>&1 < /dev/null &
tail -f run-level2-fix.log
```

The live solve-rate improvement is **not yet measured**. Compare the five-run
solve rate against an unchanged baseline. The harness uses a shared
`/tmp/_agent_level2.py` path, so separate working directories alone do not make
concurrent agent runs safe.
