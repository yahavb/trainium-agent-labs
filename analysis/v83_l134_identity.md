# v8.3 (5c3aba2) sends the same requests as v8.2 (96a9fc9) on levels 1, 3 and 4

Why it matters: the final group takes levels 1, 3 and 4 from 96a9fc9's runs and level 2 from 5c3aba2's. That
is only one configuration if L2CAT changes nothing outside level 2. Executor 1 showed this offline; this is an
independent check.

Method: both commits' `projects/02-kernel-agent` (via `git archive`), run with V8.md's v8.2 switches
(`SKELETON=0 L1FIX=1 TRUNCFIX=1 MIXSAMP=1 L2HINT=1 MIX=1`, plus `L2CAT=1` for 5c3aba2) against the same
deterministic mock model (its answer is a function of the prompt's sha256; it logs every request body).
`--level 1`, `--level 3` and `--level 4`, `--rounds 6 --samples 1 --repeat 3`, NKI 0.6.0, trn2.

| | 96a9fc9 | 5c3aba2 (L2CAT=1) |
|---|---|---|
| requests | 27 | 27 |
| full request bodies (messages, sampling, max_tokens) | | **byte-identical** (sorted multiset) |
| attempt sequences (level, run, round, reward, feedback, code) | | **identical** |
| repair-round attempts covered | | 18 (level 1 ran all 6 rounds in each of 3 runs; level 4 two rounds) |

A first try with `--samples 4` solved every level in round 0 (12 requests, also identical) and never reached a
repair, so it was rerun with one sample per round to cover the repair path. Level 2 was not compared: L2CAT is
meant to change it.

## WARM (15fb0d5) against v8.3 (5c3aba2) on levels 1-4

15fb0d5 (`feedback_v8.py` md5 6c84ec47a8a1fede6e17ddfd387216be) adds `WARM=1`, also `warm_l4_8339bbd6.py`, and
nkibench's `--check` traffic fix (3788320, which grading does not use). Same method: v8.3's switches
(`... MIX=1 L2CAT=1`) for both, plus `WARM=1` for 15fb0d5, levels 1-4, `--rounds 6 --samples 1 --repeat 3`.
The first mock solved level 2 in round 0 every time, so a third run used a mock that answers level 2 only with
two wrong transposes L2CAT recognises (x unchanged; a dropped loop variable). That forces level 2 through its
repair rounds and L2CAT's messages.

| mock | requests (each commit) | request bodies | attempts | repair attempts covered |
|---|---|---|---|---|
| hash mock (sometimes the reference) | 30 | byte-identical | identical | 18 (L1, L4) |
| never the reference | 48 | byte-identical | identical | 36 (L1, L3, L4) |
| level 2 always a wrong transpose | 63 | **byte-identical** | **identical** | **51** (L1 18, L2 15, L3 9, L4 18; all 18 L2 attempts carry L2CAT's message) |

**Result: WARM changes nothing on levels 1-4.**

## What WARM sends in round 0 of levels 5-7

The round-0 prompts that 15fb0d5 builds with `WARM=1` (`agent.first_prompt(5|6|7)`; 1,904 / 1,938 / 1,928
characters). Each carries `warm_l4_8339bbd6.py`, renamed to the level's entry point, byte for byte. That is the
agent's own level-4 solve, and the kernel file itself shares no line with any forbidden source. Scanned like
analysis/prompt_leak_check.md (lines of 40+ characters against the 7 tutorial kernels, reference_level1-4 and
ans_level9-14): levels 5 and 7, 0 lines; level 6, 1 line: `def nki_matmul_block_free_dimension_(lhsT, rhs):`.
That line is the level's required entry point. The harness names the function (nkibench's level 6) and every
level-6 prompt must state it. It matches the tutorial's line because the organizers took the name from there.
No `.ap([` in any of them.
