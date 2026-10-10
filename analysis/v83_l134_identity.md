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
