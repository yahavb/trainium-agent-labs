# v7 on the instrumented agent.py: compatibility check

Tree: team/master at `ce0403c` (feedback_v7 7417cf9 on top of e7663a3's token accounting, confidence
and held-out verdicts). NKI 0.6.0 CPU simulator in `python:3.12-slim` (Docker, `NEURON_PLATFORM_TARGET_OVERRIDE=trn2`).
No `neuronx-cc` locally, so `NKI_VERDICT_COMPILE` was left unset; everything else is V7.md's configuration:
`PYTHONPATH=.:check MESSAGES=v5 CARD=category PROMPT1=v2 SAMPLING=qwen REPAIR_PROMPT=restructure GRADE_TIMEOUT=120 GATE=static NKI_VERDICTS=... USAGE_LOG=...`.

## 1. Offline, V7.md's command

`python3 feedback_v7.py --offline --all --rounds 2 --samples 2 --repeat 2 --context 8192 --log ... --verdicts ...`

- exit 0; 32 attempts, **all 32 carry `prompt_split`** (v7's API card is recognised as the `api_card` segment)
- both verdict files written, 8 lines each: v7's `NKI_VERDICTS` (VERIFIED, 0.90/0.90/0.74/0.88) and
  agent.py's `--verdicts` (CONFIDENCE printed first, then VERIFIED 20/20, 16/16, 4/4, 16/16); calibration
  table and Brier (0.123) printed at the end

## 2. The token and verdict code changes nothing v7 sends or scores

v7 was run twice against the same deterministic mock endpoint (the answer is a function of the
prompt's sha256: a broken variant or the reference), `--all --rounds 6 --samples 3 --repeat 2`:
once on this tree, once on the same tree with `agent.py` replaced by 4350038's (before any token or
verdict code). The mock logged every request body.

| | before (4350038 agent.py) | this tree |
|---|---|---|
| requests | 60 | 60 |
| full request bodies (messages, max_tokens, temperature, top_p, top_k, chat_template_kwargs) | | **byte-identical** (sorted multiset) |
| attempt sequence (level, round, reward, feedback, code) | | **identical** |
| v7's own verdicts (level, status, confidence) | | **identical** |
| `prompt_split` on attempts | - | 60 / 60 |
| agent.py verdicts | - | 8 |

The run reached the repair prompt and the failure ledger (4 ledger rounds), so those paths are covered.

## 3. Tokens under v7: exact numbers are in USAGE_LOG, not attempts.jsonl

v5's `ask5` replaces agent.py's `ask` and returns a plain string, so attempts.jsonl's token fields
fall back to `count_method = chars/4` estimates (V7.md says the same). The exact server counts are in
`USAGE_LOG`, one line per request with `level` and `code_sha1`: **60 of 60 attempts join to it on
(level, sha1(code))**. So the final token chart should take its totals from USAGE_LOG; no agent change
is needed.
