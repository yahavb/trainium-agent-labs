# Prompt leak check: no answer kernel line reaches a prompt

**Conclusion: 0 hits.** No line of 40 or more characters from a tutorial kernel, the organizers' reference
kernels or our level 9-14 answers appears in any of the 35 distinct prompts that v7 sent
(80 requests, levels 1-4 and 9-14, first and repair rounds), nor in any of the 2851
string constants of the agent and feedback code. The one access-pattern mention, `tile.ap([[stride, count], ...])`,
is the API card's signature with placeholders, not the level-1 pattern. One worked example (a per-channel
mean) is structurally close to level 1 and is described under Limits. The same scan finds 95 hits when
answer code *is* put into prompts (the positive control below), so the 0 is not the scan missing them.

## Forbidden sources (417 distinct lines after filtering, from 716 line occurrences)

- tutorial kernels: `average_pool2d_nki_kernels.py`, `mamba_nki_kernels.py`, `matrix_multiplication_nki_kernels.py`, `spmd_multiple_nc_tensor_addition_nki_kernels.py`, `spmd_tensor_addition_nki_kernels.py`, `tensor_addition_nki_kernels.py`, `transpose2d_nki_kernels.py`
- organizers' reference kernels: `reference_level1.py`, `reference_level2.py`, `reference_level3.py`, `reference_level4.py`
- our answers (levels 9-14): `ans_level9.py`, `ans_level10.py`, `ans_level11.py`, `ans_level12.py`, `ans_level13.py`, `ans_level14.py`

## Prompts scanned

feedback_v7 with V7.md's configuration (`MESSAGES=v5 CARD=category PROMPT1=v2 SAMPLING=qwen
REPAIR_PROMPT=restructure GRADE_TIMEOUT=120 GATE=static`, trn2; only `NKI_VERDICT_COMPILE` off, since there is
no compiler here), `--all` (levels 1-4) and `--level 9` ... `--level 14`, `--rounds 4 --samples 2`, against a mock
model that logs every request body. **The mock never answers with code from a forbidden source**: each reply is
one of four broken kernels written for this check (no code; a 256-row tile; a call to the invented
`nisa.multiply`; no `@nki.jit`), chosen by the prompt's hash. So the repair prompts, which quote the previous
kernel, cannot carry a forbidden line that the mock supplied. Those four replies drive the agent through its
repair messages (partition limit, invented name, rule violation, no code / shorter re-ask).

| level | distinct first prompts | distinct repair prompts |
|---|---|---|
| 1 | 3 | 2 |
| 2 | 1 | 2 |
| 3 | 1 | 2 |
| 4 | 1 | 2 |
| 9 | 1 | 3 |
| 10 | 1 | 3 |
| 11 | 1 | 2 |
| 12 | 1 | 3 |
| 13 | 1 | 2 |
| 14 | 1 | 2 |

String constants: every `str` literal and every literal part of every f-string in
`agent.py`, `feedback_v2.py`, `feedback_v3.py`, `feedback_v4.py`, `feedback_v5.py`, `feedback_v6.py`, `feedback_v7.py`, `gate_nki.py`, `verdict_nki.py`, `hidden_eval.py`, `ops07.py`, `ops08.py`: 2851 constants, split into lines.

## Rule

A line counts if, after stripping, it is at least 40 characters, is not an import, and has more than two
words. Whitespace runs are collapsed on both sides before comparing, which can only add matches. A hit is a
line present both in a prompt (or a constant) and in a forbidden source.

## Results

| check | result |
|---|---|
| prompt lines also in a forbidden source | **0** |
| string-constant lines also in a forbidden source | **0** |
| prompts containing `.ap([` | 10 (the first prompts), all through one line |
| constants containing `.ap([` | `agent.py:174` |
| forbidden sources containing `.ap([` | `average_pool2d_nki_kernels.py`, `reference_level1.py` |

The `.ap([` line in the prompts is, verbatim, `tile.ap([[stride, count], ...])           a strided view, for
reductions`: an entry of `agent.API_CARD` (agent.py:174, the organizers' original card). It names the method and
its argument shape. It does not give level 1's strides (the reference's
`[[sz_hin * sz_win, sz_p], [sz_pool * sz_win, sz_hin // sz_pool], ...]`). We count it as documentation and
list it so a reader can judge.

## Positive control

The v7 compatibility check (analysis/v7_compat_check.md) used a mock that *does* reply with the reference
kernels, so its repair prompts quote them. The same scan on its 7 distinct prompts reports **95 hits (58
distinct lines), all in repair prompts, 0 in first prompts**: copyright lines, docstrings and code of
`reference_level*.py`. The scan finds a leak when there is one.

## Limits

- A model can still *write* a line that also appears in a tutorial; that is its output, not something we gave it.
  This check is about what we send.
- Exact lines only (after whitespace collapsing), so a paraphrase would not be caught. The worked examples
  `CARD=category` puts into first prompts were therefore read by hand (feedback_v4 `CARD_ADDITION`, a row mean
  over the free axis, for matmul levels; feedback_v5 `CARD_REDUCE3D`, the mean of each channel of a (C, A, B)
  tensor, for reduction levels). **The channel-mean example is structurally close to level 1 (average
  pooling)**: the same pipeline of one whole-tensor SBUF tile, `nl.sum(..., keepdims=True)` over the free
  axes, `nisa.tensor_scalar` by 1/(count), and a copy out. What it does not contain is what makes pooling
  pooling: the windowed `.ap()` access pattern over (H/p, W/p, p, p). It averages each whole channel plane.
  No line of it matches the reference. We report it as a strong hint for level 1, not the answer.
- Level 8 and levels 5-7 were not run; the final configuration covers levels 1-4 and 9-14.

## Reproduce

`analysis/prompt_leak_check/`: `mock_leak.py` (the mock), `run_leak.sh` (the v7 runs), `leak_scan.py` (the scan),
`scan_result.json` (this result). Paths inside are those of the Docker container (repository at `/h/tal-deliv`,
reference repository at `/h/neuron-agentic-development`).
