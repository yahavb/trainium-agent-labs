# VeriLoop — 2-minute demo script

> **DRAFT — numbers marked ⏳ come from the last runs (~4:45 pm).** The case below is a real run:
> `s31-05_mac-C-r1-182546` in `results/2026-10-10_s31_attempts.jsonl`.

## 0:00 — the one-line pitch (say it)

> "A small model on your chip designs chip hardware. A simulator grades every design down to the exact
> signal and clock cycle. We measured how much the *quality* of that feedback changes whether it succeeds."

## 0:15 — the loop (show the diagram in README, or draw it)

spec → **Qwen3-8B on Trainium** writes Verilog → **simulator** checks every test → score + feedback → back to
the model. The checker is the only judge: the agent says `SOLVED` only when every test passed.

## 0:35 — one real failure and recovery (level 5, the MAC cell)

Show, round by round (from the attempt log):

1. **Round 0 — confidently wrong.** The model wrote
   `input signed [8:0] b,  // Adjusted to 9 bits to handle signed multiplication` — the spec says 8 bits.
   It explained its mistake in a comment. Score **0.00**.
2. **The checker's feedback C:** `Port b is 9 bit(s) wide, but the spec says 8. Declare it as [7:0] b.`
   Note: iverilog itself only *warns* about this and silently drops a bit — our checker fails it.
3. **Round 1 — one line changed** (`[8:0]` → `[7:0]`), all 4 attempts pass all 242 tests:
   `SOLVED -- verified by the checker on every test`.

Say honestly: *"On this level A and B recovered on round 2 as well — the model only needed to know it was
wrong. The hard levels are where we hoped C would matter."*

**Level-4 alternative** (if asked what feedback C looks like on a hard level):
`cycle 6: inputs reset=0 ped=0 -> light = YELLOW (2), expected GREEN (1)  <-- first wrong cycle`

## 1:05 — the result (show `results/graph.png`)

- Easy blocks (mux, adder, counter): solved on the first try with **any** feedback — 72/72.
- MAC: fixed on round 2 with any feedback — 9/9.
- Traffic light and FIFO: **0 solved with any feedback**; C scores slightly higher on the traffic light
  (0.69 vs 0.65). ⏳ 12-round runs.

## 1:30 — what we learned (pick two)

- The model's mistakes are **systematic**: 67% of traffic-light failures hold a light one phase too long;
  every MAC failure was the same 9-bit port. (`results/TAXONOMY.md`)
- The model **goes in circles** without being told it repeated itself — identical design 6 rounds running.
- "8 runs" can be **2 runs**: the model is nearly deterministic, so we fingerprinted every design to prove
  our runs are independent.
- The model **does not know when it is wrong**: designs it rated 80–100% confident were right only 52%
  of the time; 16% of its wrong designs got ≥ 80. That is why only the checker may say "solved".

## 1:50 — close

> "Everything — the checker, every attempt, every number — is in the repo, and `selftest.py` proves each
> level before the model ever sees it."

## Backup if the live part fails

Show `results/graph.png` and the round-by-round story from the log file in a text editor — no live model
needed.
