# Make a correct level-4 kernel faster: which feedback works?

- **Who / seat:** Rishabh (+ Claude), seat 48
- **Commit measured:** uncommitted. The code is `roofline/optimize.py` and `roofline/instcount.py` in
  the team repo clone (`~/nyu/hackathon/hack-the-chip-team`), copied to `/root/roofline/` in seat 48
  (md5 9e75ebfb… optimize.py, 3d5028fc… instcount.py). The roofline tool itself is team/main `1b92c1c`.
- **Hypothesis:** feedback built from the roofline tool helps Qwen3-8B speed up a correct kernel
  more than predicted-latency numbers alone.
- **Change:** none to the workshop repo. This is a separate optimization loop.
- **Command:** `python optimize.py --condition <specific|roofline|control> --repeat 3 --rounds 6 --samples 4 --patience 3`
- **Log:** `/root/roofline/opt-<condition>.jsonl` in seat 48; the first attempt is in `/root/roofline/run1-generic/`

## Setup

The loop starts from a **correct but slow** level-4 matmul: `reference_level4.py` with `TILE_N = 128`.
Its predicted latency is **330.0 µs**, summed over the 4 nkibench shapes. The tutorial's reference
(`TILE_N = 512`) is 118.0 µs; the roofline model floor is 32.7 µs.

Each round, Qwen proposes 4 changes. A change is kept only if it's correct on all 4 shapes
(`nki.simulate`) and its predicted latency (compiler, no device) improves by more than 1%. The three
conditions differ in one paragraph of feedback only:

- **control:** the predicted latency numbers.
- **roofline:** the numbers plus the roofline diagnosis (floor, what limits it, wasted bytes, busiest
  engine).
- **specific:** the numbers, the floor in one line, and ONE named change from an instruction counter
  (`instcount.py` wraps `nisa.nc_matmul` in the simulator). For this kernel it says: "the kernel
  issues 64 nc_matmul instructions where 16 would do, because the tile you pass as moving= is only 128
  columns wide; nc_matmul accepts up to 512…"

## Scores

```
specific  speedups = [3.02, 3.02, 3.02]   all runs 330.0 -> 109.1 us   (118.0 by round 0, 109.1 by round 1)
control   speedups = [1.03, 1.03, 1.03]   all runs 330.0 -> 320.8 us   (one allocation moved out of the loop)
roofline  run 1:     1.00                 24/24 candidates returned the kernel UNCHANGED   [stopped after 1 run]
```

All three runs within each condition are identical. That fits greedy sampling on these seats: in the
baseline, 91 of 108 rounds returned 4 byte-identical samples. So "3 runs" here is close to one
deterministic trajectory repeated, and the result is deterministic rather than a rate.

The 109.1 µs kernel is **faster than the tutorial's reference** (118.0 µs). Qwen split the k loop in
two: it loads all the rhs tiles first, then streams the lhsT tiles and runs the matmuls. It passes all
4 fixed shapes and `stress.py --level 4` (50 generated inputs including hostile values). Kernel: `roofline/results/best_level4_specific_109us.py` in the team repo clone (and
`/tmp/opt_best_specific_0.py` in seat 48). Logs: `roofline/results/opt-*.jsonl`.

Every round took 4–7 min, because up to 3 jobs shared one model server (4 slots). Scores are
unaffected.

## Verdict

- **Kept: one named, counted change works, and generic advice doesn't.** With the instruction count
  against the minimum ("64 nc_matmul instructions where 16 would do… moving= is only 128 columns wide;
  nc_matmul accepts up to 512"), Qwen reached 3.02x in every run, beating the tutorial's reference.
  With the latency numbers alone, it reached 1.03x.
- **The roofline diagnosis on its own made Qwen freeze** (24/24 unchanged): true, but it pointed three
  ways at once and named no line of code. That is the README's "off by 341 percent" lesson again.
- **Caveat:** one starting kernel with one dominant flaw, and a checker that knows how to count that
  flaw. The next test of generality is a kernel whose waste is different, e.g. softmax with
  extra engine passes.
