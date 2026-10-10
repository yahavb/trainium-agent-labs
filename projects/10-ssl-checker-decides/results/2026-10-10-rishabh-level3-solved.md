# Level 3 solved in the normal agent loop: step list + a named PSUM fix

- **Who / seat:** Rishabh (+ Claude), seat 48 (shared with two `--samples 1`/`2` jobs; scores unaffected)
- **Commit measured:** uncommitted; `agent.py` md5 e50a1afd… (laptop and `/root/l3run` on seat 48),
  `nkibench.py` 68e795d5…
- **Hypothesis:** level 3 (one-tile matmul, 0.30 every run) is blocked by two avoidable mistakes, a
  1-D tile and a reshape, plus one error with no fix message. A step list and that one message solve it.
- **Change:** (1) `--step-list` for level 3 (`step_list_note_l3()`): "each input is already exactly
  one tile, so use it whole: no loops, no reshape, every tile 2-D… allocate one PSUM tile acc of shape
  (M, N)… move acc out to the result". It names the method and API, as the instructions card already
  does. (2) A new `enrich()` message for "dma_copy requires HBM or SBUF tensors, got src=psum", built
  from the kernel's OWN names: `acc_sbuf = nl.ndarray(acc.shape, dtype=result.dtype, buffer=nl.sbuf)` /
  `nisa.tensor_copy(dst=acc_sbuf, src=acc)` / `nisa.dma_copy(dst=result, src=acc_sbuf)`.
- **Command:** `python agent.py --level 3 --step-list --rounds 8 --samples 1 --context 8192 --repeat 2`
- **Log:** `logs/attempts-rishabh-step-list-l3-psum.jsonl`; solved kernel `logs/rishabh-l3-solved-kernel.py`

## Scores

```
baseline (no step list):               level 3: solved 0/4   all = [0.30, 0.30, 0.30, 0.30]
messages v1:                           level 3: solved 0/3   all = [0.30, 0.30, 0.30]
step list only (agent.py 5fce636f):    level 3: solved 0/3   all = [0.30, 0.30, 0.30]
step list + named PSUM fix (e50a1afd): level 3: solved 2/2   all = [1.00, 1.00]   solved on round 2 both runs
```

Verified independently with the stock `nkibench.py --level 3 --check`: rules clean, 1/1 shape.
`stress.py --level 3`: 28/28 generated inputs with hostile values. Both runs produced the identical
kernel (greedy), so this shows the result is reproducible, not robust.

## Verdict

**Kept. Level 3 goes from 0.30 to solved in the normal benchmark loop.** The two changes worked in
sequence: the step list alone got rid of the 1-D tile and reshape walls, but every attempt then DMA'd
PSUM straight to HBM (12 of 15) and got a bare assertion. Once that error named the missing
`tensor_copy` hop with the kernel's real names, round 2 solved it. That's today's lesson on a fresh
level: one named, specific fix, with no placeholders, moves the model.
