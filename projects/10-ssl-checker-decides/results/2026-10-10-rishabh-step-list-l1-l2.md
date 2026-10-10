# Step lists + named fix messages on levels 1 and 2 (last hour, not solved)

- **Who / seat:** Rishabh (+ Claude), seat 45, from the separate copy `/root/stepruns`
- **Commit measured:** uncommitted; v1 runs `agent.py` a07db68b, v2 runs `agent.py` 1c17bbaf
- **Hypothesis:** the recipe that solved level 3 (a method-level step list, then one named fix for
  the error it stalls on) also moves levels 1 and 2.
- **Change:** v1 = `--step-list` (`STEPS_L12`: method and API only, no strides or index arithmetic).
  v2 = plus two `enrich()` messages: the `ap()` partition-stride rule (level 1), and the reference's
  exact signature when the entry point takes the wrong number of arguments (level 2).
- **Command:** `python agent.py --level N --step-list --rounds 8 --samples 1 --context 8192 --repeat 2|3`
- **Log:** `logs/attempts-rishabh-step-list-l{1,2}-steps-v{1,2}.jsonl`

## Scores

```
level 1  baseline 0.30 every run     v1 [0.30, 0.30]   v2 [0.30, 0.30]   0 solved
level 2  baseline solved 3/7 today   v1 [0.30, 0.30, 0.30]   v2 [0.62, 0.50]   0 solved
```

- **Level 1:** the step list got Qwen to the strided `tile.ap` view, then every attempt (8/8 v1, 11/11
  v2) failed on the same error, all-zero strides. The stride-rule message didn't land.
- **Level 2:** v1's step list made Qwen drop the `shape2D` parameter (12/12). v2's signature message
  fixed that, and the runs climbed to 0.62 and 0.50 (now NUMERICAL MISMATCH: a real transpose, wrong
  index mapping). But the solve rate (0/5) is below the baseline's 3/7, so **the step list hurts level 2.**

## Verdict

**Reverted for levels 1 and 2, not adopted.** The level-3 recipe doesn't transfer automatically. On
level 2 the step list displaced a kernel shape that already worked about half the time. On level 1
the blocking error is the `ap()` stride arithmetic itself, which a method-level message can't supply
without giving the answer. The two new messages stay as checker improvements: each cleared its target
error (the level-2 signature error went from 12/12 to 3/15).
