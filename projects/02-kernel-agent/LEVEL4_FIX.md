# Level 4: tiled matmul repair

Branch: `fix/kernel-agent-level4`, based on `master` at `8f1ca41`.

In seat 21's first baseline pass, all 16 attempts scored 0.30. The selected
candidate copied a 128-by-512 PSUM result (65,536 elements) into the
128-by-128 stationary input buffer (16,384 elements). The old feedback
described this as a generic assignment mismatch and suggested indexing the
output, leaving the actual PSUM-to-SBUF copy unchanged.

The level-4 feedback now identifies the separate, matching result SBUF tile
needed for that copy. The initial and repair prompts explain the different
input/result tile shapes, tiling across M, N and K, and the accumulator's
lifetime across K tiles. Larger inputs cannot be loaded wholesale or handled
by only looping over K. Scoring and numeric tolerances are unchanged.

Validation performed: Python compilation and the checker self-test passed.
The pod's simulator reproduced the reported failure from the saved baseline
candidate. Giving that candidate a separate result SBUF fixed the first shape;
the remaining three shapes exposed its missing input tiling. The shipped
fully tiled reference passed 4/4. Non-target prompts, representative feedback
and scoring weights were compared against `master` and remained unchanged.
These are diagnostic checks, not successful model-generated runs.

After other model experiments have stopped, evaluate this branch separately:

```bash
python nkibench.py --selftest
python nkibench.py --level 4 --check reference_level4.py
nohup python -u agent.py --level 4 --rounds 8 --samples 4 --context 8192 --repeat 5 \
  --log attempts-level4-fix.jsonl > run-level4-fix.log 2>&1 < /dev/null &
tail -f run-level4-fix.log
```

The live solve-rate improvement is **not yet measured**. Compare five-run
solve rates against the unchanged baseline. Run agents sequentially within
the pod: the harness shares `/tmp/_agent_level4.py` even across checkouts.
