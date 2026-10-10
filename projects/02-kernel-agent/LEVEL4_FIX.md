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
lifetime across K tiles. Later feedback also diagnoses the M=128 zero-loop
case: using M//512 for tile count skips every output write. Scoring and
numeric tolerances are unchanged.

On the first level-4 repair after a nonempty failed candidate, the agent now
adds a clearly labeled three-axis loop scaffold. It asks the model to complete
the input loads, matmul, and output copies, then the normal checker grades the
candidate. There is no reference-kernel fallback. Each attempt record includes
a `scaffolded` field so scaffolded results can be reported separately from
unscaffolded runs.

Validation performed locally: Python compilation, checker self-test, and a
prompt-structure check passed. The self-test reports that its NKI simulation
portion needs verification on the pod. This does not establish that the model
will solve the level. No successful model-generated runs have been measured yet.

On the pod, evaluate five independent runs, each with at most five rounds:

```bash
python nkibench.py --selftest
python nkibench.py --level 4 --check reference_level4.py
nohup python -u agent.py --level 4 --rounds 5 --samples 4 --context 8192 --repeat 5 \
  --log attempts-level4-scaffold.jsonl > run-level4-scaffold.log 2>&1 < /dev/null &
tail -f run-level4-scaffold.log
```

The target is reward 1.00 in all five runs (summary: `solved 5/5`). The live
solve rate remains unmeasured until that pod run completes. Run agents
sequentially within the pod: the harness shares `/tmp/_agent_level4.py` even
across checkouts.
