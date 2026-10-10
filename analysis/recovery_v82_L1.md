# Correct in the simulator, rejected for the chip, rewritten: feedback v8.2, level 1

Source: `runs/seat-117/v82_L1_partial/v82_L1_s117.jsonl` (md5 `1690af776147b8860029ac59162b7aca`, 12 attempts, one run), its console log
`run_v82_L1_s117.log`, `_usage.jsonl` and both verdict files, dated 16:18. Configuration: `feedback_v8.py` at
96a9fc9 ("v8.2"): V7.md's v7 settings plus `SKELETON=0 L1FIX=1 TRUNCFIX=1 MIXSAMP=1 L2HINT=1 MIX=1`, Qwen3-8B on
the seat. Under **MIX**, from round 1 on sample 1 repairs the round's best kernel, and samples 2-4 start over from
the first prompt ("fresh"). Under **MIXSAMP**, odd samples use agent.py's sampling (temperature 0.6, top_p 0.95)
and even samples v7's (0.7, 0.8, top_k 20). Rounds count from 0; it was solved in round 2.

## The chain

| round | what scored best | score | why |
|---|---|---|---|
| 0 | sample 1 (first prompt, agent.py's sampling), line 1 | 0.50 | runs on some shapes; an axis error on others. Sample 4 was cut off at 2,500 tokens (221 s) |
| 1 | **sample 2, a fresh attempt** (v7's sampling), line 6 | **0.95** | correct on every loop shape, but **held by the compiler gate**. The repair of round 0's kernel (sample 1) scored 0.30 |
| 2 | **sample 1, the repair** of line 6 with the gate's rewrite, line 9 | **1.00** | the gate's code, applied |

Round 1's 0.95 did not come from repairing round 0's kernel. It came from a sample that MIX started from
scratch: that fresh sample wrote a level-1 kernel that is right in the simulator, and the gate then held it.
Sample 3, another fresh attempt, also reached 0.95 in that round.

## What the gate said (sent in round 2's prompt, verbatim)

The round-1 kernel looped over channels and reduced one partition at a time (`t[c, ...]`). The simulator accepts
that; the trn2 compiler does not (89417dd). `gate_nki.fixes()` flags it, and the score is held at 0.95 with this
message:

```text
Correct in the simulator on every shape, but the trn2 compiler rejects it: the chip has no such instruction, so this kernel cannot run on the device. Fix exactly these lines:

Lines 18-27 (`t[c, ...]` gives an instruction one partition starting at partition `c`; on the chip every compute instruction must start at partition 0. Do all C rows at once with `c` gone: `t[:, ...]`, and (1, ...) tiles become (C, ...)). Replace:
    for c in nl.affine_range(C):
        for h in nl.affine_range(H // p):
            for w in nl.affine_range(W // p):
                # Load the pool window
                window = t[c, h*p:(h+1)*p, w*p:(w+1)*p]
                # Compute mean
                sum_val = nl.sum(window, axis=[1, 2], keepdims=True)
                mean_val = nl.ndarray((C, 1, 1), dtype=x.dtype, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=mean_val, data=sum_val, op0=nl.multiply, operand0=1.0 / (p * p))
                out[c, h, w] = mean_val[0, 0, 0]
with:
    for h in nl.affine_range(H // p):
        for w in nl.affine_range(W // p):
            window = t[:, h * p:(h + 1) * p, w * p:(w + 1) * p]
            sum_val = nl.sum(window, axis=[1, 2], keepdims=True)
            mean_val = nl.ndarray((C, 1, 1), dtype=x.dtype, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=mean_val, data=sum_val, op0=nl.multiply, operand0=1.0 / (p * p))
            nisa.dma_copy(dst=out[:, h:h + 1, w:w + 1], src=mean_val[:, 0:1, 0:1])

Keep everything else identical.
```

Round 2's repair sample took it. Its kernel (sha1 `53900f4b`) is the round-1 kernel with exactly that replacement
applied: no `c` loop, `t[:, ...]`, and one `dma_copy` per window for all channels. The only differences from
the gate's text are two comments the model kept and its own spacing (`h*p` for `h * p`). The rebuilt round-2
repair prompt (v8.2's own `repair_prompt` from the logged kernel and message) is 2,895 characters, as the usage log
records. `gate_nki.fixes()` finds nothing in the solving kernel.

## Cost

12 requests, 13,595 prompt and 10,935 answer tokens (the server's counts, USAGE_LOG). Rounds took 225 s, 73 s and
230 s. Rounds 0 and 2 were long because three fresh answers ran to the 2,500-token limit (TRUNCFIX: not graded).

## Checked on this branch (NKI 0.6.0, `NEURON_PLATFORM_TARGET_OVERRIDE=trn2`)

| check | result |
|---|---|
| `nkibench.py --level 1 --check` | rules clean; 4/4 shapes; allocation audit 364 allocations |
| held-out set (`--eval`, 5 new shapes x 4 value kinds) | **20/20** |
| `scripts/reaudit.py` (fresh process) | **PASS** (53900f4b, first at line 9, round 2) |
| agent.py's verdict (logged) | confidence 0.90 stated first; VERIFIED, held-out 20/20 |
| v7's verdict (logged) | VERIFIED, 0.90: "passes all 35 extra cases ... and lowers for trn2" |

"Lowers for trn2" is the verdict's lowering step. A full build with birsim and a run on the chip were not part of
it, and were not done here.

## Reproduce

```bash
sed -n 9p runs/seat-117/v82_L1_partial/v82_L1_s117.jsonl | python3 -c 'import json,sys; print(json.loads(sys.stdin.read())["code"])' > /tmp/l1.py
export NEURON_PLATFORM_TARGET_OVERRIDE=trn2
python projects/02-kernel-agent/nkibench.py --level 1 --eval /tmp/l1.py
python scripts/reaudit.py runs/seat-117/v82_L1_partial/v82_L1_s117.jsonl
```
