# One failure and the recovery: feedback v8, level 1 (average pooling), solved in round 1

The first level-1 solve in the logs we pulled today (all seats, every configuration; dated 15:40). A second,
v8.2's at 16:18 by a different route (the compiler gate's rewrite), is in analysis/recovery_v82_L1.md. Source:
`runs/seat-116/v8_L1_s116/v8_L1_s116.jsonl` (md5 `73b1f62b2961215bebce66e3aeb14550`, 8 attempts, one run),
its console log `run_v8_L1_s116.log`, `v8_L1_s116_usage.jsonl` (the server's token counts) and both verdict
files, dated 15:40. Configuration: `feedback_v8.py` at 6f11031 (with 81c37cf): v7 (`PROMPT1=v2 MESSAGES=v5
CARD=category SAMPLING=qwen REPAIR_PROMPT=restructure GATE=static`) plus `SKELETON=1 L1FIX=1 TRUNCFIX=1`, and
v8's one extra prompt line for samples 2-4. Qwen3-8B on the seat. **Rounds count from 0: it failed in round 0
and was solved in round 1, the first repair** (the console log says "SOLVED on round 1"; counted from 1, that is
the second round).

Both prompts were rebuilt offline with v8's own code from the logged kernel and feedback: the first prompt is
4,160 characters and the round-1 repair prompt 2,121, exactly the logged `prompt_chars` (samples 2-4 carry the
extra line: 4,185 and 2,146 in the usage log). So the text quoted below is what the model was sent.

| round | sample (attempts line) | score | finish | prompt tokens | answer tokens | seconds | what happened |
|---|---|---|---|---|---|---|---|
| 0 | 1 | 0.30 | stop | 1205 | 442 | 80 | `nl.sum` without keepdims: a 1-D result |
| 0 | 2 | 0.00 | **length** | 1217 | 2500 | 243 | cut off at max_tokens; TRUNCFIX: not graded |
| 0 | 3 | 0.30 | stop | 1217 | 486 | 88 | the same error, on its own `nl.sum(window, axis=[1, 2])` |
| 0 | 4 | 0.00 | **length** | 1217 | 2500 | 243 | cut off at max_tokens; TRUNCFIX: not graded |
| 1 | 1-4 | **1.00** | stop | 651 / 663 | 446 each | 80 | all four return the same kernel; correct on every shape |

Each attempt is matched to its usage line by the sha1 of its code and its round's prompt length (the usage log
is in completion order, not sample order): attempts lines 1-4 are round 0, samples 1-4, and lines 5-8 round 1. Round 0 took 247 s, almost all of it the two truncated
answers (2,500 tokens each); round 1 took 81 s. In total: 7,496 prompt
tokens and 7,712 answer tokens over 8 requests.

## Round 0: 0.30

The carried attempt is sample 1 (`v8_L1_s116.jsonl` line 1; ties go to the earliest sample). It copies the
whole (C, H, W) input into one SBUF tile, loops over the output windows, and sums each window across all
channels at once. Key lines:

```python
    out = nl.ndarray(out_shape, dtype=x.dtype, buffer=nl.shared_hbm)
    t = nl.ndarray((C, H, W), dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=t, src=x)
    tile = nl.ndarray((C, H, W), dtype=x.dtype, buffer=nl.sbuf)
    # Use affine_range to loop over the tiles
    sum_tile = nl.ndarray((C, H // p, W // p), dtype=x.dtype, buffer=nl.sbuf)
    for i in nl.affine_range(H // p):
        for j in nl.affine_range(W // p):
            t_view = t[:, i*p:(i+1)*p, j*p:(j+1)*p]
            s = nl.sum(t_view, axis=[1, 2])
            sum_tile[:, i, j] = s
    nisa.tensor_scalar(dst=sum_tile, data=sum_tile, op0=nl.multiply, operand0=1.0 / (p * p))
    nisa.dma_copy(dst=out, src=sum_tile)
```

**The simulator says:** `SBUF and PSUM tensors must have at least 2 dimensions (partition-dim and free-dim)`

**Sent in round 1's prompt** (under "A checker reports:", after the kernel; verbatim, and identical to the
logged feedback):

```text
0 of 4 shapes passed. On C,H,W=(32, 32, 32) pool=2: Line 29: `s = nl.sum(t_view, axis=[1, 2])` raised AssertionError: SBUF and PSUM tensors must have at least 2 dimensions (partition-dim and free-dim) This reduction drops the axes it reduces, so the result has fewer than 2 dimensions. Add keepdims=True to this call, e.g. nl.sum(tile, axis=[1], keepdims=True) on a (rows, n) tile gives (rows, 1).
```

followed by v3's repair instruction: "Make the change the checker describes. If it gives code, use that code. If
the change needs new loops or new tiles, restructure around them; otherwise keep the rest of the kernel as it
is. Reply with ONE python code block."

The message is v2's (`feedback_v2.advise`, the reduction-drops-axes branch). SKELETON did not blank anything in
it: it carries an example call, not the model's code with slices. The two truncated samples got, instead of a
parse error, TRUNCFIX's note (`Your previous answer was cut off at the token limit before the code was complete. Reply wi...`). It was not used, because sample 1 scored higher.

## Round 1: 1.00

All four samples returned the same kernel (sha1 `1bc6322394`). Its one change from round 0 is the fix the
message named:

```diff
-            s = nl.sum(t_view, axis=[1, 2])
+            s = nl.sum(t_view, axis=[1, 2], keepdims=True)
```

The rest is unchanged, including the comments, an unused `tile` allocation, and the per-window loop.

**The compiler gate was silent.** `gate_nki.fixes()` finds nothing in this kernel: it reduces every channel
at once (`t[:, ...]`), not one partition per channel, so 89417dd's level-1 rule has nothing to hold. Otherwise
it would have scored 0.95.

## Checked again on this branch (NKI 0.6.0, `NEURON_PLATFORM_TARGET_OVERRIDE=trn2`)

| check | result |
|---|---|
| `nkibench.py --level 1 --check` | rules clean; 4/4 shapes; allocation audit 16 allocations |
| held-out set, `nkibench.py --level 1 --eval` (5 new shapes x 4 value kinds) | **20/20** |
| `scripts/reaudit.py` (fresh process) | **PASS** (kernel `1bc6322394`, seen 4x, first at line 5, round 1) |
| agent.py's verdict (logged) | confidence 0.90 stated first, then VERIFIED, held-out 20/20 |
| v7's verdict (logged) | VERIFIED, confidence 0.90: "passes all 35 extra cases ... and lowers for trn2" |
| lines shared with the tutorial / reference / answer kernels | none; it does not use `.ap()` |

Not checked here: a full trn2 build with birsim or a run on the chip (no compiler on this machine). v7's verdict
reports that it lowers for trn2 (lowering only; a full build was not part of that verdict).

## Reproduce

```bash
sed -n 5p runs/seat-116/v8_L1_s116/v8_L1_s116.jsonl | python3 -c 'import json,sys; print(json.loads(sys.stdin.read())["code"])' > /tmp/l1.py
export NEURON_PLATFORM_TARGET_OVERRIDE=trn2
python projects/02-kernel-agent/nkibench.py --level 1 --check /tmp/l1.py
python projects/02-kernel-agent/nkibench.py --level 1 --eval /tmp/l1.py
python scripts/reaudit.py runs/seat-116/v8_L1_s116/v8_L1_s116.jsonl
```
