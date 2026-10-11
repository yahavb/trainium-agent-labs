# Next experiments

Speedups are relative to the no-speculation baseline (51.7 tok/s), measured on one
trn2 chip (seat-40), TP=2, greedy, 128 new tokens. Current best: bidirectional
k=4/k=8 at 88-89 tok/s (1.70-1.73x).

## Draft-token count has limited headroom

Accepted tokens by draft position (k=8 bidirectional run):

| position | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---------:|---:|---:|--:|--:|--:|--:|--:|--:|
| accepted | 204 | 118 | 64 | 33 | 17 | 7 | 5 | 5 |

Acceptance falls off after position 4 and is near-flat (5-7) by positions 5-7.
Each additional drafted position costs a target verification while contributing
few accepted tokens, so causal k=15 (67.3 tok/s) was slower than k=8 (69.0 tok/s).
The limiting factor is draft quality, not draft-token count.

## Directions

### 1. DFlash2 draft with the candidate selector
The deployed 8B draft is unary DFlash (greedy top-1 per position). The DFlash2
draft reranks top-16 candidates per position via a predecessor/successor
transition score and walks one coherent path, which raises acceptance across
positions. Components are implemented and verified against the released weights
(`dflash2_27b/dflash2_components.py`, `DFLASH2_COMPONENTS: PASS`).
Remaining: wire into a Neuron proposer.

### 2. Qwen3.8-27B with DFlash2, end to end
The model pair the DFlash2 draft was trained for. Both Qwen3.5 attention variants
(GatedDeltaNet/SSM + full) are validated against the real 27B weights; the DFlash2
convolution and selector are validated. Remaining is the Neuron assembly of the
64-layer hybrid decoder (dual KV + SSM-state cache) and serving.

### 3. Bidirectional k=12 / k=15
Not yet measured with the bidirectional patch (only causal was run at k=15). The
per-position data suggests a plateau; running these would establish the upper bound.

### 4. Bidirectional k=5 / k=6
k=4 (88.1 tok/s) and k=8 (89.4 tok/s) are within ~1.5%. Measuring k=5/k=6 would
locate the throughput peak between them.

## Status
These require the Trainium pod. The DFlash2 draft components and both Qwen3.5
attention variants are verified in isolation against the released weights; the
end-to-end 27B serve is not yet implemented.
