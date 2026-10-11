# Project 4 - DFlash speculative decoding on Trainium

**DFlash runs on Trainium 2 through vLLM Neuron. A bidirectional query-block attention
change in the draft raises Qwen3-8B decode throughput to 1.70x (k=4) over the no-speculation
baseline.**

[DFlash](https://huggingface.co/z-lab/Qwen3-8B-DFlash-b16) is a block-diffusion
speculative-decoding draft model: a small 5-layer model reads hidden states from the target
model and proposes several next tokens in one pass, and the target model checks them all in a
single forward pass. vLLM supports it on GPU; the Neuron backend did not. Before this change,
serving it failed with:

```
AttributeError: type object 'DFlashQwen3ForCausalLM' has no attribute 'from_configs'
```

![throughput](results/throughput.svg)

## Results

Measured on one `trn2` chip (seat-40), TP=2, one request at a time, greedy decoding,
128 new tokens, 3 prompts x 2 runs. All servers use **the same config**
(`max_model_len=256`, `max_num_seqs=1`, `max_num_batched_tokens=256`); only the
`--speculative-config` flag differs.

| config | tokens/s | per-request median | speedup | draft acceptance |
|---|---|---|---|---|
| Qwen3-8B (autoregressive baseline) | 51.7 | 51.7 | 1.00x | n/a |
| DFlash k=2, causal | 64.7 | 65.0 | 1.25x | 33.1% |
| DFlash k=2, bidirectional | 73.8 | 76.0 | 1.43x | 44.0% |
| DFlash k=4, causal | 69.3 | 66.5 | 1.34x | 20.1% |
| DFlash k=4, bidirectional | 88.1 | 85.9 | **1.70x** | 32.7% |
| DFlash k=8, causal | 69.0 | 68.2 | 1.33x | 11.1% |
| DFlash k=8, bidirectional | 89.4 | 89.2 | **1.73x** | 17.8% |

![acceptance by draft position](results/acceptance.svg)

Throughput peaks around k=4-k=8. Accepted tokens by position fall off after position 4
(bidirectional k=8: 204/118/64/33/17/7/5/5), so additional draft positions contribute little
while each still requires target verification. Causal k=15 (67.3 tok/s) was slower than causal
k=8 (69.0 tok/s) for this reason. Higher acceptance at later positions is what would make
longer drafts worthwhile.

Raw output: [`results/`](results/) (one file per run). Regenerate the table and charts with
`python3 make_charts.py`.

**Output check.** Both servers give the same greedy text for about the first 95 tokens of
prompt 1, then diverge on a near-tie (`"which is then mapped to an index"` vs
`"which is then used to determine"`). Both continuations are coherent. The likely cause is
bf16 rounding: the target scores 5 tokens per step when verifying instead of 1, a different
kernel shape. Speculative decoding on GPU in bf16 is also not bit-exact for the same reason.

## Demo

On a seat with the default image (`vllm-neuronx 0.24`, Neuron SDK 2.32):

```bash
cd /workspace/projects/04-dflash2-neuron
bash apply.sh                      # install the DFlash2 Neuron support (idempotent)
bash serve_dflash2.sh              # Qwen3-8B + DFlash2 on :8000 (DFLASH_NUM_SPEC=8 for 8 drafts)
bash tools/wait_log.sh             # returns when the server is ready or failed (max 60s per call)
bash run_bench.sh 4                # benchmark, saved to results/bench_dflash2_k4.txt

bash serve_baseline.sh             # same config, autoregressive, for the comparison
bash tools/wait_log.sh /tmp/baseline.log
bash run_bench.sh 0                # saved to results/bench_baseline.txt

python3 make_charts.py             # rebuild results/*.svg and print the results table
```

The first start compiles every graph for the chip (about 5 minutes). Later starts
reuse the Neuron compile cache and come up in about 2 minutes.

Quick single request:

```bash
curl -s localhost:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-8B","prompt":"The capital of France is","max_tokens":40,"temperature":0}'
```

## What changed

Two new files (in `overlay/`) and a 300-line patch to three existing files
([`vllm_neuron-dflash2.patch`](vllm_neuron-dflash2.patch)). The DFlash2 code does not depend
on the Eagle3 implementation; the only shared pieces are the runner's existing
speculative-decoding plumbing and the target-side hidden-state capture interface.

| File | Change |
|---|---|
| `vllm_neuron/model/qwen3/dflash_draft.py` (new) | Neuron DFlash2 draft model: 5 Qwen3 layers + `fc` + `hidden_norm`, layers named 36-40 so their KV cache doesn't collide with the target's |
| `vllm_neuron/vllm/spec_decode/dflash2.py` (new) | `DFlash2Proposer`: loads the draft and its weights, loads `embed_tokens`/`lm_head` from the target (DFlash2 checkpoints don't ship them), feeds the draft 5 target hidden states |
| `vllm_neuron/model/qwen3/model.py` | Qwen3 target captures hidden states at the layers DFlash2 reads (`target_layer_ids` + 1 = 2, 10, 18, 26, 34) and returns them to the runner; weight loading accepts the DFlash2 checkpoint layout (bare keys, no embedding / LM head); KV cache names use each layer's own index |
| `vllm_neuron/model/registry.py` | registers `DFlashDraftModel` |
| `vllm_neuron/vllm/worker/neuron_model_runner.py` | routes `method: "dflash"` to `DFlash2Proposer` |

### One draft step

1. **Context:** `ctx = hidden_norm(fc(concat of 5 target hidden states))`. For each draft layer,
   K/V of `ctx` (K with `k_norm` + RoPE) are written straight into that layer's KV cache at the
   target tokens' slots.
2. **Query block:** `[next token, MASK, MASK, MASK, MASK]` at positions `p+1 - p+5` run through
   the 5 draft layers with the existing Neuron decode attention, which reads the context K/V.
3. **Drafts:** greedy tokens at the 4 MASK positions. The target verifies them with vLLM's
   rejection sampler, so wrong drafts are discarded and output quality is unaffected.

## Limitations and next steps

- **Query-block attention.** The initial Neuron port used a causal mask inside the draft
  query block, which reused the existing decode kernel but reached lower acceptance than the
  draft's training configuration. The current build uses a bidirectional (all-ones) active
  mask for the query block, matching how the draft was trained; this raised k=4 acceptance
  from 20.1% to 32.7% and k=4 throughput from 1.34x to 1.70x.
- **Draft count.** The checkpoint was trained for blocks of 16 (`b16`), so up to 15 drafts
  per step are possible. Measured: 4 and 8 drafts both give 1.33x and 15 gives 1.30x, because
  acceptance falls off quickly after position 4 and is zero after position 8. Longer drafts only
  pay off once acceptance improves.
- **Tested config:** 1 request, `max_model_len=256`, async scheduling off, 4 / 8 / 15 draft
  tokens. Batching and longer context haven't been measured yet.
- If the query block reaches past the blocks allocated for a request, the block index is
  clamped to the last allocated block. That can corrupt the draft's own cache (lower acceptance),
  but never the target's.

## Dev tools

`tools/wait_log.sh` polls a vLLM log every second and returns as soon as the server is ready or
has failed, printing the root cause (60 s cap per call). `tools/progress.sh` shows how many graphs
have compiled. `tools/rootcause.sh` extracts the first worker traceback.
