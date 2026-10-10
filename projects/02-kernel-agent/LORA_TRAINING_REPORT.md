# LoRA training evidence

Current status: completed.

- Actual base: cached Qwen/Qwen3-8B, snapshot b968826d9c46dd6066d109eabc6255188de91218. Private reconstructed index points to existing weights; no cache/SDK/server modifications.
- Tokenizer: same cached snapshot; thinking disabled; 1140 maximum actual sequence tokens; no truncation permitted.
- Training examples: 41 ({'generation': 15, 'repair': 26}); held-out: 3 ({'generation': 1, 'repair': 2}). Entire elementwise family and its clean AST excluded from training/retrieval.
- Source: synthetic_nki/data_v4, 16 independently simulator/NumPy-verified non-benchmark kernels and 28 repair pairs. No official reference kernels or hidden outputs used. The newer 38-pair corpus is excluded from this frozen training run.
- LoRA rank 8, alpha 16, dropout .05, q_proj/v_proj in all layers; 3,833,856 trainable parameters; frozen base.
- Learning rate 1e-4, AdamW, batch size 1, gradient accumulation 1, completion-only loss, gradient clipping 1.0, two epochs / 82 planned optimizer steps, seed 2026.
- CPU BF16 base, PEFT trainable adapters; torch compute threads 2, low process priority. No Neuron device file open on inspected training process; host memory/CPU are shared, so strict resource independence and timing equivalence are not claimed.
- Final-epoch mean example training loss: 0.2100287243040994 (unweighted mean of batches, not kernel accuracy).
- Completed steps: 82. Latest observed training loss: 0.24963468313217163. Final training loss: 0.24963468313217163.
- Held-out mean example loss before: 0.9951169490814209. After: 0.12747866617913436.
- Checkpoints: checkpoint-10, checkpoint-20, checkpoint-30, checkpoint-40, checkpoint-50, checkpoint-60, checkpoint-70, checkpoint-80.
- Final adapter: /tmp/trainium-kernel-dev/projects/02-kernel-agent/runs/qwen3-nki-lora-mdzxuq1k/training/adapter (saved).
- Training console: /tmp/trainium-kernel-dev/projects/02-kernel-agent/runs/qwen3-nki-lora-mdzxuq1k/console.log; real step metrics: /tmp/trainium-kernel-dev/projects/02-kernel-agent/runs/qwen3-nki-lora-mdzxuq1k/training/metrics.jsonl.

Loss reduction is not kernel correctness. Held-out loss uses only three examples from one excluded family; it is a small pipeline/generalization check, not broad NKI reliability evidence. The tiny-model training test is excluded from all Qwen/benchmark results.

## Reproduce

```bash
python -B -m training.data --source synthetic_nki/data_v4 --output runs/NEW-SFT-DATA
python -B training/train_lora.py --data runs/NEW-SFT-DATA --model training/base-local --output runs/NEW-LORA --epochs 2 --threads 2 --rank 8
```

Dependencies were installed with --no-deps into training/vendor only (PEFT 0.21.2, Accelerate 1.15.0); torch, Transformers, Neuron and the serving environment were not upgraded. Reinstall these two packages into an isolated target if reproducing elsewhere. Use training/cache_index.py to reconstruct a missing cached weight index in a new private directory.
