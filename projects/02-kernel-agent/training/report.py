"""Refresh progress reports using raw evidence; pending training is never a result."""
import collections
import json
from pathlib import Path

PROJECT=Path(__file__).resolve().parents[1]
RUN='qwen3-nki-lora-mdzxuq1k'
MATCH='lora-matched-2x2-_mu040_v'


def main():
    run=PROJECT/'runs'/RUN;training=run/'training'
    manifest=json.loads((training/'manifest.json').read_text())
    metrics=[json.loads(x) for x in (training/'metrics.jsonl').read_text().splitlines()]
    result=json.loads((training/'result.json').read_text()) if (training/'result.json').exists() else None
    data=PROJECT/'runs/lora-sft-data-20261010-v1'
    counts={split:dict(collections.Counter(json.loads(x)['kind'] for x in (data/(split+'.jsonl')).read_text().splitlines())) for split in ('train','heldout')}
    text=f'''# LoRA training evidence

Current status: {'completed' if result else 'running; final outcome pending'}.

- Actual base: cached Qwen/Qwen3-8B, snapshot b968826d9c46dd6066d109eabc6255188de91218. Private reconstructed index points to existing weights; no cache/SDK/server modifications.
- Tokenizer: same cached snapshot; thinking disabled; 1140 maximum actual sequence tokens; no truncation permitted.
- Training examples: {manifest['train_count']} ({counts['train']}); held-out: {manifest['heldout_count']} ({counts['heldout']}). Entire elementwise family and its clean AST excluded from training/retrieval.
- Source: synthetic_nki/data_v4, 16 independently simulator/NumPy-verified non-benchmark kernels and 28 repair pairs. No official reference kernels or hidden outputs used. The newer 38-pair corpus is excluded from this frozen training run.
- LoRA rank 8, alpha 16, dropout .05, q_proj/v_proj in all layers; 3,833,856 trainable parameters; frozen base.
- Learning rate 1e-4, AdamW, batch size 1, gradient accumulation 1, completion-only loss, gradient clipping 1.0, two epochs / 82 planned optimizer steps, seed 2026.
- CPU BF16 base, PEFT trainable adapters; torch compute threads 2, low process priority. No Neuron device file open on inspected training process; host memory/CPU are shared, so strict resource independence and timing equivalence are not claimed.
- Final-epoch mean example training loss: {sum(m['loss'] for m in metrics if m['epoch']==1)/sum(m['epoch']==1 for m in metrics) if result else 'pending'} (unweighted mean of batches, not kernel accuracy).
- Completed steps: {len(metrics)}. Latest observed training loss: {metrics[-1]['loss'] if metrics else 'unavailable'}. Final training loss: {metrics[-1]['loss'] if result else 'pending'}.
- Held-out mean example loss before: 0.9951169490814209. After: {result['heldout_loss_after'] if result else 'pending'}.
- Checkpoints: {', '.join(p.name for p in sorted(training.glob('checkpoint-*')))}.
- Final adapter: {training/'adapter'} ({'saved' if result else 'pending'}).
- Training console: {run/'console.log'}; real step metrics: {training/'metrics.jsonl'}.

Loss reduction is not kernel correctness. Held-out loss uses only three examples from one excluded family; it is a small pipeline/generalization check, not broad NKI reliability evidence. The tiny-model training test is excluded from all Qwen/benchmark results.

## Reproduce

```bash
python -B -m training.data --source synthetic_nki/data_v4 --output runs/NEW-SFT-DATA
python -B training/train_lora.py --data runs/NEW-SFT-DATA --model training/base-local --output runs/NEW-LORA --epochs 2 --threads 2 --rank 8
```

Dependencies were installed with --no-deps into training/vendor only (PEFT 0.21.2, Accelerate 1.15.0); torch, Transformers, Neuron and the serving environment were not upgraded. Reinstall these two packages into an isolated target if reproducing elsewhere. Use training/cache_index.py to reconstruct a missing cached weight index in a new private directory.
'''
    (PROJECT/'LORA_TRAINING_REPORT.md').write_text(text)
    matched=PROJECT/'runs'/MATCH/'evaluation'
    summaries=[]
    for p in matched.glob('level*/*/summary.json'):summaries.append(json.loads(p.read_text()))
    table=['| Level | A: original + legacy | B: original + planner | C: LoRA + legacy | D: LoRA + planner |','|---|---|---|---|---|']
    names=['A_original_legacy','B_original_planner','C_lora_legacy','D_lora_planner']
    for level in (1,2,3,4):
        cells=[]
        for name in names:
            row=next((r for r in summaries if r['level']==level and r['arm']==name),None)
            cells.append(f"{max(t['best_reward'] for t in row['trials']):.3f}; {row['solved']}/{row['trial_count']} solved" if row and row['trials'] else 'Pending')
        table.append('| '+str(level)+' | '+' | '.join(cells)+' |')
    comparison='# Matched original / LoRA / planner comparison\n\n'+'\n'.join(table)+f'''\n\nThese cells contain only the requested matched 2x2 study. Pending cells must not be substituted with historical or different-policy results. Study queued at {matched}; it preserves the previously scheduled full-adapter evaluation, then runs all four arms sequentially. A separate held-out repair worker is queued after the preserved matched study; the newer reproduction script also supports a held-out check before the arms. Same CPU Transformers model instance, tokenizer, checker, shapes, temperature .6, top_p .95, thinking off, context 8192, four candidates, eight-round ceiling, repeat one. A/C use standard generation, reward selection, standard repair, legacy feedback, examples off, no adaptive history. B/D differ only by hardware/semantic planner. The serving process switches PEFT adapter layers off/on per request; no server restart or weight merge.

Separate available evidence (not the matched 2x2): original five-repeat baseline solved zero of five on every level; historical best rewards .30/.30/.30/.625. Untuned full-agent scores are .30/.30/1.00/.625. First planner full-agent Level 1 run: .30, zero numerical cases, 20 candidates before unchanged repetition stop. Planner changed observed AST operation choice from 30/32 self-products to 0/20, but numerical pooling remains unverified. Shape-consistency refinement completed at .30 in controlled-20261010T195135-qyw82zco. A generic instruction-legalizer warm replay scored 1.00 on Level 1, 4/4 shapes; this is excluded from cold-start and LoRA results. Cold-start planner/legalizer evaluation is recorded separately in controlled-20261010T201403-vxbwe5dd.

Training effect on initial generation, mathematical operation, valid APIs, repair success and official correctness: **not yet established**. Candidate code, actual tokens and per-shape checker results will be saved for every arm; held-out repair checks use real simulator/NumPy comparisons, not loss proxies. Wall times from this CPU study cannot be compared with older Neuron vLLM runs as model speedups.

```bash
python -B training/run_matched.py --prior-evaluation runs/lora-full-evaluation-e43ejnxq/evaluation --output runs/NEW-MATCHED --rounds 8 --samples 4 --levels 1 2 3 4
python -B -m training.report
```

New studies require exclusive artifacts and no competing evaluation. The quoted command waits for the prior evaluation instead of launching competing endpoint requests. One repetition per arm is not evidence of reliable superiority.
'''
    (PROJECT/'LORA_COMPARISON.md').write_text(comparison)
    print('Updated LORA_TRAINING_REPORT.md and LORA_COMPARISON.md from raw evidence.')

if __name__=='__main__':main()
