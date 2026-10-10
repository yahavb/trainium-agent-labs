# upstream-nki

Runs of the upstream NKI kernel agent (`trainium-agent-labs/projects/02-kernel-agent`, commit
`8f1ca41`) on the seat pod, plus four prompt/feedback experiments layered on it (v1..v4, each built from the previous one's failures).

| file | what |
|---|---|
| `FEEDBACK-EXPERIMENTS.md` | what the baseline showed, what each version changes, solve rates |
| `agent_feedback_v1.py` | sidecar: one targeted hint (invented `nisa.multiply` -> `nisa.tensor_scalar`) |
| `agent_feedback_v2.py` | sidecar: shape card, rewrite-not-patch repair prompt, per-level direction. **Solved level 4.** |
| `agent_feedback_v3.py` | v2 + `.ap()` stride rule, MemoryRegion rule, nested-loop level-4 text |
| `agent_feedback_v4.py` | v3 + matmul "shape doctor" (names which tensor each element count is) |
| `logs/baseline.log` | unchanged `agent.py --all --rounds 8 --samples 4 --context 8192 --repeat 5`, Qwen3-8B |
| `logs/attempts-baseline.jsonl` | every attempt of that run: level, round, reward, code, feedback (360 rows) |
| `logs/feedback-v*-L*.log` | per-version, per-level runs (added as they finish) |

v3 imports v2 and v4 imports v3, so all four files travel together. The sidecars import the pod's `agent.py` and rebind `enrich` / `first_prompt` / `repair_prompt`;
nothing upstream is edited. Both have a `--check-feedback` mode that runs without a model.
