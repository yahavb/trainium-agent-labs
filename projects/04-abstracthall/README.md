# Scientific summarization solver

This is the **generator**, not the checker. It reads the eight real abstracts, asks the existing Qwen3-8B endpoint for summaries, and saves every reply. It does not score factuality, select key sentences, train weights, or repair drafts yet. Baseline generation is the first experiment stage.

Python 3.10+ and the standard library are sufficient. No pip installation is needed. Run inside seat 17 for access to its localhost model server. Your Mac's localhost is a different machine.

## 1. Offline validation
From this folder:

```bash
python -m unittest -v test_solver.py
python solver.py --dry-run
```

Tests use mock protocol replies only; they are not dataset examples and are never written to experiment output. Dry-run validates and displays the actual eight abstracts without calling a model or fabricating summaries.

## 2. Start the existing model
In a dedicated shell inside seat 17, run `/workspace/serve.sh` from `/workspace` if the server is not already running. Wait for READY. Stop previous kernel experiments before starting a new generation run.

## 3. One-abstract smoke test
In another shell inside seat 17, enter this folder:

```bash
python -u solver.py --id yeast_01 --samples 1 --repeat 1
```

The CLI prints the new JSONL output path. Open that file to inspect the actual summary. Each run gets a unique filename. If the pod environment is missing the endpoint setting, pass `--base http://localhost:8000/v1`.

## 4. Eight-abstract baseline
Only after the smoke test works:

```bash
nohup python -u solver.py --samples 4 --repeat 5 > baseline-console.log 2>&1 < /dev/null &
tail -f baseline-console.log
```

This requests 8 abstracts × 4 samples × 5 repetitions = 160 summaries, with at most four concurrent requests. Conditions/repetitions are otherwise sequential. Do not run another experiment against the server concurrently. Ctrl+C while tailing stops only the viewer. Reusing this console command overwrites the console log; the JSONL results always have a fresh name.

Default settings: Qwen/Qwen3-8B, temperature 0.6, top_p 0.95, 300 output tokens, thinking off. The 300-token cap follows the planned baseline and can truncate answers; those are explicitly flagged. The input-only limit is 8,100 tokens (`--input-limit 8100`), independent of the 300-token output budget. Before each batch the client calls the server’s `/tokenize` endpoint with the same messages and Qwen chat-template settings, including the generation prefix. Inputs above the limit are logged as `input_too_long`, without generation or silent truncation. The measured count is saved as `input_tokens`. If tokenization fails or the endpoint is unavailable, the batch fails explicitly; endpoint support still needs live verification on seat 17. Dry-run remains offline and does not count tokens. Server-reported generation usage is also saved when supplied. Errors are recorded without silent retries. If every request in a batch fails, the run stops.

## Reuse for context repetition and repair

```python
from solver import Arm, solve
arm = Arm(base_url="http://localhost:8000/v1")
replies = solve(prompt="Summarise: " + abstract_text, arm=arm, n=4)
```

The future conditions module will build the repeated-context prompts; the future checker will generate repair instructions. Both call this same `solve(prompt, arm, n)` function. For another OpenAI-compatible server, provide its exact API base and model and use `--no-qwen-template` if it does not support the Qwen extension. Compatibility with the shared gpt-oss endpoint has not been tested.

## Output schema

Every JSONL row records run ID, abstract ID/source/text hash, condition, round, repeat and sample indices, exact prompt, generation settings, UTC timestamp, cleaned summary (`text`), original answer-channel content (`raw_content`), finish reason, latency, usage and error status. Separate hidden reasoning fields are not collected. Complete `<think>` blocks are removed from cleaned text, and unfinished blocks are not treated as summaries.

Statuses: `complete` (stop + nonempty text), `truncated` (length limit), `empty`, `incomplete` (other finish reason), `input_too_long`, or `error`. **Complete does not mean factually correct.** There are no correctness scores yet.

## Dataset and validation

See `data/README.md` for provenance, version uncertainty, rights, and the one abstract word-count discrepancy. This folder carries the JSONL input and its provenance README; other dataset formats remain in the separate dataset package.

Local unit tests and a dry-run validate request formatting, error/truncation handling, dataset loading and result persistence. Live inference and scientific summary quality must still be checked on seat 17.

Tokenization protocol reference: https://docs.vllm.ai/en/v0.15.0/api/vllm/entrypoints/serve/tokenize/protocol/ . Eight offline tests pass, including the 8,100/8,101-token boundary and independence of the output budget.
