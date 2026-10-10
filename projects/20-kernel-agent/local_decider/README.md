# Local Decider experiment

An optional advisory repair-strategy selector for the kernel-agent project. It runs on your Mac and
replays existing attempts; it does not modify `agent.py`, execute recorded code, connect to AWS, or
override the numerical checker. Nothing is deployed or pushed by these commands.

## Setup and use

Run from the repository root. Use the existing virtual environment, or create one
with `python3 -m venv .venv` and install dependencies with
`.venv/bin/python -m pip install -r requirements.txt`:

```bash
# Already installed during local setup:
# HOMEBREW_NO_AUTO_UPDATE=1 HOMEBREW_NO_INSTALL_CLEANUP=1 brew install llama.cpp
.venv/bin/python projects/20-kernel-agent/local_decider/decider.py download
.venv/bin/python projects/20-kernel-agent/local_decider/decider.py start
.venv/bin/python projects/20-kernel-agent/local_decider/decider.py status
.venv/bin/python projects/20-kernel-agent/local_decider/decider.py decide \
  --level 8 --feedback 'Contraction axis mismatch: q needs the correct transpose for q k^T'
.venv/bin/python projects/20-kernel-agent/local_decider/decider.py replay \
  projects/20-kernel-agent/logs/attempts.tar.gz --limit 16
.venv/bin/python projects/20-kernel-agent/local_decider/e2e_check.py
.venv/bin/python projects/20-kernel-agent/local_decider/decider.py stop
```

The server loads a 2.71 GB model; total memory usage is higher. It uses Metal where supported,
a 2,048-token context and one inference slot, on port 8096. Weights and runtime/evaluation files
are Git-ignored under `models/` and `runs/`. The downloader refuses an existing partial download;
an interrupted `.partial` file can be removed explicitly before retrying. A running server can
be stopped with the command above to release its memory.

## How decisions work

The client renders the upstream **v2 plain state-first format**, ending in `Answer: (`. It checks
that all six option letters are single tokens at that position and requests one token's raw
pre-sampling log probabilities from llama.cpp's native `/completion` API. It then normalizes only
the six allowed letters at upstream v2's temperature, **1.935**. This does not use the GGUF's generic
chat template. The native endpoint produces one token to expose its probabilities; this is not the
upstream dedicated no-generation inference engine.

The suggestions are fixed IDs: `repair_layout`, `repair_api`, `repair_memory`, `reduce_traffic`,
`change_approach`, `minimal_edit`. Low confidence (default <0.6), missing option probabilities,
oversized prompts, tokenization incompatibility, malformed replies or service errors retain a
deterministic repair rule. Quantization and NKI-domain confidence have **not** been calibrated:
0.6 is an experimental filter, not a guarantee of accuracy.

Replay evaluates selected failed candidates, deduplicates identical bounded states, and logs
decision latency, probabilities, fallbacks and agreement with the simple baseline. It does not
execute a suggested repair, so a differing choice is not evidence of improvement. Completion
speed, solve rate and avoided rounds need a later controlled end-to-end comparison. For now,
importing `bounded_state` and `decide` provides a local advisory API for an orchestrator; the shipped
Trainium agent remains unchanged.

## Safety review and controls

Scope: model repository files/provenance, local runtime configuration, downloader and decision client.
This is a limited review, not an independent audit of llama.cpp or the model's training data.

* Only the GGUF tensor artifact is downloaded. No remote Python, installer script, pickle,
  `trust_remote_code`, model-provided tools or generated commands are executed.
* Model revision: `2796fac5cdad018ef6d9a4a004735ff819f424a2`.
* Exact size: **2,708,804,544 bytes**.
* SHA-256: `f7e2e510ef51d212ea9b7fb8bf27906b5f516d7939ca847428fb91f6a8acfa79`.
  Published provenance and Hugging Face's LFS hash agree. Verification prevents accidentally loading
  changed/corrupted weights; it is not proof that an artifact or native parser is free of vulnerabilities.
* Runtime installed through Homebrew; server binds only `127.0.0.1`, requires a randomly generated
  API key stored mode 0600, disables the web UI, agent tools, MCP proxy and Jinja, and uses offline mode.
  Server environment does not inherit AWS/HF credentials, proxy settings or LLAMA argument overrides.
* Client uses a fixed loopback URL, no environment proxies, and no redirects. Downloads use verified
  HTTPS for the initial pinned Hugging Face URL and follow redirects without a destination allowlist;
  final bytes must match the pinned size/hash. Inference does not call a hosted provider.
* Model responses are advisory fixed choices. Prompt injection could still distort a strategy choice;
  numerical correctness remains the responsibility of the existing deterministic checker.
* Archives are streamed as JSON data, never extracted, with member/line size limits. Source logs are
  read-only. Evaluation files do not copy full source prompts/code.

The native runtime runs with your user account's permissions, not in a separate OS sandbox. Keep it
updated, avoid loading unrelated unverified models, and use non-sensitive evaluation logs.

Tests (no model or network needed):

```bash
.venv/bin/python -m unittest discover -s projects/20-kernel-agent/local_decider -p 'test_*.py' -v
```

## Reproducible local integration check

With the local server running, `e2e_check.py` exercises three deliberately broken CPU fixtures:
attention-score orientation, transpose output shape, and an invalid softmax API. It uses the real
classifier, the existing `agent.repair_prompt`, and the project's NumPy reference functions.
The selected strategy routes to a **trusted scripted NumPy repair**, then an independent numerical
comparison checks the result. It also verifies that bad results remain rejected and a simulated
client outage retains deterministic fallback. At least one accepted real-model decision is required;
an entirely fallback-only run fails. The script emits results to stdout and writes no files.

This is not a run of `agent.solve`, model-generated repair code, NKI simulation, Trainium execution,
an accuracy benchmark, or a speedup measurement. It does not wire the classifier into the shipped
agent loop. Any future adoption needs an explicitly opt-in integration and a controlled comparison.

Sources: [GGUF model/provenance](https://huggingface.co/mindchain/decider-4b-v2-GGUF),
[v2 configuration](https://huggingface.co/Mapika/decider-4b/blob/v2/decider_config.json),
[upstream prompt format](https://github.com/Mapika/decider/blob/main/decider/prompt.py),
[llama.cpp native server API](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md).
