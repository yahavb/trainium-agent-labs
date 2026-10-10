# Local safety and runtime verification — October 10, 2026

Verdict: acceptable for the implemented **local advisory experiment** with the controls below.
No critical issue was identified in this scoped review. This is not a guarantee that the weights,
native parser or inferred repair strategy are safe/correct in every setting.

## Artifact and runtime

* Downloaded only `decider-4b.v2-Q4_K_M.gguf` from the pinned Hugging Face revision
  `2796fac5cdad018ef6d9a4a004735ff819f424a2`.
* Downloaded file has exactly 2,708,804,544 bytes and SHA-256
  `f7e2e510ef51d212ea9b7fb8bf27906b5f516d7939ca847428fb91f6a8acfa79`.
  Both match the repository provenance and Hugging Face LFS metadata.
* Homebrew llama.cpp **0.6.0**, build 11429, commit `d81235049`.
* Server successfully loaded the model with GPU offload enabled (`-ngl 99`) on the Apple Silicon Mac,
  context 2048, one slot; authenticated `/health` returned `{"status":"ok"}`.

## Verified controls

* `lsof` confirms only `127.0.0.1:8096`, not a LAN/public interface.
* Inference without the API key returns HTTP **401**.
* Web UI route returns HTTP **404**.
* API key file is mode **0600** and is ignored by Git.
* Runtime arguments disable agent mode, web UI, MCP proxy and Jinja; offline mode is enabled.
  LLAMA argument overrides, AWS/HF credentials and proxy variables are not inherited.
* Plain decision prompts match the upstream v2 configuration. Actual tokenizer checks confirm
  that all six option letters are single tokens at the answer slot. Real option probabilities
  were returned by the native completion endpoint and normalized at v2 temperature 1.935.
* Model output never executes code or commands, changes a numerical grade or marks a kernel solved.
* Weights/runtime/results are ignored by Git. The initial local validation made no commit, push,
  deployment or Trainium change. Publication of these source files does not deploy the service.

## Validation

Eleven unit tests pass: model integrity rejection before process launch, bounded input,
credential/tool environment filtering, no execution of recorded code, probability normalization,
missing/duplicate/non-finite/malformed/truncated readout refusal, and unavailable/low-confidence
fallback. `git diff --check` passes.

Local layout-repair smoke test: expected `repair_layout`, decision probability approximately 0.772,
168 prompt tokens, total client latency **0.9831 s**. This probability is not an empirically validated
NKI-domain confidence.

Replay of the first **16 distinct selected failed states** from the shipped archive (levels 4 and 1):

| measurement | observed |
|---|---:|
| median full client decision time | 3.13475 s |
| deterministic fallback used | 9/16 |
| accepted strategy differs from baseline | 2/16 |
| NKI decision accuracy measured | no |
| complete-agent speedup measured | no |

Raw local-only results: `runs/replay-1791671517033725000.jsonl` and its `.summary.json`.
Replay timing includes tokenization checks, prompt processing and option readout, not just one token
of decoding. Deduplication prevents charging twice for the same bounded state within a replay.

## Remaining limits

The server runs under the user's OS permissions, not in a separate process sandbox. Hash verification
proves identity/integrity, not absence of native parser vulnerabilities. Third-party quantization and
NKI decisions are not independently calibrated; prompt injection may affect advisory choices.
The 0.6 threshold is experimental. This small replay is not an accuracy or solve-rate benchmark.
The existing agent and deterministic checker remain the authority; end-to-end adoption needs a
controlled comparison that includes router overhead and checks that solve rate does not fall.

## Publication and local integration check

The scoped working-tree review covered `.gitignore`, `decider.py`, `test_decider.py`, this document,
and the README. No reportable security finding was identified. This conclusion is limited to the
advisory experiment, not an audit of native dependencies, training data, unrelated code or Git history.
The supplementary `e2e_check.py` is a reproducible functional check, not a security sandbox.
The publication unit suite contains 14 tests: the original 11 plus three regressions for numerical
rejection, the trusted CPU repair, and explicit failure checks that remain active under `python -O`.

Fresh checks confirmed: model size/hash passes, authenticated health HTTP 200, unauthenticated
completion HTTP 401, disabled web UI HTTP 404, localhost-only listener, private `runs/` directory
(0700) and API key (0600). No model/runtime file is tracked. A publishable-checkout pattern scan
found no AWS access-key IDs or private-key headers; that is not an exhaustive secret scan.

The three real-classifier CPU fixtures passed in 3.0578 seconds total during the initial integration
check. They selected `repair_layout`, `repair_layout`, and `repair_api` respectively. All three
used the existing repair-prompt builder, retained checker feedback, passed after a scripted repair,
and remained rejected when the broken result was checked again. Simulated outage fallback also
passed. These are small fixed fixtures, not evidence of broad decision accuracy or a faster agent.

Reviewers can reproduce them using the README commands. Stage only source/docs/tests and the ignore
rules; do not force-add `models/` or `runs/`. The main agent and numerical grader remain unchanged.

Implementation distinctions retained from review: redirects are not destination-allowlisted;
existing API-key checks reject symlinks/group-other access but do not positively verify file type;
existing directory permissions are not tightened automatically; direct imported `decide` callers
must invoke `bounded_state` and validate inputs; archive budgets apply to selected JSONL members,
not every archive byte; runtime error text/metadata is not a generic confidentiality sanitizer.
