# Model options and access audit — 2026-10-10

## Verified environment

Read-only checks on seat-100 found only `Qwen/Qwen3-8B` advertised by `/v1/models`, at `http://localhost:8000/v1`, with `MAX_MODEL_LEN=8192`. Runtime packages are vLLM 0.24.0, vllm-neuron 0.24.0.1.1.0, neuronx-cc 2.27.5334.0+f702b353. This is the newer vLLM Neuron plugin; the older NxDI package is absent. Do not assume old NxDI instructions apply directly.

The Hugging Face cache contains 16.38 GB of Qwen3-8B weights, 41.83 GB of unsloth/gpt-oss-20b-BF16 weights, 59.55 GB of meta-models/Muse-Glimmer-30B weights, and 2.10 GB of a Qwen3 DFlash draft. The openai/gpt-oss-20b directory has no weight files. File existence/size is not checksum validation or proof of a compilable checkpoint. No models were downloaded and no server was changed.

Explicit credential presence checks found no OPENAI_API_KEY or ANTHROPIC_API_KEY in either local process or seat-100. Local GPTOSS_BASE_URL is absent. No secret values were printed or copied.

The repo already documents a shared `gpt-oss-20b` server on separate hardware in `gptoss/README.md` and a Kubernetes job referencing secret `gptoss-url`. Reading even that secret's metadata is forbidden to the current role. Stop at that boundary; do not use an alternate pod or mounted secret to bypass RBAC.

AWS Bedrock `ListFoundationModels` and `ListInferenceProfiles` in ap-south-2 both returned AccessDenied for the existing role. These failures do not prove InvokeModel is denied, but they do mean no accessible hosted profile was verified. No permissions or credentials were changed and no inference request was made.

## Recommended sequence

1. **Immediate: Qwen3-8B thinking-mode treatment.** Existing trials explicitly disable thinking. Enable `chat_template_kwargs.enable_thinking=true` in a separately labeled arm; retain the same baseline/referee, sampling and candidate budget. The official model card recommends temperature 0.6, top_p 0.95, top_k 20 for thinking (versus 0.7/0.8 for non-thinking). The Qwen3 paper identifies thinking mode as its multi-step reasoning pathway. This tests an omitted capability without deployment risk, but is a mode ablation rather than a new model. [Model card](https://huggingface.co/Qwen/Qwen3-8B), [technical report](https://arxiv.org/abs/2505.09388).

   Respect the real 8192-token server context: measure prompt tokens first and reserve a bounded completion budget, e.g. up to 4096 if the prompt fits. Record finish_reason, generated tokens, and nonempty final code; a reasoning-only truncation is an inference failure, not a kernel failure. Do not confuse hidden reasoning with candidate code or silently retry without accounting.

2. **First different model: existing shared gpt-oss-20b, if its configured endpoint is made available through the authorized interface.** This avoids stealing any chip cores from current tests. Repository measurements say the shared endpoint is greedy, has an 8192 input limit, may exhaust reasoning budget before returning content, and ignores tool parameters. Use one sample per changed prompt, sufficient completion budget and explicit empty-content handling. Access remains unverified; do not claim it is running. Hosted cost is not known from the repository.

3. **Scheduled deployment option: Qwen2.5-Coder-14B-Instruct (or 7B if memory dictates).** This is a code-specialized checkpoint trained for generation/repair, a better motivated alternative than an arbitrary larger chat model. Its official checkpoint is about 29.6 GB; that is weight storage, excluding KV cache/compiler/runtime overhead. The Qwen2.5-Coder report supports the coding rationale, not a guarantee on NKI. AWS lists the Qwen2.5 architecture under NxDI, but this does NOT establish support in our installed new plugin: inspect the installed model registry and run a separate compile/serve smoke before claiming deployability. New downloads, compile latency and isolated serving capacity are required. [Paper](https://arxiv.org/abs/2409.12186), [checkpoint](https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct/tree/main), [AWS architecture reference](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.31.1/libraries/nxd-inference/developer_guides/model-reference.html).

4. **Cached local gpt-oss-20b BF16 is a fallback after current comparisons.** AWS's new vLLM Neuron tutorial supports GPT-OSS on Trainium2 in BF16. Use its tested recipe and verify required tensor-parallel memory before scheduling. A 41.8 GB checkpoint plus activations/KV/compiler buffers cannot be assumed to fit on a single free core. Do not replace the running Qwen server during a pinned experiment. [AWS tutorial](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.32.0/vllm-neuron/docs/tutorials/tutorial-gpt-oss.html).

No evidence justifies deploying the unfamiliar cached Muse checkpoint for this experiment. A speculative-decoding draft is not a substitute target generator.

## Evaluation gate

First run a small, separately logged generation/repair pilot and require valid final code, simulation correctness, device compilation and held-out chip correctness. Then repeat equal-budget trials. Log model ID/revision, mode, sampling, prompt hash, input/output tokens, generation latency and all rejected/duplicate candidates. Distinguish repairs from speedups. Do not combine a model change, seed-template change and feedback change into one causal claim. Keep the original comparison unchanged.
