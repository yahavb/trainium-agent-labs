# Prior work and dependencies

This project builds on existing kernel-fusion and online log-sum-exp algorithms.
It does not claim to invent fused log-probability/entropy scoring.

- [AWS NKI Library cross entropy](https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/experimental/loss/cross_entropy.py):
  streaming normalization and a measured baseline dependency, Apache-2.0.
- [Hugging Face TRL token scoring](https://github.com/huggingface/trl/blob/v1.14.0/trl/kernels/logprob_entropy.py):
  existing fused GPU operation, Apache-2.0; our contribution targets NKI/Trainium.
- [Neuron Agentic Development](https://github.com/aws-neuron/neuron-agentic-development):
  SDK authoring guidance. Installed `nkilib` utilities manage allocation, tiles,
  and kernel assertions.

No GPU latency from these sources is used as a Trainium baseline. SDK versions
and hashes of the installed baseline source are saved with measurement results.
