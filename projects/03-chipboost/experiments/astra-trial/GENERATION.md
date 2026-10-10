# Astra coding-agent trial

- Model label supplied for this delegated trial: `gpt-6-astra`.
- Provenance: agent-assisted source generation in Codex, 2026-10-10. This is a separately labeled coding-agent trial, not a Qwen benchmark or an equal-budget model comparison. The model label is supplied by orchestration; these artifacts contain no independent model-identity attestation.
- Requested environment: NKI 0.6.0, BF16 inputs `lhsT[K,M]` and `rhs[K,N]`, K and M multiples of 128, N a multiple of 512.

## Inputs inspected

- `projects/02-kernel-agent/reference_level4.py` in full.
- The `API_CARD` block in `projects/02-kernel-agent/agent.py`. A context search also displayed adjacent API-error explanation and agent orchestration code; none was used as an optimized kernel source. The initially specified filename `samefolderagent.py` was absent.
- The parent task's explicit guidance on partition-first resident arrays, list-comprehension frontend limitations, tile dimensions, and bounded storage.
- No external documentation was needed. No expert kernels, fixture kernels, prior optimized candidates, `search.py`, timing shapes, or held-out data were inspected.

## Candidate A

For K <= 4096, cache one RHS column panel in `(128, K/128, 512)` SBUF storage and reuse it across all M tiles. This uses at most 4 MiB for the BF16 resident panel. Relative to the reference's operand loads, the intended change eliminates repeated RHS HBM reads across M tiles. The LHS remains streamed.

For larger K, stream one RHS tile and reuse it across two M tiles when M is divisible by 256, retaining their two accumulators in PSUM. Odd M tile counts use a one-tile block. Storage remains independent of K.

## Candidate B

For K <= 16384, cache one LHS row panel in `(128, K/128, 128)` SBUF storage and reuse it across all N tiles. This uses at most 4 MiB for the BF16 resident panel. The intended change eliminates repeated LHS HBM reads across N tiles. The RHS remains streamed.

For larger K, stream one LHS tile and reuse it across two N tiles when N is divisible by 1024, retaining their two accumulators in PSUM. Odd N tile counts use a one-tile block. Storage remains independent of K.

## Validation status

Both candidates retain 128-by-128 stationary and 128-by-512 moving tiles, FP32 accumulation across K in PSUM, and final BF16 output conversion. No host execution, simulator, compiler, chip tool, public referee, timing run, or held-out evaluation was performed by this generation agent. Correctness, frontend compatibility, resource allocation, and performance remain for the parent agent's public-referee checks. These files make no measured performance claim.
