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

## Iterative candidate C

The parent requested an additional agent-assisted iteration after reporting that both unchanged candidates passed its public-referee chip correctness and held-out checks. Its supplied aggregate feedback was A: 1.51705x baseline speedup and simulator traffic 1.16x the floor; B: 1.20335x baseline speedup and simulator traffic 1.75x the floor. These are parent-reported results for A/B, not measurements performed by this generation agent and not a prediction for C. Replication was still underway when the request arrived. No timing shapes or held-out data were supplied or inspected.

`candidate_c.py` extends the agent's own A/B designs. For K <= 8192 it keeps a RHS panel resident across M, selecting one to four exact-divisor N tiles subject to an 8 MiB BF16 residency cap. A single wider contiguous DMA fills all columns for each K tile. Each streamed LHS tile updates all N accumulators before advancing K, allowing reuse of both operands and exposing independent PSUM accumulators. The maximum four accumulators occupy 1 MiB of FP32 PSUM. The three-column threshold is the largest multiple of 128 below 8192/3, derived from the same residency cap.

For larger K, C uses a streamed output block of at most two M by two N tiles. Every loaded LHS tile serves both N tiles, and every loaded RHS tile serves both M tiles. Exact divisibility determines whether each block dimension is two or one. This fallback's SBUF and PSUM allocation sizes are independent of K.

C uses only the original permitted reference/API guidance, this agent's A/B source, and the explicitly supplied aggregate feedback. No additional repository sources, expert kernels, external documentation, compiler runs, simulator runs, or hardware runs were used. C's correctness, frontend compatibility, and performance remain unvalidated; no performance improvement is claimed. This remains a model-labeled, agent-assisted iterative trial rather than an equal-budget model comparison.
