"""Deterministic prompt perspectives and structural diversity; no model calls."""

from dataclasses import dataclass
import hashlib

from failure_selection import code_fingerprint


@dataclass(frozen=True)
class PromptVariant:
    strategy: str
    prompt: str


_PERSPECTIVES = (
    ("original", ""),
    ("simple", "Prefer the simplest legal NKI operations. Remove unnecessary intermediates; "
     "do not invent APIs. Prioritize numerical correctness over optimization."),
    ("shapes_buffers", "Check the shape of every operand, allocation and slice. Keep partition "
     "and free dimensions explicit; verify required buffers and matching transfer sizes."),
    ("boundaries_coverage", "Check tiling and loop bounds against actual input dimensions. "
     "Handle partial boundary tiles and copy every computed result to its correct output slice."),
)


def build_variants(prompt, samples, policy="standard", *, repair=False, repeated=0, repair_scope=None):
    if samples < 1:
        raise ValueError("samples must be positive")
    if policy not in ("standard", "diverse"):
        raise ValueError("unsupported candidate policy")
    if policy == "standard":
        return tuple(PromptVariant("original", prompt) for _ in range(samples))
    result = []
    for index in range(samples):
        strategy, perspective = _PERSPECTIVES[index % len(_PERSPECTIVES)]
        if index == 0:
            result.append(PromptVariant(strategy, prompt))
            continue
        if not perspective:
            strategy, perspective = "local_correction", "Inspect the named failing operation first."
        base = prompt
        if repair:
            perspective += (" Preserve working code and the checker context; apply a localized correction."
                            if repair_scope in (None,"localized_api_correction") else
                            " Follow the supplied repair scope; coordinate dependent lines and preserve unrelated code.")
            if repeated >= 2 and strategy == "simple":
                # A bounded alternative is permitted only after repeated local failures.
                base = base.replace(
                    "Change exactly what the checker names and keep everything else identical.",
                    "The localized correction has repeatedly failed. You may simplify the failing "
                    "operation and its dependent shapes while preserving unrelated code.")
                perspective += " Simplify only the failing operation and its dependencies if needed."
        suffix = f"\n\nCandidate perspective ({strategy}): {perspective}"
        if index >= len(_PERSPECTIVES):
            suffix += f" Independently check this perspective for candidate {index}."
        result.append(PromptVariant(strategy, base + suffix))
    return tuple(result)


def diversity_metrics(sources):
    exact = tuple(hashlib.sha256(source.encode()).hexdigest() for source in sources)
    structural = tuple(code_fingerprint(source) for source in sources)
    total = len(sources)
    return dict(total_candidates=total, unique_exact_sources=len(set(exact)),
                unique_ast_sources=len(set(structural)),
                duplicate_fraction=(1 - len(set(structural)) / total) if total else 0.,
                exact_source_hashes=exact, ast_source_hashes=structural)
