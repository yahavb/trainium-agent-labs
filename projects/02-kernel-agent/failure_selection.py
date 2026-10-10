"""Pure failure classification and optional equal-reward candidate selection.

No NKI imports, execution, inference, or filesystem writes. Confidence describes
pattern evidence, never kernel correctness. Ranking is an unverified repairability
hypothesis: precise targets and locations may help repair; category names have no
fixed priority. Reward always dominates, with exact equality (no rounding).
"""

import ast
from dataclasses import asdict, dataclass
import hashlib
import math
import re
from typing import Sequence


@dataclass(frozen=True)
class FailureDiagnostic:
    failure_category: str
    original_feedback: str
    normalized_signature: str
    classification_confidence: float
    classification_reason: str


# Match the checker's primary verdict before its appended repair examples. Each
# regex identifies a symptom; numeric details are removed only from signatures.
_RULES = (
    ("NO_CODE", r"No code came back\.?", 0.99, "explicit empty-code verdict"),
    ("SYNTAX_ERROR", r"(?:The code does not parse|the file does not parse):[^\n]*", 0.99,
     "explicit parser failure"),
    ("STATIC_RULE_VIOLATION", r"(?:Rule violations|RULE VIOLATIONS)[^\n]*", 0.99,
     "explicit static-check verdict"),
    ("HARDWARE_CORRECTNESS_HAZARD",
     r"CORRECT ON CPU BUT WRONG ON HARDWARE:[^\n]*|incorrect results on hardware[^\n]*",
     0.99, "explicit simulator hardware-correctness warning"),
    ("MEMORY_TRAFFIC_EXCESS", r"CORRECT, BUT TOO MUCH HBM TRAFFIC FOR THIS LEVEL:[^\n]*",
     0.99, "explicit traffic-threshold failure"),
    ("INCOMPLETE_OUTPUT", r"OUTPUT IS [\d.]+% ZEROS|[\d.]+% of the output is zero",
     0.99, "explicit missing-output verdict"),
    ("NUMERICAL_MISMATCH", r"NUMERICAL MISMATCH:[^\n]*|WRONG SHAPE:[^\n]*|NON-FINITE OUTPUT:[^\n]*",
     0.99, "explicit numerical comparison failure"),
    ("DMA_SHAPE_MISMATCH",
     r"dma_copy requires src and dst to have the same number of elements,\s*got src=\d+,\s*dst=\d+",
     0.99, "DMA element-count assertion"),
    ("INVALID_BUFFER_PLACEMENT",
     r"\b(?:\w+\s+)?(?:dst|src|stationary|moving) must be in \[[^\]]+\], got \w+",
     0.99, "named operand has an explicit required memory region"),
    ("INVALID_TENSOR_DIMENSIONS",
     r"SBUF and PSUM tensors must have at least 2 dimensions[^\n]*?\)|"
     r"SBUF and PSUM tensors must have at least 2 dimensions|"
     r"dma_copy \w+ partition dimension \d+ exceeds maximum \d+|"
     r"Matmul contraction dimension \d+ exceeds pmax=\d+|"
     r"cannot reshape array of size \d+ into shape \([^)]*\)|"
     r"value array of shape \([^)]*\) could not be broadcast to indexing result of shape \([^)]*\)",
     0.99, "explicit tensor rank, tile dimension, or reshape constraint"),
    ("OUT_OF_BOUNDS",
     r"Out-of-bound access for tensor .*?exceed dimension size of \d+|"
     r"index [^\n]*out of bounds[^\n]*",
     0.99, "explicit indexing-bound failure"),
    ("INVALID_API_ARGUMENT",
     r"\w+\(\) got an unexpected keyword argument '[^']+'|"
     r"\w+\(\) got multiple values for argument '[^']+'|"
     r"\w+\(\) missing \d+ required (?:positional|keyword-only) arguments?:[^\n]*",
     0.99, "Python argument-binding failure names an API argument"),
    ("INVALID_API_FUNCTION",
     r"module '[\w.]+' has no attribute '\w+'|"
     r"'\w+' object has no attribute '\w+'|"
     r"'MemoryRegion' object is not callable|There is no module named '[\w.]+'",
     0.95, "missing API member/module or memory region called as a function"),
)

_COMPILED_RULES = tuple((c, re.compile(p, re.I), confidence, reason)
                        for c, p, confidence, reason in _RULES)
CATEGORIES = frozenset(c for c, *_ in _RULES) | {"UNKNOWN"}
_NUMBER = re.compile(r"(?<![\w.])[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?", re.I)


def _normalize(text):
    # Normalize addresses and dimensions, preserving API and operand names.
    text = re.sub(r"0x[0-9a-f]+", "<address>", text, flags=re.I)
    text = _NUMBER.sub("<n>", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def classify_failure(feedback: str) -> FailureDiagnostic:
    """Preserve feedback verbatim and classify with deterministic ordered rules.

    Signatures describe the matched symptom, excluding per-case labels and most
    appended guidance. Unknowns retain normalized text rather than being conflated.
    """
    for category, pattern, confidence, reason in _COMPILED_RULES:
        match = pattern.search(feedback)
        if match:
            return FailureDiagnostic(category, feedback,
                                     category + ":" + _normalize(match.group()),
                                     confidence, reason)
    return FailureDiagnostic("UNKNOWN", feedback, "UNKNOWN:" + _normalize(feedback),
                             0.0, "no supported failure pattern; no classification evidence")


@dataclass(frozen=True)
class Candidate:
    reward: float
    code: str
    feedback: str


def code_fingerprint(code: str) -> str:
    """Structural novelty, not semantic equivalence; ignore comments/formatting."""
    try:
        canonical = ast.dump(ast.parse(code), include_attributes=False)
    except SyntaxError:
        canonical = re.sub(r"\s+", " ", code).strip()
    return hashlib.sha256(canonical.encode()).hexdigest()


def _repair_evidence(diagnostic):
    """Score information present in the feedback, never supposed class difficulty.

    Target 2: an operand/symbol plus an explicit constraint or corrective action.
    Target 1: an identified symbol/location without an equally specific repair.
    Location 2: source line or output element; 1: named API/operand; 0: global.
    A test-case shape label alone does not localize the bug in the source/output.
    """
    text = diagnostic.original_feedback
    category = diagnostic.failure_category
    location = 0
    target = 0
    if category == "UNKNOWN":
        return target, location
    if re.search(r"\bline \d+|(?:at index|first at)\s*\([\d, ]+\)", text, re.I):
        location, target = 2, 1
    named = re.search(r"\w+\(\) (?:got|missing)|module '[\w.]+' has no attribute '\w+'|"
                      r"\b(?:dst|src|stationary|moving) must be in", text)
    if named:
        location, target = max(location, 1), max(target, 1)
    if category == "DMA_SHAPE_MISMATCH":
        # The assertion supplies both operands and their violated size invariant.
        target, location = 2, max(location, 1)
    elif category == "INVALID_BUFFER_PLACEMENT":
        target, location = 2, max(location, 1)
    elif category == "INVALID_API_ARGUMENT" and re.search(
            r"Remove the `[^`]+=` argument|Pass every argument by keyword", text):
        target = 2
    elif category == "STATIC_RULE_VIOLATION" and re.search(
            r"line \d+: (?:calls|uses|partition)|no function named|not decorated", text):
        target = 2
    return target, location


@dataclass(frozen=True)
class CandidateEvidence:
    repair_target_strength: int
    localization_strength: int
    novel_code: bool
    repeated_signature_rounds: int
    code_fingerprint: str

    @property
    def rank(self):
        return (self.repair_target_strength, self.localization_strength,
                int(self.novel_code), -self.repeated_signature_rounds)


@dataclass(frozen=True)
class SelectionDecision:
    selected_index: int
    policy: str
    diagnostics: tuple[FailureDiagnostic, ...]
    evidence: tuple[CandidateEvidence, ...]
    selection_reasons: tuple[str, ...]

    def log_metadata(self, index):
        """Additive metadata; the caller retains every original attempt-log field."""
        diagnostic = asdict(self.diagnostics[index])
        diagnostic.pop("original_feedback")  # already present as legacy 'feedback'
        return dict(diagnostic, selected=index == self.selected_index,
                    candidate_index=index, selection_policy=self.policy,
                    selection_reason=self.selection_reasons[index],
                    selection_evidence=asdict(self.evidence[index]))


def select_candidate(candidates: Sequence[Candidate], policy="reward",
                     history: Sequence[Sequence[Candidate]] = ()) -> SelectionDecision:
    """Select among EXACT maximum-reward ties; final tie is original input order.

    History contains failed candidates grouped by prior round in this solve.
    Count signatures once per prior round, so duplicate samples do not inflate
    repetition. No disk memory, cross-level memory, reward changes, or planning.
    """
    if policy not in ("reward", "diagnostic"):
        raise ValueError(f"unsupported selection policy: {policy}")
    if not candidates:
        raise ValueError("cannot select from an empty candidate list")
    if any(not math.isfinite(c.reward) for c in candidates):
        raise ValueError("candidate rewards must be finite")
    diagnostics = tuple(classify_failure(c.feedback) for c in candidates)
    previous_codes = {code_fingerprint(c.code) for rnd in history for c in rnd}
    previous_signatures = [set(classify_failure(c.feedback).normalized_signature for c in rnd)
                           for rnd in history]
    evidence = tuple(CandidateEvidence(
        *_repair_evidence(d), code_fingerprint(c.code) not in previous_codes,
        sum(d.normalized_signature in signatures for signatures in previous_signatures),
        code_fingerprint(c.code)) for c, d in zip(candidates, diagnostics))
    maximum = max(c.reward for c in candidates)
    tied = [i for i, c in enumerate(candidates) if c.reward == maximum]
    selected = (tied[0] if policy == "reward"
                else max(tied, key=lambda i: evidence[i].rank))
    reasons = []
    for i, (c, d, e) in enumerate(zip(candidates, diagnostics, evidence)):
        if c.reward < maximum:
            reason = f"excluded: reward {c.reward!r} is below maximum {maximum!r}"
        elif policy == "reward":
            reason = "maximum reward; ties retain original candidate order"
        else:
            reason = (f"maximum reward {maximum!r}; category={d.failure_category}; "
                      f"rank(target,location,novel,-repeat)={e.rank}; "
                      f"selected candidate {selected}; final ties retain original order; "
                      "repairability heuristic is unverified")
        reasons.append(("selected: " if i == selected else "not selected: ") + reason)
    return SelectionDecision(selected, policy, diagnostics, evidence, tuple(reasons))
