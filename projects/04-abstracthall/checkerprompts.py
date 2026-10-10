#!/usr/bin/env python3
"""
checkerprompts.py — Independent hallucination checker with prompt-engineered repair.

Fully self-contained: no dependency on sumcheck.py.

Grades summaries using the same claim-level approach (numeric check + NLI),
but repair prompts are built using the prompt engineering methods from:

    King (2025) "Mitigating Hallucinations in Zero-Shot Scientific
    Summarisation: A Pilot Study" arXiv:2512.00931

Repair strategy by h_score severity:
  Faithful  (h=0.0)       → None, no repair needed
  Low       (h < 0.3)     → PE-1:  explicit instruction nudge
  Medium    (h 0.3–0.6)   → CR-K1: repeat the single most relevant sentence
  High      (h > 0.6)     → CR-K2: repeat the two most relevant sentences

The repair prompt is self-contained and can be passed directly to
solver.solve() without any extra context.
"""

import re
import json
import random

# ---------------------------------------------------------------------------
# Optional deps
# ---------------------------------------------------------------------------
# NLI model is disabled on Neuron/Trainium pods where torch is compiled for
# the accelerator and cannot run standard CPU models like bart-large-mnli.
# The numeric check + string match fallback still catches fabricated numbers
# and obvious hallucinations.
hf_pipeline = None

try:
    from sentence_transformers import SentenceTransformer
    import faiss
    import numpy as np
    _SENTENCE_MODEL = None

    def _get_sentence_model():
        global _SENTENCE_MODEL
        if _SENTENCE_MODEL is None:
            _SENTENCE_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        return _SENTENCE_MODEL

    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

try:
    import nltk
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False


# ---------------------------------------------------------------------------
# NLI model (lazy load)
# ---------------------------------------------------------------------------

def _get_nli_model():
    if not hasattr(_get_nli_model, "model"):
        if hf_pipeline is None:
            raise ImportError("transformers not installed. Run: pip install transformers torch")
        _get_nli_model.model = hf_pipeline(
            "zero-shot-classification", model="facebook/bart-large-mnli"
        )
    return _get_nli_model.model


# ---------------------------------------------------------------------------
# Grading helpers
# ---------------------------------------------------------------------------

def _extract_figures(text):
    """Returns all numeric values found in text."""
    return set(re.findall(r"\b\d+(?:\.\d+)?\b", text))


def _extract_claims(summary, llm_client=None, model_name="Qwen2.5-7B-Instruct"):
    """
    Splits a summary into atomic factual claims.
    Uses an LLM if a client is provided, otherwise falls back to sentence splitting.
    """
    if not llm_client:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", summary) if s.strip()]
        return [{"claim": s} for s in sentences]

    prompt = (
        "Extract all atomic factual claims from the following summary. "
        "Return ONLY a JSON array of strings, where each string is one independent claim.\n\n"
        f"Summary: {summary}"
    )
    response = llm_client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    try:
        raw_json = re.search(r"\[.*\]", response.choices[0].message.content, re.DOTALL)
        claims = json.loads(raw_json.group(0))
        return [{"claim": c} for c in claims]
    except Exception:
        return [{"claim": summary}]


def _verify_claim(claim_text, source_text):
    """
    Checks one claim against the source.
    Returns (verdict, reason) where verdict is one of:
      'Supported', 'Contradicted', 'Not established'
    """
    # 1. Numeric check — any number in the claim must exist in the source
    unsupported_figs = _extract_figures(claim_text) - _extract_figures(source_text)
    if unsupported_figs:
        return "Contradicted", f"Fabricated number(s): {', '.join(unsupported_figs)}"

    # 2. NLI check
    try:
        nli = _get_nli_model()
        result = nli(source_text, [claim_text], hypothesis_template="This text entails that {}")
        score = result["scores"][0]
        if score >= 0.85:
            return "Supported", "NLI entails the claim."
        elif score <= 0.15:
            return "Contradicted", "NLI contradicts the claim."
        else:
            return "Not established", "Insufficient evidence to support or contradict."
    except ImportError:
        if claim_text.lower() in source_text.lower():
            return "Supported", "Exact string match found."
        return "Not established", "No NLI model loaded to verify."


# ---------------------------------------------------------------------------
# Prompt engineering helpers (from arXiv:2512.00931)
# ---------------------------------------------------------------------------

def _split_sentences(text):
    if HAS_NLTK:
        return nltk.sent_tokenize(text)
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


_KEY_TERMS = ["result", "method", "conclusion"]


def _extract_key_sentences(text, k):
    """
    Returns the K sentences most semantically relevant to
    {result, method, conclusion} using all-MiniLM-L6-v2 + FAISS L2.
    Falls back to first-K sentences if dependencies are missing.
    """
    sentences = _split_sentences(text)
    k = min(k, len(sentences))

    if not HAS_SENTENCE_TRANSFORMERS:
        return sentences[:k]

    model = _get_sentence_model()
    sentence_vecs  = model.encode(sentences,   convert_to_numpy=True).astype("float32")
    key_term_vecs  = model.encode(_KEY_TERMS,  convert_to_numpy=True).astype("float32")
    query_vec      = np.mean(key_term_vecs, axis=0, keepdims=True).astype("float32")

    index = faiss.IndexFlatL2(sentence_vecs.shape[1])
    index.add(sentence_vecs)
    _, indices = index.search(query_vec, k)

    # Return in original document order
    return [sentences[i] for i in sorted(indices[0].tolist())]


def _build_prompt(source, method):
    """
    Builds a summarisation prompt using one of the paper's seven methods.
    Supported: baseline, pe-1, pe-2, cr-k1, cr-k2, ra-k1, ra-k2
    """
    method = method.lower().strip()

    if method == "baseline":
        return f"Summarise: {source}"

    elif method == "pe-1":
        return f"Write a concise abstract summarising this text: {source}"

    elif method == "pe-2":
        return (
            f"Write a concise abstract summarising this text using the following "
            f"sections: Background, Objective, Methods, Results, Conclusion. {source}"
        )

    elif method in ("cr-k1", "cr-k2"):
        k = 1 if method == "cr-k1" else 2
        key_sentences = _extract_key_sentences(source, k)
        cr_context = " ".join(key_sentences)
        return f"Summarise: {source} {cr_context}"

    elif method in ("ra-k1", "ra-k2"):
        k = 1 if method == "ra-k1" else 2
        key_sentences = _extract_key_sentences(source, k)
        sentences = _split_sentences(source)
        pool = [s for s in sentences if s not in key_sentences]
        k = min(k, len(pool))
        random_sentences = random.sample(pool, k) if pool else key_sentences
        ra_context = " ".join(random_sentences)
        return f"Summarise: {source} {ra_context}"

    else:
        raise ValueError(f"Unknown method '{method}'.")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check(item, reply, llm_client=None):
    """
    Grade a summary for hallucinations and return a prompt-engineered
    repair prompt when needed.

    Parameters
    ----------
    item       : dict  — must contain a "text" key with the source abstract
    reply      : str   — the model's generated summary
    llm_client         — optional OpenAI-compatible client for claim extraction

    Returns
    -------
    dict with keys:
        reward          float    1.0 = fully faithful, 0.0 = total hallucination
        h_score         float    (contradicted + not_established) / total_claims
        total_claims    int
        supported       int
        contradicted    int
        not_established int
        label           str      faithful | hallucination_contradicted | hallucination_unverified
        feedback        str      human-readable summary of issues
        repair_prompt   str|None prompt-engineered retry prompt, or None if faithful
        repair_method   str|None which PE method was used (pe-1/cr-k1/cr-k2), or None
    """
    source = item.get("text", "") or item.get("passage", "")

    summary = re.sub(r"<think>.*?</think>", "", reply or "", flags=re.S).strip()
    if not summary:
        return dict(reward=0.0, label="empty", feedback="No summary provided.",
                    repair_prompt=None, repair_method=None)

    claims = _extract_claims(summary, llm_client)
    if not claims:
        return dict(reward=0.0, label="no_claims", feedback="Could not extract facts.",
                    repair_prompt=None, repair_method=None)

    total_claims    = len(claims)
    contradicted    = 0
    not_established = 0
    feedback_lines  = []

    for c in claims:
        verdict, reason = _verify_claim(c["claim"], source)
        if verdict == "Contradicted":
            contradicted += 1
            feedback_lines.append(f"Contradicted: '{c['claim']}' ({reason})")
        elif verdict == "Not established":
            not_established += 1
            feedback_lines.append(f"Not established: '{c['claim']}' ({reason})")

    h_score = (contradicted + not_established) / total_claims
    reward  = round(1.0 - h_score, 3)

    if h_score == 0:
        label    = "faithful"
        feedback = "All claims supported."
    elif contradicted > 0:
        label    = "hallucination_contradicted"
        feedback = " ".join(feedback_lines[:2])
    else:
        label    = "hallucination_unverified"
        feedback = " ".join(feedback_lines[:2])

    repair_prompt, repair_method = _build_repair_prompt(
        h_score=round(h_score, 3),
        source=source,
        feedback_lines=feedback_lines,
    )

    return dict(
        reward=reward,
        h_score=round(h_score, 3),
        total_claims=total_claims,
        supported=total_claims - contradicted - not_established,
        contradicted=contradicted,
        not_established=not_established,
        label=label,
        feedback=feedback,
        repair_prompt=repair_prompt,
        repair_method=repair_method,
    )


def _build_repair_prompt(h_score, source, feedback_lines):
    """
    Maps hallucination severity to a prompt engineering method and builds
    a self-contained repair prompt.

      Low    h < 0.3   → PE-1   explicit instruction, light touch
      Medium h 0.3–0.6 → CR-K1  repeat the single most relevant sentence
      High   h > 0.6   → CR-K2  repeat the two most relevant sentences

    Returns (repair_prompt: str | None, repair_method: str | None)
    """
    if h_score == 0:
        return None, None

    problem_summary = "\n".join(f"- {line}" for line in feedback_lines[:3])

    if h_score < 0.3:
        method = "pe-1"
        prefix = (
            f"Your previous summary contained unsupported claims:\n{problem_summary}\n\n"
            f"Please try again, staying strictly within what the source states.\n\n"
        )
    elif h_score <= 0.6:
        method = "cr-k1"
        prefix = (
            f"Your previous summary contained several unsupported or contradicted claims:\n"
            f"{problem_summary}\n\n"
            f"Rewrite the summary. The source's most relevant sentence has been repeated "
            f"at the end of the prompt to help you stay grounded.\n\n"
        )
    else:
        method = "cr-k2"
        prefix = (
            f"Your previous summary was heavily hallucinated:\n{problem_summary}\n\n"
            f"Disregard your previous attempt. Write a new summary from scratch using "
            f"only the source text. The two most relevant sentences have been repeated "
            f"at the end of the prompt — use them as your anchor.\n\n"
        )

    return prefix + _build_prompt(source, method), method
