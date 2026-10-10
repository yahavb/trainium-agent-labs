#!/usr/bin/env python3
"""
sumcheck.py — Claim-level hallucination checker.
Extracts atomic claims from a summary and verifies them independently 
using deterministic numerical checks and NLI (Natural Language Inference).
"""

import re
import json
# NLI model disabled on Neuron/Trainium pods — torch is compiled for the
# accelerator and cannot run CPU models like bart-large-mnli.
# Fallback: numeric check + string match.
pipeline = None

# Initialize NLI model lazily to avoid loading overhead during simple imports
def get_nli_model():
    if not hasattr(get_nli_model, "model"):
        if pipeline is None:
            raise ImportError("transformers not installed. Run: pip install transformers torch")
        # Using a fast, lightweight NLI model suitable for CPU/standard inference
        get_nli_model.model = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    return get_nli_model.model

def _extract_figures(text):
    """Finds exact numeric values for deterministic checking."""
    return set(re.findall(r"\b\d+(?:\.\d+)?\b", text))

def extract_claims(summary, llm_client=None, model_name="Qwen2.5-7B-Instruct"):
    """
    Extracts atomic factual claims from the summary.
    In a live environment, this uses Qwen to split the text.
    """
    if not llm_client:
        # Fallback simplistic extraction for testing without an LLM endpoint
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
        temperature=0.0
    )
    
    try:
        raw_json = re.search(r"\[.*\]", response.choices[0].message.content, re.DOTALL)
        claims = json.loads(raw_json.group(0))
        return [{"claim": c} for c in claims]
    except Exception:
        # Fallback if the LLM fails to output valid JSON
        return [{"claim": summary}]

def verify_claim(claim_text, source_text):
    """
    Checks a single atomic claim against the source.
    Returns a verdict: 'Supported', 'Contradicted', or 'Not established'.
    """
    # 1. Deterministic Numerical Check
    claim_figs = _extract_figures(claim_text)
    src_figs = _extract_figures(source_text)
    
    unsupported_figs = claim_figs - src_figs
    if unsupported_figs:
        return "Contradicted", f"Fabricated number(s): {', '.join(unsupported_figs)}"

    # 2. Semantic NLI Check
    try:
        nli = get_nli_model()
        # We frame the source text as the premise, and check if it entails the claim
        result = nli(source_text, [claim_text], hypothesis_template="This text entails that {}")
        score = result['scores'][0]
        
        # Thresholds require tuning based on the specific NLI model used
        if score >= 0.85:
            return "Supported", "NLI entails the claim."
        elif score <= 0.15:
            return "Contradicted", "NLI contradicts the claim."
        else:
            return "Not established", "Insufficient evidence to support or contradict."
    except ImportError:
        # Fallback if transformers isn't available
        if claim_text.lower() in source_text.lower():
            return "Supported", "Exact string match found."
        return "Not established", "No NLI model loaded to verify."

def check(item, reply, llm_client=None):
    """
    Main grader. Computes the hallucination score H based on atomic claims.
    """
    source = item.get("text", "") or item.get("passage", "")
    
    # Strip any thought blocks from the reply
    summary = re.sub(r"<think>.*?</think>", "", reply or "", flags=re.S).strip()
    if not summary:
        return dict(reward=0.0, label="empty", feedback="No summary provided.", repeat_targets=[])

    claims = extract_claims(summary, llm_client)
    if not claims:
        return dict(reward=0.0, label="no_claims", feedback="Could not extract facts.", repeat_targets=[])

    total_claims = len(claims)
    contradicted = 0
    not_established = 0
    feedback_lines = []

    for c in claims:
        verdict, reason = verify_claim(c["claim"], source)
        if verdict == "Contradicted":
            contradicted += 1
            feedback_lines.append(f"Contradicted: '{c['claim']}' ({reason})")
        elif verdict == "Not established":
            not_established += 1
            feedback_lines.append(f"Not established: '{c['claim']}' ({reason})")

    # Calculate Hallucination Score (H)
    # H = (Contradicted + Not established) / Total Claims
    h_score = (contradicted + not_established) / total_claims
    
    # Reward is the inverse of the hallucination score (1.0 = perfect, 0.0 = total hallucination)
    reward = round(1.0 - h_score, 3)

    if h_score == 0:
        label = "faithful"
        feedback = "All claims supported."
    elif contradicted > 0:
        label = "hallucination_contradicted"
        feedback = " ".join(feedback_lines[:2]) # Keep feedback concise
    else:
        label = "hallucination_unverified"
        feedback = " ".join(feedback_lines[:2])

    repair_prompt = _build_repair_prompt(
        h_score=round(h_score, 3),
        source=source,
        original_summary=summary,
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
        repair_prompt=repair_prompt,      # None when faithful, string otherwise
        repeat_targets=[] # Can be populated with specific source sentences using FAISS if needed
    )


def _build_repair_prompt(h_score, source, original_summary, feedback_lines):
    """
    Builds a graded repair prompt to feed back into the solver.
    Returns None if no repair is needed (h_score == 0).

    Severity tiers:
      Low    h_score < 0.3  — targeted correction of specific claims
      Medium 0.3–0.6        — directive rewrite staying close to source
      High   > 0.6          — full rewrite, treat original as unreliable
    """
    if h_score == 0:
        return None

    # Show at most 3 problem claims to keep the prompt focused
    problem_claims = "\n".join(f"- {line}" for line in feedback_lines[:3])

    if h_score < 0.3:
        # Minor issues — nudge the model to fix specific claims only
        return (
            f"Your previous summary contained a small number of unsupported claims.\n\n"
            f"Problems found:\n{problem_claims}\n\n"
            f"Please revise only those specific claims so they are directly supported "
            f"by the source text below. Keep the rest of your summary unchanged.\n\n"
            f"Source:\n{source}"
        )
    elif h_score <= 0.6:
        # Moderate issues — directive rewrite with explicit grounding instruction
        return (
            f"Your previous summary contained several unsupported or contradicted claims "
            f"and needs revision.\n\n"
            f"Problems found:\n{problem_claims}\n\n"
            f"Rewrite the summary. Every factual claim must be directly traceable to the "
            f"source text. Do not introduce numbers, findings, or conclusions that are not "
            f"explicitly stated in the source.\n\n"
            f"Source:\n{source}"
        )
    else:
        # Severe — treat original summary as unreliable, start from scratch
        return (
            f"Your previous summary was heavily hallucinated and cannot be salvaged by "
            f"targeted edits.\n\n"
            f"Problems found:\n{problem_claims}\n\n"
            f"Disregard your previous summary entirely. Write a new summary from scratch "
            f"using only the source text below. Do not add any information, numbers, or "
            f"conclusions that do not appear in the source. Be conservative — if you are "
            f"unsure whether something is stated, leave it out.\n\n"
            f"Source:\n{source}"
        )