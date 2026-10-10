import re


def normalize(text):
    return re.sub(r"\s+", " ", str(text).lower()).strip()


def check_answer(answer, question):
    text = normalize(answer)
    facts = question.get("required_facts", [])
    forbidden = question.get("forbidden_claims", [])

    found = []
    missing = []
    total_weight = sum(float(f.get("weight", 1)) for f in facts)
    earned_weight = 0

    for fact in facts:
        phrases = fact.get("match_any", [])
        matched = any(normalize(p) in text for p in phrases)

        if matched:
            found.append(fact["id"])
            earned_weight += float(fact.get("weight", 1))
        else:
            missing.append(fact["id"])

    fact_score = (
        100 * earned_weight / total_weight
        if total_weight else 100
    )

    source_url = "https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/what-is-neuron.html"
    has_citation = source_url.lower() in text

    contradictions = [
        claim for claim in forbidden
        if normalize(claim) in text
    ]

    score = (
        0.8 * fact_score
        + (10 if has_citation else 0)
        + (10 if not contradictions else 0)
    )

    passed = (
        not missing
        and has_citation
        and not contradictions
    )

    feedback_parts = []

    if missing:
        feedback_parts.append(
            "Include these missing required facts: "
            + ", ".join(missing)
        )

    if not has_citation:
        feedback_parts.append(
            "Cite the documentation URL explicitly: " + source_url
        )

    if contradictions:
        feedback_parts.append(
            "Review these potentially contradictory claims: "
            + ", ".join(contradictions)
        )

    if passed:
        feedback_parts.append(
            "All configured checks passed. Ensure the answer is factually supported by the source."
        )

    return {
        "score": round(score, 2),
        "passed": passed,
        "found_facts": found,
        "missing_facts": missing,
        "has_citation": has_citation,
        "contradictions": contradictions,
        "feedback": " ".join(feedback_parts)
    }
