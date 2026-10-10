"""Prompt conditions for the real-abstract summarization experiment.

Key sentences must be annotated by a human from the original abstract.
This module deliberately does not score summaries or implement a checker.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import random
import re

CONDITIONS = ("baseline", "key2", "random2")
ANNOTATION_PATH = Path(__file__).with_name("annotations.json")


def sentences(text: str) -> list[str]:
    """Lightweight sentence segments for *inspection*; check each boundary manually.

    Scientific abbreviations can fool simple segmentation. Human annotations are
    literal spans of the source and may be longer/differently segmented.
    """
    return [p.strip() for p in re.split(r"(?<=[.!?])\s+(?=[A-Z(])", text.strip()) if p.strip()]


def _annotations(item_id: str) -> list[str]:
    if not ANNOTATION_PATH.exists():
        raise ValueError(f"Missing {ANNOTATION_PATH.name}: annotate two source sentences before running key2/random2")
    data = json.loads(ANNOTATION_PATH.read_text(encoding="utf-8"))
    entry = data.get(item_id)
    if not isinstance(entry, dict):
        raise ValueError(f"No annotations for {item_id}")
    keys = entry.get("key_sentences")
    if not isinstance(keys, list) or len(keys) != 2 or any(not isinstance(s, str) or not s.strip() for s in keys):
        raise ValueError(f"{item_id}: set exactly two human-selected key_sentences in annotations.json")
    if keys[0] == keys[1]:
        raise ValueError(f"{item_id}: key sentences must be distinct")
    return keys


def build_prompt(condition: str, item_id: str, abstract: str, repeat: int = 1) -> dict:
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition: {condition}")
    if not abstract.strip():
        raise ValueError("Empty abstract")
    parts = sentences(abstract)
    selected: list[str] = []
    key_indices: list[int] = []
    selected_indices: list[int] = []
    seed = None
    if condition != "baseline":
        keys = _annotations(item_id)
        # Ensure the repeated evidence is unchanged text from the real abstract.
        for key in keys:
            if key not in abstract:
                raise ValueError(f"{item_id}: annotated key sentence is not an exact substring of the original abstract")
        key_indices = [i for i, s in enumerate(parts) if any(s == k for k in keys)]
        if condition == "key2":
            selected = keys
            selected_indices = [parts.index(s) if s in parts else -1 for s in selected]
        else:
            # A fixed deterministic draw for every (abstract, repeat), with no
            # replacement. Exclude any segment overlapping annotated evidence.
            pool = [(i, s) for i, s in enumerate(parts)
                    if not any(s in k or k in s for k in keys)]
            if len(pool) < 2:
                raise ValueError(f"{item_id}: fewer than two eligible non-key sentence segments")
            seed = int.from_bytes(hashlib.sha256(f"{item_id}:random2:{repeat}".encode()).digest()[:8], "big")
            chosen = random.Random(seed).sample(pool, 2)
            selected_indices = [i for i, _ in chosen]
            selected = [s for _, s in chosen]
    # Preserve the exact source: repetition is additional context, not replacement.
    prompt = "Summarise: " + abstract
    if selected:
        prompt += "\n\nSource sentences repeated for emphasis (verbatim):\n" + "\n".join(selected)
    return {
        "prompt": prompt,
        "sentence_count": len(parts),
        "key_indices": key_indices,
        "repeated_indices": selected_indices,
        "repeated_sentences": selected,
        "context_chars": len(prompt) - len("Summarise: " + abstract),
        "seed": seed,
    }
