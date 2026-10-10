"""Exact token counts for attempts logged under feedback_v5+ (USAGE_LOG), for token_budget.py and
summarize.py's --usage.

v5's ask5 replaces agent.ask and returns a plain string, so attempts.jsonl carries chars/4 estimates.
USAGE_LOG has the server's counts, one line per request, with level, prompt_chars and the sha1 of the
code extracted from the reply. An attempt matches the earliest unused line with the same
(level, sha1(code), prompt_chars). Matched attempts get the server's prompt/completion tokens, and
their prompt_split is scaled to the exact prompt total. Unmatched attempts keep their estimate and
are counted, never dropped.
"""
import collections
import hashlib
import json


def _tokens(rec):
    """(prompt, completion) for one usage line; a thinking request may have a second call."""
    p = c = 0
    for key in ("usage", "answer_usage"):
        u = rec.get(key) or {}
        p += u.get("prompt_tokens") or 0
        c += u.get("completion_tokens") or 0
    return p, c


def load(paths):
    queues = collections.defaultdict(collections.deque)
    for path in paths:
        for line in open(path):
            if line.strip():
                rec = json.loads(line)
                if rec.get("usage"):
                    queues[(rec.get("level"), rec.get("code_sha1"), rec.get("prompt_chars"))] \
                        .append(rec)
    return queues


def apply(attempts, queues):
    """Rewrite token fields in place on every matched attempt. Returns (matched, unmatched)."""
    matched = unmatched = 0
    for r in attempts:
        key = (r["level"], hashlib.sha1((r.get("code") or "").encode()).hexdigest(),
               r.get("prompt_chars"))
        q = queues.get(key)
        if not q:
            r["usage_matched"] = False
            unmatched += 1
            continue
        rec = q.popleft()
        p, c = _tokens(rec)
        split = r.get("prompt_split") or {}
        est = sum(split.values())
        if est:
            r["prompt_split"] = {k: round(v * p / est) for k, v in split.items()}
        r["prompt_tokens"], r["completion_tokens"] = p, c
        r["count_method"] = f"usage_log (split scaled from {r.get('count_method', '?')})"
        r["usage_matched"] = True
        # ask5 returns a plain string, so the attempt itself never learns it was cut off
        r["finish"] = rec.get("answer_finish") or rec.get("finish") or rec.get("think_finish")
        r["request_t"], r["request_seconds"] = rec.get("t"), rec.get("seconds")
        matched += 1
    return matched, unmatched
