"""REVIEWER: judges kernels that compute the right values, accepts them or names one improvement.

What "done" means comes from the ladder. Levels 1-4 and 8 have no speed bar, so a kernel correct on
every shape is accepted at once, with no model call. Levels 5-7 must also move few enough HBM bytes
(nkibench's max_waste: 1.6x, 1.25x, 1.05x of the floor); the checks enforce that, so a kernel over the
bar arrives here as a TRAFFIC failure, and the reviewer names the improvement.

Every number is from the simulator's byte counter, and is labelled so. Nothing here has run on the
device: see DESIGN.md, profiler layer 3.
"""

from agents2.llm import Section
from agents2.debugger import parse_change


def profile_text(check):
    """The measurements, one line per shape."""
    rows = []
    for s in check.get("shapes") or []:
        if "bytes" not in s:
            continue
        ratio = f"{s['ratio']:.2f}x the floor" if s.get("ratio") else "floor unknown"
        rows.append(f"  {s['label']}: {s['bytes']:,} HBM bytes in {s['transfers']} dma_copy calls, "
                    f"{ratio}")
    return ("Measured in the simulator (not on the device):\n" + "\n".join(rows)) if rows else ""


def worst_bytes(check):
    return max((s.get("bytes", 0) for s in check.get("shapes") or []), default=0)


class Reviewer:
    def __init__(self, llm, cfg):
        self.llm, self.cfg = llm, cfg

    def review(self, level, plan, code, check, tries, ledger, **tags):
        """Returns dict(accept, end, change, path, profile).

        tries: this thread's earlier improvement attempts as (bytes, change) pairs, oldest first."""
        prof = profile_text(check)
        if check["correct"]:
            return dict(accept=True, end=False, change=None, path="rule", profile=prof)
        # A TRAFFIC failure: right values, too many bytes (or none counted).
        now = worst_bytes(check)
        if len(tries) >= self.cfg.review_tries and all(b <= now for b, _ in tries[-self.cfg.review_tries:]):
            return dict(accept=False, end=True, change=None, path="rule", profile=prof,
                        reason=f"{self.cfg.review_tries} improvements without fewer bytes")
        if not tries or tries[-1][0] > now:
            # First try, or the last one helped: the checker's own message names the next step.
            return dict(accept=False, end=False, change=check.get("error"), path="rule", profile=prof)
        change = self._ask_model(level, plan, code, check, prof, tries[-1][1], ledger, **tags)
        return dict(accept=False, end=False, change=change or check.get("error"), path="model",
                    profile=prof)

    def _ask_model(self, level, plan, code, check, prof, previous, ledger, **tags):
        sections = [
            Section("intro", "This AWS Neuron NKI kernel computes the right values but moves too many "
                             "bytes between HBM and the chip. Name ONE change that moves fewer.",
                    required=True),
            Section("code", f"```python\n{code}\n```", required=True),
            Section("profile", prof, required=True),
            Section("check", (check.get("error") or "")[:600], required=True),
            Section("previous", f"Already tried, without fewer bytes: {previous[:400]}" if previous
                    else "", priority=4),
            Section("format", "Reply with exactly three lines:\nCAUSE: <which data is moved more than "
                              "once>\nLINE: <the line number>\nCHANGE: <the one change>", required=True),
        ]
        text, meta = self.llm.chat("reviewer", sections, tags=dict(level=level, **tags))
        ledger.note_call(meta)
        if meta.get("error"):
            return None
        return parse_change(text).get("change")
