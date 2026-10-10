"""What has been tried at one level: approaches, the attempts in each thread, calls by role.

The ledger is the memory. No prompt carries a conversation; each one is rebuilt from here, which is
what keeps every call inside an 8K context however long the level runs.
"""

import collections
import dataclasses
import hashlib
import threading


@dataclasses.dataclass
class Plan:
    approach: str
    calls: list
    layout: str = ""
    steps: str = ""
    raw: str = ""
    unknown: list = dataclasses.field(default_factory=list)
    docs: list = dataclasses.field(default_factory=list)     # what the planner looked up

    def key(self):
        return frozenset(self.calls)

    def text(self):
        out = [f"APPROACH: {self.approach}"]
        if self.calls:
            out.append(f"CALLS: {', '.join(self.calls)}")
        if self.layout:
            out.append(f"LAYOUT: {self.layout}")
        if self.steps:
            out.append(f"STEPS: {self.steps}")
        return "\n".join(out)


@dataclasses.dataclass
class Approach:
    id: str
    plan: Plan
    thread: int
    best: float = 0.0
    attempts: int = 0
    kinds: collections.Counter = dataclasses.field(default_factory=collections.Counter)
    status: str = "running"
    end_reason: str = ""
    end_detail: str = ""

    def summary(self):
        kinds = ", ".join(f"{k} x{n}" for k, n in self.kinds.most_common(3))
        why = f"{self.end_reason}" + (f" ({self.end_detail[:100]})" if self.end_detail else "")
        return (f"{self.plan.approach[:160]} -> best {self.best:.2f}"
                + (f"; failures {kinds}" if kinds else "") + (f"; ended: {why}" if why else ""))


def same_approach(p, q):
    import difflib
    sim = difflib.SequenceMatcher(None, p.approach.lower(), q.approach.lower()).ratio()
    return sim >= 0.9 or (p.key() == q.key() and sim >= 0.6)


def sha(code):
    return hashlib.sha1((code or "").encode()).hexdigest()[:12]


class Ledger:
    def __init__(self, level, run):
        self.level, self.run = level, run
        self.lock = threading.Lock()
        self.approaches = []
        self.solved = False
        self.best = (0.0, "", None)            # reward, code, approach id
        self.calls = collections.Counter()
        self.tokens = collections.Counter()
        self.seconds = collections.Counter()
        self.failed_calls = 0                  # model calls in a row that failed outright
        self.stop_reason = ""
        self.checks = 0
        self.planning = 0                      # threads still choosing their approach

    def register(self, plan, thread):
        """Add an approach, unless the ledger already has the same one: the same set of calls and a
        similar description, or a near-identical description. Calls alone are not enough, since two
        different pooling algorithms can both be nl.sum plus dma_copy."""
        with self.lock:
            if any(same_approach(a.plan, plan) for a in self.approaches):
                return None
            a = Approach(id=f"A{len(self.approaches) + 1}", plan=plan, thread=thread)
            self.approaches.append(a)
            return a

    def tried(self, exclude=None):
        with self.lock:
            return [f"{a.id}: {a.summary()}" + (f" [calls: {', '.join(a.plan.calls)}]" if a.plan.calls
                                                 else "")
                    for a in self.approaches if a is not exclude]

    def note_call(self, meta):
        with self.lock:
            role = meta.get("role", "?")
            self.calls[role] += 1
            self.tokens[role] += int(meta.get("completion_tokens") or 0)
            self.seconds[role] += float(meta.get("gen_seconds") or 0)
            self.failed_calls = self.failed_calls + 1 if meta.get("error") else 0

    def total_calls(self):
        with self.lock:
            return sum(self.calls.values())

    def note_check(self, approach, code, check):
        with self.lock:
            self.checks += 1
            approach.attempts += 1
            approach.best = max(approach.best, check["reward"])
            if not check["correct"]:
                approach.kinds[check["kind"]] += 1
            if check["reward"] > self.best[0]:
                self.best = (check["reward"], code, approach.id)

    def accept(self, approach, code, check):
        with self.lock:
            self.solved = True
            self.best = (check["reward"], code, approach.id)
            approach.status, approach.end_reason = "accepted", "accepted"

    def end(self, approach, reason, detail=""):
        with self.lock:
            if approach.status == "running":
                approach.status, approach.end_reason, approach.end_detail = "ended", reason, detail
