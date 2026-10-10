"""Test-time learning for the minimum-traffic loop: strategy bandit, attempt memory, population.

Pure controller state. No model calls, no weight updates. Deterministic given a seed so a run
can be replayed. The model still authors every candidate kernel; these tables only decide which
strategy is prompted and which kernel is improved next.

The arm menu mirrors Section 3 of the plan:
    retain_both, retain_rhs, retain_lhs, bounded_blocking, tidy_only

Reward convention used by the loop (traffic_agent.py):
    0 for an invalid candidate; otherwise the improvement in worst-case waste ratio over the
    parent, clamped to [0, 1]. Lower waste is better; the floor is 1.0.
"""

import math

ARMS = ("retain_both", "retain_rhs", "retain_lhs", "bounded_blocking", "tidy_only")


class Bandit:
    """UCB1 over the strategy arms.

    Untried arms are picked first, in alphabetical order, so the first len(ARMS) rounds are
    deterministic. Ties break alphabetically, so the same seeds reproduce the same choices.
    """

    def __init__(self, arms=ARMS, exploration=1.0):
        self.arms = tuple(arms)
        self.exploration = float(exploration)
        self.counts = {a: 0 for a in self.arms}
        self.totals = {a: 0.0 for a in self.arms}
        self.rounds = 0

    def select(self):
        untried = sorted(a for a in self.arms if self.counts[a] == 0)
        if untried:
            return untried[0]
        best, best_score = None, None
        for a in sorted(self.arms):
            mean = self.totals[a] / self.counts[a]
            bonus = self.exploration * math.sqrt(math.log(max(2, self.rounds)) / self.counts[a])
            score = mean + bonus
            if best_score is None or score > best_score:
                best, best_score = a, score
        return best

    def update(self, arm, reward):
        if arm not in self.counts:
            raise KeyError(f"unknown arm {arm!r}")
        r = min(1.0, max(0.0, float(reward)))
        self.counts[arm] += 1
        self.totals[arm] += r
        self.rounds += 1

    def state(self):
        return dict(exploration=self.exploration, rounds=self.rounds,
                    counts=dict(self.counts),
                    means={a: (self.totals[a] / self.counts[a]) if self.counts[a] else None
                           for a in self.arms})


class Memory:
    """One dict per attempt; best improvements and recent failures go back into the prompt."""

    def __init__(self, cap=200):
        self.cap = max(1, int(cap))
        self.items = []

    def append(self, item):
        self.items.append(dict(item))
        if len(self.items) > self.cap:
            del self.items[: len(self.items) - self.cap]

    def prompt_lines(self, best_k=6, failures_k=3):
        improved = [i for i in self.items
                    if i.get("ok") and float(i.get("improvement") or 0.0) > 0.0]
        improved.sort(key=lambda i: -float(i.get("improvement") or 0.0))
        recent_fail = [i for i in self.items if not i.get("ok")]

        def clean(seq):
            seen, out = set(), []
            for i in seq:
                lesson = (i.get("lesson") or "").strip()
                if lesson and lesson not in seen:
                    seen.add(lesson)
                    out.append(lesson)
            return out

        return dict(best=clean(improved[:best_k]),
                    failures=clean(recent_fail[-failures_k:]) if failures_k else [])

    def dump(self):
        return list(self.items)


class Population:
    """Parents for the next round. Every member must be valid; new valid candidates replace the
    worst member only when they are better. A slightly worse but different parent can hold the
    one trick the leader lacks, so the pool never collapses to a single kernel."""

    def __init__(self, size=3):
        self.size = max(1, int(size))
        self.members = []  # sorted by worst_waste ascending (lower is better)

    def insert(self, member):
        if not member.get("valid"):
            return False
        for m in self.members:
            if m["hash"] == member["hash"]:
                return False
        entry = dict(member)
        entry.setdefault("worst_waste", float("inf"))
        if len(self.members) < self.size:
            self.members.append(entry)
        elif entry["worst_waste"] < self.members[-1]["worst_waste"]:
            self.members[-1] = entry
        else:
            return False
        self.members.sort(key=lambda m: m["worst_waste"])
        return True

    def best(self):
        return self.members[0] if self.members else None

    def pick_parent(self, rng, second_prob=0.15):
        """Best member most rounds; the runner-up with a small logged probability."""
        if not self.members:
            return None
        if len(self.members) > 1 and rng.random() < second_prob:
            return self.members[1]
        return self.members[0]

    def state(self):
        return [dict(hash=m["hash"], worst_waste=m["worst_waste"], strategy=m["strategy"])
                for m in self.members]
