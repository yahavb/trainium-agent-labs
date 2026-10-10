"""Tabular contextual bandit; checkpoint files contain no executable objects."""
import json
import hashlib
import math
import random
from pathlib import Path

ACTIONS = ("RETRY", "FIX_PDE", "FIX_COEFF", "USE_CALCULATOR", "CHANGE_BASIS", "FIX_FORMAT")
PARTS = ("equation", "left_bc", "right_bc", "start_shape")


def state_of(result, tolerance):
    parts = result["parts"]
    valid = all(key in parts for key in PARTS)
    error = result["start_error"]
    bucket = ("unavailable" if error is None or not math.isfinite(error) else
              "passed" if parts.get("start_shape") else
              "near" if error <= 10*tolerance else "far")
    return ":".join([str(int(valid)), *(str(int(bool(parts.get(key)))) for key in PARTS), bucket])


def transition_reward(current, next_result, calls, requests):
    newly = next_result["reward"] == 1.0 and current["reward"] != 1.0
    return next_result["reward"]-current["reward"] + 0.5*newly - 0.01*calls - 0.002*requests


def better(a, b):
    def rank(result):
        error = result["start_error"]
        return result["reward"], -(error if error is not None else float("inf"))
    return rank(a) > rank(b)


def rule(result, calculator_used=False):
    p = result["parts"]
    if len(p) != 4:
        return "FIX_FORMAT"
    if not p["left_bc"] or not p["right_bc"]:
        return "CHANGE_BASIS"
    if not p["equation"]:
        return "FIX_PDE"
    if not p["start_shape"]:
        return "FIX_COEFF" if calculator_used else "USE_CALCULATOR"
    return "RETRY"


def tuples(value):
    return tuple(tuples(item) if isinstance(item, list) else item for item in value)


class Bandit:
    def __init__(self, seed=42):
        self.values, self.counts = {}, {}
        self.rng = random.Random(seed)
        self.episodes = 0
        self.config, self.dataset = {}, None
        self.provenance = None

    def select(self, state, epsilon=0):
        if self.rng.random() < epsilon:
            return self.rng.choice(ACTIONS)
        values = self.values.get(state, [0.0]*len(ACTIONS))
        maximum = max(values)
        return self.rng.choice([a for a, v in zip(ACTIONS, values) if v == maximum])

    def update(self, state, action, reward):
        if not math.isfinite(reward):
            raise ValueError("nonfinite bandit reward")
        values = self.values.setdefault(state, [0.0]*len(ACTIONS))
        counts = self.counts.setdefault(state, [0]*len(ACTIONS))
        i = ACTIONS.index(action)
        counts[i] += 1
        values[i] += (reward-values[i])/counts[i]

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(version=1, actions=ACTIONS, values=self.values, counts=self.counts,
                       episodes=self.episodes, config=self.config, dataset=self.dataset,
                       rng=self.rng.getstate(), provenance=self.provenance)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2, allow_nan=False)+"\n")
        temporary.replace(path)

    @classmethod
    def load(cls, path):
        payload = json.loads(Path(path).read_text())
        if payload["version"] != 1 or tuple(payload["actions"]) != ACTIONS:
            raise ValueError("unsupported policy version or actions")
        policy = cls()
        policy.values, policy.counts = payload["values"], payload["counts"]
        if policy.values.keys() != policy.counts.keys():
            raise ValueError("invalid policy tables")
        for state, values in policy.values.items():
            counts = policy.counts[state]
            if len(values) != len(ACTIONS) or len(counts) != len(ACTIONS) or any(not math.isfinite(v) for v in values) or any(type(c) is not int or c < 0 for c in counts):
                raise ValueError("invalid policy values/counts")
        policy.episodes = payload["episodes"]
        policy.config, policy.dataset = payload["config"], payload["dataset"]
        policy.provenance = payload.get("provenance")
        # Legacy experiments always sent seeds; preserve that sampling requirement.
        if "max_tokens" in policy.config:
            policy.config.setdefault("request_seeds", True)
            policy.config.setdefault("intervention_version", 1)
        policy.rng.setstate(tuples(payload["rng"]))
        return policy

    @classmethod
    def warm_start(cls, path, config, dataset):
        """Migrate the calculator protocol while retaining unaffected observations."""
        policy = cls.load(path)
        previous = policy.config
        if previous.get("intervention_version") != 1 or config.get("intervention_version") != 2:
            raise ValueError("warm-start supports only intervention version 1 -> 2")
        before = {k: v for k, v in previous.items() if k != "intervention_version"}
        after = {k: v for k, v in config.items() if k != "intervention_version"}
        if policy.dataset != dataset or before != after:
            raise ValueError("warm-start requires identical data and generation configuration except intervention version")
        i = ACTIONS.index("USE_CALCULATOR")
        reset_count = sum(counts[i] for counts in policy.counts.values())
        for state in policy.values:
            policy.values[state][i] = 0.0
            policy.counts[state][i] = 0
        policy.provenance = dict(kind="calculator_protocol_warm_start", source=str(path),
                                 source_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                                 source_episodes=policy.episodes, source_config=previous,
                                 reset_action="USE_CALCULATOR", reset_observations=reset_count,
                                 retained_observations=sum(sum(c) for c in policy.counts.values()))
        policy.config = config
        return policy
