"""Every setting in one place, so each run's config.json records exactly what it ran with."""

import dataclasses
import os

# Output caps, in tokens. Decode is ~99% of model time on the seat servers (measured 2026-10-10: 7.9 s of
# prefill against 1,299 s of decode), so output is what gets capped hard, while prompts can be generous.
# Temperatures: the planner is where different threads should differ, so it samples hot; top-p stays
# at 1.0 everywhere, because 0.95 collapsed 4 samples into 1 kernel in 72 of 83 rounds.
ROLE_DEFAULTS = {
    "planner":  dict(temperature=1.0, top_p=1.0, max_tokens=220, prompt_cap=3000),
    "coder":    dict(temperature=0.7, top_p=1.0, max_tokens=900, prompt_cap=4500),
    "debugger": dict(temperature=0.3, top_p=1.0, max_tokens=180, prompt_cap=4000),
    "reviewer": dict(temperature=0.3, top_p=1.0, max_tokens=180, prompt_cap=3500),
}
CODER_MAX_TOKENS_BIG = 1400          # levels 4 and up write longer kernels
CONTEXT_MARGIN = 192                 # chat template and tokenizer drift


@dataclasses.dataclass
class Role:
    name: str
    base: str
    model: str
    temperature: float
    top_p: float
    max_tokens: int
    prompt_cap: int


@dataclasses.dataclass
class Config:
    roles: dict
    threads: int = 2                 # 2 requests at once is the measured throughput peak per server
    max_approaches: int = 4          # plans tried per level, across all threads
    max_attempts: int = 10           # checks per thread
    max_no_gain: int = 6             # checks in a thread without a better reward
    same_error_limit: int = 3        # the same failure this many times in a row ends the thread
    review_tries: int = 2            # reviewer improvements without fewer bytes before it gives up
    max_calls: int = 80              # model calls per level
    check_workers: int = 2
    check_timeout: int = 180         # seconds per check (all shapes)
    retries: int = 3
    hint: bool = False               # nkibench's own level hint in the planner and coder prompts
    skeleton: bool = True            # the organisers' example kernel (end of agent.API_CARD) in write
    aws_docs: bool = False           # excerpts of third_party/ docs (nki 0.4.0) in the coder prompt
    offline: bool = False
    # LOOKUP rounds per role: how many times a role may ask the retriever for documentation before it
    # answers. 0 everywhere (--no-lookup) leaves the problem statement and the index alone.
    lookup: dict = dataclasses.field(default_factory=lambda: dict(planner=2, coder=1, debugger=1,
                                                                  reviewer=0))

    def as_dict(self):
        d = dataclasses.asdict(self)
        return d


def build(a):
    """Config from parsed arguments. --role ROLE=BASE|MODEL routes one role elsewhere, e.g. the planner
    to the Qwen3-32B server: --role planner=http://seat-198.seat:8000/v1|Qwen/Qwen3-32B"""
    base = (a.base or "").rstrip("/") + (a.path or "")
    routes = {}
    for spec in a.role or []:
        name, _, target = spec.partition("=")
        rbase, _, rmodel = target.partition("|")
        routes[name] = (rbase.rstrip("/") or base, rmodel or a.model)
    roles = {}
    for name, d in ROLE_DEFAULTS.items():
        rbase, rmodel = routes.get(name, (base, a.model))
        roles[name] = Role(name=name, base=rbase, model=rmodel, **d)
    if a.coder_max_tokens:
        roles["coder"].max_tokens = a.coder_max_tokens
    return Config(roles=roles, threads=a.threads, max_approaches=a.approaches,
                  max_attempts=a.attempts, max_no_gain=a.no_gain, max_calls=a.max_calls,
                  check_workers=a.check_workers, retries=a.retries, hint=a.hint,
                  skeleton=not a.no_skeleton,
                  aws_docs=a.aws_docs, offline=a.offline,
                  lookup=(dict(planner=0, coder=0, debugger=0, reviewer=0) if a.no_lookup else
                          dict(planner=2, coder=1, debugger=1, reviewer=0)))


def default_model():
    return os.environ.get("KERNEL_AGENT_MODEL", "Qwen/Qwen3-8B")
