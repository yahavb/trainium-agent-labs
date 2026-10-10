"""Sequential HTTP episodes with explicit call/deadline accounting."""
import json
import random
import re
import time

import httpx
import sympy as sp

from .bandit import ACTIONS, better, rule, state_of, transition_reward
from .physics import bounded, check, prompt_of
from .problems import make

TEMPLATES = {
    "RETRY": "Try a fresh derivation and a different candidate solution. Correct every reported failure.",
    "FIX_PDE": "Recheck each mode's eigenvalue and exponential time decay. Its decay rate must equal k times its own spatial frequency squared. Constant modes do not decay.",
    "FIX_COEFF": "Recompute the Fourier projection coefficients, signs, and normalization. Include missing terms and enough terms to meet the initial-profile tolerance. Do not guess coefficients.",
    "USE_CALCULATOR": "Use the calculator to evaluate the integrals you set up for needed coefficients. Choose the basis and integrals yourself; substitute all problem parameters numerically.",
    "CHANGE_BASIS": "Reconsider the spatial eigenfunctions and allowed frequencies using BOTH boundary conditions. Distinguish zero value from zero derivative and check whether a constant mode is allowed.",
    "FIX_FORMAT": "Give a finite explicit expression with numeric coefficients using valid Python arithmetic. End with exactly u(x, t) = <expression>. Remove unknown parameters, LaTeX, and unevaluated sums.",
}
INTERVENTION_VERSION = 2
CALCULATOR = (
    "For THIS turn, output ONLY 1 to 8 plain lines of the form COMPUTE: <Python expression>, "
    "then STOP and wait for calculator results. Do not give a derivation, guessed integral "
    "values, LaTeX, Markdown, or a u(x,t) answer in this turn. This overrides the final-answer "
    "instruction above for this turn only. Example syntax:\n"
    "COMPUTE: Integral(x*sin(pi*x), (x,0,1))\n"
    "Choose the integrals yourself, including normalization and all pieces/amplitudes of "
    "the initial profile. Substitute numeric problem parameters. For piecewise profiles, "
    "split integrals at the breakpoints. Allowed: arithmetic, sin, cos, exp, sqrt, pi, "
    "Integral, integrate, Sum, simplify, Rational, x, t, n."
)


def calculator_requests(answer):
    """Accept plain requests and harmless Markdown wrappers, preserving ** powers."""
    requests = []
    for line in answer.splitlines():
        line = line.strip()
        for wrapper in ("```", "**", "`"):
            if line.startswith(wrapper):
                line = line[len(wrapper):].strip()
                if line.endswith(wrapper):
                    line = line[:-len(wrapper)].strip()
                break
        match = re.fullmatch(r"COMPUTE:\s*(.+)", line)
        if match:
            requests.append(match.group(1).strip())
    return requests[:8]


class OperationalError(RuntimeError):
    """Transport/server failures are not mathematical outcomes."""


class BudgetStop(RuntimeError):
    pass


def remaining(deadline):
    value = deadline-time.monotonic()
    if value <= 0:
        raise BudgetStop("wall-time budget exhausted")
    return value


def http_request(cfg, prompt, seed, timeout):
    body = dict(model=cfg["model"], messages=[dict(role="user", content=prompt)],
                temperature=cfg["temperature"], top_p=cfg["top_p"], max_tokens=cfg["max_tokens"],
                chat_template_kwargs={"enable_thinking": False})
    # Neuron backends may lack the device Generator required by per-request seeds.
    if cfg.get("request_seeds", False):
        body["seed"] = seed
    response = httpx.post(cfg["base"].rstrip("/")+"/chat/completions", json=body, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    text = data["choices"][0]["message"].get("content") or ""
    if not isinstance(text, str):
        raise ValueError("content must be text")
    return text, data.get("usage")


class HTTPClient:
    synthetic = False

    def __init__(self, config):
        self.config = config

    def ask(self, prompt, seed, record, action, deadline):
        timeout = min(self.config["http_timeout"], remaining(deadline))
        try:
            # httpx phase timeouts alone do not bound a slowly streaming response.
            # A killable process bounds the whole request and response decoding.
            return bounded("http", (self.config, prompt, seed, timeout), timeout)
        except (ValueError, TimeoutError, EOFError) as exc:
            remaining(deadline)
            raise OperationalError(f"Qwen request failed: {str(exc)[:300]}") from exc


class OfflineClient:
    """Synthetic fixture, deliberately unrelated to real-model performance."""
    synthetic = True

    def ask(self, prompt, seed, record, action, deadline):
        remaining(deadline)
        if action == "USE_CALCULATOR" and "Calculator results:" not in prompt:
            return "COMPUTE: Integral(x*sin(pi*x), (x,0,1))", None
        fixture = make(record["level"], record["sub"], record["seed"], references=True)
        rng = random.Random(seed)
        if action is not None and fixture.get("reference") and rng.random() < 0.7:
            return "u(x,t) = "+fixture["reference"], None
        return rng.choice(["u(x,t) = 0", "u(x,t) = "+record["f"], "I need more time."]), None


def write_log(log, record):
    log.write(json.dumps(record, allow_nan=False)+"\n")
    log.flush()


def episode(record, client, policy, mode, config, seed, deadline, log, epsilon=0):
    start = time.monotonic()
    calls, requests, repairs = 0, 0, 0
    calculator_used = False
    base = prompt_of(record)
    rng = random.Random(seed)

    def completion(prompt, action):
        nonlocal calls
        if calls >= config["max_calls"]:
            raise BudgetStop("model-call budget exhausted")
        remaining(deadline)
        request_seed = seed*100 + calls
        calls += 1
        request_start = time.monotonic()
        try:
            answer, usage = client.ask(prompt, request_seed, record, action, deadline)
        except (OperationalError, BudgetStop, KeyboardInterrupt) as exc:
            write_log(log, dict(kind="request_failure", problem=record["id"], mode=mode,
                                seed=seed, request_seed=request_seed, action=action,
                                synthetic=client.synthetic, http_call=calls, prompt=prompt,
                                elapsed=time.monotonic()-request_start, reason=str(exc)))
            raise
        write_log(log, dict(kind="completion", problem=record["id"], mode=mode,
                            seed=seed, request_seed=request_seed, action=action,
                            synthetic=client.synthetic, http_call=calls, prompt=prompt,
                            answer=answer, usage=usage, elapsed=time.monotonic()-request_start))
        return answer, usage

    def grade(answer):
        return check(record, answer, timeout=min(config["symbolic_timeout"], remaining(deadline)))

    answer, usage = completion(base, None)
    best = grade(answer)
    first_score = best["reward"]
    write_log(log, dict(kind="initial", problem=record["id"], level=record["level"], mode=mode,
                        seed=seed, synthetic=client.synthetic, prompt=base, answer=answer,
                        result=best, usage=usage, http_calls=calls))
    while best["reward"] != 1.0 and repairs < config["repairs"] and calls < config["max_calls"]:
        remaining(deadline)
        state = state_of(best, record["tol"])
        if mode in ("train", "learned"):
            action = policy.select(state, epsilon if mode == "train" else 0)
        elif mode == "random":
            action = rng.choice(ACTIONS)
        elif mode == "rule":
            action = rule(best, calculator_used)
        else:
            action = "RETRY"
        prompt = (base+"\n\nPrevious candidate: "+str(best["expr"])+
                  "\nChecker feedback: "+best["feedback"]+"\n"+TEMPLATES[action])
        if action == "USE_CALCULATOR":
            prompt += "\n"+CALCULATOR
        calls_before, requests_before = calls, requests
        answer, usage = completion(prompt, action)
        exchanges = [dict(prompt=prompt, answer=answer, usage=usage)]
        asks = calculator_requests(answer) if action == "USE_CALCULATOR" else []
        if asks and calls < config["max_calls"]:
            blocks = []
            for expression in asks:
                remaining(deadline)
                requests += 1
                try:
                    value = bounded("compute", expression, min(config["symbolic_timeout"], remaining(deadline)))
                except (ValueError, TimeoutError, EOFError) as exc:
                    value = f"Calculator error: {exc}"
                write_log(log, dict(kind="calculator", problem=record["id"], mode=mode,
                                    seed=seed, synthetic=client.synthetic,
                                    expression=expression, value=value, request=requests))
                blocks.append(f"COMPUTE: {expression}\n= {value}")
            calculator_used = True
            followup = prompt+"\n\n"+answer+"\nCalculator results:\n"+"\n".join(blocks)+"\nUse these results and give the final explicit u(x,t) answer now."
            answer, usage = completion(followup, action)
            exchanges.append(dict(prompt=followup, answer=answer, usage=usage))
        candidate = grade(answer)
        reward = transition_reward(best, candidate, calls-calls_before, requests-requests_before)
        if mode == "train":
            policy.update(state, action, reward)
        write_log(log, dict(kind="transition", problem=record["id"], level=record["level"], mode=mode,
                            seed=seed, synthetic=client.synthetic, attempt=repairs+1,
                            state=state, next_state=state_of(candidate, record["tol"]), action=action,
                            before=best, result=candidate, bandit_reward=reward, exchanges=exchanges,
                            http_calls=calls-calls_before, calculator_requests=requests-requests_before))
        if better(candidate, best):
            best = candidate
        repairs += 1
    return dict(problem=record["id"], level=record["level"], seed=seed, mode=mode,
                first_score=first_score, score=best["reward"], first_solved=first_score == 1.0,
                solved=best["reward"] == 1.0, http_calls=calls, calculator_requests=requests,
                repairs=repairs, elapsed=time.monotonic()-start, synthetic=client.synthetic)
