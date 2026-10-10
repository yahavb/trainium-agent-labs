#!/usr/bin/env python3
"""
agent.py — the loop: a small model on your trn2 works a heat-equation problem until it solves
it, or runs out of rounds.

    controller  picks a problem            (level0_heatrod.py / level1_heatrod.py)
    generator   the model proposes N answers, sampling ON
    checker     grades each one; the grade is the reward   (pdecheck.py)
    loop        the best answer from the entire history anchors bounded repairs

Every attempt is appended to a JSONL file as prompt, answer and reward. That file is what an
RL trainer consumes; the trainer replaces this file's last few lines and nothing else.

    export HEATROD_BASE_URL="http://qwen3-8b:8000/v1"
    python agent.py --level 1 --sub 3
    python agent.py --level 1 --all --samples 4 --rounds 4
    python agent.py --offline --level 0 --all

Sampling must be ON, or all N answers are identical and there is nothing to compare.

# CHANGE HONESTY A8/A9: fast mode stops at the first full checker pass. Each launched
# sample has a terminal record; cancelled/error records have reward=null. Consumers
# must filter status="graded" for scores. --collect-all disables within-round cancellation.
"""

import argparse
# CHANGE A1/A3/A6/A7: local state, bounded semantic history, and explicit custom problems.
import copy
from collections import OrderedDict, deque
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import time
import uuid

import numpy as np
import sympy as sp

import pdecheck
import tool_calc
import level0_heatrod
import level1_heatrod

LEVELS = {0: level0_heatrod, 1: level1_heatrod}


# CHANGE A1: order complete grades; equal priorities retain the older candidate.
def grade_priority(grade):
    reward = float(grade.get("reward", 0.0))
    reward = reward if math.isfinite(reward) else -math.inf
    passed = sum(value is True for value in grade.get("parts", {}).values())
    error = grade.get("start_error")
    finite = isinstance(error, (int, float)) and math.isfinite(error)
    return reward, passed, int(finite), -float(error) if finite else -math.inf


# CHANGE A2: consume structured diagnostics while retaining the old checker keys.
def conditions_of(grade):
    parts = grade.get("parts", {})
    passed = grade.get("passed_conditions")
    failed = grade.get("failed_conditions")
    if not isinstance(passed, list):
        passed = [key for key, value in parts.items() if value is True]
    if not isinstance(failed, list):
        failed = [key for key, value in parts.items() if value is not True]
        if not parts and grade.get("reward", 0.0) == 0.0:
            failed = ["format_or_evaluation"]
    actions = grade.get("repair_actions")
    if not isinstance(actions, list):
        actions = [grade.get("feedback", "Write an explicit evaluable expression.")]
    return [str(item) for item in passed], [str(item) for item in failed], [str(item) for item in actions]


# CHANGE A6: symbolic structure detects duplicates; this is not full mathematical equivalence.
def candidate_key(grade, answer):
    expression = grade.get("expr")
    if expression:
        try:
            parsed = pdecheck.parse(expression)
            # No expand/factor/simplify: symbolic products must not trigger expansion.
            canonical = sp.srepr(parsed)
            return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        except Exception:
            pass
    normalized = re.sub(r"\s+", "", expression or answer)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


# CHANGE A3: bounded, deduplicated candidate/error memory contains no controller answers.
def remember_attempt(history, seen, errors, key, answer, grade, rnd, limit, error_limit):
    duplicate = key in seen
    seen[key] = rnd
    seen.move_to_end(key)
    while len(seen) > limit:
        seen.popitem(last=False)
    passed, failed, actions = conditions_of(grade)
    history[:] = [item for item in history if item["canonical_key"] != key]
    history.append(dict(round=rnd, canonical_key=key, reward=grade["reward"],
                        passed_conditions=passed, failed_conditions=failed,
                        expression=(grade.get("expr") or answer)[:240]))
    del history[:-limit]
    if failed:
        mistake = grade.get("feedback", "")[:300]
        signature = json.dumps([failed, re.sub(r"\d+(?:\.\d+)?", "#", mistake)], sort_keys=True)
        previous = next((item for item in errors if item["signature"] == signature), None)
        if previous:
            errors.remove(previous)
            previous.update(last_round=rnd, count=previous["count"] + 1)
            errors.append(previous)
        else:
            errors.append(dict(signature=signature, last_round=rnd, count=1,
                               failed_conditions=failed, mistake=mistake,
                               repair_actions=[action[:180] for action in actions[:4]]))
        del errors[:-error_limit]
    return duplicate


# CHANGE A4: the model derives every coefficient; the controller supplies only BC rules.
def calculator_plan(problem, tools_enabled):
    left, right = problem["left"], problem["right"]
    if left == right == "dirichlet":
        basis = "phi_n=sin(n*pi*x/L), omega_n=n*pi/L, n>=1"
    elif left == "dirichlet":
        basis = "phi_n=sin((n-1/2)*pi*x/L), omega_n=(n-1/2)*pi/L, n>=1"
    elif right == "dirichlet":
        basis = "phi_n=cos((n-1/2)*pi*x/L), omega_n=(n-1/2)*pi/L, n>=1"
    else:
        basis = "phi_n=cos((n-1)*pi*x/L), omega_n=(n-1)*pi/L; n=1 is the constant mode"
    compute = ("Send concrete projection integrals on COMPUTE: lines, within the allowed tool exchange."
               if tools_enabled else "Calculator calls are disabled; evaluate the projection yourself, without COMPUTE: lines.")
    return ("\n\nConstruction phases:\n"
            f"1. Use the stated boundary conditions: {basis}. Substitute the stated L.\n"
            "2. For each mode use c_n=Integral(f(x)*phi_n,(x,0,L))/Integral(phi_n**2,(x,0,L)); "
            "substitute the stated f and L before asking for any calculation. This normalization also handles the constant mode.\n"
            f"3. {compute} Fill in numerical or exact numeric coefficients; do not leave Integral or c_n unresolved.\n"
            "4. For each individual mode write omega, then rate=k*omega**2. Pair that rate with the same mode. "
            "The constant mode has rate zero.\n"
            "5. Assemble explicit terms c*exp(-rate*t)*phi. Check the equation, each boundary and u(x,0); "
            "finish with one u(x,t)=expression line. No unevaluated sums or unknown constants.")


# CHANGE A2/A3: repair the historical anchor and preserve everything already verified.
def repair_prompt(base_prompt, best, history, errors, strategy):
    grade = best["grade"]
    passed, failed, actions = conditions_of(grade)
    anchor = grade.get("expr") or best["answer"]
    lines = [base_prompt, "", "Historical best candidate (do not replace it with a worse recent attempt):",
             f"  u(x, t) = {anchor}", f"Historical best reward: {grade['reward']}",
             f"Already passed: {', '.join(passed) or 'none'}.",
             f"Failed: {', '.join(failed) or 'none'}.",
             "Preserve the passed conditions and recheck them after every change."]
    if "start_shape" in passed:
        if set(failed).issubset({"equation"}) and {"left_bc", "right_bc"}.issubset(passed):
            lines.append("Preserve the verified spatial basis, u(x,0), and coefficients. Only repair each "
                         "exponential rate=k*omega**2, matched to its own spatial mode.")
        else:
            lines.append("Preserve the accepted u(x,0) within its tolerance. If a boundary fails, re-express "
                         "that initial shape in the correct boundary basis and recompute projections if needed; "
                         "do not lock an incorrect basis or coefficients. Recheck all previously passed conditions.")
    if "equation" in passed:
        lines.append("Preserve the equation: when a failing boundary requires a changed basis, pair each new "
                     "frequency with its own k*omega**2 rate rather than reusing an incompatible old rate.")
    lines.extend(["Concrete repair actions:", *[f"- {action}" for action in actions],
                  f"Checker feedback: {grade.get('feedback', '')}", f"Current strategy: {strategy}."])
    if errors:
        lines.append("Recent distinct mistakes; do not repeat the same failed repair:")
        lines.extend(f"- {','.join(item['failed_conditions'])} ({item['count']} occurrences): {item['mistake']}"
                     for item in errors)
    if history:
        lines.append("Recent distinct attempted expressions (these are attempts, not correct answers):")
        lines.extend(f"- reward={item['reward']}, failed={','.join(item['failed_conditions'])}: {item['expression']}"
                     for item in history)
    lines.append("Apply the smallest justified repair to the historical best, then give the complete explicit expression.")
    return "\n".join(lines)


# CHANGE A5: distinct bounded sample plans; optional adaptive counts never mutate shared args.
def sample_prompt(prompt, strategy, sample_index):
    if strategy.startswith("target_repair"):
        plans = ("Repair only the named failing factor; keep every passed term and coefficient.",
                 "Independently verify each omega/rate pair, then make the smallest correction.",
                 "Check the failed residual directly before writing the minimally repaired expression.",
                 "Use an algebraically equivalent explicit repair and revalidate all passed conditions.")
    elif strategy == "diversify":
        plans = ("Re-derive the mode normalization independently rather than tweaking the previous guessed numbers.",
                 "Use a different justified truncation of the allowed modes, with individually derived coefficients.",
                 "Build separated spatial/time factors independently and verify each rate before combining them.",
                 "Derive an equivalent expanded representation; justify coefficient signs and boundary symmetry.")
    else:
        plans = ("Derive the lowest allowed modes and their exact projection coefficients.",
                 "Independently verify projection normalization and coefficient signs before assembly.",
                 "Start with each boundary and its allowed frequencies, then derive matching decay rates.",
                 "Check how many allowed modes the initial-shape tolerance requires; avoid guessing coefficients.")
    suffix = plans[sample_index % len(plans)]
    if strategy == "target_repair_diverse":
        suffix += " Take an independent derivation of the failing factor rather than repeating an unsuccessful tweak."
    return f"{prompt}\n\nSample plan {sample_index + 1}: {suffix}"


def sample_settings(a, strategy, sample_index, rnd):
    if strategy.startswith("target_repair"):
        temperature = getattr(a, "target_temperature", 0.4)
    elif strategy == "diversify":
        temperature = getattr(a, "diversify_temperature", 0.8)
    else:
        temperature = getattr(a, "temperature", 0.6)
    return min(2.0, temperature + 0.03 * (sample_index % 4) + 0.02 * (rnd % 2)), getattr(a, "top_p", 0.95)


# CHANGE A7: explicit four-BC input; no exact/series answer is read or manufactured.
def problem_from_file(path, default_seed):
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    required = {"L", "k", "f", "left", "right"}
    allowed = required | {"tol", "seed", "name"}
    if not isinstance(data, dict) or not required.issubset(data) or set(data) - allowed:
        raise ValueError("Problem JSON needs L,k,f,left,right; optional fields are tol,seed,name only.")
    numbers = {}
    for key in ("L", "k"):
        value = pdecheck.parse(str(data[key]))
        if (value.free_symbols or value.is_number is not True or value.is_real is not True
                or not math.isfinite(float(value)) or float(value) <= 0):
            raise ValueError(f"{key} must be a finite positive real number.")
        numbers[key] = value
    for side in ("left", "right"):
        if data[side] not in ("dirichlet", "neumann"):
            raise ValueError(f"{side} must be dirichlet or neumann.")
    f = pdecheck.parse(str(data["f"]))
    if f.free_symbols - {pdecheck.x} or f.has(sp.Sum, sp.Integral, sp.Derivative):
        raise ValueError("f must be an explicit expression in x only.")
    grid = np.linspace(0, float(numbers["L"]), 801)
    if not np.all(np.isfinite(pdecheck._num(f, grid, np.zeros_like(grid)))):
        raise ValueError("f must be finite and real on the rod's checker grid.")
    tol = float(data.get("tol", 1e-6))
    seed = data.get("seed", default_seed)
    if not math.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be finite and positive.")
    # CHANGE A7: reject invalid sampling seeds before any model request.
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a nonnegative integer.")
    problem = dict(name=str(data.get("name", "custom-heat-rod")), level="custom", seed=seed,
                   L=numbers["L"], k=numbers["k"], f=f, left=data["left"], right=data["right"],
                   tol=tol, exact=None)
    problem["basis"] = lambda n: pdecheck.boundary_basis(problem, n)
    problem["lam"] = lambda n: pdecheck.frequency_of_mode(problem, n)
    return problem


def ask_once(a, prompt):
    import httpx
    # CHANGE A5: only sampling parameters vary; HTTP endpoint/TLS/auth behavior is unchanged.
    body = dict(model=a.model, messages=[{"role": "user", "content": prompt}],
                max_tokens=a.max_tokens, temperature=getattr(a, "temperature", 0.6), top_p=getattr(a, "top_p", 0.95),
                chat_template_kwargs={"enable_thinking": a.think})
    r = httpx.post(f"{a.base.rstrip('/')}/chat/completions", json=body,
                   timeout=900, verify=False)
    if r.status_code != 200:
        raise SystemExit(f"the server returned HTTP {r.status_code}:\n{r.text[:800]}\n\n"
                         f"request was: {json.dumps(body)[:400]}")
    return r.json()["choices"][0]["message"].get("content") or ""


# CHANGE SHARED A11: MODE certifies a model-supplied wave, coefficient setup and rate.
# COMPUTE keeps its original meaning; arithmetic-only results are not verified PDE steps.
SHARED_INSTRUCTIONS = (
    "\n\nVerified shared calculator: send at most 8 requests in one exchange. "
    "For a complete mode, prefer this Python-syntax line:\n"
    "  MODE: <wave in x> ; <normalized projection integral> ; <decay-rate expression>\n"
    "Fill in f, L and k from this problem. The projection must include normalization. "
    "The checker verifies both boundaries, the projection setup and rate=k*omega**2 "
    "before sharing the computed coefficient. Legacy COMPUTE lines are also accepted. "
    "Shared modes are checked individually; they do not prove that enough modes were kept. "
    "Use verified values at the next exchange, then give the complete final answer."
)


def intermediate_requests(reply):
    requests = []
    for match in re.finditer(r"^\s*(COMPUTE|MODE):\s*(.+?)\s*$", reply or "", re.M):
        kind, text = match.groups()
        if kind == "COMPUTE":
            requests.append(dict(kind="compute", expression=text.strip()))
        else:
            fields = [field.strip() for field in text.split(";")]
            requests.append(dict(kind="mode", wave=fields[0], expression=fields[1] if len(fields) == 3 else "",
                                 rate=fields[2] if len(fields) == 3 else "", original=text))
        if len(requests) == 8:
            break
    return requests


def intermediate_block(requests, results, modes=()):
    lines = []
    for request, result in zip(requests, results):
        label = "MODE: " + request.get("original", "") if request["kind"] == "mode" else "COMPUTE: " + request["expression"]
        lines.extend((label, "  = " + result["text"]))
        if result.get("verified"):
            lines.append("  Verified scope: " + result["scope"])
        elif result.get("scope") != "rejected":
            lines.append("  Arithmetic only; this is not a verified coefficient or decay-rate step.")
    if modes:
        lines.append("Verified shared modes for this exact problem (partial, not a complete solution):")
        lines.extend(f"  wave={mode['wave']}; coefficient={mode['coefficient']}; rate={mode['rate']}" for mode in modes)
    return "\n".join(lines)


def _rejected_intermediate(error):
    return dict(text=f"Intermediate rejected: {error}", verified=False,
                shareable=False, scope="rejected", mode=None)


# CHANGE FAST A10: no per-request/tool/sample timers.
def one_attempt(a, problem, prompt, rnd):
    """One sample, including its tool exchanges.

    The model may ask for exact values with COMPUTE: lines; we evaluate them and hand the
    numbers back, then ask for the final answer. The model chooses which integral to set up,
    which is the part worth measuring, and sympy does the arithmetic it cannot do reliably.
    """
    if a.offline:
        # CHANGE A6: independent deterministic sample seeds; always logged as simulation.
        return fake_model(problem, 1, rnd, sample_seed=getattr(a, "_offline_sample_seed", rnd))[0], 0
    # CHANGE A4: zero exchange steps disables both tool instructions and tool requests.
    effective_tools = not a.no_tools and a.tool_steps > 0
    convo = f"{prompt}\n\n{tool_calc.INSTRUCTIONS}" if effective_tools else prompt
    shared = effective_tools and not getattr(a, "no_shared_calculations", False)
    if shared:
        convo += SHARED_INSTRUCTIONS
    used = 0
    for step in range(a.tool_steps + 1):
        reply = ask_once(a, convo)
        asks = (intermediate_requests(reply) if shared else tool_calc.requests_in(reply)) if effective_tools else []
        if not asks or step == a.tool_steps:
            return reply, used
        used += len(asks)
        if shared and getattr(a, "_shared_connection", None) is not None:
            a._shared_connection.send(dict(kind="tools", requests=asks))
            response = a._shared_connection.recv()
            if response.get("kind") != "tool_reply":
                raise RuntimeError("Shared calculator returned an invalid message")
            block = response["block"]
        elif shared:
            block = intermediate_block(asks, [pdecheck.check_intermediate(problem, request) for request in asks])
        else:
            block = tool_calc.answer_block(asks)
        convo = (f"{convo}\n\n{reply}\n\n"
                 f"{block}\n\n"
                 f"Use those values and give the final answer now, as one line "
                 f"u(x, t) = <expression> with every number filled in.")
    return reply, used


# CHANGE SHARED A12: isolated calculator processes can be stopped with the sample workers.
def _calculator_worker(connection, inherited_connections, problem):
    for inherited in inherited_connections:
        inherited.close()
    try:
        while True:
            job = connection.recv()
            try:
                if job.get("fallback") is not None:
                    result = dict(text=tool_calc.compute(job["fallback"]), verified=False,
                                  shareable=False, scope="uncertified_arithmetic", mode=None)
                else:
                    result = pdecheck.evaluate_intermediate(problem, job["prepared"])
            except Exception as error:
                result = _rejected_intermediate(f"{type(error).__name__}: {error}")
            connection.send(dict(job_id=job["job_id"], result=result))
    except (EOFError, BrokenPipeError, OSError):
        pass
    finally:
        connection.close()


# CHANGE SHARED A13: only the parent owns cache state; forked dictionaries are not shared.
# Single-flight requests wait for one certified calculation. Invalid MODE setup is rejected
# before integration. Unrecognised legacy calculations are answered privately, never cached.
class SharedCalculator:
    def __init__(self, ctx, problem, a, pending, outcomes):
        self.ctx, self.problem, self.pending, self.outcomes = ctx, problem, pending, outcomes
        self.limit = min(getattr(a, "calculator_workers", 2), a.samples)
        signature = tuple(sp.srepr(sp.sympify(problem[key])) for key in ("L", "k", "f")) + (problem["left"], problem["right"])
        state = getattr(a, "_shared_cache_state", None)
        if not state or state["signature"] != signature:
            state = dict(signature=signature, cache=OrderedDict(), aliases=OrderedDict(), modes=OrderedDict())
            a._shared_cache_state = state
        self.state = state
        self.workers, self.jobs, self.waiting, self.batches = [], {}, deque(), {}
        self.stats = dict(requests=0, calculations=0, cache_hits=0, in_flight_reuse=0, rejected=0)

    def _count(self, index, name):
        self.stats[name] += 1
        stats = self.outcomes[index].setdefault("intermediate_stats", {})
        stats[name] = stats.get(name, 0) + 1

    def submit(self, receiver, requests):
        index = self.pending[receiver]
        batch_id = uuid.uuid4().hex
        batch = dict(receiver=receiver, requests=requests, results=[None]*len(requests), remaining=len(requests))
        self.batches[batch_id] = batch
        for offset, request in enumerate(requests):
            self._count(index, "requests")
            raw_key = json.dumps(request, sort_keys=True)
            alias = self.state["aliases"].get(raw_key)
            if alias in self.state["cache"]:
                self._count(index, "cache_hits")
                self._deliver(batch_id, offset, self.state["cache"][alias])
                continue
            if alias in self.jobs:
                self._count(index, "in_flight_reuse")
                self.jobs[alias]["waiters"].append((batch_id, offset))
                continue
            try:
                prepared = pdecheck.prepare_intermediate(self.problem, request)
            except Exception as error:
                if request["kind"] == "mode":
                    self._count(index, "rejected")
                    self._deliver(batch_id, offset, _rejected_intermediate(error))
                    continue
                prepared = None  # Keep the original calculator syntax privately usable.
            shareable = bool(prepared and prepared["shareable"])
            key = prepared["canonical"] if shareable else uuid.uuid4().hex
            if shareable:
                self.state["aliases"][raw_key] = key
                while len(self.state["aliases"]) > 1024:
                    self.state["aliases"].popitem(last=False)
            if shareable and key in self.state["cache"]:
                self._count(index, "cache_hits")
                self._deliver(batch_id, offset, self.state["cache"][key])
            elif shareable and key in self.jobs:
                self._count(index, "in_flight_reuse")
                self.jobs[key]["waiters"].append((batch_id, offset))
            else:
                self.jobs[key] = dict(job_id=key, prepared=prepared,
                                      fallback=request["expression"] if prepared is None else None,
                                      owner=index, waiters=[(batch_id, offset)])
                self.waiting.append(key)
        self.dispatch()

    def _deliver(self, batch_id, offset, result):
        batch = self.batches.get(batch_id)
        if batch is None:
            return
        batch["results"][offset] = result
        batch["remaining"] -= 1
        if batch["remaining"]:
            return
        receiver = batch["receiver"]
        if receiver in self.pending:
            block = intermediate_block(batch["requests"], batch["results"], list(self.state["modes"].values())[-8:])
            try:
                receiver.send(dict(kind="tool_reply", block=block))
            except (BrokenPipeError, EOFError, OSError):
                pass  # The sample's EOF is handled by the normal result collector.
        del self.batches[batch_id]

    def dispatch(self):
        if self.waiting and len(self.workers) < self.limit:
            for _ in range(self.limit-len(self.workers)):
                parent, child = self.ctx.Pipe(duplex=True)
                inherited = list(self.pending) + [w["connection"] for w in self.workers] + [parent]
                process = self.ctx.Process(target=_calculator_worker, args=(child, inherited, self.problem))
                try:
                    process.start()
                except BaseException:
                    parent.close(); child.close()
                    raise
                child.close()
                self.workers.append(dict(process=process, connection=parent, busy=None))
        for worker in self.workers:
            if worker["busy"] is None and self.waiting:
                key = self.waiting.popleft()
                job = self.jobs[key]
                worker["connection"].send({name: job[name] for name in ("job_id", "prepared", "fallback")})
                worker["busy"] = key
                self._count(job["owner"], "calculations")

    def connections(self):
        return [worker["connection"] for worker in self.workers if worker["busy"] is not None]

    def receive(self, connection):
        worker = next(w for w in self.workers if w["connection"] is connection)
        key = worker["busy"]
        try:
            message = connection.recv()
            if message.get("job_id") != key:
                raise ValueError("Calculator job identifier mismatch")
            result = message["result"]
        except (EOFError, OSError, ValueError) as error:
            result = _rejected_intermediate(f"Calculator worker failed: {error}")
            worker["process"].join(timeout=0.2)
            if worker["process"].is_alive():
                worker["process"].terminate(); worker["process"].join()
            worker["process"].close(); connection.close()
            self.workers.remove(worker)
        worker["busy"] = None
        job = self.jobs.pop(key)
        if result.get("shareable"):
            self.state["cache"][key] = result
            while len(self.state["cache"]) > 512:
                self.state["cache"].popitem(last=False)
        mode = result.get("mode")
        if result.get("verified") and mode:
            self.state["modes"][mode["wave"]] = mode
            while len(self.state["modes"]) > 8:
                self.state["modes"].popitem(last=False)
        for batch_id, offset in job["waiters"]:
            if result.get("scope") == "rejected" and batch_id in self.batches:
                receiver = self.batches[batch_id]["receiver"]
                if receiver in self.pending:
                    self._count(self.pending[receiver], "rejected")
            self._deliver(batch_id, offset, result)
        self.dispatch()

    def shutdown(self):
        for worker in self.workers:
            if worker["process"].is_alive():
                worker["process"].terminate()
        for worker in self.workers:
            process = worker["process"]
            process.join(timeout=0.2)
            if process.is_alive():
                process.kill(); process.join()
            process.close(); worker["connection"].close()
        self.workers.clear()


# CHANGE EARLY_STOP / HONESTY A8: workers own requests; parent owns all grading/logs.
def _sample_worker(sender, inherited_receivers, a, problem, prompt, rnd):
    for receiver in inherited_receivers:
        receiver.close()
    try:
        result = one_attempt(a, problem, prompt, rnd)
        sender.send(dict(kind="result", result=result))
    except BaseException as error:
        try:
            sender.send(dict(kind="error", error=f"{type(error).__name__}: {error}"))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        sender.close()


def ask_round(a, problem, prompt, rnd, sample_count=None):
    """Grade every received final answer; optionally cancel unfinished candidates.

    Default: stop on a full pass. --collect-all: finish and grade the entire round.
    Linux fork is required for problem lambdas. Server abort after disconnect is unconfirmed.
    """
    import multiprocessing as mp
    from multiprocessing.connection import wait
    if "fork" not in mp.get_all_start_methods():
        raise RuntimeError("Run this agent inside the Linux seat pod (fork is required).")
    ctx = mp.get_context("fork")
    count = min(getattr(a, "max_samples", None) or a.samples,
                a.samples if sample_count is None else sample_count)
    if count < 1:
        raise ValueError("ask_round requires at least one sample")
    strategy = getattr(a, "_round_strategy", "derive")
    collect_all = getattr(a, "collect_all", False)
    processes, pending, completed = [], {}, []
    a._early_stop = False
    a._outcomes = {}
    a._launched_count = 0
    interrupted = False
    shared = not a.offline and not a.no_tools and a.tool_steps > 0 and not getattr(a, "no_shared_calculations", False)
    broker = SharedCalculator(ctx, problem, a, pending, a._outcomes) if shared else None
    a._shared_stats = {}

    def receive_one(receiver, allow_tools=True):
        index = pending[receiver]
        outcome = a._outcomes[index]
        try:
            message = receiver.recv()
        except (EOFError, OSError):
            if outcome["status"] == "running":
                outcome.update(status="error", error="Worker exited without a final result.")
            pending.pop(receiver)
            receiver.close()
            return
        kind = message["kind"]
        # CHANGE SHARED A14: model requests stay parallel while the parent brokers tools.
        if kind == "tools" and broker is not None:
            if allow_tools:
                broker.submit(receiver, message["requests"])
            return
        pending.pop(receiver)
        receiver.close()
        if kind == "error":
            outcome.update(status="error", error=message["error"])
            print(f"  sample {index}: worker error (ungraded): {message['error']}", flush=True)
            return
        result = message["result"]
        outcome.update(answer=result[0], tool_calls=result[1])
        try:
            grade = pdecheck.check(problem, result[0])
        except Exception as error:
            outcome.update(status="error", error=f"Checker error: {type(error).__name__}: {error}")
            return
        completed.append((index, result, grade))
        outcome.update(status="graded", grade=grade)
        print(f"  completed sample {index}: reward={grade['reward']:.1f}", flush=True)
        if grade["reward"] == 1.0:
            if not collect_all:
                a._early_stop = True

    try:
        for index in range(count):
            child = copy.copy(a)
            child.temperature, child.top_p = sample_settings(a, strategy, index, rnd)
            seed_text = f"{problem['name']}:{problem['seed']}:{rnd}:{index}"
            child._offline_sample_seed = int(hashlib.sha256(seed_text.encode("utf-8")).hexdigest()[:16], 16)
            receiver, sender = ctx.Pipe(duplex=shared)
            child._shared_connection = sender if shared else None
            process = ctx.Process(target=_sample_worker,
                                  args=(sender, list(pending) + [receiver], child, problem,
                                        sample_prompt(prompt, strategy, index), rnd))
            try:
                process.start()
            except BaseException:
                receiver.close()
                sender.close()
                raise
            sender.close()
            processes.append(process)
            pending[receiver] = index
            a._launched_count += 1
            a._outcomes[index] = dict(status="running", answer=None)
        mode = "collect-all" if collect_all else "first-pass early stop"
        print(f"  launched {count} parallel samples; mode={mode}; checking each completion immediately", flush=True)
        while pending and not a._early_stop:
            connections = list(pending) + (broker.connections() if broker else [])
            for receiver in wait(connections):
                if receiver in pending:
                    receive_one(receiver)
                else:
                    broker.receive(receiver)
                if a._early_stop:
                    print("  full checker pass: stopping unfinished local workers; "
                          "server-side cancellation is unconfirmed", flush=True)
                    break
    except BaseException:
        interrupted = True
        raise
    finally:
        # Record any already delivered messages before cancelling, including final answers.
        if not interrupted:
            for receiver in list(pending):
                while receiver in pending and receiver.poll():
                    receive_one(receiver, allow_tools=False)
        for index, process in enumerate(processes):
            if process.is_alive():
                outcome = a._outcomes[index]
                if outcome["status"] == "running":
                    outcome.update(status="cancelled", cancel_reason="interrupted" if interrupted else "another_sample_solved")
                process.terminate()
        for process in processes:
            process.join(timeout=0.2)
            if process.is_alive():
                process.kill()
                process.join()
            process.close()
        if broker is not None:
            broker.shutdown()
            a._shared_stats = dict(broker.stats)
        # Preserve final results already delivered when local workers were stopped.
        if not interrupted:
            for receiver in list(pending):
                while receiver in pending and receiver.poll():
                    receive_one(receiver, allow_tools=False)
        for receiver, index in pending.items():
            outcome = a._outcomes[index]
            if outcome["status"] == "running":
                outcome.update(status="cancelled", cancel_reason="interrupted" if interrupted else "another_sample_solved")
            receiver.close()
        # Explicit worker errors retain their status, even when another sample solves.
        a._cancelled_indices = sorted(i for i, o in a._outcomes.items() if o["status"] == "cancelled")
        a._error_indices = sorted(i for i, o in a._outcomes.items() if o["status"] == "error")
        a._uncollected_indices = sorted(a._cancelled_indices + a._error_indices)
    completed.sort(key=lambda item: item[0])
    a._sample_indices = [item[0] for item in completed]
    a._graded = [item[2] for item in completed]
    return [item[1] for item in completed]


# CHANGE HONESTY A8: one terminal record for each cancelled/error sample; no invented score.
def write_ungraded_records(log, problem, a, prompt, rnd, round_args, strategy,
                           remaining, stop_reason, include_graded=False):
    for index, outcome in sorted(round_args._outcomes.items()):
        is_graded = outcome["status"] == "graded"
        if is_graded and not include_graded:
            continue
        temperature, top_p = sample_settings(a, strategy, index, rnd)
        grade = outcome.get("grade", {})
        log.write(json.dumps(dict(
            run_id=a._run_id, problem=problem["name"], seed=problem["seed"], round=rnd, sample_index=index,
            status=outcome["status"], answer=outcome.get("answer"), reward=grade.get("reward"), parts=grade.get("parts"),
            start_error=grade.get("start_error"), prompt=sample_prompt(prompt, strategy, index), round_prompt=prompt,
            strategy=strategy, sampling=dict(temperature=temperature, top_p=top_p),
            launched_samples=round_args._launched_count, early_stop=round_args._early_stop,
            collect_all=bool(getattr(a, "collect_all", False)), stop_reason=stop_reason,
            remaining_attempt_budget=remaining, cancel_reason=outcome.get("cancel_reason"),
            error=outcome.get("error"), tool_calls=outcome.get("tool_calls"),
            intermediate_stats=outcome.get("intermediate_stats", {}),
            shared_calculator=bool(getattr(round_args, "_shared_stats", {})),
            local_usage_complete=is_graded,
            server_usage_complete=False, server_cancellation_confirmed=None,
            source="offline_simulation" if a.offline else "model_response", offline=bool(a.offline)
        )) + "\n")
    log.flush()


def fake_model(problem, n, rnd, sample_seed=None):
    """Offline stand-in, so the loop can be exercised with no model. It improves each round.
    Never report a number that came from here."""
    # CHANGE A6: reference answers are used only by this explicit offline stand-in.
    rng = random.Random(rnd if sample_seed is None else sample_seed)
    mod = LEVELS[problem["level"]]
    if problem["exact"] is not None:
        good = sp.sstr(problem["exact"])
        wrong = [sp.sstr(problem["f"]), sp.sstr(problem["exact"]).replace("sin", "cos", 1)]
    else:
        good = sp.sstr(mod.series_answer(problem, 4))
        wrong = [sp.sstr(mod.series_answer(problem, 1)), sp.sstr(problem["f"])]
    p_good = min(0.9, 0.15 + 0.25 * rnd)
    return [f"u(x, t) = {good if rng.random() < p_good else rng.choice(wrong)}"
            for _ in range(n)]


def solve(problem, a, log):
    # CHANGE HONESTY A8: distinguish repetitions appended to the same JSONL file.
    if not getattr(a, "_run_id", None):
        a._run_id = uuid.uuid4().hex
    # CHANGE A7: custom problems cannot use the controller's canned offline formulas.
    if a.offline and problem.get("level") == "custom":
        raise ValueError("Custom --problem-file input cannot use the offline reference generator.")
    # CHANGE SHARED A15: a fresh problem gets a fresh cache; rounds of that problem reuse it.
    a._shared_cache_state = None
    # CHANGE A4: generic phase guidance never computes or reads a correct solution.
    base_prompt = pdecheck.prompt_of(problem) + calculator_plan(problem, not a.no_tools and a.tool_steps > 0)
    print(f"\n=========== {problem['name']} ===========")
    print(base_prompt)
    # CHANGE A1/A3/A5/A6: bounded state and a full historical-best answer/grade/source.
    best_ever = None
    history, errors, seen = [], [], OrderedDict()
    history_limit = max(1, min(32, getattr(a, "history_limit", 8)))
    error_limit = max(1, min(16, getattr(a, "error_history_limit", 6)))
    remaining = getattr(a, "attempt_budget", None)
    if remaining is None:
        remaining = a.rounds * a.samples
    cap = getattr(a, "max_samples", None) or a.samples
    minimum = max(1, getattr(a, "min_samples", 1))
    stale, recovery_stale, recovery_used = 0, 0, False
    strategy, rounds_run, stop_reason = "derive", 0, "round_budget_exhausted"
    for rnd in range(a.rounds):
        if remaining < minimum:
            stop_reason = "attempt_budget_exhausted"
            break
        # CHANGE A5: adaptive allocation is optional and stays within the original total budget.
        sample_count = a.samples
        if getattr(a, "adaptive_samples", False):
            sample_count = (max(minimum, min(2, cap)) if strategy.startswith("target_repair")
                            else cap if strategy == "diversify" else a.samples)
        sample_count = min(cap, sample_count, remaining)
        prompt = base_prompt if best_ever is None else repair_prompt(base_prompt, best_ever, history, errors, strategy)
        round_args = copy.copy(a)
        round_args._round_strategy = strategy
        t0 = time.perf_counter()
        try:
            attempts = ask_round(round_args, problem, prompt, rnd, sample_count=sample_count)
        except BaseException:
            # CHANGE HONESTY A8: preserve records for locally interrupted requests too.
            write_ungraded_records(log, problem, a, prompt, rnd, round_args, strategy,
                                   remaining-round_args._launched_count, "interrupted_or_controller_error",
                                   include_graded=True)
            raise
        remaining -= round_args._launched_count
        a._shared_cache_state = getattr(round_args, "_shared_cache_state", None)
        rounds_run = rnd + 1
        answers = [ans for ans, _ in attempts]
        tool_calls = sum(used for _, used in attempts)
        # CHANGE EARLY_STOP: reuse grades; do not evaluate candidates a second time.
        graded = round_args._graded
        sample_indices = round_args._sample_indices
        if not graded:
            write_ungraded_records(log, problem, a, prompt, rnd, round_args, strategy,
                                   remaining, "no_graded_candidates")
            raise RuntimeError("No candidate could be graded; inspect error records in the attempt log.")
        # CHANGE A1: priority breaks reward ties; exact priority ties retain the earlier best.
        previous_priority = grade_priority(best_ever["grade"]) if best_ever else None
        round_best = max(range(len(graded)), key=lambda index: grade_priority(graded[index]))
        round_priority = grade_priority(graded[round_best])
        improved = previous_priority is None or round_priority > previous_priority
        if improved:
            best_ever = dict(answer=answers[round_best], grade=copy.deepcopy(graded[round_best]),
                             source=dict(round=rnd, sample_index=sample_indices[round_best], strategy=strategy,
                                         offline=bool(a.offline), origin="offline_simulation" if a.offline else "model_response"))
        duplicates = []
        for ans, grade in zip(answers, graded):
            key = candidate_key(grade, ans)
            duplicates.append(remember_attempt(history, seen, errors, key, ans, grade, rnd, history_limit, error_limit))
        stale = 0 if improved else stale + 1
        if recovery_used:
            recovery_stale = 0 if improved else recovery_stale + 1
        near = best_ever["grade"]["reward"] >= getattr(a, "near_success_reward", 0.8)
        next_strategy = "target_repair" if near else "derive"
        switch_now = False
        # CHANGE A6: one recovery switch; subsequent stagnation can end the bounded loop.
        duplicate_fraction = sum(duplicates) / len(duplicates)
        if (not recovery_used and rnd + 1 < a.rounds
                and (stale >= getattr(a, "stagnation_rounds", 2)
                     or (rnd > 0 and not improved and duplicate_fraction >= getattr(a, "duplicate_threshold", 0.75)))):
            recovery_used, recovery_stale, switch_now = True, 0, True
        if recovery_used:
            next_strategy = "target_repair_diverse" if near else "diversify"
        terminal_reason = None
        if best_ever["grade"]["reward"] == 1.0:
            terminal_reason = "solved"
        elif rnd + 1 >= a.rounds:
            terminal_reason = "round_budget_exhausted"
        elif remaining < minimum:
            terminal_reason = "attempt_budget_exhausted"
        elif (recovery_used and not switch_now and getattr(a, "stagnation_patience", 2) > 0
              and recovery_stale >= getattr(a, "stagnation_patience", 2)):
            terminal_reason = "stagnation_after_strategy_switch"
        # CHANGE A1/A3/A6: retain legacy attempt fields plus actual sample prompt and stop provenance.
        for index, ((ans, used), grade) in enumerate(zip(attempts, graded)):
            passed, failed, actions = conditions_of(grade)
            temperature, top_p = sample_settings(a, strategy, sample_indices[index], rnd)
            log.write(json.dumps(dict(run_id=a._run_id, problem=problem["name"], seed=problem["seed"],
                                      round=rnd, prompt=sample_prompt(prompt, strategy, sample_indices[index]), answer=ans,
                                      # CHANGE FAST A10: keep final answer/score and cancellation audit only.
                                      status="graded", collect_all=bool(getattr(a, "collect_all", False)),
                                      local_usage_complete=True, server_usage_complete=False,
                                      tool_calls_observed_completed=used,
                                      intermediate_stats=round_args._outcomes[sample_indices[index]].get("intermediate_stats", {}),
                                      shared_calculator=bool(round_args._shared_stats),
                                      tool_calls=used, reward=grade["reward"], parts=grade["parts"],
                                      start_error=grade["start_error"], passed_conditions=passed,
                                      failed_conditions=failed, repair_actions=actions,
                                      diagnostics=grade.get("diagnostics", {}), round_prompt=prompt,
                                      sample_index=sample_indices[index], sample_count=len(attempts), strategy=strategy,
                                      launched_samples=round_args._launched_count,
                                      early_stop=round_args._early_stop,
                                      uncollected_sample_indices=round_args._uncollected_indices,
                                      server_cancellation_confirmed=None,
                                      next_strategy=None if terminal_reason else next_strategy,
                                      sampling=dict(temperature=temperature, top_p=top_p),
                                      duplicate_candidate=duplicates[index], canonical_key=candidate_key(grade, ans),
                                      history_best=copy.deepcopy(best_ever), history=list(history),
                                      mistake_history=[{key: value for key, value in item.items() if key != "signature"} for item in errors],
                                      stagnation_rounds=stale, recovery_switch_used=recovery_used,
                                      recovery_switch_for_next_round=switch_now and terminal_reason is None,
                                      remaining_attempt_budget=remaining, stop_reason=terminal_reason or "continue",
                                      source="offline_simulation" if a.offline else "model_response", offline=bool(a.offline))) + "\n")
        write_ungraded_records(log, problem, a, prompt, rnd, round_args, strategy,
                               remaining, terminal_reason or "continue")
        log.flush()
        rewards = [g["reward"] for g in graded]
        print(f"\nround {rnd}: rewards {rewards}  graded mean {sum(rewards) / len(rewards):.2f}  "
              f"history best {best_ever['grade']['reward']:.1f}  {tool_calls} tool call(s)  "
              f"({time.perf_counter() - t0:.1f}s)")
        if round_args._shared_stats:
            stats = round_args._shared_stats
            print(f"  shared calculator: requests={stats['requests']} calculations={stats['calculations']} "
                  f"cache_hits={stats['cache_hits']} in_flight_reuse={stats['in_flight_reuse']} "
                  f"rejected={stats['rejected']}; individually verified modes do not replace final grading.")
        print(f"  samples: launched={round_args._launched_count} graded={len(attempts)} "
              f"cancelled={round_args._cancelled_indices} errors={round_args._error_indices}; "
              "tool count covers graded samples only; server token cost is unmeasured.")
        print(f"  historical anchor: {best_ever['grade'].get('expr')}")
        print(f"  checker: {best_ever['grade'].get('feedback', '')}")
        if terminal_reason:
            stop_reason = terminal_reason
            break
        strategy = next_strategy
    reward = best_ever["grade"]["reward"] if best_ever else 0.0
    print(f"stop_reason={stop_reason}; rounds={rounds_run}; historical best reward={reward:.3f}")
    if stop_reason == "solved":
        print(f"SOLVED: u(x, t) = {best_ever['grade']['expr']}")
    # CHANGE A1/A6: keep the original solve return interface and report actual rounds run.
    return reward, rounds_run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=0, choices=sorted(LEVELS))
    ap.add_argument("--sub", type=int)
    ap.add_argument("--all", action="store_true", help="every sub-problem in this level")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=4)
    # gpt-oss reasons before it answers and needs ~6000 on level 1.3.
    ap.add_argument("--max-tokens", type=int, default=int(os.environ.get("HEATROD_MAX_TOKENS", 1200)))
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--tool-steps", type=int, default=1,
                    help="rounds of COMPUTE: exchanges allowed per attempt")
    ap.add_argument("--no-tools", action="store_true",
                    help="withhold the calculator, to measure what it buys")
    ap.add_argument("--model", default=os.environ.get("HEATROD_MODEL", "Qwen/Qwen3-8B"))
    ap.add_argument("--base", default=os.environ.get("HEATROD_BASE_URL"))
    ap.add_argument("--log", default="attempts.jsonl")
    ap.add_argument("--offline", action="store_true")
    # CHANGE HONESTY A9: complete-round evaluation keeps every candidate and score.
    ap.add_argument("--collect-all", action="store_true",
                    help="Finish and grade all launched samples in each round, even after a full pass")
    # CHANGE SHARED A16: baseline switch and bounded calculator concurrency for comparisons.
    ap.add_argument("--no-shared-calculations", action="store_true",
                    help="Use the previous private calculator and original prompts, without MODE requests")
    ap.add_argument("--calculator-workers", type=int, default=2,
                    help="Concurrent isolated calculator workers for shared mode (1..8; default 2)")
    # CHANGE A3/A5/A6: explicit bounded memory, sampling allocation, and stop controls.
    ap.add_argument("--history-limit", type=int, default=8, help="Distinct recent attempts retained (1..32)")
    ap.add_argument("--error-history-limit", type=int, default=6, help="Distinct recent mistakes retained (1..16)")
    ap.add_argument("--stagnation-rounds", type=int, default=2, help="No-improvement rounds before the one recovery switch")
    ap.add_argument("--stagnation-patience", type=int, default=2, help="Stale recovery rounds before stopping; 0 disables early stop")
    ap.add_argument("--duplicate-threshold", type=float, default=0.75, help="Duplicate fraction triggering recovery during stagnation")
    ap.add_argument("--near-success-reward", type=float, default=0.8, help="Reward at which repair focuses on failed conditions")
    ap.add_argument("--adaptive-samples", action="store_true", help="Allocate samples by strategy within an explicit total budget")
    ap.add_argument("--min-samples", type=int, default=1)
    ap.add_argument("--max-samples", type=int, help="Per-round cap; defaults to --samples (4)")
    ap.add_argument("--attempt-budget", type=int, help="Total candidate cap; defaults to --rounds times --samples")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--target-temperature", type=float, default=0.4)
    ap.add_argument("--diversify-temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    # CHANGE A7: four-BC JSON input is separate from the original level/sub selection.
    ap.add_argument("--problem-file", type=Path, help="JSON L,k,f,left,right; optional tol,seed,name; no exact answer")
    a = ap.parse_args()
    a._run_id = uuid.uuid4().hex

    # CHANGE A7: checker sampling requires a nonnegative seed for built-in problems too.
    if a.seed < 0:
        ap.error("seed must be a nonnegative integer")
    # CHANGE A5/A6: requests remain bounded even when adaptive sampling is enabled.
    a.max_samples = a.samples if a.max_samples is None else a.max_samples
    a.attempt_budget = a.rounds * a.samples if a.attempt_budget is None else a.attempt_budget
    if a.rounds < 1 or a.samples < 1 or a.tool_steps < 0 or a.max_tokens < 1:
        ap.error("rounds, samples and max-tokens must be positive; tool-steps cannot be negative")
    if not 1 <= a.calculator_workers <= 8:
        ap.error("calculator-workers must be 1..8")
    if not 1 <= a.min_samples <= a.samples <= a.max_samples or a.attempt_budget < a.min_samples:
        ap.error("require 1 <= min-samples <= samples <= max-samples and attempt-budget >= min-samples")
    if not 1 <= a.history_limit <= 32 or not 1 <= a.error_history_limit <= 16:
        ap.error("history-limit must be 1..32 and error-history-limit 1..16")
    if a.stagnation_rounds < 1 or a.stagnation_patience < 0:
        ap.error("stagnation-rounds must be positive; stagnation-patience cannot be negative")
    if not 0 <= a.duplicate_threshold <= 1 or not 0 <= a.near_success_reward <= 1:
        ap.error("duplicate-threshold and near-success-reward must be between 0 and 1")
    if not 0 < a.top_p <= 1 or any(not math.isfinite(value) or not 0 <= value <= 2 for value in
                                 (a.temperature, a.target_temperature, a.diversify_temperature)):
        ap.error("top-p must be in (0,1]; temperatures must be finite and in [0,2]")
    if a.problem_file and (a.all or a.sub is not None):
        ap.error("--problem-file cannot be combined with --all or --sub")
    if a.problem_file and a.offline:
        ap.error("--problem-file cannot use --offline; no custom reference answer is generated")

    if not a.base and not a.offline:
        sys.exit("set HEATROD_BASE_URL to your vLLM endpoint, or pass --offline")
    if a.offline:
        print("*** OFFLINE: fake generator, numbers are meaningless ***")

    # CHANGE A7: instantiate only the supplied equation/data, never a hidden correct answer.
    if a.problem_file:
        try:
            problems = [problem_from_file(a.problem_file, a.seed)]
        except (ValueError, TypeError, AttributeError, SyntaxError, OverflowError, OSError) as error:
            ap.error(f"invalid problem-file: {error}")
    else:
        mod = LEVELS[a.level]
        subs = mod.SUBS if (a.all or not a.sub) else (a.sub,)
        problems = [mod.make(sub, a.seed) for sub in subs]
    results = []
    with open(a.log, "a", encoding="utf-8") as log:
        for problem in problems:
            reward, rounds = solve(problem, a, log)
            results.append((problem["name"], reward, rounds))

    summary_label = "custom problem" if a.problem_file else f"level {a.level}"
    print(f"\n=========== summary: {summary_label} ===========")
    for name, reward, rounds in results:
        print(f"  {name:<12} reward {reward:.1f} after {rounds} round(s)"
              + ("  SOLVED" if reward == 1.0 else ""))
    solved = sum(1 for _, r, _ in results if r == 1.0)
    print(f"  solved {solved}/{len(results)}")
    print(f"\nattempts logged to {a.log}")


if __name__ == "__main__":
    main()
