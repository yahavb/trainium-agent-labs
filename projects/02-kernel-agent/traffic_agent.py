#!/usr/bin/env python3
"""
traffic_agent.py — the optimizer loop for the minimum-traffic experiment.

A model rewrites a correct matmul kernel so it moves fewer HBM bytes; the checker measures
against the byte floor; test-time learning (bandit, memory, population) steers which strategy
is prompted and which kernel is improved next. The model authors every candidate. The learning
layer adds no model calls and never updates model weights.

    python traffic_agent.py --seed-kernel reference_level4.py --rounds 6 --samples 2 \
        --repeat 1 --context 8192 --max-tokens 3000 --cases traffic_cases.json \
        --population-size 3 --memory-k 6 --learning on --output runs/traffic/pilot

Measuring happens in the simulator on the CPU, where an optimizer belongs (seconds per attempt,
hundreds of attempts possible). Device validation of the winner is a separate step and is never
claimed here. Every attempt is appended to rounds.jsonl; raw replies and candidate files are kept.

Vocabulary used below:
    valid      the candidate is correct: rules clean, numerics, inputs untouched, no hazard
    accepted   valid AND the traffic gate is met on every shape
    at_floor   accepted with every shape exactly at the byte floor (the stopping rule)
"""

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import random
import re
import time

import nkibench
import test_time_learning as ttl
import traffic_eval
from agent import API_CARD, extract_code
from agent import enrich as _agent_enrich

MODEL = os.environ.get("KERNEL_AGENT_MODEL", "Qwen/Qwen3-8B")

# Integer traffic thresholds: bytes/floor <= num/den. Same values as nkibench's max_waste,
# compared with integer arithmetic so no rounding can hide a miss.
LEVEL_BARS = {5: (8, 5), 6: (5, 4), 7: (21, 20)}


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def compact(text, n=110):
    return " ".join(str(text).split())[:n]


def first_failure(ev):
    return next((c["failure"] for c in ev["per_case"] if c.get("failure")),
                ev.get("feedback") or "")


def enrich_feedback(text):
    """Turn simulator exceptions into instructions.

    Measured in the pilot and the tuned pilot: raw exceptions make the small model abandon a
    working approach ("partition dimension 256 exceeds maximum 128" -> it tried PSUM straight
    to HBM) or fix the wrong axis ("same number of elements" -> it kept a wrong destination
    shape). The repo's recurring lesson: a verdict is not an instruction.

    The partition wall gets the task-specific recipe that the probe validated; everything else
    delegates to the general agent's enrich(), which already translates element-count
    mismatches, out-of-bounds indexing, broadcast mismatches, invented APIs, and memory
    placement errors into named changes.
    """
    m = re.search(r"dma_copy (\w+) partition dimension (\d+) exceeds maximum (\d+)", text)
    if m:
        which, got, mx = m.group(1), int(m.group(2)), int(m.group(3))
        return (text + f" A tile's FIRST dimension is the partition dimension and holds at most "
                f"{mx} rows, but this allocation asked for {got}. Do not allocate one cache "
                f"shaped [K, ...] when K > {mx}. Keep all K rows resident by stacking k-chunks "
                f"along the free (second) dimension: one cache shaped [128, k_tiles * M], loaded "
                f"chunk by chunk with dma_copy(dst=cache[:, kk*M:(kk+1)*M], "
                f"src=lhsT[kk*128:(kk+1)*128, :]), and sliced for the matmul as "
                f"cache[:, kk*M + m0*128 : kk*M + (m0+1)*128].")
    out = _agent_enrich(text)
    if "requires HBM or SBUF tensors" in text and "psum" in text:
        out += (" PSUM cannot be copied to HBM directly. Copy PSUM to SBUF with "
                "nisa.tensor_copy, then SBUF to HBM with nisa.dma_copy.")
    if "requires src and dst to have the same number of elements" in text:
        out += (" Destination and source slices must have the SAME shape. For a cache stacked "
                "along the free dimension: cache = nl.ndarray((128, k_tiles * M), ...); "
                "nisa.dma_copy(dst=cache[:, kk*M:(kk+1)*M], src=lhsT[kk*128:(kk+1)*128, :]); "
                "then slice operands as cache[:, kk*M + m0*128 : kk*M + (m0+1)*128].")
    return out


# ---------------------------------------------------------------- token accounting

class TokenCounter:
    """Exact chat-template token counts from the server's /tokenize endpoint.

    `/tokenize` with `messages` + `add_generation_prompt` counts the rendered chat prompt
    exactly, including template tokens. If the endpoint cannot answer, it falls back to a
    chars/4 estimate and marks itself inexact so no number is presented as exact when it is not.
    """

    def __init__(self, base, model):
        root = base.rstrip("/")
        if root.endswith("/v1"):
            root = root[:-3]
        self.tokenize_url = root + "/tokenize"
        self.model = model
        self.exact = None
        self._overhead = None

    def count(self, prompt):
        import httpx
        try:
            r = httpx.post(self.tokenize_url,
                           json=dict(model=self.model,
                                     messages=[{"role": "user", "content": prompt}],
                                     add_generation_prompt=True),
                           timeout=30, verify=False)
            if r.status_code == 200:
                self.exact = True
                return int(r.json()["count"])
        except Exception:
            pass
        self.exact = False
        return max(1, len(prompt) // 4)

    def overhead(self):
        """Template tokens added around any single message; cached once.

        Each section is counted through the chat template, so every per-section count carries
        this overhead. Subtracting it keeps the per-section numbers summing to the prompt total.
        """
        if self._overhead is None:
            n = self.count("")
            self._overhead = n if self.exact else 0
        return self._overhead


# ---------------------------------------------------------------- the model

def ask(a, prompt, max_tokens):
    import httpx
    body = dict(model=a.model, messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens, temperature=0.6, top_p=0.95,
                chat_template_kwargs={"enable_thinking": False})
    r = httpx.post(f"{a.base.rstrip('/')}/chat/completions", json=body,
                   timeout=900, verify=False)
    if r.status_code != 200:
        raise SystemExit(f"the endpoint returned HTTP {r.status_code}:\n{r.text[:600]}")
    payload = r.json()
    ch = payload["choices"][0]
    content = ch.get("message", {}).get("content") or ""
    if ch.get("finish_reason") == "length":
        print(f"    (TRUNCATED: finish_reason=length after {len(content)} chars)")
    return content, (payload.get("usage") or {})


def ask_parallel(a, prompt, max_tokens, n):
    with cf.ThreadPoolExecutor(max_workers=max(1, n)) as ex:
        return [f.result() for f in [ex.submit(ask, a, prompt, max_tokens) for _ in range(n)]]


def parse_reply(text):
    strategy, conf = "", None
    m = re.search(r"STRATEGY:\s*(.+?)\s*$", text, re.M)
    if m:
        strategy = m.group(1)[:80]
    m = re.search(r"CONFIDENCE:\s*(\d{1,3})", text, re.I)
    if m:
        conf = max(0, min(100, int(m.group(1))))
    return strategy, conf


# ---------------------------------------------------------------- prompts

def strategy_instruction(arm):
    base = ("Rewrite the kernel so it moves fewer bytes from HBM. Correctness must not change. "
            "The goal is the byte floor: every input element read once and the output written "
            "once (1.00x).")
    guide = {
        "retain_both": ("Load BOTH inputs into on-chip SBUF buffers exactly once, then compute "
                        "every output tile from those resident tiles. Slice the resident buffers "
                        "as the matmul operands."),
        "retain_rhs": ("Keep the `rhs` operand resident on chip: load each rhs tile exactly "
                       "once, then sweep all lhsT blocks against it."),
        "retain_lhs": ("Keep the `lhsT` operand resident on chip: load each lhsT tile exactly "
                       "once, then sweep all rhs blocks against it."),
        "bounded_blocking": ("Block the M, N and K loops so tiles stay in SBUF and are reused "
                             "across iterations; never reload a tile that is still resident."),
    }[arm]
    return (base + "\n" + guide + "\n\nReply with ONE complete python code block (imports + the "
            "full kernel). After the code block, add exactly two lines:\n"
            "STRATEGY: <a few words for the change you made>\n"
            "CONFIDENCE: <0-100, percent chance this is correct on unseen shapes>")


def kernel_block(level, src):
    entry = nkibench.LEVELS[level]["entry"]
    return (f"Kernel to improve. Keep the entry point `{entry}` decorated with `@nki.jit`, keep "
            f"float32 inputs and float32 accumulators, and keep the computed result identical.\n"
            f"Capacity facts: the largest supplied input pair totals 1.5 MiB in float32 and the "
            f"on-chip SBUF is far larger, so keeping both inputs resident can fit. Tile limits: "
            f"partition dimension <= 128; a matmul stationary operand is [K<=128, M<=128]; a "
            f"moving operand is [K<=128, N<=512].\n\n"
            f"```python\n{src}\n```")


def report_block(level, ev, tag):
    bar = nkibench.LEVELS[level].get("max_waste")
    gate = f"{bar:.2f}x" if bar is not None else "n/a (this level has no traffic gate)"
    lines = [f"{tag} measured HBM traffic (lower is better; 1.00x is the floor; the level "
             f"{level} gate is {gate}):"]
    for c in ev["per_case"]:
        w = f"{c['waste']:.2f}x" if c.get("waste") is not None else "?"
        line = f"- {c['case']}: {c.get('bytes')} bytes vs floor {c.get('floor')} = {w}"
        if c.get("failure"):
            line += f"  [not accepted: {compact(c['failure'])}]"
        lines.append(line)
    return "\n".join(lines)


def repair_prompt(src, feedback, arm):
    return ("This NKI matmul kernel is not right yet.\n\n```python\n" + src + "\n```\n\n"
            "A checker reports:\n" + feedback + "\n\n"
            "Change exactly what the checker names and keep everything else identical.\n\n"
            + strategy_instruction(arm))


def demo_block(demo_path):
    """A worked example of the loop, injected at the start of every prompt.

    Two sources, both labelled in the prompt text itself: the team's calibration example
    (`demo.json` at the project root) shows what attempt -> checker report -> one named change
    -> verified result looks like; once the agent produces its own first verified improvement,
    that trace (written to the output directory) can replace it. The label always says which
    one it is, so no hand-written kernel is ever presented as the model's work.
    """
    if not demo_path:
        return ""
    try:
        d = json.load(open(demo_path))
    except Exception:
        return ""
    if not d.get("kernel"):
        return ""
    label = d.get("label") or ("A verified example from an earlier run of this same agent "
                               "(same model, same checker).")
    return (f"{label}\n"
            f"- change: {d.get('change', 'n/a')}\n"
            f"- measured: {d.get('measured', 'n/a')}\n\n"
            f"```python\n{d['kernel']}\n```")


def assemble_prompt(a, counter, api_card, demo_txt, kernel_txt, report_txt, failure_txt,
                    memory_best, memory_failures, instruction):
    """Join prompt sections, trimming (memory first, the demo last) to fit the budget.

    Never trims the kernel block. Raises if the kernel alone cannot fit.
    """
    while True:
        mem_parts = []
        if memory_best:
            mem_parts.append("Closest attempts so far in this run (best progress first):\n"
                             + "\n".join(f"- {l}" for l in memory_best))
        if memory_failures:
            mem_parts.append("Recent failures in this run (avoid repeating them):\n"
                             + "\n".join(f"- {l}" for l in memory_failures))
        parts = [("api_card", api_card), ("demo", demo_txt), ("kernel", kernel_txt),
                 ("report", report_txt), ("failure", failure_txt),
                 ("memory", "\n\n".join(mem_parts)), ("instruction", instruction)]
        prompt = "\n\n".join(text for _, text in parts if text)
        total = counter.count(prompt)
        if total + a.max_tokens + 64 <= a.context:
            ovh = counter.overhead()
            sections = {name: max(1, counter.count(text) - ovh) for name, text in parts if text}
            sections["_template_overhead"] = ovh
            return prompt, total, sections
        if memory_failures:
            memory_failures = memory_failures[1:]
            continue
        if memory_best:
            memory_best = memory_best[:-1]
            continue
        if api_card:
            api_card = ""
            continue
        if failure_txt:
            failure_txt = ""
            continue
        if demo_txt:
            demo_txt = ""
            continue
        raise SystemExit(f"the kernel block alone needs {total} tokens, which does not fit the "
                         f"{a.context}-token context with a {a.max_tokens}-token answer")


# ---------------------------------------------------------------- acceptance helpers

def candidate_valid(ev):
    """Correct on every shape (traffic gate excluded; above-bar candidates are the search space)."""
    if not (ev["rules_ok"] and ev["numerics_ok"] and ev["inputs_ok"] and ev["hazards_ok"]):
        return False
    if ev["passed"] == 0:
        return False
    for c in ev["per_case"]:
        ch = c.get("checks")
        if ch is None:
            return False
        if not (ch["inputs_ok"] and ch["numerics_ok"] and ch["hazard_ok"]):
            return False
        if c.get("bytes") is not None and c.get("floor") and c["bytes"] < c["floor"]:
            return False  # below the floor means incomplete accounting
    return True


def at_floor(ev):
    return ev["accepted"] and all(
        c.get("bytes") is not None and c.get("floor") and c["bytes"] == c["floor"]
        for c in ev["per_case"])


def level_status(ev):
    st = {}
    for lv, (num, den) in LEVEL_BARS.items():
        st[lv] = bool(candidate_valid(ev)) and all(
            c.get("bytes") is not None and c.get("floor")
            and c["bytes"] * den <= c["floor"] * num
            for c in ev["per_case"])
    return st


def improvement(parent_worst, ev):
    if not candidate_valid(ev) or ev["worst_waste"] is None or parent_worst is None:
        return 0.0
    return max(0.0, min(1.0, float(parent_worst) - float(ev["worst_waste"])))


def progress(ev):
    """Tiered progress for the bandit only; population and winner stay valid-only.

    rules clean 0.2, fraction of shapes that ran 0.3, fraction numerically correct 0.3, and
    traffic credit 0.2 on the correct shapes (0 at 2.00x, 1 at the 1.00x floor). Measured:
    with a valid-only reward the bandit's landscape was flat 0.00 for 80 attempts; this gives
    it a gradient from near misses without ever counting a wrong kernel as a win.
    """
    if not ev.get("rules_ok"):
        return 0.0
    total = max(1, int(ev.get("total") or 1))
    ran = correct = 0
    credit = []
    for c in ev.get("per_case", []):
        ch = c.get("checks")
        if ch is None:
            continue
        ran += 1
        if ch["inputs_ok"] and ch["numerics_ok"] and ch["hazard_ok"]:
            correct += 1
            w = c.get("waste")
            if w is not None:
                credit.append(max(0.0, min(1.0, 2.0 - w)))
    traffic = (sum(credit) / len(credit)) if credit else 0.0
    return round(0.2 + 0.3 * (ran / total) + 0.3 * (correct / total) + 0.2 * traffic, 4)


def bandit_reward(ev):
    """The bandit's reward: absolute tiered progress of the candidate.

    The first version rewarded progress OVER THE PARENT (the seed at 0.80). Measured in
    pilot_v2: every partial kernel sits just below the seed (0.55-0.775), so the bandit stayed
    flat at 0.00 and could not rank its arms. Absolute progress ranks arms by how close their
    candidates get -- retain_lhs 0.775 vs retain_both 0.55 vs retain_rhs 0.20 in that pilot --
    which is what the arm choice needs to learn from.
    """
    return float(progress(ev))


def lesson_for(arm, valid, ev, imp):
    p = progress(ev)
    if not ev["rules_ok"]:
        return f"{arm}: rule violation: {compact(ev['feedback'], 90)}"
    if not valid:
        ran = sum(1 for c in ev["per_case"] if c.get("checks") is not None)
        corr = sum(1 for c in ev["per_case"]
                   if c.get("checks") and c["checks"]["inputs_ok"]
                   and c["checks"]["numerics_ok"] and c["checks"]["hazard_ok"])
        return (f"{arm}: partial (progress {p:.2f}): ran {ran}/{ev['total']}, correct "
                f"{corr}/{ev['total']}; wall: {compact(ev['feedback'], 70)}")
    if ev["worst_waste"] is None:
        return f"{arm}: correct but traffic unmeasured"
    return (f"{arm}: worst-case {ev['worst_waste']:.2f}x (improvement {imp:+.2f}, "
            f"progress {p:.2f})")


# ---------------------------------------------------------------- evaluation plumbing

def write_and_eval(out_dir, tag, src, level, tol, seed):
    path = os.path.join(out_dir, "candidates", tag + ".py")
    with open(path, "w") as f:
        f.write(src)
    ev = traffic_eval.evaluate_file(path, level, tol, seed)
    return path, ev


def adapt_seed(seed_src, level):
    """Mechanical, disclosed adaptation: rename the shipped reference's entry point only."""
    entry = nkibench.LEVELS[level]["entry"]
    return re.sub(r"\bnki_matmul_tiled_\b", entry, seed_src)


# ---------------------------------------------------------------- the loop

def run_once(a, level, out_dir, rng, counter, opt_seeds):
    os.makedirs(out_dir, exist_ok=True)
    for sub in ("learning", "replies", "candidates"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
    log_path = os.path.join(out_dir, "rounds.jsonl")
    open(log_path, "w").close()   # a rerun into the same output must not mix runs
    logf = open(log_path, "a")

    learning = a.learning == "on"
    bandit = ttl.Bandit(exploration=a.exploration) if learning else None
    memory = ttl.Memory(cap=a.memory_cap)
    population = ttl.Population(size=a.population_size if learning else 1)

    seed_original = open(a.seed_kernel).read()
    seed_src = adapt_seed(seed_original, level)
    with open(os.path.join(out_dir, "seed_original.py"), "w") as f:
        f.write(seed_original)
    _, seed_ev = write_and_eval(out_dir, "seed", seed_src, level, a.tol, opt_seeds[0])
    with open(os.path.join(out_dir, "seed_eval.json"), "w") as f:
        json.dump(seed_ev, f, indent=2)
    if candidate_valid(seed_ev):
        population.insert(dict(hash=sha(seed_src), source=seed_src, strategy="seed",
                               worst_waste=seed_ev["worst_waste"], valid=True,
                               progress=progress(seed_ev), eval=seed_ev,
                               path=os.path.join(out_dir, "candidates", "seed.py")))
    seen = {sha(seed_src)}
    last_failure = ""
    pending = None
    rounds_used = 0
    tokens_in = tokens_out = 0
    hit_floor = False
    best_improvement = None   # (improvement, source, label, eval): first verified win -> demo.json
    run_id = os.path.basename(out_dir)

    for rnd in range(a.rounds):
        t0 = time.perf_counter()
        rounds_used = rnd + 1
        parent = None
        if pending and pending["remaining"] > 0:
            arm = pending["arm"]
            parent_worst = pending["worst_waste"]
            parent_hash = sha(pending["source"])
            prompt = repair_prompt(pending["source"], pending["feedback"], arm)
            pending["remaining"] -= 1
            prompt_tokens = counter.count(prompt)
            sections = {"repair": prompt_tokens}
            if prompt_tokens + a.max_tokens + 64 > a.context:
                print(f"round {rnd}: repair prompt too long ({prompt_tokens} tokens); dropping it")
                pending = None
                continue
        else:
            pending = None
            arm = bandit.select() if learning else rng.choice(ttl.ARMS)
            parent = population.pick_parent(rng, a.second_parent_prob)
            if parent is None:
                raise SystemExit("no parent kernel available")
            parent_worst = parent["worst_waste"]
            parent_hash = parent["hash"]
            rp = report_block(level, parent["eval"], "Parent")
            lines = memory.prompt_lines(a.memory_k) if learning else dict(best=[], failures=[])
            prompt, prompt_tokens, sections = assemble_prompt(
                a, counter, API_CARD, demo_block(a.demo), kernel_block(level, parent["source"]),
                rp, last_failure, lines["best"], lines["failures"], strategy_instruction(arm))

        budget = min(a.max_tokens, max(256, a.context - prompt_tokens - 64))
        try:
            replies = ask_parallel(a, prompt, budget, a.samples)
        except SystemExit:
            raise
        except Exception as e:
            print(f"round {rnd}: model call failed: {type(e).__name__}: {e}")
            continue

        rewards, round_valid, round_invalid = [], [], []
        for i, (reply, usage) in enumerate(replies):
            tokens_in += int(usage.get("prompt_tokens") or 0)
            tokens_out += int(usage.get("completion_tokens") or 0)
            raw_path = os.path.join(out_dir, "replies", f"r{rnd}_s{i}.txt")
            with open(raw_path, "w") as f:
                f.write(reply)
            src = extract_code(reply)
            label, conf = parse_reply(reply)
            record = dict(run_id=run_id, level=level, round=rnd, sample=i, arm=arm,
                          parent_hash=parent_hash, strategy=label, confidence=conf,
                          raw_reply=os.path.relpath(raw_path, out_dir),
                          prompt_tokens=prompt_tokens,
                          completion_tokens=int(usage.get("completion_tokens") or 0),
                          prompt_section_tokens=sections, retried=False,
                          elapsed_s=round(time.perf_counter() - t0, 1))

            # One nudge retry when the reply is empty or byte-identical to an earlier candidate.
            # Measured: 44% of the frozen-batch attempts were duplicates, so half the round
            # budget re-tested the same kernel. Changing the prompt is the only lever with a
            # deterministic server.
            if (not src.strip()) or (sha(src) in seen):
                nudge = ("\n\nNOTE: your previous reply was the same kernel as an earlier "
                         "attempt in this run. Keep the same goal, but change at least the tile "
                         "allocation sizes or the indexing. Reply with ONE complete python code "
                         "block, then the STRATEGY and CONFIDENCE lines."
                         if src.strip() else
                         "\n\nNOTE: your previous reply contained no python code block. Reply "
                         "with ONE complete python code block, then STRATEGY and CONFIDENCE.")
                try:
                    reply2, usage2 = ask(a, prompt + nudge, budget)
                    tokens_in += int(usage2.get("prompt_tokens") or 0)
                    tokens_out += int(usage2.get("completion_tokens") or 0)
                    raw2 = os.path.join(out_dir, "replies", f"r{rnd}_s{i}_retry.txt")
                    with open(raw2, "w") as f:
                        f.write(reply2)
                    src2 = extract_code(reply2)
                    if src2.strip() and sha(src2) not in seen:
                        reply, src = reply2, src2
                        label, conf = parse_reply(reply2)
                        record.update(raw_reply=os.path.relpath(raw2, out_dir),
                                      completion_tokens=int(usage2.get("completion_tokens") or 0),
                                      retried=True, strategy=label, confidence=conf)
                except Exception as e:
                    print(f"    retry failed: {type(e).__name__}: {e}")

            if not src.strip():
                record.update(decision="empty", failure_kind="empty")
                logf.write(json.dumps(record) + "\n")
                logf.flush()
                memory.append(dict(ok=False, improvement=0.0, progress=0.0,
                                   lesson=f"{arm}: empty reply"))
                continue
            code_hash = sha(src)
            if code_hash in seen:
                record.update(decision="duplicate", source_hash=code_hash,
                              failure_kind="duplicate")
                logf.write(json.dumps(record) + "\n")
                logf.flush()
                memory.append(dict(ok=False, improvement=0.0, progress=0.0,
                                   lesson=f"{arm}: repeated an earlier candidate (same source)"))
                continue
            seen.add(code_hash)
            code_path, ev = write_and_eval(out_dir, f"r{rnd}_s{i}", src, level, a.tol,
                                           opt_seeds[0])
            valid = candidate_valid(ev)
            imp = improvement(parent_worst, ev)
            prog = progress(ev)
            reward = bandit_reward(ev)
            inserted = False
            if valid:
                inserted = population.insert(dict(
                    hash=code_hash, source=src, strategy=arm,
                    worst_waste=ev["worst_waste"], valid=True, progress=prog,
                    eval=ev, path=code_path))
                round_valid.append(dict(src=src, ev=ev, code_path=code_path))
                if imp > 0 and (best_improvement is None or imp > best_improvement[0]):
                    best_improvement = (imp, src, label or arm, ev)
            else:
                round_invalid.append(dict(src=src, ev=ev, code_path=code_path))
            memory.append(dict(ok=valid, improvement=imp, progress=prog, arm=arm,
                               source_hash=code_hash, worst_waste=ev["worst_waste"],
                               failure_kind=ev["failure_kind"],
                               lesson=lesson_for(arm, valid, ev, imp)))
            record.update(source_hash=code_hash, code=os.path.relpath(code_path, out_dir),
                          bandit_state="learning/bandit.json",
                          population_snapshot="learning/population.json",
                          per_case=[dict(case=c["case"], bytes=c.get("bytes"),
                                         floor=c.get("floor"), waste=c.get("waste"),
                                         failure=c.get("failure"))
                                    for c in ev["per_case"]],
                          failure_kind=ev["failure_kind"], accepted=ev["accepted"],
                          valid=valid, worst_waste=ev["worst_waste"], improvement=imp,
                          progress=prog, progress_delta=reward,
                          decision="inserted" if inserted else "evaluated",
                          at_floor=at_floor(ev), level_status=level_status(ev))
            logf.write(json.dumps(record) + "\n")
            logf.flush()
            rewards.append(reward)
            if at_floor(ev):
                hit_floor = True
                break

        if bandit is not None:
            bandit.update(arm, max(rewards) if rewards else 0.0)
            with open(os.path.join(out_dir, "learning", "bandit.json"), "w") as f:
                json.dump(bandit.state(), f, indent=2)
        with open(os.path.join(out_dir, "learning", "memory.jsonl"), "w") as f:
            for item in memory.dump():
                f.write(json.dumps(item) + "\n")
        with open(os.path.join(out_dir, "learning", "population.json"), "w") as f:
            json.dump(population.state(), f, indent=2)

        best_worst = population.best()["worst_waste"] if population.best() else float("nan")
        print(f"round {rnd}: arm={arm} reward={(max(rewards) if rewards else 0.0):.2f} "
              f"best={best_worst:.2f}x ({time.perf_counter() - t0:.0f}s)")

        if hit_floor:
            print(f"  at the byte floor on round {rnd}; stopping")
            break

        if round_valid:
            pending = None            # a repair succeeded; stop repairing
        elif pending is not None:
            # Mid-repair: keep the two-round budget, but a spent repair returns to normal
            # exploration instead of chaining a new repair (measured: chained repairs ate the
            # whole round budget and the bandit never tried its other arms).
            if pending["remaining"] <= 0:
                pending = None
        else:
            # Normal round: queue one bounded repair of the closest invalid attempt this
            # round (rules-clean and simulated, so it is close rather than nonsense).
            promising = [c for c in round_invalid if c["ev"]["rules_ok"] and c["ev"]["per_case"]]
            if promising:
                promising.sort(key=lambda c: (c["ev"]["worst_waste"] is None,
                                              c["ev"]["worst_waste"] or 0.0))
                best = promising[0]
                fb = enrich_feedback(first_failure(best["ev"]))
                pending = dict(source=best["src"], feedback=fb, arm=arm,
                               remaining=3, worst_waste=best["ev"]["worst_waste"],
                               progress=progress(best["ev"]))
                last_failure = compact(fb, 220)
                memory.append(dict(ok=False, improvement=0.0,
                                   lesson=f"repairing the last attempt: {last_failure}"))

    logf.close()
    if best_improvement is not None:
        imp_best, demo_src, demo_change, demo_ev = best_improvement
        demo = dict(change=demo_change or "structural change",
                    measured=f"worst-case {demo_ev['worst_waste']:.2f}x, every shape accepted",
                    run=run_id, model=a.model, kernel=demo_src)
        demo_path = os.path.join(os.path.dirname(out_dir), "demo.json")
        with open(demo_path, "w") as f:
            json.dump(demo, f, indent=2)
        print(f"  wrote {demo_path}: the agent's first verified improvement at "
              f"{demo_ev['worst_waste']:.2f}x becomes a worked example for later runs")
    winner = population.best()
    if winner is None:
        raise SystemExit("no valid candidate was produced (even the seed failed)")
    with open(os.path.join(out_dir, "winner.py"), "w") as f:
        f.write(winner["source"])
    hold_seed = opt_seeds[1] if len(opt_seeds) > 1 else opt_seeds[0]
    _, winner_ev = write_and_eval(out_dir, "winner_holdout_seed", winner["source"], level,
                                  a.tol, hold_seed)
    with open(os.path.join(out_dir, "winner_eval.json"), "w") as f:
        json.dump(dict(seed=hold_seed, evaluation=winner_ev,
                       level_status=level_status(winner_ev)), f, indent=2)
    summary = dict(run=run_id, rounds_used=rounds_used, at_floor=hit_floor,
                   winner_hash=winner["hash"], best_worst=winner["worst_waste"],
                   accepted_seed0=winner["eval"]["accepted"],
                   accepted_holdout_seed=winner_ev["accepted"],
                   level_status_holdout=level_status(winner_ev),
                   tokens_in=tokens_in, tokens_out=tokens_out,
                   token_counting="exact" if counter.exact else "estimate")
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


# ---------------------------------------------------------------- cli

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-kernel", default="reference_level4.py")
    ap.add_argument("--level", type=int, default=5, choices=sorted(nkibench.LEVELS))
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--context", type=int, default=8192)
    ap.add_argument("--max-tokens", type=int, default=3000)
    ap.add_argument("--cases", default="traffic_cases.json")
    ap.add_argument("--output", default="runs/traffic/pilot")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--base", default=os.environ.get("KERNEL_AGENT_BASE_URL")
                    or os.environ.get("GPTOSS_BASE_URL"))
    ap.add_argument("--population-size", type=int, default=3)
    ap.add_argument("--memory-k", type=int, default=6)
    ap.add_argument("--memory-cap", type=int, default=200)
    ap.add_argument("--learning", choices=("on", "off"), default="on")
    ap.add_argument("--exploration", type=float, default=1.0)
    ap.add_argument("--second-parent-prob", type=float, default=0.15)
    ap.add_argument("--tol", type=float, default=2e-2)
    ap.add_argument("--rng-seed", type=int, default=0)
    ap.add_argument("--demo", default="", help="path to a verified demo.json written by an "
                    "earlier successful run; reused as a worked example in prompts")
    a = ap.parse_args()
    if not a.demo and os.path.exists("demo.json"):
        a.demo = "demo.json"   # the team calibration example, injected by default when present

    if not a.base:
        raise SystemExit("set KERNEL_AGENT_BASE_URL (or GPTOSS_BASE_URL) to the model endpoint")
    if not os.path.exists(a.seed_kernel):
        raise SystemExit(f"seed kernel not found: {a.seed_kernel}")
    opt_seeds = [0]
    if os.path.exists(a.cases):
        try:
            opt_seeds = json.load(open(a.cases)).get("optimization", {}).get("seeds", [0]) or [0]
        except Exception:
            pass

    counter = TokenCounter(a.base, a.model)
    os.makedirs(a.output, exist_ok=True)
    summaries = []
    for rep in range(a.repeat):
        rng = random.Random(a.rng_seed + rep)
        run_dir = os.path.join(a.output, f"run{rep + 1}")
        print(f"\n===== repeat {rep + 1}/{a.repeat} -> {run_dir} =====")
        summaries.append(run_once(a, a.level, run_dir, rng, counter, opt_seeds))
    solved = sum(1 for s in summaries if s["at_floor"])
    with open(os.path.join(a.output, "summary.json"), "w") as f:
        json.dump(dict(model=a.model, base=a.base, level=a.level, repeats=a.repeat,
                       learning=a.learning, cases=a.cases, tol=a.tol,
                       token_counting="exact" if counter.exact else "estimate",
                       runs=summaries), f, indent=2)
    best = min((s["best_worst"] for s in summaries if s["best_worst"] is not None),
               default=None)
    print(f"\n=== {solved}/{a.repeat} runs reached the byte floor; best worst-case waste "
          f"={best if best is None else round(best, 2)}x; token counting "
          f"{'exact' if counter.exact else 'estimated'} ===")


if __name__ == "__main__":
    main()
