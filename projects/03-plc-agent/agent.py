"""agent.py - two-agent IEC 61131-3 generator with a deterministic gate.

    Agent 1 (generator)  writes a Structured Text program for a spec.
    Gate (plc_checker)   IronPLC compile + spec checks. The ONLY thing that can say PASS.
    Agent 2 (critic)     turns the gate's verdict into concrete line-level edits.
    Ledger               remembers rejected programs and failure signatures.

Every attempt is logged. Use --feedback raw for the single-agent baseline (compiler
output goes straight back to the generator), and --repeat N to report a rate.

    python agent.py --n 10 --repeat 3                       # local Qwen3 on localhost:8000
    python agent.py --n 10 --repeat 3 --feedback raw        # baseline, same specs
    python agent.py --n 10 --repeat 3 --bootstrap           # carries verified experience spec to spec
    python agent.py --n 30 --curriculum                     # easy specs first, level up as it solves them
    python agent.py --base-url https://api.openai.com/v1 --model gpt-5.4-mini --samples 1 --flex
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import random
import re
import statistics
import sys
import time
import uuid

from openai import OpenAI, APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

import plc_checker as pc

GEN_SYSTEM = ("You are an automation engineer writing IEC 61131-3 Structured Text for the IronPLC compiler. "
              "Reply with the program only: no explanations, no markdown fences.")

CRITIC_SYSTEM = """You review IEC 61131-3 Structured Text that failed automated checks. The compiler and checker have already decided it fails; do not argue with them or judge it yourself. Turn each problem into one concrete edit.
Rules:
- One numbered line per edit, at most 8 lines.
- Cite the line number from the listing (L12) and say exactly what to add, delete or change, with the corrected text.
- Fix compiler errors first. One mistake often causes several messages, so fix the first one.
- Then cover every failed requirement check.
- No full program, no explanations, no markdown."""

TRUNCATED_MSG = ("Your previous answer was cut off at the token limit before END_PROGRAM. Write a shorter program: "
                 "no unused variables, minimal comments, only the logic the requirements ask for.")
EMPTY_MSG = "Your previous answer contained no Structured Text program. Output only the program."


# =====================================================================
# LLM client (any OpenAI-compatible endpoint)
# =====================================================================
def normalize_base_url(url):
    url = url.rstrip("/")
    return url if url.endswith("/v1") else url + "/v1"


class LLM:
    def __init__(self, base_url, model, api_key, think, temperature, max_tokens, flex):
        self.is_openai = "api.openai.com" in base_url
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=900 if flex else 300)
        self.model = model or self.client.models.list().data[0].id
        self.think, self.temperature, self.max_tokens, self.flex = think, temperature, max_tokens, flex

    def chat(self, system, user, max_tokens=None, temperature="default"):
        temperature = self.temperature if temperature == "default" else temperature
        kw = dict(model=self.model, messages=[{"role": "system", "content": system},
                                              {"role": "user", "content": user}])
        mt = max_tokens or self.max_tokens
        if self.is_openai:
            kw["max_completion_tokens"] = mt
            if self.flex:
                kw["service_tier"] = "flex"
        else:
            kw["max_tokens"] = mt
            if "qwen" in self.model.lower():
                kw["extra_body"] = {"chat_template_kwargs": {"enable_thinking": self.think}}
        if temperature is not None:
            kw["temperature"] = temperature
        t0, last_err = time.time(), ""
        for attempt in range(4):
            try:
                r = self.client.chat.completions.create(**kw)
                ch = r.choices[0]
                return ch.message.content or "", ch.finish_reason or "", time.time() - t0
            except (RateLimitError, APITimeoutError, APIConnectionError) as e:
                last_err = repr(e)
            except APIStatusError as e:
                last_err = repr(e)
                if e.status_code < 500:
                    break
            time.sleep(10 * 2 ** attempt)
        return "", "error: " + last_err[:300], time.time() - t0


# =====================================================================
# Helpers
# =====================================================================
def extract_code(text):
    if not text:
        return ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    m = re.search(r"```[A-Za-z0-9_+-]*[ \t]*\n(.*?)```", text, re.S)
    if m:
        text = m.group(1)
    else:
        text = re.sub(r"```[A-Za-z0-9_+-]*", "", text)
    m = re.search(r"\b(TYPE|FUNCTION_BLOCK|PROGRAM)\b", text)
    if m:
        text = text[m.start():]
    m = re.search(r"\bEND_PROGRAM\b", text)
    if m:
        text = text[:m.end()]
    return text.strip()


def numbered(code):
    return "\n".join(f"L{i:<3} {line}" for i, line in enumerate(code.splitlines(), 1))


def trim(s, n):
    return s if len(s) <= n else s[:n] + "\n...[truncated]"


def code_hash(code):
    norm = re.sub(r"\s+", " ", pc.strip_comments(code)).strip().lower()
    return hashlib.sha1(norm.encode()).hexdigest()[:12]


def signature(v):
    if v.truncated:
        return "truncated output"
    if v.empty:
        return "no program in output"
    if not v.compiled:
        return "compiler: " + norm_error(v.compiler_output)
    return "checks failed: " + ", ".join(r.id for r in v.failed)


def raw_feedback(v):
    if v.truncated:
        return TRUNCATED_MSG
    if v.empty:
        return EMPTY_MSG
    parts = []
    if not v.compiled:
        parts.append("IronPLC rejected the program:\n" + pc.explain_compiler(v.compiler_output))
    if v.failed:
        parts.append("Failed requirement checks:\n" + "\n".join(f"- {r.id}: {v.hint(r)}" for r in v.failed))
    return "\n\n".join(parts)


def critic_feedback(llm, code, v, critic_max_tokens):
    """Agent 2. Advises only; never decides pass/fail. Falls back to raw feedback if it returns nothing."""
    if v.truncated or v.empty:
        return raw_feedback(v), 0.0
    compiler = "IronPLC: OK" if v.compiled else "IronPLC rejected it:\n" + pc.explain_compiler(v.compiler_output)
    checks = "\n".join(f"- {r.id}: {v.hint(r)}" for r in v.failed) or "- none"
    user = f"Program listing:\n{numbered(code)}\n\n{compiler}\n\nFailed requirement checks:\n{checks}\n\nWrite the edits."
    text, fr, secs = llm.chat(CRITIC_SYSTEM, user, max_tokens=critic_max_tokens,
                              temperature=None if llm.is_openai else 0.2)
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
    lines = [ln for ln in text.splitlines() if ln.strip()][:10]
    if not lines:
        return raw_feedback(v), secs
    return "\n".join(lines), secs


def repair_prompt(spec, prev_code, feedback, failures, repeated):
    s = pc.spec_prompt(spec) + f"\n\nYour previous program was rejected:\n{prev_code}\n\nRequired fixes:\n{feedback}"
    if failures:
        s += "\n\nEarlier rounds already failed on:\n" + "\n".join("- " + f for f in failures[-4:])
    if repeated:
        s += "\n\nYou returned a program identical to one already rejected. You must change it."
    return s + "\n\nOutput the complete corrected program only."


class Logger:
    def __init__(self, path):
        self.f = open(path, "a", encoding="utf-8")

    def write(self, rec):
        self.f.write(json.dumps(rec) + "\n")
        self.f.flush()


# =====================================================================
# Bootstrap memory: what Agent 1 carries from one spec to the next
# =====================================================================
def norm_error(compiler_output):
    """A compiler error with positions removed, so the same mistake matches across programs."""
    problems = pc.parse_compiler(compiler_output)
    if problems:
        return pc.problem_key(problems[0])
    lines = [ln.strip() for ln in compiler_output.splitlines() if ln.strip()]
    first = next((ln for ln in lines if "error" in ln.lower()), lines[0] if lines else "")
    return re.sub(r"\s+", " ", re.sub(r"\d+", "#", first)).strip()[:140]


class Memory:
    """Experience that only ever comes from the gate, never from an LLM's opinion:
      exemplars  programs that PASSED, retrieved by requirement overlap
      fixes      compiler errors seen before, with the feedback that made them go away
      missed     how often each requirement fails on round 0, to warn about it up front
    Bad code cannot get in: an exemplar exists only if the compiler and every check accepted it."""

    def __init__(self, max_exemplar_chars=3000):
        self.exemplars, self.fixes, self.errors = [], {}, {}
        self.missed, self.seen = {}, {}
        self.max_exemplar_chars = max_exemplar_chars

    # ---- learning ----
    def add_exemplar(self, spec, code, origin):
        if len(code) <= self.max_exemplar_chars:
            self.exemplars.append({"reqs": sorted(r.id for r in spec.reqs), "code": code, "origin": origin})

    def observe_round0(self, spec, v):
        for r in spec.reqs:
            self.seen[r.id] = self.seen.get(r.id, 0) + 1
        for r in v.failed:
            self.missed[r.id] = self.missed.get(r.id, 0) + 1

    def observe_error(self, err):
        if err:
            self.errors[err] = self.errors.get(err, 0) + 1

    def observe_fix(self, err, feedback):
        if err and err not in self.fixes:
            self.fixes[err] = " ".join(feedback.split())[:300]

    # ---- recall ----
    def best_exemplar(self, spec):
        want = {r.id for r in spec.reqs} - {"structure"}
        scored = [(len(want & set(e["reqs"])), -len(e["code"]), i) for i, e in enumerate(self.exemplars)]
        scored = [x for x in scored if x[0] >= 2]
        return self.exemplars[max(scored)[2]] if scored else None

    def lessons(self, spec, k_errors=3, k_missed=3):
        out = []
        for err, _ in sorted(self.errors.items(), key=lambda x: -x[1]):
            if err in self.fixes:
                out.append(f'- IronPLC error "{err}" was fixed by: {self.fixes[err]}')
            if len(out) >= k_errors:
                break
        by_id = {r.id: r for r in spec.reqs}
        rates = sorted(((self.missed.get(i, 0) / self.seen[i], i) for i in by_id
                        if self.seen.get(i, 0) >= 2 and self.missed.get(i, 0)), reverse=True)
        for rate, rid in rates[:k_missed]:
            if rate >= 0.3:
                out.append(f"- Requirement '{rid}' failed {rate:.0%} of first attempts. {by_id[rid].hint}")
        return out

    def prompt_section(self, spec):
        parts, ex = [], self.best_exemplar(spec)
        if ex:
            parts.append("A program from an earlier task that passed IronPLC and every check. Reuse its syntax "
                         "patterns; implement only the requirements listed above:\n" + ex["code"])
        lessons = self.lessons(spec)
        if lessons:
            parts.append("Lessons from earlier attempts:\n" + "\n".join(lessons))
        return ("\n\n" + "\n\n".join(parts)) if parts else "", (ex["origin"] if ex else None), len(lessons)

    def stats(self):
        return {"exemplars": len(self.exemplars), "known_fixes": len(self.fixes), "errors_seen": len(self.errors)}

    # ---- persistence ----
    def save(self, path):
        with open(path, "w") as f:
            json.dump({"exemplars": self.exemplars, "fixes": self.fixes, "errors": self.errors,
                       "missed": self.missed, "seen": self.seen}, f, indent=1)

    @classmethod
    def load(cls, path, max_exemplar_chars=3000):
        m = cls(max_exemplar_chars)
        with open(path) as f:
            d = json.load(f)
        m.exemplars, m.fixes, m.errors = d["exemplars"], d["fixes"], d["errors"]
        m.missed, m.seen = d["missed"], d["seen"]
        return m


# =====================================================================
# One sample: up to --rounds rounds of --samples candidates each
# =====================================================================
def solve(spec, sample_id, run, args, llm, log, memory=None):
    seen, failures = set(), []
    prev_code, feedback, repeated = None, None, False
    pending_err, pending_fb = None, None
    best_score, gen_calls, keep = 0.0, 0, None
    learn = memory is not None and not args.freeze_memory
    for rnd in range(args.rounds):
        exemplar_from, n_lessons = None, 0
        if rnd == 0:
            prompt = pc.spec_prompt(spec)
            if memory is not None:
                section, exemplar_from, n_lessons = memory.prompt_section(spec)
                prompt += section
        else:
            prompt = repair_prompt(spec, prev_code or "(none)", feedback, failures, repeated)
            if memory is not None:
                lessons = memory.lessons(spec, k_missed=0)
                if lessons:
                    prompt += "\n\nLessons from earlier attempts:\n" + "\n".join(lessons)
        with cf.ThreadPoolExecutor(max_workers=args.samples) as ex:
            outs = list(ex.map(lambda _: llm.chat(GEN_SYSTEM, prompt), range(args.samples)))
        gen_calls += len(outs)

        cands = []
        for k, (text, finish, secs) in enumerate(outs):
            code = extract_code(text)
            v = pc.grade(code, spec, args.compiler, truncated=(finish == "length"))
            h = code_hash(code) if code else ""
            log.write({"type": "attempt", "run": run, "sample": sample_id, "round": rnd, "candidate": k,
                       "model": llm.model, "feedback_mode": args.feedback, "bootstrap": memory is not None,
                       "exemplar_from": exemplar_from, "lessons_in_prompt": n_lessons, "score": v.score,
                       "passed": v.ok, "compiled": v.compiled, "failed_checks": [r.id for r in v.failed],
                       "compiler_output": trim(v.compiler_output, 2000), "finish_reason": finish,
                       "gen_seconds": round(secs, 2), "prompt_chars": len(prompt), "code_hash": h,
                       "repeat_of_rejected": h in seen, "code": code})
            cands.append((v.score, v.frac, k, code, v, h))

        score, _, k, code, v, h = max(cands, key=lambda c: (c[0], c[1], -c[2]))
        best_score = max(best_score, score)
        worse = keep is not None and (score, v.frac) <= (keep["score"], keep["frac"]) and not v.ok
        print(f"    round {rnd}: best {score:.2f} of {args.samples}  [{signature(v) if not v.ok else 'PASS'}]"
              + (f"  (worse than {keep['score']:.2f}; repairing that one)" if worse else ""))

        err = norm_error(v.compiler_output) if (code and not v.compiled and not v.truncated) else None
        if learn:
            if rnd == 0:
                memory.observe_round0(spec, v)
            memory.observe_error(err)
            if pending_err and pending_err != err:      # last round's error is gone: the feedback worked
                memory.observe_fix(pending_err, pending_fb)

        if v.ok:
            if learn:
                memory.add_exemplar(spec, code, {"run": run, "sample": sample_id, "round": rnd})
            return {"solved": True, "rounds": rnd + 1, "best_score": 1.0, "code": code, "gen_calls": gen_calls}

        # Always repair from the best program so far. If this round got worse, discard it and
        # send the best version back with its feedback, instead of repairing the regression.
        regressed = keep is not None and (score, v.frac) <= (keep["score"], keep["frac"])
        if regressed:
            feedback = (keep["feedback"] + f"\n\nYour latest version scored {score:.2f}, worse than this one "
                        f"({keep['score']:.2f}), and was discarded. Apply the fixes to the program shown above.")
            critic_secs = 0.0
        else:
            if args.feedback == "critic":
                feedback, critic_secs = critic_feedback(llm, code, v, args.critic_max_tokens)
            else:
                feedback, critic_secs = raw_feedback(v), 0.0
            if code:
                keep = {"score": score, "frac": v.frac, "code": code, "feedback": feedback}
        pending_err, pending_fb = err, feedback
        repeated = h in seen
        seen.update(c[5] for c in cands if c[5])
        failures.append(signature(v))
        if keep:
            prev_code = keep["code"]
        log.write({"type": "feedback", "run": run, "sample": sample_id, "round": rnd, "mode": args.feedback,
                   "regressed": regressed, "repairing_score": keep["score"] if keep else None,
                   "critic_seconds": round(critic_secs, 2), "feedback": feedback})
    return {"solved": False, "rounds": args.rounds, "best_score": best_score, "code": None, "gen_calls": gen_calls}


# =====================================================================
# Main
# =====================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=10, help="specs per run")
    ap.add_argument("--rounds", type=int, default=4, help="max rounds per spec")
    ap.add_argument("--samples", type=int, default=4, help="candidates per round (use 1 for greedy models)")
    ap.add_argument("--repeat", type=int, default=1, help="repeat the same specs N times to report a rate")
    ap.add_argument("--seed", type=int, default=0, help="spec seed (same seed = same specs)")
    ap.add_argument("--feedback", choices=["critic", "raw"], default="critic",
                    help="critic = two-agent loop; raw = single-agent baseline")
    ap.add_argument("--base-url", default=os.environ.get("PLC_BASE_URL")
                    or os.environ.get("KERNEL_AGENT_BASE_URL") or "http://localhost:8000/v1")
    ap.add_argument("--model", default=os.environ.get("PLC_MODEL"), help="default: first model the server lists")
    ap.add_argument("--api-key-env", default="OPENAI_API_KEY", help="env var holding the API key")
    ap.add_argument("--temperature", type=float, default=None, help="default 0.7 local, unset for OpenAI")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--critic-max-tokens", type=int, default=400)
    ap.add_argument("--think", action="store_true", help="enable Qwen3 thinking (measured: much slower, worse)")
    ap.add_argument("--flex", action="store_true", help="OpenAI flex service tier")
    ap.add_argument("--compiler", default="ironplcc check")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--bootstrap", action="store_true",
                    help="carry gate-verified exemplars and lessons from each spec to the next")
    ap.add_argument("--memory-in", default=None, help="start every run from this saved memory.json")
    ap.add_argument("--freeze-memory", action="store_true",
                    help="use --memory-in without updating it (held-out evaluation)")
    ap.add_argument("--max-exemplar-chars", type=int, default=4000)
    ap.add_argument("--curriculum", action="store_true",
                    help="start at level 1 and level up as specs get solved (implies --bootstrap)")
    ap.add_argument("--start-level", type=int, default=1)
    ap.add_argument("--promote-window", type=int, default=3,
                    help="level up when this many of the last specs at a level were solved in <= 2 rounds, "
                         "minus one (default: 2 of the last 3)")
    args = ap.parse_args()

    base_url = normalize_base_url(args.base_url)
    is_openai = "api.openai.com" in base_url
    api_key = os.environ.get(args.api_key_env) or ("" if is_openai else "EMPTY")
    if not api_key:
        sys.exit(f"Set {args.api_key_env} in the environment (never hardcode keys in the script).")
    temperature = args.temperature if args.temperature is not None else (None if is_openai else 0.7)

    print("Probing compiler...")
    try:
        unsupported, gate_ok, details = pc.probe_compiler(args.compiler)
    except pc.CompilerMissing as e:
        sys.exit(str(e))
    if not gate_ok:
        sys.exit("Compiler gate is not trustworthy; run: python plc_checker.py --selftest")
    if unsupported:
        excluded = sorted(v for v, d in pc.CAPABILITY_DEPS.items() if d & unsupported)
        print(f"  ironplcc cannot compile: {', '.join(sorted(unsupported))} -> not sampling {', '.join(excluded)}")

    llm = LLM(base_url, args.model, api_key, args.think, temperature, args.max_tokens, args.flex)
    if args.memory_in or args.freeze_memory or args.curriculum:
        args.bootstrap = True
    if args.freeze_memory and not args.memory_in:
        sys.exit("--freeze-memory needs --memory-in")
    tag = run_tag(args)
    out_dir = args.out_dir or os.path.join("runs", time.strftime("%Y%m%d-%H%M%S") + f"-{tag}")
    os.makedirs(out_dir, exist_ok=True)
    log = Logger(os.path.join(out_dir, "attempts.jsonl"))
    dataset = Logger(os.path.join(out_dir, "dataset.jsonl"))
    config = {**vars(args), "base_url": base_url, "model": llm.model, "temperature": temperature,
              "unsupported_capabilities": sorted(unsupported)}
    log.write({"type": "config", **config})
    print(f"Model {llm.model} at {base_url}, mode={tag}, logging to {out_dir}/")

    results = []  # (run, sample, result)
    try:
        for run in range(args.repeat):
            rng = random.Random(args.seed)
            memory = None
            if args.bootstrap:   # each run learns from scratch (or from --memory-in), so runs stay independent
                memory = (Memory.load(args.memory_in, args.max_exemplar_chars) if args.memory_in
                          else Memory(args.max_exemplar_chars))
            level = args.start_level if args.curriculum else pc.MAX_LEVEL
            history = []
            print(f"\n=== run {run + 1}/{args.repeat} ===")
            for i in range(args.n):
                spec = pc.build_spec(rng, unsupported, level)
                md = spec.metadata
                print(f"  sample {i} [L{level}]: {md['control_strategy'] or '-'} | {', '.join(md['iec_features']) or '-'} | "
                      f"{', '.join(md['safety_features'] + md['failure_scenarios'])} | {md['security_profile'] or '-'}")
                t0 = time.time()
                res = solve(spec, i, run, args, llm, log, memory)
                res["seconds"] = round(time.time() - t0, 1)
                res["level"] = level
                if args.curriculum:
                    history.append((res["solved"], res["rounds"]))
                    if level < pc.MAX_LEVEL and promoted(history, args.promote_window):
                        level, history = level + 1, []
                        print(f"  ** level up -> L{level}")
                        log.write({"type": "level_up", "run": run, "after_sample": i, "level": level})
                results.append((run, i, res))
                log.write({"type": "result", "run": run, "sample": i, **{k: v for k, v in res.items() if k != "code"},
                           "spec": md})
                if res["solved"]:
                    dataset.write({"id": str(uuid.uuid4()), "run": run, **md, "model": llm.model,
                                   "rounds": res["rounds"], "code": res["code"]})
                print(f"  -> {'SOLVED in ' + str(res['rounds']) + ' round(s)' if res['solved'] else 'NOT SOLVED'}"
                      f"  best {res['best_score']:.2f}  {res['seconds']}s"
                      + (f"  memory {memory.stats()}" if memory is not None else ""))
            if memory is not None and not args.freeze_memory:
                memory.save(os.path.join(out_dir, f"memory_run{run}.json"))
    except KeyboardInterrupt:
        print("\nInterrupted; summarizing what finished.")

    summarize(results, args, config, out_dir)


def run_tag(args):
    return (args.feedback + ("-curriculum" if args.curriculum else "-bootstrap" if args.bootstrap else "")
            + ("-frozen" if args.freeze_memory else ""))


def promoted(history, window):
    """history: [(solved, rounds)] at the current level. 2 of the last 3 solved in <= 2 rounds."""
    recent = history[-window:]
    return len(recent) == window and sum(s and r <= 2 for s, r in recent) >= window - 1


def summarize(results, args, config, out_dir):
    if not results:
        return
    runs = sorted({r for r, _, _ in results})
    per_run = [sum(res["solved"] for r, _, res in results if r == run) for run in runs]
    per_run_n = [sum(1 for r, _, _ in results if r == run) for run in runs]
    rates = [s / n for s, n in zip(per_run, per_run_n)]
    solved = [res for _, _, res in results if res["solved"]]
    calls = sum(res.get("gen_calls", 0) for _, _, res in results)

    # learning curve: position in the sequence of specs, averaged over runs
    by_pos = {}
    for _, i, res in results:
        by_pos.setdefault(i, []).append(res)
    curve = []
    for i in sorted(by_pos):
        rs = by_pos[i]
        curve.append({"sample": i,
                      "solve_rate": round(sum(r["solved"] for r in rs) / len(rs), 3),
                      "first_round_rate": round(sum(r["solved"] and r["rounds"] == 1 for r in rs) / len(rs), 3),
                      "mean_rounds": round(statistics.mean(r["rounds"] for r in rs), 2),
                      "mean_gen_calls": round(statistics.mean(r.get("gen_calls", 0) for r in rs), 2),
                      "mean_best_score": round(statistics.mean(r["best_score"] for r in rs), 3)})
    with open(os.path.join(out_dir, "learning_curve.csv"), "w") as f:
        f.write(",".join(curve[0]) + "\n")
        for row in curve:
            f.write(",".join(str(v) for v in row.values()) + "\n")
    half = len(curve) // 2 or 1

    def avg(rows, key):
        return round(statistics.mean(r[key] for r in rows), 3) if rows else None

    tag = run_tag(args)
    levels = {}
    for _, _, res in results:
        levels.setdefault(res.get("level", pc.MAX_LEVEL), []).append(res)
    summary = {
        "model": config["model"], "mode": tag, "runs": len(runs), "specs_per_run": args.n, "seed": args.seed,
        "solved_per_run": per_run, "solve_rate_mean": round(statistics.mean(rates), 3),
        "solve_rate_min": round(min(rates), 3), "solve_rate_max": round(max(rates), 3),
        "first_round_solve_rate": round(sum(r["rounds"] == 1 for r in solved) / len(results), 3),
        "mean_rounds_when_solved": round(statistics.mean(r["rounds"] for r in solved), 2) if solved else None,
        "generator_calls_total": calls,
        "generator_calls_per_solve": round(calls / len(solved), 2) if solved else None,
        "mean_best_score": round(statistics.mean(res["best_score"] for _, _, res in results), 3),
        "first_half": {"first_round_rate": avg(curve[:half], "first_round_rate"),
                       "mean_rounds": avg(curve[:half], "mean_rounds")},
        "second_half": {"first_round_rate": avg(curve[half:], "first_round_rate"),
                        "mean_rounds": avg(curve[half:], "mean_rounds")},
        "per_level": {str(lv): {"specs": len(rs), "solved": sum(r["solved"] for r in rs),
                                "first_round": sum(r["solved"] and r["rounds"] == 1 for r in rs)}
                      for lv, rs in sorted(levels.items())},
        "per_sample_solved": {str(i): f"{sum(r['solved'] for r in v)}/{len(v)}" for i, v in sorted(by_pos.items())},
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\n=== summary ===")
    print(f"mode {tag}: solved per run {per_run}  rate {summary['solve_rate_mean']:.0%} "
          f"(min {summary['solve_rate_min']:.0%}, max {summary['solve_rate_max']:.0%})")
    print(f"solved on round 1: {summary['first_round_solve_rate']:.0%}   mean rounds when solved: "
          f"{summary['mean_rounds_when_solved']}   generator calls per solve: {summary['generator_calls_per_solve']}")
    print(f"first half vs second half of the sequence: round-1 rate {summary['first_half']['first_round_rate']} -> "
          f"{summary['second_half']['first_round_rate']}, mean rounds {summary['first_half']['mean_rounds']} -> "
          f"{summary['second_half']['mean_rounds']}")
    if args.curriculum:
        print("per level: " + "  ".join(f"L{lv}: {d['solved']}/{d['specs']} solved"
                                        for lv, d in summary["per_level"].items()))
    print("per spec across runs: " + "  ".join(f"#{k}:{v}" for k, v in summary["per_sample_solved"].items()))
    print(f"wrote {out_dir}/summary.json and learning_curve.csv")


if __name__ == "__main__":
    main()
