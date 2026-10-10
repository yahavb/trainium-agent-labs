"""MANAGER: starts threads, ends them, ends the level. Code, not a model: its decisions are rules on
measured signals, so they cost no tokens and can be explained.

One thread follows one approach:

    planner -> coder (write) -> checks -+- fail ----> debugger -> coder (apply) -> checks ...
                                        |                 '-- APPROACH WRONG: thread ends
                                        +- TRAFFIC --> reviewer -> coder (improve) -> checks ...
                                        '- correct --> reviewer: accept, and the level ends

A thread ends when:
  - the same failure (kind, line, message) comes back same_error_limit times in a row ("cycling");
  - the coder hands back the kernel it was given, after the debugger has already named a second,
    different change for that same failure ("echo");
  - max_no_gain checks pass without progress ("no_gain"), or max_attempts checks in all. Progress
    is the check stage reached as well as the reward (see progress()): the reward is coarse, and a
    kernel moving from an invented name to a shape error to wrong values is getting somewhere at the
    same 0.30 -- "expect the failure to move rather than vanish" (README-task.md, hint 9);
  - the debugger says the approach is wrong; the reviewer gives up on bytes;
  - three answers in a row have no code ("empty_answers");
  - the level's call budget runs out, or model calls keep failing ("budget", "http_errors"): checked
    after every check, so a running thread stops too, not only the next one to start.
An ended thread is replaced by a new plan, told what failed, until max_approaches plans have run.
"""

import concurrent.futures as cf
import threading
import time
import traceback

from agents2.coder import Coder
from agents2.debugger import Debugger
from agents2.ledger import Ledger, sha
from agents2.llm import head
from agents2.planner import Planner
from agents2.reviewer import Reviewer, worst_bytes


class Manager:
    def __init__(self, cfg, llm, checks, retriever, events, agent_mod, say=print):
        self.cfg, self.llm, self.checks, self.events = cfg, llm, checks, events
        self.retriever = retriever
        self.say = say
        self.planner = Planner(llm, retriever, cfg, events)
        self.coder = Coder(llm, retriever, cfg, agent_mod.extract_code, events)
        self.debugger = Debugger(llm, retriever, cfg, agent_mod.enrich, fix_examples(agent_mod), events)
        self.reviewer = Reviewer(llm, cfg)
        self.say_lock = threading.Lock()

    def log(self, msg):
        with self.say_lock:
            self.say(msg)

    # ------------------------------------------------------------ the level

    def run_level(self, level, run=0):
        led = Ledger(level, run)
        stop = threading.Event()
        t0 = time.perf_counter()
        started = 0
        with cf.ThreadPoolExecutor(max_workers=self.cfg.threads) as ex:
            live = {}
            while started < min(self.cfg.threads, self.cfg.max_approaches):
                live[ex.submit(self._thread_safe, led, stop, started)] = started
                started += 1
            while live:
                done, _ = cf.wait(live, return_when=cf.FIRST_COMPLETED)
                for f in done:
                    live.pop(f)
                if led.solved:
                    stop.set()
                    continue
                why = self._out_of_budget(led)
                if why:
                    led.stop_reason = led.stop_reason or why
                    stop.set()
                    continue
                # A thread whose planner found nothing new registers no approach, so count approaches,
                # not threads; the thread cap stops a planner that only repeats itself.
                if (len(led.approaches) + led.planning < self.cfg.max_approaches
                        and started < 2 * self.cfg.max_approaches and not stop.is_set()):
                    live[ex.submit(self._thread_safe, led, stop, started)] = started
                    started += 1
        if led.solved:
            led.stop_reason = "solved"
        elif not led.stop_reason:
            led.stop_reason = "exhausted"
        result = dict(level=level, run=run, reward=led.best[0], solved=led.solved,
                      stop_reason=led.stop_reason, approaches=[a.summary() for a in led.approaches],
                      checks=led.checks, calls=dict(led.calls), tokens=dict(led.tokens),
                      model_seconds={k: round(v, 1) for k, v in led.seconds.items()},
                      wall_seconds=round(time.perf_counter() - t0, 1), best_code=led.best[1])
        self.events.write("level_end", **{k: v for k, v in result.items() if k != "best_code"})
        return result

    def _out_of_budget(self, led):
        if led.failed_calls >= 6:
            return "http_errors"
        if led.total_calls() >= self.cfg.max_calls:
            return "budget"
        return ""

    def _thread_safe(self, led, stop, tid):
        try:
            self._thread(led, stop, tid)
        except Exception as e:
            self.events.write("thread_crash", level=led.level, thread=tid, error=repr(e),
                              tb=traceback.format_exc())
            self.log(f"[L{led.level} T{tid}] crashed: {e!r}")
            if "ENV:" in str(e):
                led.stop_reason = "environment"
                stop.set()

    # ------------------------------------------------------------ one thread

    def _thread(self, led, stop, tid):
        level = led.level
        tag = f"[L{level} T{tid}]"
        with led.lock:
            led.planning += 1
        try:
            approach, plan = self._plan(led, tid, tag)
        finally:
            with led.lock:
                led.planning -= 1
        if approach is None:
            return
        self.events.write("plan", level=level, thread=tid, approach=approach.id, plan=plan.text(),
                          raw=plan.raw, unknown=plan.unknown)
        self.log(f"{tag} {approach.id}: {plan.approach[:140]}  calls: {', '.join(plan.calls) or '-'}")
        try:
            self._work(led, stop, tid, tag, approach, plan)
        except Exception as e:
            self._end(led, approach, "crashed", tag, repr(e))     # not left "running" in the summary
            raise

    def _plan(self, led, tid, tag):
        """A plan registered as a new approach, or (None, None)."""
        level, approach, note, plan = led.level, None, "", None
        for _ in range(3):
            plan, meta = self.planner.plan(level, led, note=note)
            if plan is None:
                self.log(f"{tag} no plan ({meta.get('error') or 'unparseable reply'})")
                return None, None
            if plan.unknown and not note:
                close = {u: self.retriever.close(u) for u in plan.unknown}
                note = ("These names do not exist in nki " + self.retriever.version + ": "
                        + "; ".join(f"{u} (closest: {', '.join(c) or 'none'})" for u, c in close.items())
                        + ". Plan again using only real names.")
                continue
            approach = led.register(plan, tid)
            if approach:
                break
            self.events.write("plan_rejected", level=level, thread=tid, plan=plan.text(), raw=plan.raw)
            note = ("Another thread is already trying this approach: " + plan.approach[:200]
                    + ". Choose a different algorithm.")
        if approach is None:
            self.log(f"{tag} no new approach")
            return None, None
        return approach, plan

    def _work(self, led, stop, tid, tag, approach, plan):
        level, cfg = led.level, self.cfg
        ids = dict(thread=tid, approach=approach.id)

        code, meta = self.coder.write(level, plan, led, **ids)
        mode = "write"
        history, review_tries = [], []
        last_sig, same, best, no_gain, empties = None, 0, None, 0, 0
        rescued = set()                  # failures whose echo the debugger has already answered
        while True:
            if stop.is_set():
                return led.end(approach, "stopped")
            if approach.attempts >= cfg.max_attempts:
                return self._end(led, approach, "max_attempts", tag)
            check = self.checks.run(code, level)
            if check["kind"] == "ENV":
                raise RuntimeError(f"ENV: {check['error']}")
            led.note_check(approach, code, check)
            self._record(led, approach, tid, mode, code, check)
            self.log(f"{tag} #{approach.attempts} {mode:<7} {check['reward']:.2f}  {check['kind']}"
                     + (f" line {check['line']}" if check.get("line") else "")
                     + (f"  {(check.get('error') or '')[:90]}" if not check["correct"] else ""))

            if check["correct"]:
                verdict = self.reviewer.review(level, plan, code, check, review_tries, led, **ids)
                self.events.write("verdict", level=level, **ids, **_slim(verdict))
                if verdict["accept"]:
                    led.accept(approach, code, check)
                    self.log(f"{tag} ACCEPTED {approach.id} after {approach.attempts} check(s)")
                    return

            # Before any more model calls: the kernel just written has been checked (checks cost none).
            why = self._out_of_budget(led)
            if why:
                led.stop_reason = led.stop_reason or why
                stop.set()
                return self._end(led, approach, why, tag)

            if best is None or progress(check) > best:
                best, no_gain = progress(check), 0
            else:
                no_gain += 1
            if no_gain >= cfg.max_no_gain:
                return self._end(led, approach, "no_gain", tag)
            sig = (check["kind"], check.get("line"), head(check.get("error")))
            same = same + 1 if sig == last_sig else 1
            last_sig = sig
            if same >= cfg.same_error_limit:
                return self._end(led, approach, "cycling", tag, check.get("error") or "")
            empties = empties + 1 if check["kind"] == "EMPTY" else 0
            if empties >= 3:
                return self._end(led, approach, "empty_answers", tag)

            if check["kind"] == "TRAFFIC":
                verdict = self.reviewer.review(level, plan, code, check, review_tries, led, **ids)
                self.events.write("verdict", level=level, **ids, **_slim(verdict))
                if verdict["end"]:
                    return self._end(led, approach, "reviewer", tag, verdict.get("reason", ""))
                review_tries.append((worst_bytes(check), verdict["change"]))
                new, mode = self._next(level, plan, code, check, led, ids, improve=verdict)
            else:
                change = self.debugger.debug(level, plan, code, check, history, led,
                                             approach.attempts, **ids)
                self.events.write("change", level=level, **ids, **_slim(change))
                if change.get("approach_wrong"):
                    return self._end(led, approach, "approach_wrong", tag, change["approach_wrong"])
                history.append((sig, change))
                self.log(f"{tag}    debugger ({change['path']}): {change['change'][:110]}")
                new, mode = self._next(level, plan, code, check, led, ids, change=change, tries=1)
                if new is None and sig not in rescued:
                    # The coder returned the kernel unchanged. Re-sending the same change, hotter, did
                    # not help: on seat-35 (agent2-v3-l1-1010-2153) all 6 such retries echoed again,
                    # each after a change that only described the failure. So the debugger, told the
                    # change was tried, names a different one on a specific line; once per failure, so
                    # a thread holding the best kernel is not given up on the first echo.
                    rescued.add(sig)
                    change = self.debugger.debug(level, plan, code, check, history, led,
                                                 approach.attempts, **ids)
                    self.events.write("change", level=level, rescue=True, **ids, **_slim(change))
                    if change.get("approach_wrong"):
                        return self._end(led, approach, "approach_wrong", tag, change["approach_wrong"])
                    history.append((sig, change))
                    self.log(f"{tag}    debugger ({change['path']}, after an echo): "
                             f"{change['change'][:100]}")
                    new, mode = self._next(level, plan, code, check, led, ids, change=change, tries=1,
                                           echo=True)
            if new is None:
                return self._end(led, approach, "echo", tag)
            code = new

    def _next(self, level, plan, code, check, led, ids, change=None, improve=None, tries=2, echo=False):
        """The coder's next kernel, or None when every try returned it unchanged. Each try after the first
        is hotter and told the last answer was unchanged (`echo` makes the first one so too). The
        debugger path uses one try and asks the debugger for a different change instead; the reviewer
        path has no debugger, so it keeps two. An empty answer is passed on, so the check records it."""
        for i in range(tries):
            hot = echo or i > 0
            if improve:
                new, meta = self.coder.improve(level, plan, code, improve["profile"], improve["change"],
                                               led, echo=hot, **ids)
                mode = "improve"
            else:
                new, meta = self.coder.apply(level, plan, code, check, change, led, echo=hot, **ids)
                mode = "apply"
            if not new.strip() or new.strip() != code.strip():
                return new, mode
            self.events.write("echo", level=level, **ids, retry=i + 1 < tries, told=hot)
        return None, mode

    def _end(self, led, approach, reason, tag, detail=""):
        led.end(approach, reason, detail)
        self.events.write("thread_end", level=led.level, approach=approach.id, reason=reason,
                          detail=detail[:300], best=approach.best, attempts=approach.attempts)
        self.log(f"{tag} {approach.id} ended: {reason} (best {approach.best:.2f}, "
                 f"{approach.attempts} checks)")

    def _record(self, led, approach, tid, mode, code, check):
        slim = {k: v for k, v in check.items() if k not in ("parts",)}
        self.events.write("check", level=led.level, thread=tid, approach=approach.id, mode=mode,
                          code_sha=sha(code), code=code, **slim)
        self.events.attempt(run=led.run, level=led.level, round=approach.attempts - 1, sample=tid,
                            approach=approach.id, mode=mode, reward=check["reward"],
                            parts=check["parts"], kind=check["kind"], line=check.get("line"),
                            code_sha=sha(code), code=code, feedback=check.get("error") or "Correct.")


# How far a check got, in the order the checks run. A later stage is progress even at the same reward.
STAGES = ("empty", "parse", "rules", "lint", "load", "run", "values", "traffic", "correct")


def progress(check):
    """(reward, stage reached, shapes passed): compared as a tuple, so reward counts first."""
    stage = check.get("stage", "empty")
    return (round(check.get("reward", 0.0), 6), STAGES.index(stage) if stage in STAGES else 0,
            check.get("passed", 0))


def fix_examples(agent_mod):
    """The debugger's checked example for an error: nki_fix_examples.example_for (one example per common
    error, level-restricted, checked in nki.simulate by nki_fix_examples.py), else agent.fragment_note.
    Same signature and format either way: (error_text, level) -> note."""
    try:
        from nki_fix_examples import example_for, load_examples
        examples = load_examples()
        return lambda error_text, level: example_for(error_text, level, examples)
    except Exception:
        return agent_mod.fragment_note


def _slim(d):
    """Drop what is too big for every log line; the prompt stays, it is the experiment."""
    return {k: v for k, v in (d or {}).items() if k not in ("best_code",)}
