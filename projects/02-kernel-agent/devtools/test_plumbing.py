#!/usr/bin/env python3
"""
test_plumbing.py -- does switching a checker feature on change ONLY what the model is told?

Runs the real loop in agent.py against scripted model replies and the stand-in simulator in
devtools/fake_nki, once without features and once with each. The claim under test is the one the
measurement depends on: with --features locate the model receives the located line, and the loop
still takes the same number of rounds, adds its ledger on the same round and stops for the same
reason. If that fails, a before/after comparison is measuring the loop, not the feedback.

    PYTHONPATH=devtools/fake_nki python devtools/test_plumbing.py

Plumbing only. The stand-in is not the simulator; this proves nothing about NKI.
"""

import contextlib
import io
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
os.chdir(os.path.dirname(HERE))

import agent      # noqa: E402
import diagnose   # noqa: E402


def planted(name, shift=False):
    """A reference kernel with one known bug. `shift` moves every line down by one, so the same
    bug is reported on a different line number."""
    p = next(p for p in diagnose.PLANTS if p["name"] == name)
    src = open(f"reference_level{p['level']}.py").read()
    assert src.count(p["old"]) == 1
    src = src.replace(p["old"], p["new"])
    return ("# moved down one line\n" + src) if shift else src


def run(features, script, level=3, rounds=8, hide=False):
    """Drive agent.solve with `script[round]` as every sample of that round. `hide` keeps the
    features on but withholds their text from the model, which must be the baseline exactly."""
    agent.FEATURES.clear()
    agent.FEATURES.update(features)
    prompts = []
    real_shown = diagnose.shown
    if hide:
        diagnose.shown = diagnose.key

    def scripted(a, prompt, n):
        prompts.append(prompt)
        return [f"```python\n{script[min(len(prompts) - 1, len(script) - 1)]}\n```"] * n

    real, agent.ask_parallel = agent.ask_parallel, scripted
    a = types.SimpleNamespace(rounds=rounds, samples=2, offline=False, terse=0,
                              give_up_after=4, rep=0)
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            result = agent.solve(a, level, io.StringIO())
    finally:
        agent.ask_parallel = real
        diagnose.shown = real_shown
        agent.FEATURES.clear()
    stopped = [l.strip()[:60] for l in out.getvalue().splitlines() if "STOPPING" in l]
    return result, prompts, stopped


def main():
    rc = 0
    scenarios = {
        "one bug whose line number moves every round":
            [planted("result tile of the wrong shape", shift=i % 2 == 1) for i in range(8)],
        "two bugs alternating":
            [planted("1-D tile") if i % 2 else planted("tile too small for the copy")
             for i in range(8)],
    }
    for name, script in scenarios.items():
        base_result, base_prompts, base_stop = run(set(), script)
        print(f"  {name}")
        for feats in ({"locate"}, {"locate", "state"}, {"locate", "state", "origin"},
                      {"locate", "state", "origin", "pieces", "shapes"}):
            # (internal and facts remove text, so they are checked separately below)
            tag = "+".join(sorted(feats))
            result, prompts, stop = run(feats, script)
            ok = base_result == result and base_stop == stop and len(prompts) == len(base_prompts)
            print(f"      {tag:13s} same reward, rounds and stop reason   -> {'ok' if ok else 'FAIL'}"
                  f"  {result}")
            rc |= 0 if ok else 1

            _, hidden, _ = run(feats, script, hide=True)
            ok = hidden == base_prompts
            print(f"      {tag:13s} identical prompts once its text is hidden -> "
                  f"{'ok' if ok else 'FAIL'}")
            rc |= 0 if ok else 1

            told = sum(1 for b, p in zip(base_prompts, prompts) if b != p)
            ok = told == len(prompts) - 1 and prompts[0] == base_prompts[0]
            print(f"      {tag:13s} reaches the model in every repair prompt  -> "
                  f"{'ok' if ok else 'FAIL'}  ({told} of {len(prompts) - 1})")
            rc |= 0 if ok else 1

    # facts and ahead REMOVE text, so "identical once hidden" does not apply to them. What must
    # still hold: the loop takes the same path, since which failures count as the same has not
    # changed.
    for name, script in scenarios.items():
        base_result, base_prompts, base_stop = run(set(), script)
        for feats in ({"locate", "state", "facts"}, {"locate", "state", "internal", "origin", "ahead"}):
            result, prompts, stop = run(feats, script)
            ok = base_result == result and base_stop == stop and len(prompts) == len(base_prompts)
            tag = "facts" if "facts" in feats else "ahead"
            print(f"  {tag}, {name}: same reward, rounds and stop reason -> {'ok' if ok else 'FAIL'}")
            rc |= 0 if ok else 1
    agent.FEATURES.update({"locate", "state", "facts"})
    _, _, feedback, _ = agent.grade(planted("result tile of the wrong shape"), 3)
    agent.FEATURES.clear()
    ok = "Do not reshape" not in feedback and "At that line: " in feedback and "raised " in feedback
    print(f"  grade() with facts: the state replaces the canned advice -> {'ok' if ok else 'FAIL'}")
    print(f"      {feedback}")
    rc |= 0 if ok else 1

    # internal: the advice goes only when the error came from inside the simulator.
    agent.FEATURES.update({"locate", "state", "internal"})
    _, _, numpy_error, _ = agent.grade(planted("result tile of the wrong shape"), 3)
    _, _, own_check, _ = agent.grade(planted("1-D tile"), 3)
    agent.FEATURES.clear()
    ok = ("Do not reshape" not in numpy_error and "At that line" in numpy_error
          and "A 1-D tile is not allowed" in own_check and "At that line" in own_check)
    print(f"  grade() with internal: advice dropped for a numpy error, kept for the simulator's "
          f"own check -> {'ok' if ok else 'FAIL'}")
    rc |= 0 if ok else 1

    # An invented function name breaks no shape rule, so the advice (the real names) must stay.
    invented = open("reference_level3.py").read().replace(
        "nisa.tensor_copy(dst=result_sbuf, src=result_psum)",
        "nisa.copy_out(dst=result_sbuf, src=result_psum)")
    agent.FEATURES.update({"locate", "state", "facts"})
    _, _, feedback, _ = agent.grade(invented, 3)
    agent.FEATURES.clear()
    _, _, plain, _ = agent.grade(invented, 3)
    advice = plain.split("has no attribute 'copy_out'", 1)[-1].strip()
    ok = bool(advice) and advice in feedback and "line " in feedback
    print(f"  grade() with facts: no rule broken, so the advice stays -> {'ok' if ok else 'FAIL'}")
    print(f"      {feedback[:260]}")
    rc |= 0 if ok else 1

    for feats, needle in (({"locate"}, "line "), ({"locate", "state"}, "At that line: ")):
        agent.FEATURES.clear()
        agent.FEATURES.update(feats)
        _, _, feedback, key = agent.grade(planted("result tile of the wrong shape"), 3)
        agent.FEATURES.clear()
        _, _, plain, plain_key = agent.grade(planted("result tile of the wrong shape"), 3)
        ok = (plain == plain_key == key and feedback != key and "\x00" not in feedback
              and needle in feedback and needle not in plain)
        print(f"  grade() with {'+'.join(sorted(feats))}: key is the plain text, no marks leak -> "
              f"{'ok' if ok else 'FAIL'}")
        print(f"      {feedback}")
        rc |= 0 if ok else 1

    print("\n" + ("Features change the feedback and nothing else." if rc == 0 else "SOMETHING FAILED."))
    sys.exit(rc)


if __name__ == "__main__":
    main()
