#!/usr/bin/env python3
"""FastLoop: alternative controller for the workshop NKI kernel agent.

Place this file in /workspace/projects/03-trainium-fastloop/.
No edits to Project 2. All acceptance decisions are from its grade().
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import time

ORIGINAL = Path(__file__).resolve().parent.parent / '02-kernel-agent'
if not (ORIGINAL / 'agent.py').exists():
    raise SystemExit(f'Missing workshop project: {ORIGINAL}')
sys.path.insert(0, str(ORIGINAL))
# Run with workshop cwd because the offline reference files use relative paths.
import agent as original  # noqa: E402


def category(parts, feedback):
    """Classify the *first observed error*, not words from explanatory advice.

    The workshop checker appends examples containing shape/dimension terms even
    when the actual exception is an invalid API call.
    """
    f = (feedback or '').lower()
    if not parts['parses']:
        return 'no_code' if 'no code' in f else 'syntax'
    if not parts['rules']:
        return 'rule'
    if not parts['runs']:
        if re.search(r"(?:attributeerror:|has no attribute)\s*.*?(?:has no attribute|\ballocate\b)", f):
            return 'invalid_api'
        if 'dma_copy requires src and dst to have the same number of elements' in f:
            return 'dma_size_mismatch'
        if 'out-of-bound access' in f or 'out of bounds' in f:
            return 'out_of_bounds'
        if 'must be in' in f and any(x in f for x in ('sbuf', 'psum', 'shared_hbm')):
            return 'buffer_mismatch'
        if any(x in f for x in ('broadcast', 'shape mismatch', 'dimension mismatch', 'indexerror:')):
            return 'shape_or_index'
        return 'execution'
    if not parts['correct']:
        return 'numerical'
    return 'correct'


HINTS = {
    'no_code': 'Return one complete Python code block with imports and the @nki.jit function.',
    'syntax': 'Fix the Python syntax error without redesigning the kernel.',
    'rule': 'Fix the named NKI rule violation before changing the algorithm.',
    'invalid_api': 'Replace the nonexistent NKI API with supported primitives. For tile allocation use nl.ndarray(shape, dtype=..., buffer=nl.sbuf or nl.psum); do not use nl.allocate.',
    'dma_size_mismatch': 'Make source slice and destination tile contain EXACTLY the same number of elements; derive tile sizes from the actual input dimensions.',
    'out_of_bounds': 'Derive every slice bound from the actual input shape, including partial final tiles.',
    'buffer_mismatch': 'Use nl.sbuf for operands, nl.psum for matmul accumulation, and dma_copy for transfers involving HBM.',
    'shape_or_index': 'Correct the indexed slice or tensor shape that the exception names.',
    'execution': 'Fix the FIRST simulator exception; keep the required function signature.',
    'numerical': 'Correct the math or output indexing and preserve valid memory operations.',
}


# Only add constraints we can justify from the observed NKI failures.
# A strategy switch restarts from the reference, rather than asking the model
# to repeatedly edit a kernel that may have multiple independent defects.
SWITCH_GUIDANCE = {
    'dma_size_mismatch': (
        'For EVERY nisa.dma_copy, source and destination must have the same '
        'element count and compatible shape. Allocate each SBUF tile with '
        'nl.ndarray using the actual source slice dimensions. '
        'Do not allocate a full 128x512 tile for a 128x64 slice.'
    ),
    'invalid_api': (
        'Use ONLY actual NKI allocation APIs: nl.ndarray(shape, dtype=..., '
        'buffer=nl.sbuf) or buffer=nl.psum. '
        'Do not call nl.allocate or invent Neuron functions.'
    ),
    'buffer_mismatch': (
        'nc_matmul accumulates into nl.psum; operands must be in nl.sbuf. '
        'Use nisa.tensor_copy for PSUM-to-SBUF and nisa.dma_copy for HBM transfers.'
    ),
    'out_of_bounds': (
        'Derive slice extents from real operand shapes. Handle final partial tiles; '
        'never read or write beyond a dimension.'
    ),
    'no_code': 'Answer immediately with only imports and one complete @nki.jit kernel.',
}


def digest(src):
    # Exact normalized whitespace, not AST rewriting: different NKI source can be meaningful.
    normalized = src.replace('\r\n', '\n').strip()
    return hashlib.sha256(normalized.encode()).hexdigest()


def response_log(file, **data):
    file.write(json.dumps(data, default=str) + '\n')
    file.flush()


def solve(a, level, log, repetition):
    prompt = original.first_prompt(level, a.terse)
    full = sum(original.WEIGHTS.values())
    cache = {}
    fingerprints = {}
    history = []
    category_streak = 0
    last_selected_category = None
    transitions = []
    best = 0.0
    validations = 0
    cache_hits = 0
    generations = 0
    switches = 0
    started = time.perf_counter()
    print(f'\n[FastLoop] level={level}, repetition={repetition}')
    for rnd in range(a.rounds):
        answers = (original.offline_answers(level, a.samples, rnd) if a.offline
                   else original.ask_parallel(a, prompt, a.samples))
        generations += len(answers)
        candidates = []
        for slot, reply in enumerate(answers):
            source = original.extract_code(reply)
            key = digest(source)
            cached = key in cache
            if cached:
                reward, parts, feedback = cache[key]
                cache_hits += 1
            else:
                reward, parts, feedback = original.grade(source, level)
                cache[key] = (reward, parts, feedback)
                validations += 1
            tag = category(parts, feedback)
            record = dict(event='attempt', mode='fastloop', repetition=repetition,
                          level=level, round=rnd, sample=slot, reward=reward,
                          parts=parts, category=tag, cache_hit=cached,
                          code_sha256=key, prompt_chars=len(prompt),
                          reply_chars=len(reply), code=source, feedback=feedback)
            response_log(log, **record)
            candidates.append((reward, source, feedback, parts, key, tag, cached))
        # Prefer the highest score; on ties prefer fresh candidates over repeats.
        top = max(candidates, key=lambda x: (x[0], not x[6]))
        reward, source, feedback, parts, key, tag, cached = top
        best = max(best, reward)
        if last_selected_category is not None:
            transition = {'from': last_selected_category, 'to': tag,
                          'changed': last_selected_category != tag}
            transitions.append(transition)
            response_log(log, event='failure_transition', mode='fastloop',
                         repetition=repetition, level=level, round=rnd,
                         **transition)
            if transition['changed']:
                print(f'  failure transition: {last_selected_category} -> {tag}')
        category_streak = category_streak + 1 if tag == last_selected_category else 1
        last_selected_category = tag
        fingerprint = (tag, feedback.strip()[:180])
        fingerprints[fingerprint] = fingerprints.get(fingerprint, 0) + 1
        history.append((key, fingerprint))
        print(f'round={rnd} score={reward:.2f} best={best:.2f} '
              f'category={tag} checked={validations} cached={cache_hits}')
        if reward >= full - 1e-9:
            result = dict(event='summary', mode='fastloop', repetition=repetition,
                          level=level, solved=True, reward=reward, rounds=rnd + 1,
                          generations=generations, validations=validations,
                          cache_hits=cache_hits, strategy_switches=switches,
                          failure_transitions=transitions,
                          elapsed_sec=round(time.perf_counter()-started, 3))
            response_log(log, **result)
            return result
        # Adapt by failure category. Keep prompt concise and bounded.
        if not source.strip():
            prompt = original.first_prompt(level, min(2, a.terse + 1))
            action = 'shorten_prompt'
        else:
            hint = HINTS.get(tag, 'Address the named checker error.')
            instruction = f'{hint}\nChecker observation: {feedback[:a.feedback_chars]}'
            prompt = original.repair_prompt(level, source, instruction)
            action = 'targeted_repair'
        recent = history[-a.stagnation_window:]
        stuck = (len(recent) == a.stagnation_window and
                 len({k for k, _ in recent}) == 1)
        repeat_class = fingerprints[fingerprint] >= a.stagnation_window
        repeated_category = category_streak >= a.stagnation_window
        observed_streak = category_streak
        if stuck or repeat_class or repeated_category:
            if switches < a.max_switches:
                switches += 1
                # Restart from short reference rather than appending an unbounded ledger.
                # Explicitly name the latest fault; avoid repeating identical prompt.
                # Failure-class-aware restart: small, concrete constraints, no broken code pasted.
                fresh = original.first_prompt(level, 1)
                instruction = SWITCH_GUIDANCE.get(tag, HINTS.get(tag, 'Avoid the prior failure.'))
                prompt = (fresh + '\n\nThe previous approach repeatedly failed. ' +
                          instruction + '\nWrite a fresh implementation; do not reuse prior code. ' +
                          'Return one Python code block only.')
                action = 'strategy_switch'
                print(f'  adaptive switch #{switches}: {tag} repeated; restart with focused constraints')
                fingerprints.clear()
                category_streak = 0
            else:
                action = 'stagnation_stop'
        response_log(log, event='decision', mode='fastloop', repetition=repetition,
                     level=level, round=rnd, category=tag, action=action,
                     evidence={'repeat_code': stuck, 'repeat_failure': repeat_class,
                               'repeated_category': repeated_category, 'category_streak': observed_streak},
                     prompt_chars_next=len(prompt))
        if action == 'stagnation_stop':
            print('Stopped: repeated failure after exhausting strategy switches')
            break
    result = dict(event='summary', mode='fastloop', repetition=repetition, level=level,
                  solved=False, reward=best, rounds=rnd + 1,
                  generations=generations, validations=validations,
                  cache_hits=cache_hits, strategy_switches=switches,
                  failure_transitions=transitions,
                  elapsed_sec=round(time.perf_counter()-started, 3))
    response_log(log, **result)
    return result


def main():
    ap = argparse.ArgumentParser(description='FastLoop controller (original checker unchanged)')
    ap.add_argument('--level', type=int, default=3, choices=sorted(original.nkibench.LEVELS))
    ap.add_argument('--rounds', type=int, default=4)
    ap.add_argument('--samples', type=int, default=2)
    ap.add_argument('--repeat', type=int, default=1)
    ap.add_argument('--max-tokens', type=int, default=2500)
    ap.add_argument('--context', type=int, default=8192)
    ap.add_argument('--terse', type=int, default=0, choices=[0, 1, 2])
    ap.add_argument('--feedback-chars', type=int, default=320)
    ap.add_argument('--stagnation-window', type=int, default=2)
    ap.add_argument('--max-switches', type=int, default=1)
    ap.add_argument('--model', default=os.getenv('KERNEL_AGENT_MODEL', original.MODEL))
    ap.add_argument('--base', default=os.getenv('KERNEL_AGENT_BASE_URL') or 'http://localhost:8000/v1')
    ap.add_argument('--log', default='fastloop_attempts.jsonl')
    ap.add_argument('--think', action='store_true')
    ap.add_argument('--offline', action='store_true')
    a = ap.parse_args()
    if min(a.rounds, a.samples, a.repeat, a.stagnation_window, a.feedback_chars) < 1:
        ap.error('rounds, samples, repeat, stagnation-window and feedback-chars must be positive')
    if a.max_switches < 0:
        ap.error('max-switches must be >= 0')
    a.base = a.base.rstrip('/')
    # Guard against running offline from the project folder with no references.
    os.chdir(ORIGINAL)
    print('FastLoop uses workshop grade() and model:', a.model)
    summaries = []
    with open(a.log if os.path.isabs(a.log) else str(Path(__file__).resolve().parent / a.log), 'a') as log:
        for rep in range(1, a.repeat + 1):
            summaries.append(solve(a, a.level, log, rep))
    successes = sum(s['solved'] for s in summaries)
    print(f'VERIFIED SOLVE RATE: {successes}/{a.repeat}')
    print('Validation cache hits:', sum(s['cache_hits'] for s in summaries))
    if a.offline:
        print('NOTE: offline mode is a harness self-test; its solve rate is not experimental evidence.')


if __name__ == '__main__':
    main()
