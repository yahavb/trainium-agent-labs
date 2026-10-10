"""Feedback v8: v7, plus a different prompt per sample (E-div), plus two switches that are off by default.

E-div (always on). Measured on the seats (2026-10-10, 14:30): the vLLM-Neuron server returns
byte-identical completions for identical requests whatever the sampling settings -- 4 of 4 identical at
temperature 0.7, and the same with n=4 in one request; a `seed` field returns HTTP 500 and kills the
engine. In the logs, 89 of 106 baseline rounds had four identical samples, and v7's level-4 runs were one
trajectory replayed five times. So --samples 4 bought one sample per round, and --repeat 5 one run.
Sample 1 keeps the prompt byte-for-byte; sample k >= 2 gets one line appended to the user message,
"(attempt k of n, run r)". Ties still go to the earliest sample, i.e. sample 1.

SKELETON=1. Every repair message that carries code keeps its loops, calls, API names and variable names,
but the slices and indices in [...], the shapes given to nl.ndarray / nl.zeros, and the arguments of
affine_range(...) become <…>, and one sentence after the code says what goes there. The prose of the
message is unchanged; messages without code are unchanged; the API-name corrections (v3's R7, "Only the
token ... is wrong", and agent.py's KNOWN_FIXES) are unchanged. `.shape[i]` (choosing a dimension) is
not a slice and stays.

L1FIX=1. Three level-1 errors that v7 still answers with a generic sentence get the real usage, each name
checked with inspect in nki 0.6.0 on a seat: nisa.memset(dst, value), nl.zeros(shape, dtype, buffer),
nisa.tensor_tensor(dst, data1, data2, op) with op=nl.add, and nl having no reshape.

Run it exactly like feedback_v7.py, with the exports in V7.md (plus SKELETON / L1FIX when testing them):

    python3 feedback_v8.py --level 4 --rounds 8 --samples 4 --context 8192 --repeat 5 \\
        --log Ediv_L4.jsonl --verdicts verdicts_div_L4.jsonl

The token split in attempts.jsonl is computed from sample 1's prompt; samples k >= 2 carry about ten
more prompt tokens than it says.
"""
import concurrent.futures as cf
import os
import re

SKELETON = os.environ.get("SKELETON", "0") == "1"
L1FIX = os.environ.get("L1FIX", "0") == "1"

# ------------------------------------------------------------------ pure text helpers (no nki needed)

HOLE = "<…>"
SKELETON_NOTE = ("Each <…> is left for you to fill: a slice, a tile shape or a loop count that follows "
                 "from the limits named above (at most 128 rows per tile, at most 512 columns for the "
                 "moving operand). Keep the structure.")


def _blank_first_arg(code, call, all_args=False):
    """Replace the first argument (or all of them) of every `call(` in code with HOLE."""
    out, i = [], 0
    while True:
        j = code.find(call, i)
        if j < 0:
            return "".join(out) + code[i:]
        start = j + len(call)
        depth, k = 0, start
        while k < len(code):
            c = code[k]
            if c in "([{":
                depth += 1
            elif c in ")]}":
                if depth == 0:
                    break
                depth -= 1
            elif c == "," and depth == 0 and not all_args:
                break
            k += 1
        out.append(code[i:start] + HOLE)
        i = k


def skeleton_line(code):
    """One line of suggested code with its slices, tile shapes and loop counts blanked."""
    for call in ("nl.ndarray(", "nl.zeros("):
        code = _blank_first_arg(code, call)
    code = _blank_first_arg(code, "affine_range(", all_args=True)
    return re.sub(r"(?<!\.shape)\[[^\[\]]*\]", f"[{HOLE}]", code)


def skeletonize(feedback):
    """The feedback with every indented code block turned into a skeleton, plus one sentence after it."""
    if not feedback or "Only the token `" in feedback:
        return feedback
    lines = feedback.split("\n")
    if not any(l.startswith("    ") and l.strip() for l in lines):
        return feedback
    out, in_block = [], False
    for l in lines:
        is_code = l.startswith("    ") and bool(l.strip())
        if in_block and not is_code:
            out += ["", SKELETON_NOTE]
        out.append(skeleton_line(l) if is_code else l)
        in_block = is_code
    if in_block:
        out += ["", SKELETON_NOTE]
    return "\n".join(out)


L1_FIXES = [
    (r"module 'nki\.isa' has no attribute 'nc_clear'",
     " There is no nc_clear. To zero a tile write nisa.memset(dst=t, value=0.0), or allocate it "
     "already zeroed with nl.zeros(shape, dtype, buffer=nl.sbuf) (buffer=nl.psum for a psum tile)."),
    (r"module 'nki\.language' has no attribute 'reshape'",
     " nl has no reshape. Do not reshape: take the piece you need by slicing the tensor or the tile, "
     "and allocate the destination with exactly that slice's shape."),
    (r"unsupported operand type\(s\) for \+=: 'float' and 'NkiTensor'",
     " A tile cannot be added to a Python number. Keep the running sum in a tile: allocate it once "
     "with nl.zeros(shape, dtype=nl.float32, buffer=nl.sbuf) and add each tile into it with "
     "nisa.tensor_tensor(dst=acc, data1=acc, data2=t, op=nl.add)."),
]


def l1_fix(error_text):
    """The L1FIX sentence for this error, or None."""
    for pattern, fix in L1_FIXES:
        if re.search(pattern, error_text):
            return fix
    return None


# ------------------------------------------------------------------ v7 wiring

import feedback_v7 as v7  # noqa: E402  (after the pure helpers, so they can be tested without nki)

agent = v7.agent


def variant(prompt, k, n, run):
    """Sample k's prompt: sample 1 unchanged, the others with one line naming the attempt."""
    return prompt if k == 1 else f"{prompt}\n\n(attempt {k} of {n}, run {run + 1})"


def ask_parallel_div(a, prompt, n):
    run = getattr(a, "run", 0)
    prompts = [variant(prompt, k, n, run) for k in range(1, n + 1)]
    with cf.ThreadPoolExecutor(max_workers=n) as ex:
        return [f.result() for f in [ex.submit(agent.ask, a, p) for p in prompts]]


agent.ask_parallel = ask_parallel_div   # solve() looks it up at call time

if L1FIX:
    _enrich = agent.enrich

    def enrich_l1(error_text, *args, **kw):
        fix = l1_fix(error_text)
        return error_text + fix if fix else _enrich(error_text, *args, **kw)

    agent.enrich = enrich_l1            # v2's feedback() and v6's tips call agent.enrich at call time

if SKELETON:
    _grade = agent.grade                # v7's outermost grade (v6's timeout, then the gate)

    def grade_skeleton(source, level):
        reward, parts, feedback = _grade(source, level)
        return reward, parts, skeletonize(feedback)

    agent.grade = grade_skeleton

if __name__ == "__main__":
    print(f"feedback v8: v7 (PROMPT1={v7.PROMPT1} MESSAGES={v7.v6.MESSAGES} CARD={v7.v5.CARD} "
          f"SAMPLING={v7.v5.SAMPLING} GATE={v7.GATE or 'off'}) + one prompt line per sample k>=2"
          f" SKELETON={int(SKELETON)} L1FIX={int(L1FIX)}")
    agent.main()
