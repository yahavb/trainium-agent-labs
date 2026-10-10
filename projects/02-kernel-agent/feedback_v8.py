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
not a slice and stays. Messages from the compiler gate (they contain gate_nki.MARK, "the trn2 compiler
rejects") are left whole: they only come after a kernel is correct on every shape, and their code is the
model's own lines rewritten for the trn2 compiler, so the slices are the fix. Level 1's rewrite is
`t[:, ...]` (all channels); blanked to `t[<…>]`, the model has only its per-channel form to go back to.

L1FIX=1. Three level-1 errors that v7 still answers with a generic sentence get the real usage, each name
checked with inspect in nki 0.6.0 on a seat: nisa.memset(dst, value), nl.zeros(shape, dtype, buffer),
nisa.tensor_tensor(dst, data1, data2, op) with op=nl.add, and nl having no reshape.

TRUNCFIX=1. v5's ask5 replaced agent.ask, and agent.ask's finish_reason check went with it: in v7's
level-1 run 1, rounds 5-7 had all four answers end at max_tokens=2500 (finish=length), the cut-off code
was graded, and the checker replied "does not parse: '(' was never closed", which the model cannot act
on. With TRUNCFIX an answer that ends with finish=length is not graded and gets no parse error; its
feedback is one instruction, TRUNC_NOTE below, and attempts.jsonl records finish / max_tokens /
completion_tokens from the server. (Wrappers on v5._post, agent.ask and agent.grade, installed only
when the switch is on.)

MIXSAMP=1. The server answers a given request body with the same text every time, and other sampling
settings give other text (measured: level-2 round 0 is 0/20 with v7's Qwen settings and 2/20 with
agent.py's, prompts byte-identical). Odd samples use agent.py's settings, even samples v7's, so each round
covers both. L2HINT=1. In v7's and the baseline's level-2 repair rounds, 81 of 134 failures were the model
transposing x itself (an index past x.shape[1] on axis 1, or the partition size changed); the messages
named the symptom only. On those errors at level 2 the feedback adds one sentence restating what the
level asks (rows stay put; each row's F1-by-F2 matrix is transposed), with no code.

Run it exactly like feedback_v7.py, with the exports in V7.md (plus SKELETON / L1FIX / TRUNCFIX / MIXSAMP /
L2HINT when testing them):

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
TRUNCFIX = os.environ.get("TRUNCFIX", "0") == "1"
MIXSAMP = os.environ.get("MIXSAMP", "0") == "1"
L2HINT = os.environ.get("L2HINT", "0") == "1"
L2_NOTE = (" This level keeps every row where it is: row p of x holds an F1-by-F2 matrix stored row-major "
           "(F1, F2 = shape2D), and row p of the output holds the same F1*F2 values of that small matrix "
           "transposed, i.e. stored column-major. Nothing moves between rows, so the first (partition) "
           "index of every read and write is the row's own index, and no tile is wider than x.shape[1]. "
           "Rewrite the kernel around that, not as a transpose of x itself.")
L2_TRIGGERS = (r"Out-of-bound access for tensor .* on dimension 1",
               r"Partition dim size must be preserved",
               r"dma_copy requires src and dst to have the same number of elements")
TRUNC_NOTE = ("Your previous answer was cut off at the token limit before the code was complete. "
              "Reply with a shorter kernel: the code block only, no comments, no explanation.")

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


import threading  # noqa: E402

_sample = threading.local()             # which sample (1..n) the current worker thread is asking for


def _ask_k(a, prompt, k):
    _sample.k = k
    return agent.ask(a, prompt)


def ask_parallel_div(a, prompt, n):
    run = getattr(a, "run", 0)
    prompts = [variant(prompt, k, n, run) for k in range(1, n + 1)]
    with cf.ThreadPoolExecutor(max_workers=n) as ex:
        return [f.result() for f in [ex.submit(_ask_k, a, p, k) for k, p in enumerate(prompts, 1)]]


agent.ask_parallel = ask_parallel_div   # solve() looks it up at call time

if MIXSAMP:
    # The server is deterministic per request: same body, same text; other sampling settings, other text.
    # So the sampling settings are a second axis of variation. Odd samples (1, 3) get their agent.py's
    # (temperature 0.6, top_p 0.95: with PROMPT1=theirs sample 1 is byte-for-byte the baseline request),
    # even samples (2, 4) Qwen3's thinking-off recipe that v7 ran with (0.7, 0.8, top_k 20).
    _post_m = v7.v5._post

    def post_mixed(a, body):
        k = getattr(_sample, "k", 1)
        body = {key: val for key, val in body.items() if key not in ("temperature", "top_p", "top_k")}
        body.update(v7.v5.THEIRS if k % 2 == 1 else v7.v5.QWEN_NOTHINK)
        return _post_m(a, body)

    v7.v5._post = post_mixed            # ask5 looks _post up at call time

if L2HINT:
    _grade_l2 = agent.grade

    def grade_l2(source, level):
        reward, parts, feedback = _grade_l2(source, level)
        if level == 2 and reward < 1 and feedback and any(re.search(t, feedback) for t in L2_TRIGGERS):
            feedback = feedback.rstrip() + L2_NOTE
        return reward, parts, feedback

    agent.grade = grade_l2

if L1FIX:
    _enrich = agent.enrich

    def enrich_l1(error_text, *args, **kw):
        fix = l1_fix(error_text)
        return error_text + fix if fix else _enrich(error_text, *args, **kw)

    agent.enrich = enrich_l1            # v2's feedback() and v6's tips call agent.enrich at call time

if SKELETON:
    import gate_nki                     # its messages carry gate_nki.MARK
    _grade = agent.grade                # v7's outermost grade (v6's timeout, then the gate)

    def grade_skeleton(source, level):
        reward, parts, feedback = _grade(source, level)
        # The compiler gate's code is the model's own lines rewritten for trn2 (all channels at once,
        # a reciprocal for divide); its slices are the fix, so its messages are left whole.
        if gate_nki.MARK in (feedback or ""):
            return reward, parts, feedback
        return reward, parts, skeletonize(feedback)

    agent.grade = grade_skeleton

if TRUNCFIX:
    import threading

    _last = threading.local()           # the samples of a round are asked from parallel threads
    _truncated = set()                  # code extracted from answers that hit max_tokens
    _post = v7.v5._post

    def post_capture(a, body):
        payload = _post(a, body)
        ch = (payload.get("choices") or [{}])[0]
        _last.meta = dict(finish=ch.get("finish_reason"), max_tokens=body.get("max_tokens"),
                          completion_tokens=(payload.get("usage") or {}).get("completion_tokens"))
        return payload

    v7.v5._post = post_capture          # ask5 looks _post up at call time

    class Reply(str):
        """The answer text, carrying the .meta that agent.solve() logs (finish, max_tokens, tokens)."""

    _ask = agent.ask                    # v5's ask5

    def ask_trunc(a, prompt):
        _last.meta = {}
        text = _ask(a, prompt)
        reply = Reply(text)
        reply.meta = dict(getattr(text, "meta", None) or {}, **_last.meta)
        if reply.meta.get("finish") == "length":
            _truncated.add(agent.extract_code(text))
        return reply

    agent.ask = ask_trunc               # ask_parallel (E-div) looks agent.ask up at call time
    _grade_t = agent.grade

    def grade_trunc(source, level):
        if source in _truncated:        # never grade a cut-off kernel, never report its parse error
            return 0.0, dict(parses=False, rules=False, runs=False, correct=False), TRUNC_NOTE
        return _grade_t(source, level)

    agent.grade = grade_trunc

if __name__ == "__main__":
    print(f"feedback v8: v7 (PROMPT1={v7.PROMPT1} MESSAGES={v7.v6.MESSAGES} CARD={v7.v5.CARD} "
          f"SAMPLING={v7.v5.SAMPLING} GATE={v7.GATE or 'off'}) + one prompt line per sample k>=2"
          f" SKELETON={int(SKELETON)} L1FIX={int(L1FIX)} TRUNCFIX={int(TRUNCFIX)} MIXSAMP={int(MIXSAMP)}"
          f" L2HINT={int(L2HINT)}")
    agent.main()
