"""Feedback v5 for the kernel agent: four ideas from other agents, each behind its own switch.

Run from the event repo's projects/02-kernel-agent directory, with that directory and this file's
directory on PYTHONPATH (this imports feedback_v4, v3 and v2, which sit next to it). With every switch
at its default this is task 04's condition E exactly:

    MESSAGES=v4 CARD=reduce LOOP=theirs SAMPLING=theirs THINK=off REPAIR_PROMPT=restructure \\
        python feedback_v5.py --all --rounds 8 --samples 4 --context 8192 --repeat 5 --log ...

The switches, and where each idea comes from (sources in the Mac's work/research/):

  CARD=category   The worked example in the first prompt is chosen by the operation's category,
                  read from the level's NumPy reference: a matmul gets E's row-mean example (it took
                  level 3 to 10/10 in task 04), a reduction gets a 3-D channel mean with the channels
                  on the partition axis and the reduction over the two free axes, and anything else
                  gets no example (E's example may have cost level 2). Category-matched one-shot
                  examples are what moved low-resource kernel languages in MultiKernelBench and GEAK.
                  CARD=reduce is E's single example for every level; CARD=theirs is their card.
  MESSAGES=v5     When nc_matmul rejects an operand for being too large in any dimension (v3/v4's
                  R4K, R5 or R5b), give the whole three-loop tiling at once, with tile sizes taken
                  from the shapes, instead of one dimension per round. Compilers report every
                  diagnostic at once; in task 04 every level-4 solve walked K, then M, then N.
                  Signature phrase: "Tile all three dimensions at once". Everything else is v4.
  LOOP=lineage    Each of the --samples requests repairs its OWN previous kernel, instead of all of
                  them repairing the round's top kernel (KernelFalcon's isolated workers). A lineage
                  that returns the same failure 3 rounds running starts over from the first prompt
                  (GEAK's "debugging trap"). The level stops at the first solve; there is no other
                  early stop. Each log row also gets `lineage` and `restarts`.
  SAMPLING=qwen   Qwen3's recommended sampling for thinking-off: temperature 0.7, top_p 0.8, top_k 20.
                  Their agent sends 0.6 / 0.95, which is Qwen's recipe for thinking ON.
  THINK=round0    The first prompt of a level (round 0, and a lineage's restart) is answered with
                  thinking on, capped at THINK_BUDGET tokens (default 4000). If the thinking hits
                  the cap, Qwen's documented budget recipe closes it ("Considering the limited time
                  by the user, ...</think>") and asks for the answer in a second request
                  (continue_final_message). Repair prompts stay thinking-off.

Always on, whatever the switches:
  * the per-process grade path from task 05 (feedback_v2.py here is task 05's copy);
  * USAGE_LOG=<path>, if set: one JSON line per model request with the token counts the server
    reports, the seconds it took, the level, whether it was a first prompt, whether it thought, how the
    thinking ended, and the sha1 of the code extracted from the reply (to join with the attempts log).
"""
import ast
import concurrent.futures as cf
import hashlib
import inspect
import json
import os
import sys
import textwrap
import threading
import time

MESSAGES = os.environ.get("MESSAGES", "v4")
CARD = os.environ.get("CARD", "reduce")
LOOP = os.environ.get("LOOP", "theirs")
SAMPLING = os.environ.get("SAMPLING", "theirs")
THINK = os.environ.get("THINK", "off")
THINK_BUDGET = int(os.environ.get("THINK_BUDGET", "4000"))
ANSWER_BUDGET = int(os.environ.get("ANSWER_BUDGET", "1500"))
USAGE_LOG = os.environ.get("USAGE_LOG", "")
_ok = dict(MESSAGES=("v4", "v5"), CARD=("theirs", "reduce", "category"), LOOP=("theirs", "lineage"),
           SAMPLING=("theirs", "qwen"), THINK=("off", "round0"))
for _k, _v in _ok.items():
    if globals()[_k] not in _v:
        sys.exit(f"{_k} must be one of {_v}, got {globals()[_k]!r}")

# feedback_v4 installs v4's messages and (with CARD=reduce) E's card at import; v5 decides the card.
os.environ["MESSAGES"], os.environ["CARD"] = "v4", "theirs"
import feedback_v4 as v4  # noqa: E402  (first: it sets up v3's and v2's switches before importing them)
import feedback_v3 as v3  # noqa: E402
import feedback_v2 as v2  # noqa: E402
import agent            # their agent.py, unchanged  # noqa: E402
import nkibench         # their checker, unchanged   # noqa: E402
os.environ["MESSAGES"], os.environ["CARD"] = MESSAGES, CARD


# ------------------------------------------------------------------ CARD=category

CARD_REDUCE3D = """
More real functions, and one worked example:

  nisa.tensor_tensor(dst=, data1=, data2=, op=nl.add)            two tiles of the same shape, elementwise
  nisa.tensor_scalar(dst=, data=, op0=nl.multiply, operand0=0.5) a tile and a number, elementwise
  Python operators (+ - * /) do not work on tiles; use the two calls above. The ops are nl.add,
  nl.subtract, nl.multiply and nl.maximum.

Mean of each channel of a (C, A, B) tensor x, C <= 128. The channels go on the partition axis, and
the sum runs over the two free axes. Never reduce axis 0, the partition axis.

    C, A, B = x.shape
    out = nl.ndarray((C, 1, 1), dtype=x.dtype, buffer=nl.shared_hbm)
    t = nl.ndarray((C, A, B), dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=t, src=x)
    s = nl.sum(t, axis=[1, 2], keepdims=True)        # (C, 1, 1)
    m = nl.ndarray((C, 1, 1), dtype=x.dtype, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=m, data=s, op0=nl.multiply, operand0=1.0 / (A * B))
    nisa.dma_copy(dst=out, src=m)
"""

BASE_CARD = agent.API_CARD
ADDITION = {"matmul": v4.CARD_ADDITION, "reduce": CARD_REDUCE3D, "other": ""}
REDUCERS = {"sum", "mean", "max", "min", "prod", "amax", "amin", "average"}
MATMULS = {"matmul", "dot", "einsum", "tensordot", "inner", "vdot"}


def category(level):
    """'matmul', 'reduce' or 'other', read from the code of the level's NumPy reference (docstring
    dropped). A matmul anywhere wins, so attention counts as a matmul."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(nkibench.LEVELS[level]["ref"])))
    fn = tree.body[0]
    if fn.body and isinstance(fn.body[0], ast.Expr) and isinstance(fn.body[0].value, ast.Constant):
        fn.body = fn.body[1:]
    names = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
            return "matmul"
        if isinstance(node, ast.Call):
            f = node.func
            names.add(f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", ""))
    if names & MATMULS:
        return "matmul"
    if names & REDUCERS:
        return "reduce"
    return "other"


_their_first_prompt = agent.first_prompt


def first_prompt_category(level, terse=0):
    agent.API_CARD = BASE_CARD + ADDITION[category(level)]   # first_prompt reads it at call time
    return _their_first_prompt(level, terse)


# ------------------------------------------------------------------ MESSAGES=v5

ONE_DIM = ("hold the whole contraction dimension",       # R4K / R4K+
           "but nc_matmul takes at most 128 there",       # R5
           "but nc_matmul takes at most 512 there")       # R5b


def tile_all(source, line):
    """The full M x N x K tiling as code in the kernel's own names, or None if the kernel is not a
    two-input lhsT/rhs-style matmul we can read.

    Task 07 fixes two things task 06 found. If the kernel passes ONE tile as both operands, the code
    now uses two tiles (lhs_t, rhs_t). And the code now includes the output allocation and replaces the
    whole body, because "keep the allocation of out" kept a hard-coded (128, 512) output once."""
    ps = v3.params(source)
    names = v3.matmul_names(source, line if "nc_matmul" in line else None)
    if len(ps) != 2 or not names:
        return None
    S0, Mv0, P = names["stationary"], names["moving"], names["dst"]
    sS, sM = v3.hbm_source(source, S0, ps[0]), v3.hbm_source(source, Mv0, ps[1])
    if sS == sM and S0 == Mv0:          # one tile for both operands: sources can't be told apart
        sS, sM = ps[0], ps[1]
    S, Mv = (("lhs_t", "rhs_t") if S0 == Mv0 else (S0, Mv0))
    out = v3.output_name(source)
    if sS == sM or "res" in (S, Mv, P, out):
        return None
    return (f"`{S0}` and `{Mv0}` are bigger than one nc_matmul takes: at most 128 rows of K, at most "
            f"128 columns of the stationary operand, and at most 512 columns of the moving operand. "
            f"Tile all three dimensions at once. Replace everything between the def line and the "
            f"return with this code, so the return stays `return {out}`:\n\n"
            f"    {out} = nl.ndarray(({sS}.shape[1], {sM}.shape[1]), dtype={sS}.dtype, buffer=nl.shared_hbm)\n"
            f"    TK = min(128, {sS}.shape[0])\n"
            f"    TM = min(128, {sS}.shape[1])\n"
            f"    TN = min(512, {sM}.shape[1])\n"
            f"    for m in nl.affine_range({sS}.shape[1] // TM):\n"
            f"        for n in nl.affine_range({sM}.shape[1] // TN):\n"
            f"            {P} = nl.ndarray((TM, TN), dtype=nl.float32, buffer=nl.psum)\n"
            f"            for k in nl.affine_range({sS}.shape[0] // TK):\n"
            f"                {S} = nl.ndarray((TK, TM), dtype={sS}.dtype, buffer=nl.sbuf)\n"
            f"                {Mv} = nl.ndarray((TK, TN), dtype={sM}.dtype, buffer=nl.sbuf)\n"
            f"                nisa.dma_copy(dst={S}, src={sS}[k * TK:(k + 1) * TK, m * TM:(m + 1) * TM])\n"
            f"                nisa.dma_copy(dst={Mv}, src={sM}[k * TK:(k + 1) * TK, n * TN:(n + 1) * TN])\n"
            f"                nisa.nc_matmul(dst={P}, stationary={S}, moving={Mv})\n"
            f"            res = nl.ndarray((TM, TN), dtype={out}.dtype, buffer=nl.sbuf)\n"
            f"            nisa.tensor_copy(dst=res, src={P})\n"
            f"            nisa.dma_copy(dst={out}[m * TM:(m + 1) * TM, n * TN:(n + 1) * TN], src=res)\n")


def advise_v5(err, line, source):
    tip = v4.advise_safe(err, line, source)
    if tip and any(s in tip for s in ONE_DIM):
        try:
            return tile_all(source, line) or tip
        except Exception as e:  # noqa: BLE001
            print(f"    [v5 advise failed, using v4: {type(e).__name__}: {e}]")
    return tip


# ------------------------------------------------------------------ requests: sampling, thinking, usage

THEIRS = dict(temperature=0.6, top_p=0.95)                 # exactly what their ask() sends
QWEN_NOTHINK = dict(temperature=0.7, top_p=0.8, top_k=20)  # Qwen3 model card, thinking off
THINKING = dict(temperature=0.6, top_p=0.95, top_k=20)     # Qwen3 model card, thinking on
FORCE_END = ("\n\nConsidering the limited time by the user, I have to give the solution based on the "
             "thinking directly now.\n</think>\n\n")
_usage_lock = threading.Lock()


def _post(a, body):
    import httpx
    r = httpx.post(f"{a.base.rstrip('/')}/chat/completions", json=body, timeout=900, verify=False)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:400]}")
    return r.json()


def _split_think(msg, finish):
    """(thinking, answer, closed) from a reply, with or without a reasoning parser on the server.
    closed: the thinking ended by itself, so whatever follows it is the answer."""
    content = msg.get("content") or ""
    reasoning = next((msg[k] for k in agent.REASONING_KEYS if msg.get(k)), "")
    if "</think>" in content:
        r, _, ans = content.partition("</think>")
        return (reasoning + r.replace("<think>", "", 1)).strip("\n"), ans, True
    if content.lstrip().startswith("<think>"):          # stopped while still thinking
        return (reasoning + content.replace("<think>", "", 1)).strip("\n"), "", False
    # A reasoning parser on the server split them. Its thinking ended if any answer came, or if the
    # model stopped by itself.
    return reasoning.strip("\n"), content, bool(content.strip()) or finish == "stop"


def _level_of(prompt):
    for n, s in nkibench.LEVELS.items():
        if f"`{s['entry']}`" in prompt or f"for {s['op']} is not right" in prompt:
            return n
    return None


def ask5(a, prompt):
    """Their ask(), plus the SAMPLING and THINK switches and the usage log. With SAMPLING=theirs and
    a repair prompt (or THINK=off) the request body is byte-for-byte theirs."""
    t0 = time.time()
    first = not prompt.startswith("This NKI kernel for")
    think = THINK == "round0" and first
    est_prompt = len(prompt) // 4
    msgs = [{"role": "user", "content": prompt}]
    rec = dict(t=round(t0, 2), pid=os.getpid(), level=_level_of(prompt), first=first, think=think,
               prompt_chars=len(prompt))
    if not think:
        budget = min(a.max_tokens, max(256, a.context - est_prompt - 64))
        body = dict(model=a.model, messages=msgs, max_tokens=budget,
                    **(QWEN_NOTHINK if SAMPLING == "qwen" else THEIRS),
                    chat_template_kwargs={"enable_thinking": False})
        try:
            payload = _post(a, body)
        except RuntimeError as e:
            raise SystemExit(f"the endpoint returned {e}") from e
        ch = payload["choices"][0]
        reply = (ch.get("message") or {}).get("content") or ""
        rec.update(finish=ch.get("finish_reason"), usage=payload.get("usage"))
        if ch.get("finish_reason") == "length":
            print(f"    (TRUNCATED: finish_reason=length after {len(reply)} chars; prompt is "
                  f"{len(prompt)} chars)")
    else:
        budget = max(256, min(THINK_BUDGET, a.context - est_prompt - 64 - ANSWER_BUDGET))
        body = dict(model=a.model, messages=msgs, max_tokens=budget, **THINKING,
                    chat_template_kwargs={"enable_thinking": True})
        try:
            payload = _post(a, body)
        except RuntimeError as e:
            raise SystemExit(f"the endpoint returned {e}") from e
        ch = payload["choices"][0]
        fin = ch.get("finish_reason")
        thinking, answer, closed = _split_think(ch.get("message") or {}, fin)
        rec.update(think_finish=fin, usage=payload.get("usage"), think_chars=len(thinking),
                   think_closed=closed, think_budget=budget)
        if fin == "stop":
            reply, rec["path"] = answer, "finished"
        else:
            if closed:
                # It finished thinking but the answer hit the cap: let it finish the answer.
                cont = "<think>\n" + thinking.strip("\n") + "\n</think>\n\n" + answer.lstrip("\n")
                rec["path"] = "answer-continued"
            else:
                # Qwen's thinking-budget recipe: close the thinking for it, then let it answer.
                cont = "<think>\n" + thinking.strip("\n") + FORCE_END
                rec["path"] = "forced"
            body2 = dict(model=a.model, max_tokens=ANSWER_BUDGET, **THINKING,
                         messages=msgs + [{"role": "assistant", "content": cont}],
                         continue_final_message=True, add_generation_prompt=False,
                         chat_template_kwargs={"enable_thinking": True})
            try:
                p2 = _post(a, body2)
                ch2 = p2["choices"][0]
                m2 = ch2.get("message") or {}
                more = (m2.get("content") or "") or next(
                    (m2[k] for k in agent.REASONING_KEYS if m2.get(k)), "")
                reply = (answer.lstrip("\n") + more) if closed else more
                rec.update(answer_finish=ch2.get("finish_reason"), answer_usage=p2.get("usage"))
            except RuntimeError as e:
                print(f"    [{rec['path']} request failed: {e}]")
                reply = answer if closed else ""
                rec.update(path=rec["path"] + "-failed", error=str(e)[:300])
    rec["seconds"] = round(time.time() - t0, 2)
    rec["reply_chars"] = len(reply)
    rec["code_sha1"] = hashlib.sha1(agent.extract_code(reply).encode()).hexdigest()
    if USAGE_LOG:
        with _usage_lock, open(USAGE_LOG, "a") as f:
            f.write(json.dumps(rec) + "\n")
    return reply


# ------------------------------------------------------------------ LOOP=lineage

def solve_lineage(a, level, log):
    print(f"\n=========== level {level}: {nkibench.LEVELS[level]['op']} ===========")
    full = sum(agent.WEIGHTS.values())
    n = a.samples
    first = agent.first_prompt(level, a.terse)
    lines = [dict(prompt=first, terse=a.terse, streak=0, last=None, tried=[], restarts=0)
             for _ in range(n)]
    best = 0.0
    for rnd in range(a.rounds):
        t0 = time.perf_counter()
        prompts = [L["prompt"] for L in lines]
        if a.offline:
            replies = [agent.offline_answers(level, 1, rnd)[0] for _ in range(n)]
        else:
            with cf.ThreadPoolExecutor(max_workers=n) as ex:
                replies = list(ex.map(lambda p: agent.ask(a, p), prompts))
        top = (-1.0, "", "")
        for i, (L, reply) in enumerate(zip(lines, replies)):
            src = agent.extract_code(reply)
            reward, parts, fb = agent.grade(src, level)
            log.write(json.dumps(dict(level=level, round=rnd, reward=reward, parts=parts,
                                      prompt_chars=len(prompts[i]), reply_chars=len(reply),
                                      code=src, feedback=fb, lineage=i,
                                      restarts=L["restarts"])) + "\n")
            if reward > top[0]:
                top = (reward, src, fb)
            if not src.strip():                    # nothing to repair: re-ask, shorter (as theirs)
                L.update(terse=min(L["terse"] + 1, 2), streak=0, last=None)
                L["prompt"] = agent.first_prompt(level, L["terse"])
                continue
            L["streak"] = L["streak"] + 1 if fb == L["last"] else 1
            L["last"] = fb
            L["tried"].append(fb)
            if L["streak"] >= 3:                   # the debugging trap: start this lineage over
                L.update(prompt=first, terse=a.terse, streak=0, last=None, tried=[],
                         restarts=L["restarts"] + 1)
                continue
            p = agent.repair_prompt(level, src, fb)
            if L["streak"] == 2:                   # their ledger, per lineage
                ledger = "\n".join(f"- {t[:160]}" for t in dict.fromkeys(L["tried"]))
                p += (f"\n\nThese approaches have already failed, so do something different:\n"
                      f"{ledger}")
            L["prompt"] = p
        log.flush()
        best = max(best, top[0])
        print(f"round {rnd}: this round {top[0]:.2f}  best so far {best:.2f}  "
              f"({time.perf_counter() - t0:.1f}s)")
        print(f"  {top[2][:400]}")
        if top[0] >= full - 1e-9:
            print(f"  SOLVED on round {rnd}. {top[2]}")
            print("  ---------------- the kernel ----------------")
            print(textwrap.indent(top[1], "  "))
            print("  -------------------------------------------")
            return top[0], rnd + 1
    print(f"  not solved in {a.rounds} rounds; best reward {best:.2f}")
    return best, a.rounds


# ------------------------------------------------------------------ install

if MESSAGES == "v5":
    v2.advise = advise_v5            # v2.feedback() looks advise up at call time
if CARD == "reduce":
    agent.API_CARD = BASE_CARD + v4.CARD_ADDITION
elif CARD == "category":
    agent.first_prompt = first_prompt_category   # solve() looks it up at call time
if LOOP == "lineage":
    agent.solve = solve_lineage      # main() looks it up at call time
_their_ask = agent.ask               # kept for test_requests.py's comparison
agent.ask = ask5                     # ask_parallel() looks it up at call time

if __name__ == "__main__":
    print(f"feedback v5: MESSAGES={MESSAGES} CARD={CARD} LOOP={LOOP} SAMPLING={SAMPLING} "
          f"THINK={THINK}" + (f" THINK_BUDGET={THINK_BUDGET}" if THINK != "off" else "")
          + f" REPAIR_PROMPT={v3.PROMPT}" + (f" USAGE_LOG={USAGE_LOG}" if USAGE_LOG else ""))
    agent.main()
