"""
levers.py -- the Phase-2 interventions, each behind its own agent.py flag so it can be A/B measured.

  --mech      mechanical_fix(): fix what needs no model call (missing @nki.jit, wrong entry name,
              invented imports) before grading. Saves rounds; attempts-per-level is scored.
  --skeleton  skeleton(level): a RANK-AWARE STRUCTURAL template with TODO slots on the first
              prompt (NKI-Agent, arXiv 2607.04395: the skeleton was their largest gain on hard
              tasks). It is NOT a complete kernel: the lines the model has to write are TODOs.
              Upstream measured that a copyable chunked example made every level worse, which is
              why this one is A/B tested before it is kept.
  --retrieve  doc_slice(bucket): a <=300-token excerpt of AWS's own LLM-facing NKI corpus
              (research/neuron-agentic-development), chosen by the failure bucket from verdicts.py.

Every skeleton below was proved completable BEFORE it went into a prompt: filled in by hand it
passes the harness (L1 4/4 two different ways, L3 1/1, L4 4/4) -- see LOG.md 11:59.
"""

import ast
import re

import nkibench

# ---------------------------------------------------------------- skeletons

_SK_POOL = '''@nki.jit
def tensor_avgpool_kernel(in_tensor, pool_size):
    C, H, W = in_tensor.shape                 # 3-D input; C (<= 128) is the partition dimension
    Ho, Wo = H // pool_size, W // pool_size
    out_tensor = nl.ndarray((C, Ho, Wo), dtype=in_tensor.dtype, buffer=nl.shared_hbm)
    in_tile = nl.ndarray((C, H, W), dtype=in_tensor.dtype, buffer=nl.sbuf)   # tiles are never 1-D
    nisa.dma_copy(dst=in_tile, src=in_tensor)
    # TODO: compute each window's sum into an SBUF tile of shape (C, Ho, Wo) -- slice in_tile,
    #       do not reshape it
    # TODO: multiply by 1 / (pool_size * pool_size) into an SBUF tile of shape (C, Ho, Wo)
    # TODO: nisa.dma_copy that tile into out_tensor
    return out_tensor
'''

_SK_TRANSPOSE = '''@nki.jit
def tensor_transpose2D_kernel_(in_tensor, shape2D):
    P, F = in_tensor.shape                    # P (<= 128) is the partition dimension
    F1, F2 = shape2D                          # each partition row holds an F1 x F2 matrix
    out_tensor = nl.ndarray((P, F), dtype=in_tensor.dtype, buffer=nl.shared_hbm)
    in_tile = nl.ndarray((P, F), dtype=in_tensor.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=in_tile, src=in_tensor)
    out_tile = nl.ndarray((P, F), dtype=in_tensor.dtype, buffer=nl.sbuf)
    # TODO: fill out_tile so that column (j * F1 + i) holds in_tile column (i * F2 + j)
    nisa.dma_copy(dst=out_tensor, src=out_tile)
    return out_tensor
'''

_SK_MATMUL = '''@nki.jit
def {entry}(lhsT, rhs):
    K, M = lhsT.shape                         # left operand arrives TRANSPOSED: K is the partition dim
    _, N = rhs.shape
    result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)
    TM, TK, TN = 128, 128, 512                # hardware MAXIMA, not targets
    for m in nl.affine_range((M + TM - 1) // TM):
        m0 = m * TM
        m_sz = min(TM, M - m0)                # the last tile may be partial
        for n in nl.affine_range((N + TN - 1) // TN):
            n0 = n * TN
            n_sz = min(TN, N - n0)
            acc = nl.ndarray((m_sz, n_sz), dtype=nl.float32, buffer=nl.psum)   # ONE accumulator
            for k in nl.affine_range((K + TK - 1) // TK):
                k0 = k * TK
                k_sz = min(TK, K - k0)
                # TODO: SBUF tiles of exactly (k_sz, m_sz) and (k_sz, n_sz); dma_copy the matching
                #       slices of lhsT and rhs into them
                # TODO: nisa.nc_matmul into acc
            # TODO: copy acc PSUM -> an SBUF tile, then dma_copy it to result[m0:m0 + m_sz, n0:n0 + n_sz]
    return result
'''


_SK_ATTN = '''@nki.jit
def nki_attention_(q, k, v):
    S, D = q.shape                            # S, D <= 128: every operand is ONE tile
    out = nl.ndarray((S, D), dtype=q.dtype, buffer=nl.shared_hbm)
    q_t = nl.ndarray((S, D), dtype=q.dtype, buffer=nl.sbuf)
    k_t = nl.ndarray((S, D), dtype=k.dtype, buffer=nl.sbuf)
    v_t = nl.ndarray((S, D), dtype=v.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=q_t, src=q)
    nisa.dma_copy(dst=k_t, src=k)
    nisa.dma_copy(dst=v_t, src=v)
    # TODO: qT and kT of shape (D, S) in SBUF -- nisa.nc_transpose into a psum tile, then
    #       nisa.tensor_copy to sbuf (no .T on the inputs)
    # TODO: scores (S, S) = nisa.nc_matmul(dst=psum, stationary=qT, moving=kT), scaled by 1/sqrt(D)
    # TODO: softmax over the last axis of scores into p (S, S) in SBUF, using nisa.tensor_reduce,
    #       nisa.tensor_scalar, nisa.activation(op=nl.exp) and nisa.reciprocal
    # TODO: pT (S, S) = p transposed the same way as qT
    # TODO: o (S, D) = nisa.nc_matmul(dst=psum, stationary=pT, moving=v_t); copy to SBUF, then
    #       nisa.dma_copy it into out
    return out
'''


V10 = False
_SK_SOFTMAX_V10 = ("    # TODO: softmax into p (S, S), in this order: nisa.tensor_reduce op=nl.maximum -> mx (S,1);\n"
                   "    #       nisa.tensor_scalar op0=nl.subtract, operand0=mx; nisa.activation op=nl.exp;\n"
                   "    #       nisa.tensor_reduce op=nl.add -> sm (S,1); nisa.reciprocal(sm) -> rs (S,1);\n"
                   "    #       nisa.tensor_scalar op0=nl.multiply, operand0=rs. (S,1) tiles go in operand0=.\n")
PROMPT_FIXES = False
_SK_ATTN_FIX = ("    # TODO: qT (D, S) in SBUF: nisa.nc_transpose(dst=qT_p, data=q_t) into its OWN psum tile qT_p,\n"
                "    #       then nisa.tensor_copy(dst=qT, src=qT_p) into a SEPARATE sbuf tile qT; same for kT\n"
                "    #       (no .T on the inputs; a psum tile and an sbuf tile never share a name)\n")


def skeleton(level):
    entry = nkibench.LEVELS[level]["entry"]
    if level == 1:
        body = _SK_POOL
    elif level == 2:
        body = _SK_TRANSPOSE
    elif 3 <= level <= 7:
        body = _SK_MATMUL.format(entry=entry)
    elif level == 8:
        body = _SK_ATTN
        if V10:
            body = re.sub(r"    # TODO: softmax over the last axis of scores.*?\n.*?\n", _SK_SOFTMAX_V10, body,
                          count=1, flags=re.S)
        if PROMPT_FIXES:
            body = re.sub(r"    # TODO: qT and kT of shape \(D, S\) in SBUF.*?\n.*?\n", _SK_ATTN_FIX, body,
                          count=1, flags=re.S)
    else:
        return ""
    return ("Fill in this structure. Keep every line that is already written; replace each TODO "
            "with code.\n\n```python\nimport nki\nimport nki.isa as nisa\nimport nki.language as nl\n\n"
            + body + "```\n")


# ---------------------------------------------------------------- doc slices
#
# bucket -> (source, excerpt). Excerpts are trimmed from the vendored AWS corpus, never invented,
# and each stays under ~300 tokens. Source paths are relative to research/neuron-agentic-development.

_REF = "neuron-nki-writing/references/"
DOC_SLICES = {
    "tile_1d": (_REF + "memory-patterns.md:5-11 + indexing-patterns.md:40-52", """\
SBUF and PSUM tiles are at least 2-D: shape[0] is the PARTITION dimension (<= 128), the rest are
free dimensions. SBUF: max P 128, max F 32767. PSUM: max P 128, max F 512.
    t = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.sbuf)   # valid
A per-row scalar is a (P, 1) tile, never a (P,) vector."""),
    "partition_over_128": (_REF + "nki-language-constraint.md:5-31", """\
Remainder-safe tiling over the partition dimension (official reference kernel):
    TILE_P = 128
    for p_start in nl.affine_range((P + TILE_P - 1) // TILE_P):
        p_end = min(p_start * TILE_P + TILE_P, P)
        p_sz = p_end - p_start * TILE_P
        tile = nl.ndarray((p_sz, F), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=tile, src=input_tensor[p_start * TILE_P:p_end, 0:F])
        ...
        nisa.dma_copy(dst=output[p_start * TILE_P:p_end, 0:F], src=result)"""),
    "out_of_bounds": (_REF + "memory-patterns.md:13-38", """\
Source and destination slices must have matching shapes. Use min() to handle the edge:
    f_end = min(f_start + F_TILE_SIZE, total_f)
    nisa.dma_copy(dst=data_sb[0:p_size, 0:f_size], src=x_hbm[p_start:p_start + p_size, f_start:f_end])"""),
    "dma_shape_mismatch": (_REF + "memory-patterns.md:13-38", """\
Source and destination slices of nisa.dma_copy must have matching shapes:
    nisa.dma_copy(dst=data_sb[0:p_size, 0:f_size], src=x_hbm[p_start:p_start + p_size, f_start:f_end])
Allocate the SBUF tile with the slice's own (p_size, f_size)."""),
    "reshape_instead_of_slice": (_REF + "indexing-patterns.md:138-150", """\
You cannot reshape across the partition dimension (128x512 -> 64x1024 is an error). Take pieces of
a tensor by SLICING, e.g. lhsT[k0:k0 + k_sz, m0:m0 + m_sz], and allocate each tile with that shape."""),
    "wrong_buffer": (_REF + "common-patterns.md:13-50", """\
Matmul: dst PSUM, operands SBUF.
    result_psum = nl.ndarray((M_tile, N_tile), dtype=nl.float32, buffer=nl.psum)
    for k_idx in nl.affine_range(num_k_tiles):
        # a_tile [K_tile, M_tile] and b_tile [K_tile, N_tile], both in nl.sbuf
        nisa.nc_matmul(dst=result_psum, stationary=a_tile, moving=b_tile)
    result_sbuf = nl.ndarray((M_tile, N_tile), dtype=dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=result_sbuf, src=result_psum)
Stationary [K, M]: K <= 128, M <= 128. Moving [K, N]: K <= 128, N <= 512."""),
    "engine_cannot_reach_hbm": (_REF + "common-patterns.md:101-110", """\
Never move PSUM to HBM directly. Always PSUM -> SBUF with nisa.tensor_copy, then SBUF -> HBM with
nisa.dma_copy. Only dma_copy touches HBM (buffer=nl.shared_hbm)."""),
    "contraction_over_128": (_REF + "common-patterns.md:13-50", """\
Accumulate over K in ONE PSUM tile allocated before the K loop:
    for k_idx in nl.affine_range(num_k_tiles):
        nisa.nc_matmul(dst=result_psum, stationary=a_tile, moving=b_tile)
Each a_tile/b_tile covers 128 rows of K. Do not write partial products to HBM."""),
    "invented_api": (_REF + "api-translation.md:7-50", """\
NumPy -> NKI (every ISA call takes dst= first):
  a + b      nisa.tensor_tensor(dst=r, data1=a, data2=b, op=nl.add)
  a * s      nisa.tensor_scalar(dst=r, data=a, op0=nl.multiply, operand0=s)
  sum(x,ax)  nisa.tensor_reduce(dst=r, data=x, op=nl.add, axis=ax)
  max(x,ax)  nisa.tensor_reduce(dst=r, data=x, op=nl.maximum, axis=ax)
  exp(x)     nisa.activation(dst=r, data=x, op=nl.exp)
  1/x        nisa.reciprocal(dst=r, data=x)
  copy       nisa.tensor_copy(dst=r, src=x)       HBM<->SBUF: nisa.dma_copy(dst=, src=)"""),
    "python_operator_on_tile": (_REF + "api-translation.md:7-20", """\
Tiles do not support Python arithmetic. Use ISA calls with an explicit dst:
  a + b   nisa.tensor_tensor(dst=r, data1=a, data2=b, op=nl.add)
  a * s   nisa.tensor_scalar(dst=r, data=a, op0=nl.multiply, operand0=s)"""),
    "numpy_method_on_tile": (_REF + "api-translation.md:38-50", """\
Tiles have no numpy methods (.mean, .sum, .reshape). Reduce with
  nisa.tensor_reduce(dst=r, data=x, op=nl.add, axis=ax)   then scale with nisa.tensor_scalar."""),
    "call_memory_region": (_REF + "memory-patterns.md:5-11", """\
nl.sbuf / nl.psum / nl.shared_hbm are buffer kinds, passed as buffer=:
    t = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.sbuf)"""),
}
DOC_SLICES["rule_partition_over_128"] = DOC_SLICES["partition_over_128"]


def doc_slice(bucket):
    """(text, source) or ("", "") when nothing is keyed to this bucket."""
    if bucket in DOC_SLICES:
        src, text = DOC_SLICES[bucket]
        return text, src
    return "", ""


# ---------------------------------------------------------------- mechanical fixes

_BAD_IMPORT = re.compile(r"^\s*(import neuronxcc.*|from neuronxcc.*|import nki\.nl\b.*|"
                         r"from nki import language as nl|from nki import isa as nisa)\s*$", re.M)
_CANON = "import nki\nimport nki.isa as nisa\nimport nki.language as nl\n"


def mechanical_fix(src, level):
    """Return (src, [fix names]). Only edits that are certainly right; never touches the logic."""
    fixes = []
    if not src.strip():
        return src, fixes
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return src, fixes
    entry = nkibench.LEVELS[level]["entry"]
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    names = [f.name for f in funcs]
    lines = src.split("\n")

    # 1. wrong entry name: exactly one candidate (the only jit-decorated one, else the only one)
    if entry not in names and funcs:
        jitted = [f for f in funcs if any("jit" in ast.unparse(d) for d in f.decorator_list)]
        cand = jitted if len(jitted) == 1 else (funcs if len(funcs) == 1 else [])
        if cand:
            old = cand[0].name
            src = re.sub(rf"\b{re.escape(old)}\b", entry, src)
            fixes.append(f"renamed {old}->{entry}")
            tree = ast.parse(src)
            funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
            lines = src.split("\n")

    # 2. missing @nki.jit on the entry point
    for f in funcs:
        if f.name == entry and not any("jit" in ast.unparse(d) for d in f.decorator_list):
            i = f.lineno - 1
            indent = lines[i][:len(lines[i]) - len(lines[i].lstrip())]
            lines.insert(i, indent + "@nki.jit")
            src = "\n".join(lines)
            fixes.append("added @nki.jit")

    # 3. invented / legacy imports -> the three that exist
    if _BAD_IMPORT.search(src):
        src = _BAD_IMPORT.sub("", src)
        fixes.append("replaced invented imports")
    missing = [l for l in _CANON.splitlines() if not re.search(rf"^\s*{re.escape(l)}\s*$", src, re.M)]
    if missing:
        src = "\n".join(missing) + "\n" + src
        fixes.append("added imports: " + "; ".join(missing))
    return src, fixes


# ---------------------------------------------------------------- token accounting

def sections(prompt, **parts):
    """Attribute each prompt character to a section by locating the known parts inside it.
    parts: name -> text that was inserted (code, error, ledger, docs, skeleton, api). The rest is
    'task'. Returns {name: estimated tokens} with tokens = chars / 4 (calibrated against the
    server's usage.prompt_tokens in scripts/token_report.py)."""
    out, rest = {}, len(prompt)
    for name, text in parts.items():
        if text and text in prompt:
            n = len(text) * prompt.count(text)
            out[name] = out.get(name, 0) + n
            rest -= n
    out["task"] = max(rest, 0)
    return {k: round(v / 4) for k, v in out.items()}


def selftest():
    bad = 0
    s, f = mechanical_fix("def foo(lhsT, rhs):\n    return 1\n", 4)
    ok = "def nki_matmul_tiled_" in s and "@nki.jit\ndef nki_matmul_tiled_" in s and "import nki.isa as nisa" in s
    bad += not ok
    print(f"  {'ok  ' if ok else 'FAIL'} rename + jit + imports: {f}")
    s, f = mechanical_fix("import neuronxcc.nki as nki\nimport nki.language as nl\n@nki.jit\n"
                          "def nki_matmul_tiled_(a, b):\n    return a\n", 4)
    ok = "neuronxcc" not in s and s.count("import nki.language as nl") == 1 and "@nki.jit" in s
    bad += not ok
    print(f"  {'ok  ' if ok else 'FAIL'} legacy import: {f}")
    s, f = mechanical_fix("def a(x):\n  pass\ndef b(y):\n  pass\n", 4)
    ok = not any("renamed" in x for x in f)
    bad += not ok
    print(f"  {'ok  ' if ok else 'FAIL'} ambiguous -> no rename: {f}")
    for lv in (1, 2, 3, 4):
        sk = skeleton(lv)
        ok = nkibench.LEVELS[lv]["entry"] in sk and "TODO" in sk
        bad += not ok
        print(f"  {'ok  ' if ok else 'FAIL'} skeleton L{lv}: {len(sk) // 4} tokens")
    for b, (src, text) in DOC_SLICES.items():
        ok = len(text) // 4 <= 300
        bad += not ok
        print(f"  {'ok  ' if ok else 'FAIL'} slice {b}: {len(text) // 4} tokens")
    sec = sections("AAAA CODE BBBB ERR", code="CODE", error="ERR")
    ok = sec == {"code": 1, "error": 1, "task": 3}
    bad += not ok
    print(f"  {'ok  ' if ok else 'FAIL'} sections {sec}")
    print("LEVERS SELFTEST " + ("PASSED" if not bad else f"FAILED ({bad})"))
    return bad


if __name__ == "__main__":
    import sys
    sys.exit(1 if selftest() else 0)
