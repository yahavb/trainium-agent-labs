"""Inputs for the v0 parity test: offline-style (the --offline replay kernels, broken and intact)."""
import hashlib, os


def h(s):
    return hashlib.sha256(s.encode()).hexdigest()


def cases(agent, have_nki=None):
    """Returns {case_name: sha256 or text}. Cases needing the simulator are only included when
    nki imports."""
    if have_nki is None:
        try:
            import nki  # noqa: F401
            have_nki = True
        except ImportError:
            have_nki = False
    out = {}
    for lv in (1, 2, 3, 4):
        ref = open(f"reference_level{lv}.py").read()
        broken = ref.replace("@nki.jit", "", 1)
        for terse in (0, 1, 2):
            out[f"first_L{lv}_t{terse}"] = h(agent.first_prompt(lv, terse))
        out[f"repair_L{lv}"] = h(agent.repair_prompt(lv, broken, "FEEDBACK"))
        r, p, fb = agent.grade(broken, lv)
        out[f"grade_broken_L{lv}"] = h(f"{r}|{sorted(p.items())}|{fb}")
        out[f"grade_syntax_L{lv}"] = h(repr(agent.grade("def (:", lv)))
        out[f"grade_empty_L{lv}"] = h(repr(agent.grade("", lv)))
        # raises inside the simulator -> exercises the exception feedback path
        if have_nki:
            r, p, fb = agent.grade(ref, lv)
            out[f"grade_ref_L{lv}"] = h(f"{r}|{sorted(p.items())}|{fb}")
            bad = ref.replace("nisa.dma_copy", "nisa.dma_copyy", 1)
            r, p, fb = agent.grade(bad, lv)
            out[f"grade_raises_L{lv}"] = h(f"{r}|{sorted(p.items())}|{fb}")
    return out
