"""agent_feedback_v4.py -- v3 plus a shape doctor for the matmul levels.

Place beside agent.py, agent_feedback_v2.py and agent_feedback_v3.py.

    python agent_feedback_v4.py --check-feedback
    python agent_feedback_v4.py --level 3 --rounds 8 --samples 4 --context 8192 --repeat 5 \
        --log attempts-feedback-v4-L3.jsonl

Why. Under v3 the level-3 kernel was right except for one line, M, K = lhsT.shape, which swaps K
and M. Every tile downstream was then half the right size and the checker said only:

    dma_copy requires src and dst to have the same number of elements, got src=65536, dst=32768

The checker's feedback already carries "On K=128 M=64 N=512", so the sidecar can work out that
65536 is rhs, the whole (K, N) tensor, while 32768 matches no legal shape -- and therefore that K
or N was derived wrongly. That inference is what this version adds to the repair prompt, plus the
explicit unpacking line in the level-3/4 direction text.
"""

import re
import sys

import agent_feedback_v2 as v2
import agent_feedback_v3 as v3

UNPACK_LINE = (" Read the sizes with K, M = lhsT.shape and K2, N = rhs.shape -- K is the FIRST "
               "dimension of both inputs, M and N are the second.")

for _lv in (3, 4):
    v2.DIRECTION[_lv] = v2.DIRECTION[_lv] + UNPACK_LINE
for _lv in (5, 6, 7):
    v2.DIRECTION[_lv] = v2.DIRECTION[4]


def shape_doctor(feedback):
    """For a matmul-level dma_copy size mismatch, name which tensor the counts belong to."""
    lab = re.search(r"On K=(\d+) M=(\d+) N=(\d+)", feedback)
    cnt = re.search(r"got src=(\d+), dst=(\d+)", feedback)
    if not (lab and cnt):
        return ""
    K, M, N = (int(g) for g in lab.groups())
    src, dst = (int(g) for g in cnt.groups())
    names = {K * M: f"lhsT, the whole (K, M) = ({K}, {M}) tensor",
             K * N: f"rhs, the whole (K, N) = ({K}, {N}) tensor",
             M * N: f"the result, (M, N) = ({M}, {N})"}

    def what(n):
        return names.get(n, "no legal shape in this problem")

    lines = [f"Shape doctor for this case: K={K}, M={M}, N={N}, so lhsT.shape is (K, M) = ({K}, {M}), "
             f"rhs.shape is (K, N) = ({K}, {N}), and the result is (M, N) = ({M}, {N}).",
             f"The source of the failing copy has {src} elements: that is {what(src)}.",
             f"The tile you allocated for it has {dst} elements: that is {what(dst)}."]
    if dst not in names:
        lines.append("A tile whose size matches none of the three shapes means a size variable was "
                     "derived wrongly before any copy happened. The usual cause is unpacking in the "
                     "wrong order: it must be K, M = lhsT.shape (K first) and K2, N = rhs.shape. "
                     "Fix the unpacking, then make every tile exactly the shape of what it holds.")
    elif src != dst:
        lines.append("Both sizes are legal shapes but different ones. Either you are copying one "
                     "tensor into a tile meant for another -- give each operand its own tile and the "
                     "result its own (M, N) tiles in psum and sbuf -- or, if the tile is already "
                     "written symbolically as (K, N) or (K, M), then K itself holds the wrong number: "
                     "it must come from K, M = lhsT.shape (K FIRST), never M, K = lhsT.shape.")
    return "\n".join(lines)


def install_feedback(base):
    originals = v3.install_feedback(base)
    prev_repair = base.repair_prompt

    def repair_prompt(level, source, feedback):
        p = prev_repair(level, source, feedback)
        if level >= 3 and "dma_copy requires src and dst" in feedback:
            doc = shape_doctor(feedback)
            if doc:
                i = p.rfind("Reply with")
                p = (p[:i] + doc + "\n\n" + p[i:]) if i >= 0 else p + "\n\n" + doc
        return p

    base.repair_prompt = repair_prompt
    return originals


def check_feedback(base):
    originals = install_feedback(base)
    try:
        fb = ("0 of 1 shapes passed. On K=128 M=64 N=512: raised AssertionError: dma_copy requires "
              "src and dst to have the same number of elements, got src=65536, dst=32768 The tile ...")
        doc = shape_doctor(fb)
        assert "that is rhs, the whole (K, N) = (128, 512)" in doc
        assert "the result, (M, N) = (64, 512)" in doc and "K FIRST" in doc
        fb0 = fb.replace("dst=32768", "dst=16384")
        assert "no legal shape" in shape_doctor(fb0) and "K first" in shape_doctor(fb0)
        p = base.repair_prompt(3, "def f(): pass", fb)
        assert "Shape doctor" in p and p.find("Shape doctor") < p.rfind("Reply with")
        print("repair_prompt: shape doctor attached for matmul dma_copy mismatches")
        fb2 = ("1 of 4 shapes passed. On K=256 M=256 N=1024: raised AssertionError: dma_copy requires "
               "src and dst to have the same number of elements, got src=65536, dst=16384")
        d2 = shape_doctor(fb2)
        assert "lhsT, the whole (K, M) = (256, 256)" in d2 and "no legal shape" in d2
        print("shape doctor: level-4 case resolves src to lhsT")
        assert "Shape doctor" not in base.repair_prompt(1, "x", fb)
        assert "Shape doctor" not in base.repair_prompt(3, "x", "NUMERICAL MISMATCH")
        print("shape doctor: silent on level 1 and on non-dma errors")
        for lv in (3, 4):
            assert v2.DIRECTION[lv].endswith(UNPACK_LINE)
        assert "THREE NESTED LOOPS" in v2.DIRECTION[4]
        print("direction: unpack line appended; v3 nested-loop text intact")
        print("\n=== shape doctor text ===")
        print(doc)
        print("\nFEEDBACK CHECK PASSED (routing only; no kernel/model run)")
    finally:
        base.enrich, base.first_prompt, base.repair_prompt = originals


if __name__ == "__main__":
    import agent as base

    if sys.argv[1:] == ["--check-feedback"]:
        check_feedback(base)
    else:
        install_feedback(base)
        print("Feedback experiment v4: v3 + matmul shape doctor", flush=True)
        base.main()
