"""Run the STOCK level-8 graders (nkibench.verify and agent.grade) unchanged.

The only intervention: the level-8 shape dicts carry seq/dim but the stock flops line reads
case["M"], case["K"], case["N"] and raises KeyError after the first correct case. We add those
keys (M=seq, K=dim, N=seq, the QK^T matmul) before calling. Inputs are built from seq/dim only,
so the extra keys change nothing the kernel sees.
"""
import sys
import nkibench


def patch_shapes():
    for c in nkibench.LEVELS[8]["shapes"]:
        c.setdefault("M", c["seq"]); c.setdefault("K", c["dim"]); c.setdefault("N", c["seq"])


if __name__ == "__main__":
    patch_shapes()
    path = sys.argv[1]
    rc = nkibench.verify(path, 8)
    import agent
    r, parts, fb = agent.grade(open(path).read(), 8)
    print("\nagent.grade:", r, parts, fb[:400])
    sys.exit(rc)
