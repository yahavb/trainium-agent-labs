"""skills.py -- a curriculum memory: kernels the agent SOLVED become examples for later levels.

The harmful prompt example that got reverted was hand-written and off-target. A kernel the agent
itself solved on a NEIGHBOURING level is on-distribution instead: level 2's solved transpose can
help level 1, level 3's single-tile matmul can help level 4. This is the Voyager / AccelOpt
"skill library" idea, scoped to one run.

Hard rule to stay honest: NEVER seed a level with a reference_level*.py kernel or a same-level
tutorial kernel -- that is leakage, and the write-up says so out loud. We only ever reuse what the
agent produced itself, and only on a DIFFERENT level.

Off by default; enabled with --skills. Skills live under skills/ so a run's curriculum is on disk
and reproducible.
"""
import os

SKILL_DIR = "skills"


def save(level, source):
    """Persist a solved kernel. Called only when a level reaches reward 1.0."""
    if not (source or "").strip():
        return
    os.makedirs(SKILL_DIR, exist_ok=True)
    with open(os.path.join(SKILL_DIR, f"level{level}.py"), "w") as f:
        f.write(source)


def nearest(level):
    """The solved kernel from the nearest OTHER level, or '' if none. Nearest by |distance|,
    preferring a lower level (a simpler solved kernel is the safer example)."""
    if not os.path.isdir(SKILL_DIR):
        return 0, ""
    have = {}
    for fn in os.listdir(SKILL_DIR):
        if fn.startswith("level") and fn.endswith(".py"):
            try:
                have[int(fn[5:-3])] = os.path.join(SKILL_DIR, fn)
            except ValueError:
                pass
    have.pop(level, None)
    if not have:
        return 0, ""
    best = min(have, key=lambda n: (abs(n - level), n > level, abs(n - level)))
    with open(have[best]) as f:
        return best, f.read()


def example_block(level):
    """A labelled example for level N's first prompt, or '' if the library has nothing to offer.
    Labelled as a DIFFERENT operation, never 'the answer', so the model adapts rather than copies.
    """
    src_level, src = nearest(level)
    if not src:
        return ""
    return (f"Here is a correct NKI kernel you wrote earlier for a DIFFERENT operation "
            f"(level {src_level}). Use it as a template for the NKI idioms -- the allocate/dma/"
            f"compute/copy-out shape -- not as the answer, since the operation here is different:\n\n"
            f"```python\n{src.strip()}\n```")
