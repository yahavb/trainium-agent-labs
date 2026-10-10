import re

_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)


def extract_code(text):
    """Last fenced block that defines a function; None if there isn't one (never the raw reply)."""
    blocks = [b for b in _FENCE.findall(text) if "def " in b]
    return blocks[-1].strip() + "\n" if blocks else None
