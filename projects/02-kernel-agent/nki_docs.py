"""Retrieve bounded local excerpts from the neuron-nki-docs skill indices.

No model calls or SDK dependency: the controller supplies documentation to the
existing code-generation loop, rather than asking the model to execute SKILL.md.
"""

from functools import lru_cache
from pathlib import Path
import re

DEFAULT_ROOT = Path(__file__).resolve().parent / "skills" / "neuron-nki-docs"
LEVEL_SYMBOLS = {
    1: ("nl.sum", "nisa.tensor_scalar", "nisa.dma_copy"),
    2: ("nisa.dma_copy", "nl.ds", "nisa.tensor_copy"),
    8: ("nisa.activation", "nisa.nc_matmul", "nisa.tensor_reduce"),
}
SYMBOL = re.compile(r"\b(?:nisa|nl|nki\.isa|nki\.language)\.[A-Za-z_]\w*")
LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")


def _read(path):
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return ""


def _canonical(symbol):
    return symbol.replace("nisa.", "nki.isa.").replace("nl.", "nki.language.")


@lru_cache(maxsize=8)
def _symbols(root):
    index = root / "references/indices/symbol-lookup.md"
    result = {}
    for line in _read(index).splitlines():
        cells = line.split("|")
        link = LINK.search(line)
        if len(cells) < 6 or not link:
            continue
        name, module = cells[1].strip().strip("`"), cells[2].strip().strip("`")
        if module in ("nki.isa", "nki.language"):
            result[f"{module}.{name}"] = (index.parent, link.group(1))
    return result


def _excerpt(root, parent, target, symbol=""):
    relative, _, anchor = target.partition("#")
    path = (parent / relative).resolve()
    if not path.is_relative_to(root / "references"):
        return "", ""
    content = _read(path)
    if not content:
        return "", ""
    # Grouped API pages use level-three headings; their contents can also have
    # level-one headings and Python comments, so do not stop at every '#'.
    headings = list(re.finditer(r"^### .+$", content, re.M))
    for i, heading in enumerate(headings):
        title = heading.group()
        if (anchor and "{#" + anchor + "}" in title) or (
            symbol and re.match(r"^### " + re.escape(symbol) + r"(?:\s|$)", title)
        ):
            content = content[heading.end():headings[i + 1].start() if i + 1 < len(headings) else len(content)]
            break
    else:
        if symbol:
            # Some index entries point to a module overview with no signature.
            # Only use it if it contains a heading for the requested symbol.
            match = re.search(r"^#{1,6} " + re.escape(symbol) + r"(?:\s.*)?$", content, re.M)
            if not match:
                return "", ""
            start = match.end()
            end = re.search(r"^#{1,6} ", content[start:], re.M)
            content = content[start:start + end.start()] if end else content[start:]
    content = re.sub(r"\[\[source\]\]\([^)]*\)", "", content).strip()
    return str(path.relative_to(root)) + ("#" + anchor if anchor else ""), content


def docs_context(level, feedback="", source="", root=DEFAULT_ROOT, max_chars=1600):
    """Prioritize compiler errors and APIs named in feedback, then level APIs.

    Every character, including source labels, counts toward the supplied budget.
    Missing or partial skill downloads simply yield no documentation.
    """
    if max_chars < 300:
        return ""
    root = Path(root).expanduser().resolve()
    if not (root / "SKILL.md").is_file():
        return ""
    candidates = []
    errors = list(dict.fromkeys(re.findall(r"\bNCC_([A-Z]+\d+)\b", feedback)))
    for error in errors:
        candidates.append((root / "references/debugging", f"error-codes/{error}.md", ""))
    symbols = list(SYMBOL.findall(feedback))
    # Handle errors such as "nc_matmul() got an unexpected keyword argument".
    index = _symbols(root)
    for name in re.findall(r"\b([A-Za-z_]\w*)\(\)", feedback):
        symbols.extend(key for key in index if key.rsplit(".", 1)[-1] == name)
    if feedback and not symbols and not errors:
        symbols.extend(SYMBOL.findall(source))
    symbols.extend(LEVEL_SYMBOLS.get(level, ("nisa.nc_matmul", "nisa.dma_copy", "nisa.tensor_copy")))
    for symbol in dict.fromkeys(_canonical(s) for s in symbols):
        if symbol in index:
            parent, target = index[symbol]
            candidates.append((parent, target, symbol))
    header = (
        "Local NKI documentation excerpts. Installed SDK signatures, checker limits, "
        "and supplied references take precedence if this documentation differs.\n"
    )
    blocks = []
    remaining = max_chars - len(header)
    seen = set()
    for parent, target, symbol in candidates:
        label, content = _excerpt(root, parent, target, symbol)
        if not content or label in seen:
            continue
        seen.add(label)
        prefix = f"\nSource: {label}\n"
        allowance = min(1000, remaining - len(prefix))
        if allowance < 150:
            break
        excerpt = content[:allowance]
        if len(content) > allowance:
            excerpt = excerpt.rsplit("\n", 1)[0] if "\n" in excerpt else excerpt
        block = prefix + excerpt
        blocks.append(block)
        remaining -= len(block)
        if len(blocks) >= 3:
            break
    return header + "".join(blocks) if blocks else ""


def augment_prompt(prompt, level, feedback="", source="", *, root=DEFAULT_ROOT,
                   max_chars=1600, context=4096, answer_tokens=2500, terse=0):
    """Reserve answer space using the same chars/4 estimate as agent.ask()."""
    allowance = max(0, (context - answer_tokens - 128) * 4 - len(prompt) - 80)
    limit = min(max_chars, allowance, (1600, 800, 400)[min(max(terse, 0), 2)])
    docs = docs_context(level, feedback, source, root, limit)
    if not docs:
        return prompt
    return prompt + "\n\n" + docs + "\nReply with ONE python code block. No prose."
