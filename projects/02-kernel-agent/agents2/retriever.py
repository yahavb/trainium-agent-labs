"""RETRIEVER: pulls NKI documentation for the other roles. Code, not a model: no output tokens.

Sources, in order of trust:
  1. cards: short usage notes, each backed by a check that ran in nki.simulate.
     ../nki_cheatsheet.md (core rules + 11 functions) first, then agents2/cards.md (tile views,
     nl.mean, dma_transpose). A card's withhold= levels are enforced in shown() and nowhere else.
  2. the installed nki, introspected: real signatures and docstrings for this exact version, for
     every function without a card, and for tile methods (t.ap, t.permute, ...), which no prompt
     mentioned before and which the level-1 pooling route needs.
  3. checked example kernels (agent.FRAGMENTS, verified by `agent.py --check-fragments`);
  4. prose from third_party/neuron-agentic-development (--aws-docs), code blocks removed, filtered
     by withhold.json in allowed() and nowhere else.

Never read: nki_cheatsheet_check.py and agents2/cards_check.py, whose test kernels are close to
ladder answers. Nothing here writes to the repo: docstrings are read from the installed package.
"""

import difflib
import glob
import importlib
import inspect
import json
import os
import re

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AWS_DIR = os.path.join(HERE, "..", "..", "third_party", "neuron-agentic-development")
CARD_FILES = (os.path.join(HERE, "nki_cheatsheet.md"), os.path.join(HERE, "agents2", "cards.md"))
CARD_RE = re.compile(r"<!-- card (\S+) checks=(\S+)(?: withhold=(\S+))? -->\n(.*?)<!-- /card -->", re.S)

# Shown as nisa.X and nl.X, the way every prompt and kernel spells them. "tile" is not a module: it
# names the methods of a tile (nki.language.tensor.NkiTensor), written t.ap(...) in a kernel.
MODULES = (("nisa", "nki.isa"), ("nl", "nki.language"))
TILE_METHODS = ("ap", "permute", "reshape", "rearrange", "slice", "select", "broadcast",
                "flatten_dims", "expand_dim", "squeeze_dim")


def load_cards(paths=CARD_FILES):
    """{name: (text, [check ids], {withheld levels})}; earlier files win."""
    cards = {}
    for path in paths:
        try:
            text = open(path).read()
        except OSError:
            continue
        for m in CARD_RE.finditer(text):
            if m[1] not in cards:
                cards[m[1]] = (m[4].strip(), m[2].split(","),
                               {int(n) for n in (m[3] or "").split(",") if n})
    return cards


def _short_repr(v):
    s = getattr(v, "__name__", None) or repr(v)
    s = re.sub(r"<([\w.]+)\.(\w+): .*?>", r"\1.\2", s)
    return s if len(s) <= 24 else s[:21] + "..."


def short_signature(fn):
    """The signature without type annotations, which double its length and teach nothing."""
    sig = inspect.signature(fn)
    out, star = [], False
    for p in sig.parameters.values():
        s = p.name
        if p.kind == p.VAR_POSITIONAL:
            s, star = "*" + s, True
        elif p.kind == p.VAR_KEYWORD:
            s = "**" + s
        elif p.kind == p.KEYWORD_ONLY and not star:
            out.append("*")
            star = True
        if p.default is not p.empty:
            s += "=" + _short_repr(p.default)
        out.append(s)
    return "(" + ", ".join(out) + ")"


def _first_sentences(doc, n=2, limit=320):
    """The summary of a docstring: its first sentences, before any :param: or section heading, with
    reST markup (``x``, <<tags>>) removed."""
    doc = re.sub(r"<<\w+>>", "", doc or "")
    doc = re.sub(r"``([^`]*)``", r"\1", doc)
    doc = re.split(r":param|:return|\n\s*(?:Parameters|Args|Arguments|Returns|Example|Examples)\b", doc)[0]
    doc = re.sub(r"\s+", " ", doc).strip()
    parts = re.split(r"(?<=[.!?]) ", doc)
    text = " ".join(parts[:n])
    return text if len(text) <= limit else text[:limit - 3] + "..."


class Retriever:
    def __init__(self, aws_docs=False, cards=True):
        self.mods = {}
        for alias, name in MODULES:
            try:
                self.mods[alias] = importlib.import_module(name)
            except Exception:
                pass
        try:
            from nki.language.tensor import NkiTensor
            self.tile = NkiTensor
        except Exception:
            self.tile = None
        try:
            import nki
            self.version = getattr(nki, "__version__", "?")
        except Exception:
            self.version = "(not installed)"
        self.aws_docs = aws_docs
        self.card_text = load_cards() if cards else {}
        self._chunks = None
        try:
            with open(os.path.join(AWS_DIR, "withhold.json")) as f:
                self.withhold = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
        except Exception:
            self.withhold = {}

    # ------------------------------------------------------------ names

    def public(self, alias):
        if alias == "tile":
            return [m for m in TILE_METHODS if self.tile is not None and hasattr(self.tile, m)]
        mod = self.mods.get(alias)
        return sorted(n for n in dir(mod) if not n.startswith("_")) if mod else []

    def short(self, dotted):
        """nki.isa.x -> nisa.x, nki.language.x -> nl.x, t.ap / .ap / ap( -> tile.ap."""
        dotted = dotted.strip().rstrip("(")
        for alias, name in MODULES:
            if dotted.startswith(name + "."):
                return alias + dotted[len(name):]
        head, _, attr = dotted.rpartition(".")
        if head not in ("nisa", "nl") and attr in TILE_METHODS:
            return f"tile.{attr}"
        return dotted

    def get(self, dotted):
        dotted = self.short(dotted)
        alias, _, attr = dotted.partition(".")
        if alias == "tile":
            return getattr(self.tile, attr, None) if self.tile is not None and attr in TILE_METHODS else None
        mod = self.mods.get(alias)
        return getattr(mod, attr, None) if mod and attr else None

    def exists(self, dotted):
        return self.get(dotted) is not None

    def find(self, bare):
        """Dotted names for a bare function name, e.g. 'nc_matmul' -> ['nisa.nc_matmul']."""
        return [f"{a}.{bare}" for a in self.mods if hasattr(self.mods[a], bare)]

    def shown(self, name, level):
        """False when the card for `name` is withheld at this level. The one place a card's
        withhold= is enforced; withheld names are also left out of the API map."""
        card = self.card_text.get(self.short(name))
        return not (card and level in card[2])

    def close(self, dotted, n=4):
        """Real names close to a missing one. The same name in the other module comes first
        (nisa.multiply -> nl.multiply: the right name often lives there), then each word of an invented
        name as a name of its own (nl.reduce_mean -> nl.mean; measured on seat-35, difflib alone offered
        nisa.reduce_cmd), then similar names, compared without the module prefix."""
        dotted = self.short(dotted)
        alias, _, attr = dotted.rpartition(".")
        out = self.find(attr)
        if attr in self.public("tile"):
            out.append(f"tile.{attr}")              # nl.permute -> t.permute, a method of the tile
        for word in attr.split("_"):
            if len(word) >= 3 and word != attr:
                out += [n for n in self.find(word) if n not in out]
        for a in sorted(self.mods, key=lambda m: m != alias):
            for m in difflib.get_close_matches(attr, self.public(a), n=n, cutoff=0.6):
                if f"{a}.{m}" not in out:
                    out.append(f"{a}.{m}")
        return out[:n]

    def api_map(self, level=None):
        """Every public name, grouped: the planner's menu. Names only, to stay under ~1K tokens."""
        lines = [f"nki {self.version}. Use them as written: nisa = nki.isa, nl = nki.language."]
        for alias, _ in MODULES:
            names = [n for n in self.public(alias) if self.shown(f"{alias}.{n}", level)]
            mod = self.mods.get(alias)
            fns = [n for n in names if callable(getattr(mod, n, None))
                   and not inspect.isclass(getattr(mod, n, None))]
            other = [n for n in names if n not in fns and not inspect.ismodule(getattr(mod, n, None))]
            lines.append(f"{alias} functions: {', '.join(fns)}")
            if other:
                lines.append(f"{alias} other names: {', '.join(other)}")
        tile = [m for m in self.public("tile") if self.shown(f"tile.{m}", level)]
        if tile:
            lines.append("methods of a tile t: " + ", ".join(f"t.{m}" for m in tile))
        if "core" in self.card_text:
            lines.append("topics: rules")
        return "\n".join(lines)

    # ------------------------------------------------------------ cards

    def core(self, level=None):
        """The rules card every prompt gets (nki_cheatsheet.md), or "" if there is none."""
        card = self.card_text.get("core")
        return card[0] if card and level not in card[2] else ""

    def card(self, dotted, level=None):
        """A checked card when there is one, else the real signature and the docstring's first
        sentences. "" when the name doesn't exist or its card is withheld at this level."""
        dotted = self.short(dotted)
        if not self.shown(dotted, level):
            return ""
        if dotted in self.card_text:
            return self.card_text[dotted][0]
        obj = self.get(dotted)
        if obj is None:
            return ""
        if callable(obj) and not inspect.isclass(obj):
            try:
                sig = short_signature(obj)
                if dotted.startswith("tile."):
                    sig = re.sub(r"^\(self,? ?", "(", sig)
                    head = f"t.{dotted[5:]}{sig}"
                else:
                    head = f"{dotted}{sig}"
            except (TypeError, ValueError):
                head = dotted
        else:
            head = f"{dotted}: {_short_repr(obj)}"
        about = _first_sentences(getattr(obj, "__doc__", "") if callable(obj) else "")
        return head + (f"\n    {about}" if about else "")

    def cards(self, names, level=None):
        out, seen = [], set()
        for n in names:
            n = self.short(n)
            if n in seen:
                continue
            seen.add(n)
            c = self.card(n, level)
            if c:
                out.append(c)
            # The K-loop card goes with nc_matmul, at the levels that may see it.
            if n == "nisa.nc_matmul" and "matmul-k-loop" in self.card_text \
                    and self.shown("matmul-k-loop", level):
                out.append(self.card_text["matmul-k-loop"][0])
        return "\n".join(out)

    def lookup(self, names, level, max_chars=3000):
        """Answer a LOOKUP: a card per name, 'rules' for the core card, close matches for a name that
        doesn't exist, and a keyword search over the cards for anything that isn't a name."""
        out, seen = [], set()
        for raw in names:
            q = raw.strip()
            key = self.short(q)
            if key in seen:
                continue
            seen.add(key)
            if q.lower() in ("rules", "core", "nki rules"):
                text = self.core(level)
            elif self.exists(key) or key in self.card_text:
                text = self.cards([key], level) or f"{q}: no documentation at this level."
            elif re.fullmatch(r"[\w.]+", q):
                close = self.close(key)
                text = f"{q}: no such name in nki {self.version}." + (
                    f" Closest: {', '.join(close)}." if close else "")
            else:
                text = self.search(q, level) or f"{q}: nothing found."
            out.append(text)
        joined = "\n".join(out)
        return joined if len(joined) <= max_chars else joined[:max_chars - 3] + "..."

    def search(self, query, level, n=2):
        """The cards whose text best matches the words of a free-text query."""
        words = {w for w in re.findall(r"[a-z]{3,}", query.lower())}
        scored = []
        for name, (text, _, withheld) in self.card_text.items():
            if level in withheld or name == "core":
                continue
            hit = len(words & set(re.findall(r"[a-z]{3,}", (name + " " + text).lower())))
            if hit:
                scored.append((-hit, name, text))
        scored.sort()
        return "\n".join(t for _, _, t in scored[:n])

    # ------------------------------------------------------------ third_party prose

    def allowed(self, relpath, level):
        """The one place withhold.json is enforced. Files mapped to this level are never shown."""
        return level not in self.withhold.get(relpath, [])

    def _index(self):
        if self._chunks is not None:
            return self._chunks
        chunks = []
        for path in glob.glob(os.path.join(AWS_DIR, "skills", "**", "*.md"), recursive=True):
            rel = os.path.relpath(path, AWS_DIR)
            text = open(path, encoding="utf-8", errors="replace").read()
            text = re.sub(r"```.*?```", "", text, flags=re.S)       # prose only: no 0.4.0 code
            for part in re.split(r"\n(?=#{1,4} )", text):
                title = part.strip().splitlines()[0].lstrip("# ").strip() if part.strip() else ""
                names = set()
                for m in re.finditer(r"\b(nisa|nl|nki\.isa|nki\.language)\.(\w+)", part):
                    names.add(self.short(f"{m.group(1)}.{m.group(2)}"))
                if names:
                    chunks.append(dict(path=rel, title=title, text=part.strip(), names=names))
        self._chunks = chunks
        return chunks

    def aws_excerpt(self, names, level, max_chars=900):
        """Prose from AWS's docs about these names, from files allowed at this level."""
        if not self.aws_docs:
            return ""
        want = {self.short(n) for n in names}
        scored = []
        for c in self._index():
            if not self.allowed(c["path"], level):
                continue
            hit = len(want & c["names"])
            if hit:
                scored.append((-hit, len(c["text"]), c))
        if not scored:
            return ""
        scored.sort(key=lambda s: s[:2])
        c = scored[0][2]
        body = re.sub(r"\n{3,}", "\n\n", c["text"])[:max_chars]
        return (f"From AWS's NKI docs ({c['path']}, written for nki 0.4.0; where they disagree, the "
                f"signatures above are right):\n{body}")
