"""Tests for agents2/index.py: the described index and fix_name(). No nki or model needed, except
test_every_described_name_exists, which only runs where the real nki is installed.

    python tests/test_index.py          (or pytest tests/test_index.py)
"""

import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
os.chdir(HERE)

from agents2.index import DESCRIBED, HEADER, described, fix_name  # noqa: E402


class Stub:
    """Just enough of a Retriever: a set of real names, and cards withheld at some levels."""

    def __init__(self, names, withheld=None, mode=None):
        self.names, self.withheld = set(names), withheld or {}
        if mode:
            self.index_mode = mode

    def short(self, dotted):
        return dotted

    def get(self, dotted):
        return (lambda *a: None) if dotted in self.names else None

    def exists(self, dotted):
        return dotted in self.names

    def shown(self, dotted, level):
        return level not in self.withheld.get(dotted, ())

    def api_map(self, level=None):
        return "every name"


ALL = {n for _, _, _, names, _, _ in DESCRIBED for n in names}


def test_fix_name_moves_a_real_name_to_its_module():
    r = Stub({"nisa.tensor_reduce", "nl.multiply", "tile.reshape", "nl.mean"})
    assert fix_name(r, "nl.tensor_reduce") == "nisa.tensor_reduce"
    assert fix_name(r, "nisa.multiply") == "nl.multiply"
    assert fix_name(r, "nl.reshape") == "tile.reshape"
    assert fix_name(r, "nl.mean") == "nl.mean"                     # already right
    assert fix_name(r, "nl.tensor_load") == "nl.tensor_load"       # exists nowhere: left for the re-plan
    r = Stub({"tile.permute"}, withheld={"tile.permute": (2,)})
    assert fix_name(r, "nl.permute", 1) == "tile.permute"
    assert fix_name(r, "nl.permute", 2) == "nl.permute"             # withheld here: not offered


def test_described_follows_levels_withholding_and_installed_names():
    r = Stub(ALL, withheld={"tile.permute": (2,)})
    l1, l2, l4 = described(r, 1), described(r, 2), described(r, 4)
    assert l1.startswith(HEADER) and l1.startswith("NKI names")    # test_agents2 looks for this label
    assert "nc_matmul" not in l1 and "nc_matmul" in l4             # matmul lines from level 3
    assert "t.permute" in l1 and "t.permute" not in l2             # withheld at level 2
    assert "ACROSS partitions" in l1                               # the description that steers away
    r = Stub(ALL - {"nisa.tensor_partition_reduce"})
    assert "tensor_partition_reduce" not in described(r, 1)        # not installed: not offered
    assert "for i in range(n)" in described(r, 1)                  # lines without names always show


def test_index_section_modes():
    from agents2.lookup import index_section
    assert "every name" in index_section(Stub(ALL, mode="names"), 1).text
    assert index_section(Stub(ALL), 1).text.startswith(HEADER)     # described is the default


def test_every_described_name_exists():
    try:
        import nki
    except ImportError:
        return
    if getattr(nki, "__version__", "") == "0.0-fake":
        return
    from agents2.retriever import Retriever
    r = Retriever()
    missing = sorted(n for n in ALL if not r.exists(n))
    assert not missing, f"not in nki {r.version}: {missing}"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"ok    {t.__name__}")
        except Exception as e:
            failed += 1
            import traceback
            print(f"FAIL  {t.__name__}: {e!r}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
