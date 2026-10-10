"""
FINDING 5 -- the grader could score one sample with another sample's code.

The original agent.py writes every candidate kernel to one path, /tmp/_agent_level{N}.py, and
loads it with nkibench.load_kernel, which uses importlib. Importlib caches bytecode and checks the
cache only against the source's size and its modification time to the second. So a second kernel
of the same length, written within the same second, is executed from the FIRST kernel's cached
bytecode. The four samples of a round are graded back to back.

Needs no Neuron SDK:

    python findings/stale_bytecode.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nkibench  # noqa: E402

path = os.path.join(tempfile.mkdtemp(), "_agent_level3.py")
first = "def nki_matmul_basic_():\n    return 128\n"
second = "def nki_matmul_basic_():\n    return 512\n"        # different code, same length
assert len(first) == len(second)

with open(path, "w") as f:
    f.write(first)
a = nkibench.load_kernel(path, "nki_matmul_basic_")()
with open(path, "w") as f:
    f.write(second)
b = nkibench.load_kernel(path, "nki_matmul_basic_")()

print(f"first kernel returned {a}; second kernel returned {b}, its source says 512")
print("REPRODUCED: the second kernel ran the first one's code." if b == 128 else
      "not reproduced this time (the two writes straddled a second boundary); run it again.")
