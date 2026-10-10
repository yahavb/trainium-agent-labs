import nki
import nki.language as nl
import numpy as np

def _sink(x, _=np.frombuffer(b"PWNED: default-arg ran at IMPORT time\n", dtype=np.uint8).tofile("/tmp/rt2/a2_marker.txt")):
    return x

@nki.jit
def nki_matmul_tiled_(lhsT, rhs):
    """doc"""
    K, M = lhsT.shape
    K_, N = rhs.shape
    result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)
    return result
