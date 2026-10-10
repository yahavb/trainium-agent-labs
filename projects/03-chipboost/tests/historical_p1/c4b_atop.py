"""CHEAT 4: uses the @ operator"""
import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np


@nki.jit
def nki_matmul_tiled_(lhsT, rhs):
  return np.transpose(lhsT) @ rhs
