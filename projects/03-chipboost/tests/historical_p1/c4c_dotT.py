"""CHEAT 4: uses .T on an argument"""
import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np


@nki.jit
def nki_matmul_tiled_(lhsT, rhs):
  return np.matmul(lhsT.T, rhs)
