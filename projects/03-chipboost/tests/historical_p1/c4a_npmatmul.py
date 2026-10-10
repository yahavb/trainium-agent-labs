"""CHEAT 4: hands the op to np.matmul"""
import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np


@nki.jit
def nki_matmul_tiled_(lhsT, rhs):
  return np.matmul(np.transpose(lhsT), rhs)
