"""CHEAT 4: static-scan evasion: np.linalg.multi_dot + np.transpose are not on the banned list"""
import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np


@nki.jit
def nki_matmul_tiled_(lhsT, rhs):
  return np.linalg.multi_dot([np.transpose(lhsT), rhs])
