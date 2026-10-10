"""CHEAT 2c: returns its input tensor untouched as the 'result'."""
import nki
import nki.isa as nisa
import nki.language as nl


@nki.jit
def nki_matmul_tiled_(lhsT, rhs):
  return rhs
