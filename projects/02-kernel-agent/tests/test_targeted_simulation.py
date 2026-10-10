"""Tiny independent CPU SDK reproductions; no benchmark answers or device use."""
import unittest
import numpy as np
import nki
import nki.language as nl
import nki.isa as ni

@nki.jit
def reduction_sum(a, keep):
    tile=nl.ndarray(a.shape,a.dtype,buffer=nl.sbuf)
    ni.dma_copy(dst=tile,src=a)
    result=nl.sum(tile,axis=1,keepdims=keep)
    out=nl.ndarray((2,1),a.dtype,buffer=nl.shared_hbm)
    ni.dma_copy(dst=out,src=result)
    return out

@nki.jit
def reduction_max(a, keep):
    tile=nl.ndarray(a.shape,a.dtype,buffer=nl.sbuf)
    ni.dma_copy(dst=tile,src=a)
    result=nl.max(tile,axis=1,keepdims=keep)
    out=nl.ndarray((2,1),a.dtype,buffer=nl.shared_hbm)
    ni.dma_copy(dst=out,src=result)
    return out

@nki.jit
def wrong_result(a,b):
    left=nl.ndarray(a.shape,a.dtype,buffer=nl.sbuf)
    right=nl.ndarray(b.shape,b.dtype,buffer=nl.sbuf)
    ni.dma_copy(dst=left,src=a)
    ni.dma_copy(dst=right,src=b)
    dst=nl.ndarray((1,2),nl.float32,buffer=nl.psum)
    ni.nc_matmul(dst=dst,stationary=left,moving=right)
    return dst

class SimulatorCauseTests(unittest.TestCase):
    def test_reduction_rank_loss_and_valid_keepdims(self):
        a=np.arange(6,dtype=np.float32).reshape(2,3)
        for kernel,ref in ((reduction_sum,np.sum),(reduction_max,np.max)):
            with self.assertRaisesRegex(AssertionError,'at least 2 dimensions'):
                nki.simulate(kernel)(a,False)
            np.testing.assert_allclose(nki.simulate(kernel)(a,True),ref(a,axis=1,keepdims=True))
    def test_matmul_internal_reshape_without_explicit_source_reshape(self):
        a=np.ones((4,2),np.float32);b=np.ones((4,3),np.float32)
        with self.assertRaisesRegex(ValueError,'cannot reshape array of size 6'):
            nki.simulate(wrong_result)(a,b)

if __name__=='__main__':unittest.main()
