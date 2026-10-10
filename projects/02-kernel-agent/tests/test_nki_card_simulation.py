"""Tiny generic API exercises on CPU, not benchmark kernels or live grade()."""
import os
import unittest
from unittest.mock import patch
import numpy as np
import nki
import nki.isa as nisa
import nki.language as nl


@nki.jit
def vector_api_exercise(a):
    tile=nl.ndarray(a.shape,nl.float32,buffer=nl.sbuf)
    nisa.dma_copy(dst=tile,src=a)
    scaled=nl.ndarray(a.shape,nl.float32,buffer=nl.sbuf)
    nisa.tensor_scalar(dst=scaled,data=tile,op0=nl.multiply,operand0=2.)
    combined=nl.ndarray(a.shape,nl.float32,buffer=nl.sbuf)
    nisa.tensor_tensor(dst=combined,data1=tile,data2=scaled,op=nl.add)
    copied=nl.ndarray(a.shape,nl.float32,buffer=nl.sbuf)
    nisa.tensor_copy(dst=copied,src=combined)
    reduced=nl.ndarray((a.shape[0],1),nl.float32,buffer=nl.sbuf)
    nisa.tensor_reduce(dst=reduced,op=nl.add,data=copied,axis=(1,),keepdims=True)
    output=nl.ndarray((a.shape[0],1),nl.float32,buffer=nl.shared_hbm)
    nisa.dma_copy(dst=output,src=reduced)
    return output


@nki.jit
def matmul_api_exercise(a,b):
    left=nl.ndarray(a.shape,nl.float32,buffer=nl.sbuf)
    right=nl.ndarray(b.shape,nl.float32,buffer=nl.sbuf)
    nisa.dma_copy(dst=left,src=a)
    nisa.dma_copy(dst=right,src=b)
    result=nl.ndarray((a.shape[1],b.shape[1]),nl.float32,buffer=nl.psum)
    nisa.nc_matmul(dst=result,stationary=left,moving=right,accumulate=False)
    nisa.nc_matmul(dst=result,stationary=left,moving=right,accumulate=True)
    copied=nl.ndarray(result.shape,nl.float32,buffer=nl.sbuf)
    nisa.tensor_copy(dst=copied,src=result)
    output=nl.ndarray(result.shape,nl.float32,buffer=nl.shared_hbm)
    nisa.dma_copy(dst=output,src=copied)
    return output


@nki.jit
def bad_rank(a):
    return nl.ndarray((4,),nl.float32,buffer=nl.sbuf)


@nki.jit
def bad_dma(a):
    tile=nl.ndarray((2,4),nl.float32,buffer=nl.sbuf)
    nisa.dma_copy(dst=tile,src=a)
    return tile


class CardSimulationTests(unittest.TestCase):
    def test_vector_calls(self):
        a=np.arange(6,dtype=np.float32).reshape(2,3)
        with patch.dict(os.environ,{'NEURON_PLATFORM_TARGET_OVERRIDE':'trn2'}):
            result=nki.simulate(vector_api_exercise)(a)
        np.testing.assert_allclose(result,3*a.sum(axis=1,keepdims=True))

    def test_matmul_layout_buffers_and_accumulation(self):
        a=np.arange(8,dtype=np.float32).reshape(4,2)
        b=np.arange(12,dtype=np.float32).reshape(4,3)
        with patch.dict(os.environ,{'NEURON_PLATFORM_TARGET_OVERRIDE':'trn2'}):
            result=nki.simulate(matmul_api_exercise)(a,b)
        np.testing.assert_allclose(result,2*(a.T@b))

    def test_rank_and_dma_rejections(self):
        with patch.dict(os.environ,{'NEURON_PLATFORM_TARGET_OVERRIDE':'trn2'}):
            with self.assertRaisesRegex(AssertionError,'at least 2 dimensions'):
                nki.simulate(bad_rank)(np.ones((2,2),np.float32))
            with self.assertRaisesRegex(AssertionError,'same number of elements'):
                nki.simulate(bad_dma)(np.ones((2,2),np.float32))


if __name__ == '__main__': unittest.main()
