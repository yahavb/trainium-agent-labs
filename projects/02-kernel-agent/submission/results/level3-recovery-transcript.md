# Level 3, one run, round by round

Generated from `results/seat-153/checker-l3.jsonl`, run 1 of 5. Flags `internal,origin` (the reportable checker), commit 39b8e5a, Qwen3-8B on seat 153, 4 samples per round. Each round shows the best sample's allocations, then exactly what the checker sent back. Nothing is paraphrased.

For contrast, the baseline on the same level: in run 1 of `results/seat-154/baseline.jsonl` the model was told "Do not reshape" and resent identical code until the loop stopped. Its kernel contains no reshape.


## Round 0: score 0.30

Tiles the kernel allocated:

```python
out = nl.ndarray(shape=lhsT.shape[1:], dtype=lhsT.dtype, buffer=nl.shared_hbm)
sbuf_lhsT = nl.ndarray(shape=lhsT.shape, dtype=lhsT.dtype, buffer=nl.sbuf)
sbuf_rhs = nl.ndarray(shape=rhs.shape, dtype=rhs.dtype, buffer=nl.sbuf)
psum = nl.ndarray(shape=out.shape, dtype=out.dtype, buffer=nl.psum)
```


Failed at line 16, `psum = nl.ndarray(shape=out.shape, dtype=out.dtype, buffer=nl.psum)`. The checker replied:


> 0 of 1 shapes passed. On K=128 M=64 N=512: line 16, `psum = nl.ndarray(shape=out.shape, dtype=out.dtype, buffer=nl.psum)`, raised AssertionError: SBUF and PSUM tensors must have at least 2 dimensions (partition-dim and free-dim) Every SBUF and PSUM tile needs two dimensions: a partition dimension first, then a free dimension. A 1-D tile is not allowed, so write nl.ndarray((rows, cols), ...) and give a length-N vector the shape (1, N) or (N, 1) depending on which axis you are reducing over. At that line: `out` is (64,) in HBM. A tile on the chip needs at least 2 dimensions, the partition dimension first: this shape has 1.


## Round 1: score 0.30

Tiles the kernel allocated:

```python
out = nl.ndarray(shape=(lhsT.shape[1],), dtype=lhsT.dtype, buffer=nl.shared_hbm)
sbuf_lhsT = nl.ndarray(shape=lhsT.shape, dtype=lhsT.dtype, buffer=nl.sbuf)
sbuf_rhs = nl.ndarray(shape=rhs.shape, dtype=rhs.dtype, buffer=nl.sbuf)
psum = nl.ndarray(shape=(out.shape[0], 1), dtype=out.dtype, buffer=nl.psum)
```


Failed at line 18, `nisa.nc_matmul(dst=psum, stationary=sbuf_lhsT, moving=sbuf_rhs)`. The checker replied:


> 0 of 1 shapes passed. On K=128 M=64 N=512: line 18, `nisa.nc_matmul(dst=psum, stationary=sbuf_lhsT, moving=sbuf_rhs)`, raised ValueError: cannot reshape array of size 32768 into shape (64,1) At that line: `psum` is (64, 1) in psum; `sbuf_lhsT` is (128, 64) in sbuf; `sbuf_rhs` is (128, 512) in sbuf. The result of nc_matmul has stationary's second dimension as its first and moving's second dimension as its second, and dst must be exactly that shape: dst is (64, 1). The tile to change is `psum`: its second dimension is 1, too small: it must equal the second dimension of `sbuf_rhs`. It got its shape on line 16, `psum = nl.ndarray(shape=(out.shape[0], 1), dtype=out.dtype, buffer=nl.psum)`, so that is the line to change, not this one.


## Round 2: score 0.30

Tiles the kernel allocated:

```python
out = nl.ndarray(shape=(lhsT.shape[1],), dtype=lhsT.dtype, buffer=nl.shared_hbm)
sbuf_lhsT = nl.ndarray(shape=lhsT.shape, dtype=lhsT.dtype, buffer=nl.sbuf)
sbuf_rhs = nl.ndarray(shape=rhs.shape, dtype=rhs.dtype, buffer=nl.sbuf)
psum = nl.ndarray(shape=(lhsT.shape[0], rhs.shape[1]), dtype=out.dtype, buffer=nl.psum)
```


Failed at line 18, `nisa.nc_matmul(dst=psum, stationary=sbuf_lhsT, moving=sbuf_rhs)`. The checker replied:


> 0 of 1 shapes passed. On K=128 M=64 N=512: line 18, `nisa.nc_matmul(dst=psum, stationary=sbuf_lhsT, moving=sbuf_rhs)`, raised ValueError: cannot reshape array of size 32768 into shape (128,512) At that line: `psum` is (128, 512) in psum; `sbuf_lhsT` is (128, 64) in sbuf; `sbuf_rhs` is (128, 512) in sbuf. The result of nc_matmul has stationary's second dimension as its first and moving's second dimension as its second, and dst must be exactly that shape: dst is (128, 512). The tile to change is `psum`: its first dimension is 128, too large: it must equal the second dimension of `sbuf_lhsT`. It got its shape on line 16, `psum = nl.ndarray(shape=(lhsT.shape[0], rhs.shape[1]), dtype=out.dtype, buffer=nl.psum)`, so that is the line to change, not this one.


## Round 3: score 0.30

Tiles the kernel allocated:

```python
out = nl.ndarray(shape=(lhsT.shape[1],), dtype=lhsT.dtype, buffer=nl.shared_hbm)
sbuf_lhsT = nl.ndarray(shape=lhsT.shape, dtype=lhsT.dtype, buffer=nl.sbuf)
sbuf_rhs = nl.ndarray(shape=rhs.shape, dtype=rhs.dtype, buffer=nl.sbuf)
psum = nl.ndarray(shape=(lhsT.shape[1], rhs.shape[1]), dtype=out.dtype, buffer=nl.psum)
```


Failed at line 20, `nisa.tensor_copy(dst=sbuf_lhsT, src=psum)`. The checker replied:


> 0 of 1 shapes passed. On K=128 M=64 N=512: line 20, `nisa.tensor_copy(dst=sbuf_lhsT, src=psum)`, raised ValueError: shape mismatch: value array of shape (32768,) could not be broadcast to indexing result of shape (8192,) At that line: `sbuf_lhsT` is (128, 64) in sbuf; `psum` is (64, 512) in psum. The two sides of a tensor_copy must hold the same number of elements, because it does not slice, reshape or broadcast: dst holds 8192 elements and src holds 32768. `sbuf_lhsT` is the wrong size here: it holds 8192 elements, too small for the 32768 on the other side. But do not resize it: line 13, `nisa.dma_copy(dst=sbuf_lhsT, src=lhsT)`, already ran with the shape it has, so it is being used for two different things. Leave `sbuf_lhsT` as it is and allocate a separate tile for this line, sized to match the other side.


## Round 4: score 0.30

Tiles the kernel allocated:

```python
out = nl.ndarray(shape=(lhsT.shape[1],), dtype=lhsT.dtype, buffer=nl.shared_hbm)
sbuf_lhsT = nl.ndarray(shape=lhsT.shape, dtype=lhsT.dtype, buffer=nl.sbuf)
sbuf_rhs = nl.ndarray(shape=rhs.shape, dtype=rhs.dtype, buffer=nl.sbuf)
psum = nl.ndarray(shape=(lhsT.shape[1], rhs.shape[1]), dtype=out.dtype, buffer=nl.psum)
sbuf_psum = nl.ndarray(shape=psum.shape, dtype=psum.dtype, buffer=nl.sbuf)
```


Failed at line 24, `nisa.dma_copy(dst=out, src=sbuf_psum)`. The checker replied:


> 0 of 1 shapes passed. On K=128 M=64 N=512: line 24, `nisa.dma_copy(dst=out, src=sbuf_psum)`, raised AssertionError: dma_copy requires src and dst to have the same number of elements, got src=32768, dst=64 The tile you allocated holds 64 elements but you copied 32768 into it. nisa.dma_copy does not slice or broadcast: allocate the destination with EXACTLY the shape of the slice you are moving. If you want a 128x512 piece of a bigger tensor, write t = nl.ndarray((128, 512), dtype=a.dtype, buffer=nl.sbuf) and then nisa.dma_copy(dst=t, src=a[0:128, 0:512]) -- the slice on the right must have the same shape as the tile on the left. At that line: `out` is (64,) in HBM; `sbuf_psum` is (64, 512) in sbuf. The two sides of a dma_copy must hold the same number of elements, because it does not slice, reshape or broadcast: dst holds 64 elements and src holds 32768. The tile to change is `out`: it holds 64 elements, too small for the 32768 on the other side. It got its shape on line 8, `out = nl.ndarray(shape=(lhsT.shape[1],), dtype=lhsT.dtype, buffer=nl.shared_hbm)`, so that is the line to change, not this one.


## Round 5: score 1.00

Tiles the kernel allocated:

```python
out = nl.ndarray(shape=(lhsT.shape[1], rhs.shape[1]), dtype=lhsT.dtype, buffer=nl.shared_hbm)
sbuf_lhsT = nl.ndarray(shape=lhsT.shape, dtype=lhsT.dtype, buffer=nl.sbuf)
sbuf_rhs = nl.ndarray(shape=rhs.shape, dtype=rhs.dtype, buffer=nl.sbuf)
psum = nl.ndarray(shape=(lhsT.shape[1], rhs.shape[1]), dtype=out.dtype, buffer=nl.psum)
sbuf_psum = nl.ndarray(shape=psum.shape, dtype=psum.dtype, buffer=nl.sbuf)
```


**Solved.** The checker replied: Correct on every shape.


The solved kernel:

```python
import nki
import nki.isa as nisa
import nki.language as nl

@nki.jit
def nki_matmul_basic_(lhsT, rhs):
    # Allocate output buffer in shared HBM
    out = nl.ndarray(shape=(lhsT.shape[1], rhs.shape[1]), dtype=lhsT.dtype, buffer=nl.shared_hbm)
    # Allocate SBUF for lhsT and rhs
    sbuf_lhsT = nl.ndarray(shape=lhsT.shape, dtype=lhsT.dtype, buffer=nl.sbuf)
    sbuf_rhs = nl.ndarray(shape=rhs.shape, dtype=rhs.dtype, buffer=nl.sbuf)
    # Copy lhsT and rhs from HBM to SBUF
    nisa.dma_copy(dst=sbuf_lhsT, src=lhsT)
    nisa.dma_copy(dst=sbuf_rhs, src=rhs)
    # Allocate PSUM for result
    psum = nl.ndarray(shape=(lhsT.shape[1], rhs.shape[1]), dtype=out.dtype, buffer=nl.psum)
    # Perform matrix multiplication
    nisa.nc_matmul(dst=psum, stationary=sbuf_lhsT, moving=sbuf_rhs)
    # Allocate temporary SBUF for psum result
    sbuf_psum = nl.ndarray(shape=psum.shape, dtype=psum.dtype, buffer=nl.sbuf)
    # Copy result from PSUM to temporary SBUF
    nisa.tensor_copy(dst=sbuf_psum, src=psum)
    # Copy result from temporary SBUF to HBM
    nisa.dma_copy(dst=out, src=sbuf_psum)
    return out
```
