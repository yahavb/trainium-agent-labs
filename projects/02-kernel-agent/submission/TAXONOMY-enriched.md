# Failure taxonomy

56 attempts, levels [2, 3, 4], 1 session(s): 20261010-195043

| failure mode | what it means | L2 | L3 | L4 | total | share |
|---|---|---|---|---|---|---|
| tile/slice size mismatch | source and destination hold different element counts | 16 | 3 | 1 | 20 | 36% |
| result does not fit its tile | an NKI call's result is a different size from its destination; the simulator reports it as a reshape error |  | 16 |  | 16 | 29% |
| no tiling: partition over 128 | one tile for the whole tensor |  |  | 15 | 15 | 27% |
| 1-D tile | allocated an SBUF/PSUM tile with one dimension |  | 5 |  | 5 | 9% |

## How the failure moved, round to round

11 repair steps: 10 repeated the same failure (91%), 1 changed it, and 0 made the score go DOWN.

| after this failure | the next round hit | times |
|---|---|---|
| 1-D tile | result does not fit its tile | 1 |

## Where the prompt went (characters, mean per prompt)

| prompt kind | prompts | reference | docs | code | feedback | ledger | instructions | prompt tokens | answer tokens |
|---|---|---|---|---|---|---|---|---|---|
| first | 3 | 455 | 1581 | 0 | 0 | 0 | 505 | 725 | 325 |
| repair | 4 | 0 | 0 | 1043 | 458 | 0 | 200 | 527 | 338 |
| repair+ledger | 7 | 0 | 0 | 1050 | 464 | 209 | 266 | 617 | 339 |

## Line-located feedback

The checker quoted the failing line on 0 of 56 attempts that raised (0%).

## One real checker message per failure mode

**tile/slice size mismatch** (level 2, round 0, reward 0.30)

> 0 of 4 shapes passed. On shape=(32, 12) as 3x4: raised AssertionError: dma_copy requires src and dst to have the same number of elements, got src=384, dst=16384 The tile you allocated holds 16384 elements but you copied 384 into it. nisa.dma_copy does not slice or broadcast: allocate the destination with EXACTLY the shape of the slice you are moving. If you want a 128x512 piece of a bigger tensor, write t = nl.ndarray((128, 512), dtype=a.dtype, buffer=nl.sbuf) and then nisa.dma_copy(dst=t, src=a

**1-D tile** (level 3, round 0, reward 0.30)

> 0 of 1 shapes passed. On K=128 M=64 N=512: raised AssertionError: SBUF and PSUM tensors must have at least 2 dimensions (partition-dim and free-dim) Every SBUF and PSUM tile needs two dimensions: a partition dimension first, then a free dimension. A 1-D tile is not allowed, so write nl.ndarray((rows, cols), ...) and give a length-N vector the shape (1, N) or (N, 1) depending on which axis you are reducing over.

**result does not fit its tile** (level 3, round 2, reward 0.30)

> 0 of 1 shapes passed. On K=128 M=64 N=512: raised ValueError: cannot reshape array of size 32768 into shape (1,64) Do not reshape. Work with the shapes you were given and slice them into tiles, e.g. src=a[0:128, 0:64].

**no tiling: partition over 128** (level 4, round 0, reward 0.62)

> 1 of 4 shapes passed. On K=256 M=256 N=1024: raised AssertionError: dma_copy dst partition dimension 256 exceeds maximum 128 A tile may have at most 128 rows, and you asked for 256. Do not allocate one tile for the whole tensor: loop over the partition dimension in chunks of at most 128 with nl.affine_range, allocate the tile inside the loop with the chunk's own size, and copy one chunk at a time, e.g. src=a[i*128:(i+1)*128, :]. If a dimension is already 128 or smaller, use it whole -- do NOT pa

