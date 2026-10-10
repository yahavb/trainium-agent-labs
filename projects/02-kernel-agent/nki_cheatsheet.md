# NKI cheat-sheet (nki 0.6.0)

Short, checked NKI facts for the kernel agent's prompts. They are cut into **cards**: one `core` card
that every prompt gets, and one card per function, which the retriever hands over only when a plan
names that function (`DESIGN.md`, "Retriever"). That split matters:
- With the old API card, which explains matmul to every level, 100% of level-1 attempts used
  `nc_matmul` for pooling.
- With the matmul text removed, 0% did.
- So matmul facts live only in the `nisa.nc_matmul` card.

Every card names the checks in `nki_cheatsheet_check.py` that back it. The script runs each check in
`nki.simulate`, then fails if any card names a check that did not hold:

    PYTHONDONTWRITEBYTECODE=1 python nki_cheatsheet_check.py

`load_cards()` in that script returns `{name: (text, check ids)}`. Re-run the script after any nki
upgrade. Results so far:
- **2026-10-10, seat-35** (nki `0.6.0+31049202112.g85070674`): every claim held.
- **Sizes in Qwen3 tokens:**
  - `core` is 251;
  - each function card is 34–117 (`nisa.nc_matmul` is the largest);
  - all 12 together are 1,043.
  - A typical prompt (`core` plus the 3–4 cards a plan names) is about 550, against 451 for today's
    `API_CARD`, which shows matmul to every level.
- The cards are untested as prompts; whether they help the agent is still to measure.

**Never put a level's answer in a card.** Two rules keep the cards apart from the answers:
- **`withhold=`:** a card with `withhold=` must not be shown at those levels, the same rule as the team's
  `third_party/neuron-agentic-development/withhold.json`. `matmul-k-loop` is the K-loop accumulation at
  the heart of level 4, so it is withheld for levels 3–7, matching that file's "matmul loops". The
  retriever has to enforce it.
  - `enrich()` already tells the agent this pattern when it hits the K > 128 error.
  - Whether that is acceptable is the team's call, not this file's.
- **Answer-like checks back no card.** Check `strided-view` proves the pooling pattern, which is level 1
  in miniature, so no card cites it. The `.ap()` example in `nl.sum` is a generic strided view instead.
- **Never show `nki_cheatsheet_check.py` to the agent.** Its test kernels include a K-loop matmul and
  the pooling pattern.

None of the cards suggest a call the rule checker bans. That includes NKI's own `nl.mean`,
`nl.transpose`, `nl.matmul` and `nl.softmax`, which pass the checker but would do a level's whole job.

<!-- card core checks=out-hbm,tile-1d,tile-3d,pmax,region-call,ragged,affine-range,no-dst,py-operator,nisa-multiply -->
NKI rules (nki 0.6.0, each checked in the simulator)
import nki; import nki.isa as nisa; import nki.language as nl
- Inputs and output live in HBM; all work happens on on-chip tiles.
- Output: out = nl.ndarray(shape, dtype=x.dtype, buffer=nl.shared_hbm); fill it; return out.
- Tile: t = nl.ndarray((P, F), dtype=x.dtype, buffer=nl.sbuf). At least 2 dims; the first
  (partition) dim is at most 128. nl.sbuf is not a function.
- To take part of a tensor, slice it: x[a:b, c:d] or x[nl.ds(start, size), :].
- Loops: for i in range(n). Last partial tile: rows = min(128, P - i * 128).
- Every nisa call writes into dst= and returns nothing. Tiles have no + - * or +=; use nisa ops.
- Ops are passed as op=: nl.add, nl.subtract, nl.multiply, nl.maximum, nl.exp. There is no nisa.multiply.
<!-- /card -->

<!-- card nisa.dma_copy checks=out-hbm,dma-count,ragged -->
nisa.dma_copy(dst=, src=): moves data between HBM and SBUF.
  e.g. nisa.dma_copy(dst=t, src=x[0:128, 0:64]); nisa.dma_copy(dst=out[0:128, 0:64], src=t)
  dst and src must hold the same number of elements; it never broadcasts or reshapes.
<!-- /card -->

<!-- card nisa.tensor_scalar checks=tensor-scalar,row-broadcast -->
nisa.tensor_scalar(dst=, data=, op0=, operand0=): dst = data <op0> operand0, where data is a tile.
  e.g. nisa.tensor_scalar(dst=r, data=t, op0=nl.multiply, operand0=0.25)
  operand0 may be a [P,1] tile, giving one value per row: op0=nl.subtract, operand0=row_max.
<!-- /card -->

<!-- card nisa.tensor_tensor checks=tensor-tensor -->
nisa.tensor_tensor(dst=, data1=, data2=, op=): elementwise on two tiles of the same shape.
  e.g. nisa.tensor_tensor(dst=r, data1=a, data2=b, op=nl.add)
<!-- /card -->

<!-- card nisa.activation checks=activation-exp,activation-rsqrt,nisa-rsqrt -->
nisa.activation(dst=, op=, data=): applies a function to every element.
  e.g. nisa.activation(dst=r, op=nl.exp, data=t). Also op=nl.rsqrt, nl.sqrt, ... (there is no nisa.rsqrt)
<!-- /card -->

<!-- card nisa.reciprocal checks=reciprocal -->
nisa.reciprocal(dst=, data=): 1 / x for every element.  e.g. nisa.reciprocal(dst=r, data=t)
<!-- /card -->

<!-- card nisa.tensor_reduce checks=reduce-sum,reduce-max,reduce-no-keepdims,reduce-nl-max -->
nisa.tensor_reduce(dst=, op=, data=, axis=, keepdims=): reduces along the free axis.
  e.g. r = nl.ndarray((P, 1), dtype=t.dtype, buffer=nl.sbuf)
       nisa.tensor_reduce(dst=r, op=nl.add, data=t, axis=(1,), keepdims=True)
  op=nl.maximum gives the row max.
<!-- /card -->

<!-- card nl.sum checks=nl-sum,ap-every-other -->
s = nl.sum(view, axis=[...], keepdims=False): returns a new SBUF tile, e.g. nl.sum(t, axis=[1], keepdims=True).
  view can be a strided view with no copying: t.ap([[stride, count], ...]), one pair per axis,
  partition first, strides in elements. e.g. every other column of a (P, F) tile:
  t.ap([[F, P], [2, F // 2]])
<!-- /card -->

<!-- card nisa.nc_matmul checks=matmul,moving-fmax,stationary-fmax,matmul-dst-sbuf,psum-to-hbm -->
nisa.nc_matmul(dst=, stationary=, moving=): dst[M,N] = stationary[K,M].T @ moving[K,N].
  stationary and moving are SBUF tiles; K <= 128 (their partition dim), M <= 128, N <= 512.
  dst is a PSUM tile: p = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.psum)
  PSUM cannot go to HBM: tensor_copy p into an SBUF tile, then dma_copy that out.
<!-- /card -->

<!-- card matmul-k-loop checks=k-accumulate,k-accumulate-flag,k-overwrite withhold=3,4,5,6,7 -->
K > 128 in a matmul: loop over K in chunks of 128, calling nc_matmul into the SAME psum tile; it adds
  up (accumulate=False would overwrite instead).
<!-- /card -->

<!-- card nisa.tensor_copy checks=matmul,psum-to-hbm,ap-every-other -->
nisa.tensor_copy(dst=, src=): copies between on-chip tiles, casting to dst's dtype. It is the way out of PSUM.
  e.g. s = nl.ndarray((M, N), dtype=out.dtype, buffer=nl.sbuf); nisa.tensor_copy(dst=s, src=p)
<!-- /card -->

<!-- card nisa.nc_transpose checks=transpose,transpose-sbuf -->
nisa.nc_transpose(dst=, data=): dst[F,P] = data[P,F] transposed. Put dst in PSUM, then tensor_copy it
  to SBUF. With dst in SBUF it only handles up to 32x32.
<!-- /card -->

## Where AWS's NKI rules are wrong or misleading on our nki 0.6.0

The source was `neuron-agentic-development` 1.3, skill `neuron-nki-writing` (`SKILL.md` and
`references/`). Its rules file says it is written for NKI 0.4.0.

| AWS says | On our 0.6.0 | Check |
|---|---|---|
| Use `nisa.rsqrt(dst=..., data=...)` | Does not exist. Use `nisa.activation(op=nl.rsqrt)` | `nisa-rsqrt`, `activation-rsqrt` |
| Never `nl.max` / `nl.min`; use `nl.maximum` | `nl.max` works as a `tensor_reduce` op too, and `nl.max(..., keepdims=True)` runs | `reduce-nl-max` |
| Never `nl.load` / `nl.store` | Both run. Keep `nisa.dma_copy` anyway: the harness counts HBM bytes only through it, so `nl.load` would hide traffic from the levels 5–7 limits | `nl-load-store` |
| `nl.tile_size.psum_fmax_bytes` | Does not exist. The names are `psum_bank_fmax`, `psum_bank_fmax_bytes`, `psum_fmax` | (from `dir(nl.tile_size)`) |
| Use `affine_range`, NOT `sequential_range` | Both are deprecated aliases of `range()` in 0.6.0, so there is no difference | `affine-range` |
| MatMul K ≤ 2048 (`SKILL.md:109`, `indexing-patterns.md`, `api-translation.md`) | One `nc_matmul` contracts at most 128: K is the partition axis of both operands. A 3-D `[128, k, N]` tile layout was not tested | `pmax` |
| Use `accumulate=(k_idx > 0)` in the K loop | Works, but so does passing no flag, which is what the shipped reference kernels do | `k-accumulate`, `k-accumulate-flag` |

## Smaller findings
- **Reshaping an HBM input works when the element count is unchanged** (`reshape`). The agent's "do not
  reshape" advice in `enrich()` is a simplification. The real level-3 error is reshaping to a different
  size, and slicing is the right way to take part of a tensor.
- **`nc_transpose` into SBUF runs on the Vector engine and only handles up to 32×32**
  (`transpose-sbuf`). Into PSUM, a 64×128 tile transposed fine.
- **`tensor_reduce` into a `[P, 1]` tile works with or without `keepdims=True`** (`reduce-no-keepdims`).
