# Extra cards for agents2 (nki 0.6.0)

Same format as `../nki_cheatsheet.md`, which the retriever reads first. These cover what agent2's first
runs on seat-35 (2026-10-10) kept tripping over and the cheat-sheet does not have:
- 30 of 37 level-1 kernels built 5-D shapes, and 24 called `nisa.dma_transpose` with axes it does not
  support;
- the best kernel (0.50) reshaped a tile and reduced the wrong axes, not knowing a reshape keeps
  row-major order or that a view can reorder axes without copying;
- `nl.mean` was used on the partition axis.

Every card names the checks in `cards_check.py` that back it; run that in a seat pod after any nki
upgrade. `withhold=` lists levels at which the card must not be shown: `tile.permute` reorders a row's
elements in one call, which is level 2 (2D transpose) whole.

<!-- card tile.reshape checks=reshape-row-major -->
t.reshape(shape): the same data with a new shape, in row-major order like numpy. It copies and reorders
  nothing: elements that are neighbours stay neighbours. Keep the partition axis (axis 0) first.
<!-- /card -->

<!-- card tile.permute checks=permute,sum-permuted withhold=2 -->
t.permute(dims): reorders axes as a view, like np.transpose, without copying. e.g. t.permute((0, 2, 1)).
  nl.sum / nl.mean / nisa.tensor_copy can read a permuted view directly. Keep axis 0 (partition) first.
<!-- /card -->

<!-- card nl.mean checks=mean-free,mean-partition -->
m = nl.mean(t, axis=[...], keepdims=False): returns a new SBUF tile. Like nl.sum it reduces only free
  axes (never axis 0, the partition axis), e.g. nl.mean(t, axis=[1], keepdims=True).
<!-- /card -->

<!-- card nisa.dma_transpose checks=dma-transpose-2d,dma-transpose-axes withhold=2 -->
nisa.dma_transpose(dst=, src=, axes=): only axes=(1, 0) for 2-D, (2, 1, 0) for 3-D and (3, 1, 2, 0) for
  4-D; dst has exactly the transposed shape. Any other axes order is an error.
<!-- /card -->
