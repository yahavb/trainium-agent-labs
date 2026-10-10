# NKI API card (nki 0.6.0, verified against the installed SDK on the seat pod)

Extracted with `dir(nki.language)` / `inspect.signature` from the actual package -- every
name below exists. When a checker reports "no attribute X", pick from here; do not guess
nearby names.

## Allocation and memory regions

- `nl.ndarray(shape, dtype, buffer=nl.sbuf)` -- allocate an on-chip tile
- buffers: `nl.sbuf` (on-chip scratchpad), `nl.psum` (matmul accumulator), `nl.shared_hbm`
  (the tensor you return), `nl.hbm`, `nl.private_hbm`
- `nl.full(shape, fill_value, dtype, buffer=...)`, `nl.zeros(shape, dtype, buffer=...)`
- `nl.tile_size` -- constants: `.pmax` (128), `.gemm_stationary_fmax` (128),
  `.gemm_moving_fmax` (512)
- dtypes: `nl.float32`, `nl.float16`, `nl.bfloat16`, `nl.int8/16/32`, `nl.bool_`

## Loops

- `nl.affine_range(n)` -- the tile loop (also start/stop/step)
- `nl.sequential_range`, `nl.dynamic_range`, `nl.static_range`, `nl.fori_loop`
- `nl.ds(start, size)` -- dynamic slice, for column slices inside a partition

## Math on tiles

- elementwise: `nl.add subtract multiply divide power abs exp log sqrt rsqrt square
  reciprocal maximum minimum where sign sigmoid tanh relu gelu silu erf floor ceil
  round-free: trunc fmod` (all take tile operands)
- reductions: `nl.sum(x, axis, dtype=None, keepdims=False)`, `nl.max`, `nl.min`,
  `nl.mean`, `nl.prod`, `nl.var`, `nl.average`, `nl.all` -- axis may be a LIST
- `nl.transpose(x)`, `nl.copy(x)`, `nl.matmul` (high-level), `nl.rms_norm`, `nl.softmax`
  (high-level helpers exist; the ladder's NKI levels ask you to build them from nisa)

## Data movement (nki.isa)

- `nisa.dma_copy(dst=, src=)` -- HBM <-> SBUF. src and dst must have the SAME shape; it
  does not slice, broadcast, or transpose. Partition dim <= 128.
- `nisa.tensor_copy(dst=, src=)` -- SBUF <-> SBUF / PSUM -> SBUF
- `nisa.tensor_scalar(dst=, data=, op0=nl.multiply, operand0=0.5)` -- scale by a constant
- `nisa.tensor_tensor(dst=, data1=, data2=, op=nl.add)` -- elementwise tile-tile
- `nisa.nc_matmul(dst=, stationary=, moving=)` -- dst MUST be in nl.psum; both operands
  MUST be in nl.sbuf; contraction dim (partition axis) <= 128; accumulates into dst
  across calls. The left operand arrives transposed (K on the partition axis).
- `nisa.tensor_reduce(dst=, op=nl.add, data=, axis=)`, `nisa.dma_transpose`,
  `nisa.nc_transpose`, `nisa.activation(dst=, op=, data=, bias=, scale=)`

## The canonical matmul sequence

dma_copy HBM->sbuf both operands; allocate psum; nc_matmul(dst=psum, stationary=sbuf_t,
moving=sbuf_t) once per K chunk into the SAME psum; tensor_copy psum->sbuf; dma_copy
sbuf->the shared_hbm output; return it.

## Simulator (no device needed)

`nki.simulate(kernel)(*args)` runs a kernel on CPU. (In older SDKs:
`nki.simulate_kernel(kernel, *args)`.) The harness in project 02 counts HBM traffic by
wrapping `nisa.dma_copy` around the simulation.
