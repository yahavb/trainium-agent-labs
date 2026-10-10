# Hack the Chip: the kernel agent, all seven levels solved

This is a fork of [yahavb/trainium-agent-labs](https://github.com/yahavb/trainium-agent-labs), the
repo for the NYU × Annapurna Labs "Hack the Chip" workshop. It branches from upstream commit
`8f1ca41`. All the work is in **Project 2, the kernel agent**
([`projects/02-kernel-agent/`](projects/02-kernel-agent/)). The cluster setup, `serve.sh`, Project 1
and the `gptoss/` client are unchanged. The
[upstream README](https://github.com/yahavb/trainium-agent-labs#readme) explains how to get onto a
seat pod and start the model.

---

## The objective

Project 2 asks for an agent that writes **NKI kernels** for AWS Trainium using a small model,
Qwen3-8B, served on the seat's own chip. The checker, [`nkibench.py`](projects/02-kernel-agent/nkibench.py),
simulates each attempt on the CPU and compares it with a NumPy reference. Its grade, and the reason
for it, go into the next prompt. A reward of 1.0 means the kernel is correct on every test shape:

| part | weight |
|---|---|
| the code parses | 0.1 |
| no rule violations (no framework shortcuts, ≤128 partitions, `@nki.jit`, right entry point) | 0.2 |
| it runs in the simulator | 0.2 |
| it is correct, scaled by the fraction of test shapes that pass | 0.5 |

The ladder has seven levels:

| level | operation | graded on |
|---|---|---|
| 1 | average pooling | correctness |
| 2 | 2D transpose | correctness |
| 3 | matmul, one tile | correctness |
| 4 | matmul, tiled | correctness |
| 5 | matmul, loads hoisted | correctness, and HBM traffic ≤ 1.6× the byte floor |
| 6 | matmul, M and N blocked | correctness, and traffic ≤ 1.25× |
| 7 | matmul, M, N and K blocked | correctness, and traffic ≤ 1.05× |

The byte floor is the traffic of reading each input once and writing the output once.

Upstream shipped this as an unsolved problem. Levels 1–4 had a five-run measurement, and levels 5–7
had no reference kernels and had never been run:

| level | upstream, 5 runs |
|---|---|
| 1 average pooling | 0/5, 0.30 every run |
| 2 transpose | 4/5, scores 0.50–1.00 (2/5 in a second measurement) |
| 3 matmul, single tile | 0/5, 0.30 every run |
| 4 matmul, tiled | 0/5, 0.62 every run |
| 5–7 | not attempted |

Levels 1, 3 and 4 scored the same in every run, so they were capability walls, not bad luck. This
fork set out to get the agent past them and then up the optimisation half of the ladder.

---

## Results

Five runs per level (`--repeat 5`) against Qwen3-8B on the seat's chip, with 4 samples per round:

| level | upstream | this fork | HBM traffic, largest shape |
|---|---|---|---|
| 1 average pooling | 0/5 | **5/5**, solved on round 0 in every run | |
| 2 transpose | 4/5 | **5/5**, round 0 | |
| 3 matmul, single tile | 0/5 | **5/5**, round 0 | |
| 4 matmul, tiled | 0/5 | **5/5**, round 0 | |
| 5 matmul, loads hoisted | not attempted | **5/5**, round 0 | 1.14× the floor (bar ≤ 1.6×) |
| 6 matmul, M and N blocked | not attempted | **5/5**, round 0 | 1.00× (bar ≤ 1.25×) |
| 7 matmul, M, N and K blocked | not attempted | **5/5**, round 0 | 1.00× (bar ≤ 1.05×) |

The winning kernels move whole tiles. Levels 1 and 2 use 2 DMA transfers per shape. Levels 3, 6 and
7 read every input exactly once. On the largest shape, levels 6 and 7 still run memory-bound, at 73.1
Flops/Byte against a ridge of 222. That is the most any kernel can reach at that size, because there
is no reuse left to find.

**What these numbers do not show:**

- **Level 1 got harder after its measurement.** Two shapes were added later: 200 channels, and an
  8×240×240 image. The kernel that won the five runs loads each image whole, so it now passes 4 of 6
  shapes. It fails on the 128-partition limit and on SBUF capacity (225 KB per partition against
  208 KB). The level-1 card was rewritten to move the input in channel chunks and row bands, and the
  reference kernel passes all six shapes. The agent has not been re-measured on them yet.
- **The test shapes do not exercise blocking.** Levels 4–7 share four test shapes, the largest with
  K = 256, M = 512 and N = 1024 (the deepest has K = 512). Level 6's block is 512 × 1024 and level
  7's K block is 1,024 deep, so every test fits in one block and each block loop runs once. The
  checker confirms that both kernels read each input once, but it never makes them reuse data
  across blocks. Adding a shape larger than one block, such as K = 2048, M = 1024, N = 2048, would
  test that.
- **The cards decide the kernel.** The level 1 and 2 cards spell out the algorithm step by step. The
  level 5–7 cards are code skeletons with the copies and matmuls left blank, and each of those
  levels produced one identical winning kernel in every run. A solve shows that the model can carry
  out a given strategy in valid NKI, not that it found the strategy itself.
- **Rounds are slower.** Round 0 took 41–94 s on levels 1–4 and 86–115 s on levels 5–7, against the
  ~8 s per round upstream measured. Every prompt now carries a strategy card.
- **Only Qwen3-8B was re-measured.** The upstream comparison with `gpt-oss-20b` has not been re-run.

---

## What changed

### 1. The checker catches more (`nkibench.py`)

Two upstream "solves" were wrong in ways the simulator could not see. The checker now catches them
and several other failures like them.

| change | what it caught |
|---|---|
| Counts writes by assignment (`out[p, k] = tile[p, j]`) as DMAs | These go through `NkiTensor.__setitem__`, which upstream did not count. An element-at-a-time transpose that made 8,193 transfers on the 128×64 shape was reported as 1 transfer at 0.50× the byte floor, "essentially optimal". |
| Checks SBUF capacity | The simulator does not. A kernel that held a 64×224×224 image on chip (294 KB per partition against 208 KB) was reported correct. Each SBUF allocation is now charged to the kernel line that made it, and the total is compared with `nl.tile_size.sbuf_fmax_bytes`. |
| Enforces the `nc_matmul` shape contract | The simulator only checks the element count. A PSUM tile allocated (N, M) instead of (M, N) passed the matmul and failed two lines later in `tensor_copy`. Now it fails on the matmul line. |
| Counts matmuls into each PSUM output tile, including regions of one block-sized PSUM tensor | More than K/128 means output tiles are adding into each other. Fewer means the tile was allocated inside the K loop. Both run cleanly and return wrong numbers. |
| Counts HBM reads per input | On levels 5–7 the traffic verdict now names the operand that is re-read, e.g. `rhs (256, 1024) 4.0 times`, not only the total. |
| `--check` applies the level 5–7 traffic bars | The agent's grader already applied them; the command-line check did not. |
| Two new level-1 shapes: 200 channels, 8×240×240 | 200 channels do not fit 128 partitions, and a 240×240 float32 image is 225 KB per partition. The old shapes passed a kernel that could not run on the device. |
| `--selftest` covers the SBUF check | Whole-image pooling must be refused, and the banded reference must pass the same shape. |

### 2. The feedback names the fix (`agent.py`)

Upstream's feedback quoted the simulator's error. That error often pointed at the wrong place or
none at all, so the model either guessed or sent the same code back.

**The failing line, with its live tiles.** The simulator runs the kernel as plain Python, so the
traceback still holds the failing line and its live tiles. `locate_failure` reads them, and
`shape_advice` turns each common shape error into one change, written with the model's own variable
names. Here is the full feedback for a level-3 kernel that allocated its PSUM tile (N, M):

```
0 of 1 shapes passed. On K=128 M=64 N=512: raised AssertionError: nc_matmul dst has shape (512, 64),
but stationary (128, 64) = [K, M] and moving (128, 512) = [K, N] produce [M, N] = (64, 512)
  The failing line is 15: `nisa.nc_matmul(dst=res, stationary=a, moving=b)`
  where `res` is (512, 64) in psum; `a` is (128, 64) in sbuf; `b` is (128, 512) in sbuf
  FIX: nc_matmul writes stationary.T @ moving into dst. stationary `a` is (128, 64) = [K, M] and
  moving `b` is (128, 512) = [K, N], so dst must be [M, N] = (64, 512), but dst `res` is (512, 64).
  Allocate dst as nl.ndarray((64, 512), dtype=nl.float32, buffer=nl.psum).
```

Upstream's generic hints now run only when there is no specific diagnosis, because several pointed
the wrong way. For example, "cannot reshape array" got the advice "do not reshape" on a kernel that
never called reshape.

**Specific diagnoses for mistakes that run cleanly and return wrong numbers:**

- `M, K = lhsT.shape` written backwards. It passes every shape where K == M.
- PSUM copied straight to HBM. The message gives the three-line fix.
- PSUM accumulation errors, from the matmul counts above.
- On levels 5–7, the K loop folded into a block slice (`lhsT_block[:, :, ...]`). That passes only
  when K is one tile.
- A transpose that returns its input unchanged, or swaps F1 and F2.
- A pooling result that is the window sum, divided by p instead of p², or has rows and columns
  swapped.
- A real function in the wrong module: `nisa.multiply` gets "write `nl.multiply`".
- Pooling inputs too big for the 128 partitions or for SBUF.

**Strategy cards.** Each level now gets a short card on how its operation maps onto NKI. The API
card already listed every primitive; what the model lacked was how they fit together. In upstream's
last level-1 measurement, 54 of 80 samples used `nc_matmul` to compute a mean.

| level | card |
|---|---|
| 1 | channel chunks × row bands, a strided `.ap()` view, one `nl.sum`, one scale |
| 2 | one DMA in, F1 row copies on chip, one DMA out |
| 3 | which operand supplies M and which N; PSUM → SBUF → HBM |
| 4 | the tiling contract up front: which loop owns which slice, and where the PSUM tile lives |
| 5 | skeleton: N outermost, so each column of rhs tiles is loaded once and reused for every m |
| 6 | skeleton: a block of up to 4 × 2 output tiles, with its operands loaded once |
| 7 | skeleton: the level 6 blocks plus K blocks of up to 8 tiles, and the output block kept in PSUM across them |

Level 4 needed the tiling contract up front because repairs fixed one dimension per round, K
tiling arrived in round 2, and M and N never did.

Levels 5–7 took two failed attempts first. With the level-4 card, the model wrote the plain tiled
loop, which moves 2.00× the byte floor on the largest shape. Level 5 scored 0.875, passing only the
three shapes small enough to fit in one tile. Prose cards that described each technique passed 0 of
18 samples, because the model un-nested the loops and allocated result tiles twice. The cards that
work are code skeletons. The loop structure and every allocation are written out, and only the
copies and matmuls are left blank. Level 6's matmul line is written out too: left blank, 8 of 8
fresh samples folded the K loop into the slice.

The repair prompt carries the card as well, so repairs stay on the strategy.

### 3. The search loop wastes fewer attempts (`agent.py`)

- **It detects echoes.** On level 3, every repair came back byte-for-byte unchanged, 16 samples of
  16 across four runs. Upstream's prompt said "keep everything else identical", and the easiest way
  to obey that is to change nothing. Code already graded in an earlier round now reuses its grade,
  and the next prompt says it was an echo. The repair prompt now says that unchanged code fails
  again.
- **It breaks ties by progress.** Below 0.5, rewards tie constantly, and the tie used to go to
  whichever reply came back first. Ties now go to the attempt that got furthest: further into the
  kernel before raising, or more of the output correct.
- **Half the samples repair, half start fresh.** Repairs converge on one kernel, and fresh samples
  are how a round escapes a kernel whose structure is wrong. Fresh prompts carry the last three
  error headlines so they do not replay round 0. With one sample, as on the greedy `gpt-oss`
  endpoint, it only repairs.
- **A grading bug is fixed.** Every candidate used to be written to the same `/tmp` path, and Python
  reuses a cached `.pyc` when the size and the mtime second match. A same-length repair graded
  within a second, such as swapping `stationary=` and `moving=`, was silently graded as the old
  code. Each kernel now gets its own file, named by its hash, in a private temp directory.

### 4. Reference kernels

- [`reference_level1.py`](projects/02-kernel-agent/reference_level1.py) goes beyond the tutorial,
  which loads the whole input into one tile. It now loops over chunks of up to 128 channels and over
  row bands that fit in SBUF.
- [`reference_level5.py`](projects/02-kernel-agent/reference_level5.py),
  [`6`](projects/02-kernel-agent/reference_level6.py) and
  [`7`](projects/02-kernel-agent/reference_level7.py) are new. Upstream had none, so the
  optimisation half of the ladder could not be checked. Each passes its level at the same traffic
  as the agent's winning kernel: 1.14×, 1.00× and 1.00× the byte floor on the largest shape.

Level 5 departs from the tutorial on purpose. The tutorial's hoisting kernel keeps M outermost and
moves 1.86× the byte floor on the largest shape, which fails the level's own 1.6× bar. Putting N
outermost hoists the bigger operand instead.

---

## What we learned

1. **Check the checker before you trust a solve.** Upstream's level-2 solves made 8,193 DMAs and were
   reported optimal. The kernel that first solved level 1 could not fit on the device. Both looked
   fine in the simulator.
2. **The model knew the API but not how the pieces combine.** Every level moved once the prompt said
   how the primitives fit together for that operation.
3. **Put loop nesting and allocation in code, not prose.** On levels 5–7, prose cards passed 0 of 18
   samples and skeleton cards solved every run. On the fork's `kernel-feedback` branch, level 4
   solved 1 of 3 runs with a prose card and 3 of 3 with a skeleton.
4. **Never tell the model to keep everything else identical.** It will keep all of it.
5. **A useful error names a line and a change, in the model's own variable names.** A bare verdict
   like "shapes differ" gets the same kernel back.

---

## Reproduce

Start the model first (`./serve.sh` in `/workspace`; see the upstream README). Then:

```bash
cd /workspace/projects/02-kernel-agent

python nkibench.py --selftest
for l in 1 2 3 4 5 6 7; do python nkibench.py --level $l --check reference_level$l.py; done

# levels 1-4, five runs each (--all covers levels 1-4 only)
nohup python agent.py --all --rounds 4 --samples 4 --context 8192 --repeat 5 > run.log 2>&1 < /dev/null &

# levels 5-7, five runs each
nohup sh -c 'for l in 5 6 7; do python agent.py --level $l --rounds 4 --samples 4 --context 8192 --repeat 5 > run$l.log 2>&1; done' < /dev/null > /dev/null 2>&1 &
```

`nkibench.py --selftest` and all seven reference checks pass on the seat pod (nki 0.6.0).

---

## Credits and license

The level-2 transpose card started on this fork's `level2` branch. The skeleton-over-prose finding
for level 4 came from the `kernel-feedback` branch. Everything else in the original workshop is the
work of the upstream authors.

The upstream terms are unchanged: sample code, provided as-is, free to reuse.
[`Qwen/Qwen3-8B`](https://huggingface.co/Qwen/Qwen3-8B) and
[`openai/gpt-oss-20b`](https://huggingface.co/openai/gpt-oss-20b) are under their own licenses.
