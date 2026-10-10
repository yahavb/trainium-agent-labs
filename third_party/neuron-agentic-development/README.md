# neuron-agentic-development (a subset)

A curated copy of AWS's `neuron-agentic-development` package, version 1.3: NKI documentation and NKI agent
definitions. It is here so the kernel agent can look NKI up for itself, instead of relying only on the
hand-written API card in `projects/02-kernel-agent/agent.py`.

Licensed under Apache 2.0 by Amazon.com, Inc. or its affiliates: see `LICENSE.txt` and `NOTICE`. The files are
unchanged; this directory is a subset of the package.

## Where it came from

The package is installed in every seat's image, at
`/opt/conda/lib/python3.13/site-packages/neuron_agentic_development/artifacts/`. Copied from seat-35 on
2026-10-10.

## What is here

| Path | What it is |
|---|---|
| `skills/neuron-nki-docs/` | NKI documentation: the API reference (`references/programming/api/`), concept guides (indexing, access patterns in `nki-aps.md`, tiling, memory hierarchy, DMA), Trainium architecture, error codes, optimization guides, and three indexes (`references/indices/`: symbol lookup, task routing, table of contents) |
| `skills/neuron-nki-writing/` | A guide to writing kernels, pattern references (indexing, memory, NumPy/PyTorch-to-NKI translation, common patterns, language constraints) and two small examples |
| `skills/neuron-nki-debugging/` | Compiler error codes, flags and artifacts, for on-device work |
| `agents/` | AWS's NKI, NKI writer and NKI debugger agent definitions. They're written for a frontier model with file and shell tools, not for our 8B agent; kept as a reference for their workflows |

## What was left out, and why

The kernel ladder's answers, so a lookup can never hand the agent a solution:

- `skills/neuron-nki-docs/references/downloads/`: complete tutorial kernels for average pooling (nki-L1),
  transpose (nki-L2) and matmul, including the hoisted, blocked and fully optimized versions (nki-L3 to L7),
  plus mamba and tensor addition.
- `skills/neuron-nki-docs/references/programming/tutorials/` `average_pool2d.md`, `transpose2d.md`,
  `matrix_multiplication.md` and `kernel-optimization.md`: walkthroughs of the same kernels.
- `skills/neuron-nki-writing/examples/simple_matmul.py`: essentially nki-L3.

Also left out: `skills/neuron-nki-writing/references/nkilib/` (library internals we don't need) and
`__pycache__/`. The indexes in `references/indices/` still link to 10 of the removed files; those links are
dangling on purpose.

## Withheld per level: `withhold.json`

Some kept files are general guidance but carry code close to a ladder answer. For example,
`common-patterns.md` has a complete tiled matmul with PSUM accumulation, which is nki-L4. `withhold.json` maps
each such file to the levels at which the agent must not be shown it, and any lookup tool has to enforce that.
The list came from scanning code blocks (not prose) for loops around `nc_matmul` and attention-style code.
`transpose-and-layout.md` is also withheld at nki-L2, as a precaution.

## Version caveat

These docs target **NKI 0.4.0** (Neuron SDK 2.30). The seats run **nki 0.6.0**. Differences found by a team
session on seat-35 on 2026-10-10:

- `nisa.rsqrt` does not exist; `nisa.activation(op=nl.rsqrt)` works.
- `nl.tile_size.psum_fmax_bytes` does not exist.
- `nl.load`, `nl.store` and `nl.max(..., keepdims=True)` all simulate fine, although
  `nki-language-constraint.md` bans them.
- "MatMul K <= 2048" is wrong for a single `nc_matmul`: K is the partition axis, so 128 at most.
- The core-detection snippets in `skills/neuron-nki-debugging/references/neuron-core-isolation.md` find no
  cores on the seats.

So when they disagree, trust the installed nki's own signatures and docstrings (`inspect.signature`,
`inspect.getdoc`): they are 0.6.0 by construction, and every `nki.isa` function has one. Run any example
through `nki.simulate` before showing it to the agent.
