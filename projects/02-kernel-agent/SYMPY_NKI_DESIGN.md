# Host-side symbolic NKI diagnostics

SymPy 1.14.0 confirmed on seat-265; NKI 0.6.0+31049202112.g85070674. `symbolic_shapes.py` never executes generated source. No SymPy imports are inserted in NKI kernels. `--shape-analysis off|sympy` defaults off and composes with existing feedback, repair, example, selection and candidate policies.

The AST translator accepts bounded integers, externally established input dimensions, shape tuples/unpacking, addition/subtraction/multiplication, nonzero-denominator division/floor division, arithmetic ceiling division, math.ceil, safe min/max, unit-stride nonnegative NKI slices with explicit bounds checks, named views, simple range/affine/sequential loops (bounded enumeration up to 64 small loop combinations), ndarray allocation and sum/max reductions. Unsupported calls, advanced/negative indexing, specialized matmul modes, conditional state merges and missing dimensions become UNKNOWN. It never calls eval, source-string sympify/parse_expr or lambdify. Source length 50,000, AST nodes 2,000, expression nodes/operations 80, depth 24 and integer literal magnitude below 2^31 bound work. These limits reduce risk; they are not an adversarial CPU-time proof.

Equality simplifies a bounded difference: zero proves equality; a proven nonzero property proves mismatch; an unresolved symbolic difference stays UNKNOWN. M-N does not prove mismatch merely because symbols differ. DMA checks only element-count agreement, leaving memory layout/engine legality to the simulator. Matmul checks contraction and each output axis separately, buffers, ordinary K/M <=128 and N/destination free extent <=512. Specialized options are excluded.

Installed primary sources: `/opt/conda/lib/python3.13/site-packages/nki/isa/_validation.py`, validate_matmul_shapes and validate_matmul_dst_shape; `nki/language/_core.py`, rank check; `nki/language/_tile_size.py`, constants. The deprecated psum_fmax=512 denotes elements per bank, not every legal PSUM allocation. Larger PSUM allocations do not by themselves trigger a 512 violation. The ordinary Trainium2 nc_matmul destination instruction does. [Official architecture](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/arch/trainium2_arch.html), [nc_matmul](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/generated/nki.isa.nc_matmul.html).

Tiling helper computes ceiling counts/final extents and whether multiple K contributions are needed. Count agreement is not proof of complete writes. Source loop coverage remains UNKNOWN unless established separately; zero-trip loop bodies are not analyzed as executed. Reduction rank is propagated from actual axes and keepdims: rank-3 axis-2 reduction can remain legal without keepdims. No unconditional reduction rule.

Feedback remains error-triggered, preserves checker text/source and adds at most 120 locally counted tokens within the available window. DMA feedback routes to transfer violations rather than later matmul hypotheses. PROVEN_VIOLATION, UNKNOWN and NO_STATIC_VIOLATION describe supported shape evidence only; no status certifies a correct kernel. Possible violations are left UNKNOWN until proved.

`synthetic_nki/symbolic_generate.py` substitutes host-side K/M/N dimensions into an original tiny offset-matmul template, validates symbolic constraints, injects wrong PSUM shape, captures actual simulator feedback and verifies three NumPy input cases after restoration. Its separate demonstration is not indexed as retrieval data and does not change prior corpus evidence. `symbolic_eval.py` evaluates preserved train/heldout mutations offline, without indexing heldout records or model requests.

Live S_control versus S_sympy uses the same diverse/diagnostic/grounded/targeted/synthetic/adaptive configuration; only the treatment adds --shape-analysis sympy. Both use frozen sources, private grading, the unchanged checker and actual endpoint usage. A solve during initial generation does not exercise SymPy and cannot establish its benefit.

```bash
cd /tmp/trainium-kernel-dev/projects/02-kernel-agent
python -B -m unittest discover -s tests -q
python -B nkibench.py --selftest
python -B kernelbench.py --selftest
output_dir=$(mktemp -d "$PWD/runs/symbolic-rebuild.XXXXXX")
python -B -m synthetic_nki.symbolic_generate --output "$output_dir/data"
python -B -m synthetic_nki.symbolic_eval --output "$output_dir/diagnoses.json"
python -B run_controlled.py --run --synthetic-pilot sympy-smoke --rounds 2 --samples 4 --repeat 1 --levels 3 4
```

NKI slicing differs from NumPy: the installed simulator raises on an oversized explicit endpoint. The analyzer now rejects such slices rather than silently clipping them; proven known loop instances can expose a final out-of-bounds tile. Unsupported or unresolved bounds remain UNKNOWN. Generic ceiling-count agreement is still not proof of full output coverage.
