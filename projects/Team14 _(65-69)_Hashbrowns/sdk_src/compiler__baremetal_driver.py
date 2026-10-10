"""[INTERNAL] Baremetal NKI compilation driver.

INTERNAL MODULE - Not part of public API. May change without notice.

Wraps the standard ncc_driver compilation pipeline with baremetal-specific
settings: direct BIR emission (no wrapper), NKI schedule times preserved,
and additional ncc backend passes skipped.

This driver is used by kernel_builder and fuzz tests. It does not affect
production compilation paths in ncc_driver.py.
"""

import json
import os
import shlex
import subprocess
import tempfile
import time
from dataclasses import replace
from typing import Dict, Iterable, List, Optional

import numpy as np

from nki.compiler._errors import NCCError
from nki.compiler._internal import run_baremetal_compilation_pipeline
from nki.compiler._ncc_error_translator import build_ncc_error_message
from nki.compiler.ncc_driver import (
    MUST_ALIAS_SUFFIX,
    CompiledKernel,
    CompileOptions,
    NirResult,
    _build_birsim_backend_options,
    _build_kernel_descriptor,
    _copy_npy_files,
    _load_birsim_outputs,
    setup_sg00_directory,
)

# Baseline passes NKI handles at compile time — must be skipped in neuronx-cc.
_BASELINE_SKIP_PASSES = [
    "unroll",
    "pre_sched",
    "non_ssa_legalization",
    "tensor_copy_elim",
    "address_rotation_sb",
    "address_rotation_psum",
    "vn_splitter",
]

# Baremetal feeds NKI's schedule straight to the no-spill backend REG allocator,
# so default to min-reg-pressure: it keeps register live ranges short and avoids
# the GpSimd register overflow plain ASAP causes on gather-heavy kernels.
_SCHED_HEURISTIC_OPT = "nisa-list-scheduling-heuristic"
_DEFAULT_SCHED_HEURISTIC = "min-reg-pressure"


def _baremetal_pipeline_options(compile_opts: CompileOptions) -> Optional[str]:
    """Build the nki-baremetal-pipeline options string for compile_opts.

    Enables the min-reg-pressure scheduling heuristic unless the caller already
    pinned a heuristic via nki_opt_pipeline_options. Returns None when there are
    no options (matching the pipeline's "unset" contract).
    """
    opts = list(compile_opts.nki_opt_pipeline_options)
    if compile_opts.enable_nisa_func_multi_core:
        opts.append("enable-nisa-func=true")
    if not any(o.startswith(_SCHED_HEURISTIC_OPT) for o in opts):
        opts.append(f"{_SCHED_HEURISTIC_OPT}={_DEFAULT_SCHED_HEURISTIC}")
    return " ".join(opts) or None


def _build_neuronx_cc_args(
    compile_opts: CompileOptions,
    extra_skip_passes: Optional[List[str]] = None,
) -> List[str]:
    """Build neuronx-cc command-line arguments for baremetal compilation."""
    neuronx_cc_args = list(compile_opts.neuronx_cc_args)
    if compile_opts.target in ("trn2", "trn3pre", "trn3"):
        neuronx_cc_args.append(f"--lnc={compile_opts.lnc}")

    # Passes that are fundamentally incompatible with NKI-emitted BIR and must
    # always be skipped, even when the caller opts out of normal pass skipping.
    # dynamic_dma_setup creates DynamicDMAScratchLoc which NKI already emits;
    # running it produces a duplicate pinned memloc that fails the verifier.
    # shrink_ml would shrink the NKI-emitted DynamicDMAScratchLoc (unreferenced
    # by instructions, used directly by DGE hardware). dynamic_dma_cleanup
    # validates DGE types which NKI already assigns.
    _NKI_INCOMPATIBLE_PASSES = [
        "dynamic_dma_setup",
        "shrink_ml",
        "dynamic_dma_cleanup",
    ]

    # If the caller provides --skip-pass via neuronx_cc_args, respect it
    # instead of adding our own. --skip-pass= (empty) means skip nothing
    # (except NKI-incompatible passes that must always be skipped).
    caller_skip = [arg for arg in neuronx_cc_args if arg.startswith("--skip-pass")]
    if caller_skip:
        # Remove caller's --skip-pass from neuronx_cc_args (it goes into
        # backend_options, not top-level args).
        neuronx_cc_args = [
            arg for arg in neuronx_cc_args if not arg.startswith("--skip-pass")
        ]
        # Collect caller-specified passes and always add NKI-incompatible ones
        passes = list(_NKI_INCOMPATIBLE_PASSES)
        for arg in caller_skip:
            value = arg.split("=", 1)[1] if "=" in arg else ""
            if value:
                passes.extend(value.split(","))
        backend_options = [
            "--skip-pass=" + ",".join(passes),
            "--dge-levels=io,spill_reload",
        ]
    else:
        skip_passes = list(_BASELINE_SKIP_PASSES)
        if extra_skip_passes:
            skip_passes.extend(extra_skip_passes)
        backend_options = [
            "--skip-pass=" + ",".join(skip_passes),
            "--dge-levels=io,spill_reload",
        ]

    if compile_opts.enable_simulation:
        backend_options.extend(_build_birsim_backend_options(compile_opts))
        if compile_opts.verbose:
            print("Enabled BIRSim for numerical validation")

    # Move walrus-internal flags from neuronx_cc_args to backend_options
    walrus_flags = [a for a in neuronx_cc_args if a.startswith("--enable-data-race")]
    if walrus_flags:
        backend_options.extend(walrus_flags)
        neuronx_cc_args = [a for a in neuronx_cc_args if a not in walrus_flags]

    if compile_opts.enable_barrier_checker:
        backend_options.extend(
            ["--enable-barrier-checker=true", "--skip-barrier-checker=false"]
        )

    if backend_options:
        neuronx_cc_args.append(
            f"--internal-backend-options={' '.join(backend_options)}"
        )

    return neuronx_cc_args


def run_neuronx_cc(
    bir_path: str,
    target: str,
    neff_name: str,
    additional_args: Optional[Iterable[str]] = (),
    verbose: bool = False,
    neuronx_cc_path: str = "neuronx-cc",
    lnc_dirs: Optional[List[str]] = None,
    cwd: Optional[str] = None,
):
    """Run neuronx-cc to compile BIR JSON to .neff file."""
    assert not cwd or os.path.isabs(neff_name), (
        f"neff_name must be absolute when cwd is set, got: {neff_name}"
    )
    cmd = [
        neuronx_cc_path,
        "compile",
        "--framework",
        "XLA",
    ]

    if lnc_dirs:
        cmd.extend(lnc_dirs)
    else:
        cmd.append(bir_path)

    cmd.extend(
        [
            "--verbose=info",
            "--pipeline",
            "compile",
            "SaveTemps",
            "--enable-internal-bir-e2e-compilation",
            "--target",
            target,
        ]
    )

    if additional_args:
        cmd.extend(additional_args)

    neuronxcc_flags = os.environ.get("NEURON_CC_FLAGS")
    if neuronxcc_flags:
        cmd.extend(shlex.split(neuronxcc_flags))

    cmd.extend([f"--output={neff_name}"])

    try:
        if verbose:
            subprocess.check_call(cmd, cwd=cwd)
        else:
            result = subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                cwd=cwd,
            )
            if result.returncode != 0:
                stderr_output = result.stderr.strip() if result.stderr else ""
                raise subprocess.CalledProcessError(
                    result.returncode, cmd, output=None, stderr=stderr_output
                )
    except subprocess.CalledProcessError as e:
        stderr_text = e.stderr if hasattr(e, "stderr") and e.stderr else ""
        raise NCCError(build_ncc_error_message(e.returncode, stderr_text, cwd)) from e

    if not os.path.exists(neff_name):
        raise RuntimeError(
            f"neuronx-cc compilation failed: output file '{neff_name}' not found"
        )


# Additional ncc backend passes skipped in baremetal mode (on top of baseline).
# These fall into several categories:
#   1. NKI-redundant: NKI handles scheduling, deps, allocation, legalization
#   2. Optimization: pure perf passes with no correctness impact
#   3. Reporting/testing/debug: metrics, stats, error injection, device print, debug alloc
#   4. No-op/unused: placeholder passes, BIR templates NKI doesn't use
#   5. Single-core without collectives: collective/shared-memory passes unused in baremetal
BAREMETAL_SKIP_PASSES = [
    # -- Scheduling (NKI emits scheduled_start/scheduled_end) --
    "post_sched",
    "expand_scheduling_units",
    # -- Dependency analysis (NKI emits deps, build_fdeps adds the rest) --
    "anti_dependency_analyzer",
    # -- Allocation (NKI allocates SB/PSUM via LinearScanAllocation) --
    "coloring_allocator_psum",
    "coloring_allocator_sb",
    # -- Legalization --
    # No-op without collectives (CCE DMA only used for collective compute).
    "legalize_cce_dma",
    # NKI allocates PSUM as fp32 with bank-aligned sizes in LinearScanAllocation.
    # Caveat: if a kernel allocates PSUM with non-fp32 dtype, ncc backend would
    # normally widen it to fp32 here. NKI kernels always use fp32 PSUM since
    # PSUM hardware inherently operates in fp32 (matmul output, accumulation).
    # TODO: add assertion in LinearScanAllocation or BIREmitter to reject
    # non-fp32 PSUM allocations, since psum_tensor() accepts any dtype.
    "psum_legalization",
    # NKI emits has_psum_accumulate_flag attribute and all matmul calls have
    # explicit accum=True/False, so in principle ncc backend's accumulation group
    # inference is unnecessary (BIRSim uses has_psum_accumulate_flag to respect
    # NKI's flags; see BIREmitter.cpp:emitFunction for where it is emitted).
    # However, skipping legalize_mm_accumulation_groups triggers NCC_IBTN177
    # (invalid matmul-accumulation-flag) backend failures, so we keep it enabled
    # and let ncc infer the accumulation groups as a workaround.
    # "legalize_mm_accumulation_groups",
    # ncc backend lowers InstSelect -> GenericCopy + CopyPredicated. NKI doesn't
    # emit InstSelect; NKI's affine_select emits TensorScalarAffineSelect
    # instead. If NKI ever emits InstSelect, it should emit the lowered
    # GenericCopy + CopyPredicated pair directly in BIREmitter.
    "lower_select",
    # ncc backend expands matmul replication metadata into DMACopy Replicate ops.
    # NKI always sets replication_num_rows/resolution/shift_amnt=0 on matmuls
    # (no replication). If NKI ever needs replication, it should emit the
    # expanded DMACopy Replicate with correct memory layout directly.
    "expand_replication",
    # ncc backend lowers GenericIndirectLoad/Save -> ReadVarAddr + TensorScalarPtr
    # + IndirectLoad/Save with linearized address computation. NKI doesn't
    # emit GenericIndirect*; NKI's dma_copy_indirect emits DMACopy with
    # indirect access patterns (IndirectArgId), and local_gather emits
    # IndirectCopy — both are already hardware-level instructions that bypass
    # this pass. If NKI ever emits GenericIndirect*, it should emit the
    # hardware indirect sequence directly instead.
    "lower_generic_indirect",
    # ncc backend lowers AbstractCopy -> Load/Save/GenericCopy based on src/dst
    # memory type. NKI doesn't emit AbstractCopy; it emits concrete DMA
    # instructions directly (DMACopy, Load, Save, TensorCopy).
    "lower_ac",
    # -- Optimization passes (pure perf, safe to skip) --
    "dead_code_elim_o1",
    "dead_code_elim_o0",
    "instruction_reorder",
    "input_dma_coalescing",
    "remat_optimization",
    "dma_optimization_psum",
    "dma_optimization_sb",
    "tensorcopy_accel",
    "peephole_opts",
    "address_rotation_dram",
    "address_rotation_psum_post_schedule",
    "remove_redundancies",
    "prefetch_scheduling_before_sched",
    "prefetch_scheduling_after_sched",
    "order_column_tiled_mms",
    # dep_opt intentionally NOT skipped: it inserts the PSUM-bank anti-
    # dependencies that serialize concurrent accumulating matmuls writing the
    # same PSUM bank -- without it baremetal kernels (e.g. mlp_cte) fault at
    # runtime with TRAINIUM_NC_ERROR_TYPE_PSUM_COLLISION.
    "dep_reduction",
    # chain_dma_transposes intentionally NOT skipped: it is a hardware-error
    # mitigation and must run in baremetal.
    # coalesce_dma_blocks intentionally NOT skipped: delegate DMA coalescing to
    # the backend (see AssignDgeType commented out in buildBaremetalPipeline).
    # "coalesce_dma_blocks",
    "mem2reg",
    "seq_inst_opt",
    "optimize_prefetch_act_control",
    "optimize_act_control",
    "branch_hint",
    "label_dma_qos",
    "optimize_queue_switch",
    "constant_propagate",
    # -- Reporting / testing / debug (no functional impact) --
    "report_stats",
    "dma_metrics",
    "error_injector",
    "expand_device_print",
    "coloring_allocator_dram_debug",
    "hbm_usage",
    # -- No-op / unused passes --
    "do_nothing",
    "inline_bir_kernel",
    # ncc backend validates DgeType assignments and forces DgeType=None when DGE
    # is disabled. DGE is always enabled for NKI (--dge-levels=io,spill_reload),
    # so the force-to-None path never triggers. The remaining logic is pure
    # validation (NEURON_ASSERT). NKI sets dge_mode on DMA ops directly.
    # -- Translation bypass: skip NKIKernel wrap/unwrap --
    # Module::load() deserializes NKI BIR JSON into flat ncc backend IR.
    # NKI always emits complete flat BIR, so translate_nki_ast_to_bir
    # (wraps in NKIKernel) and inline_nki_kernel (unwraps) are redundant.
    # Skipping these also enables dynamic_dma_setup skip below, because
    # Module::load() correctly deserializes runtime_reserved/pinned flags
    # that translate_nki_ast_to_bir would drop.
    "translate_nki_ast_to_bir",
    "inline_nki_kernel",
    # dynamic_dma_setup + shrink_ml must be skipped together.
    # The BIR emitter emits DynamicDMAScratchLoc (DGE scratch at SB addr 0,
    # 128 partitions, 16384 bytes/partition). shrink_ml would shrink this
    # memloc because no BIR instructions reference it (DGE hardware uses it
    # directly), corrupting the NEFF.
    "dynamic_dma_setup",
    "shrink_ml",
    "dynamic_dma_cleanup",
    # dynamic_dma_scan intentionally NOT skipped: NKI leaves DMAs with
    # Unassigned DgeType (AssignDgeType commented out in buildBaremetalPipeline)
    # so the ncc backend assigns dge_mode.
    # "dynamic_dma_scan",
    # NOTE: insert_dma_switch_queue_instance and lower_control are NOT
    # skipped — they are needed for dynamic loops (multi-BB kernels).
    # insert_dma_switch_queue_instance: inserts queue switch at BB entry.
    # lower_control: inserts Drain+Barrier+SemaReset+Barrier before loop
    # body terminators to synchronize all engines and reset semaphore state
    # between iterations. See NKI-1699.
    # NKI's LinearScanAllocation allocates DRAM (HBM) addresses for private
    # hbm allocs (nisa.alloc<hbm>). The BIR emitter sets internal_dram_allocated
    # in baremetal mode when any HBM bind_memloc is present, telling ncc not to
    # overwrite those addresses.
    # Kernel output shared_hbm allocs emit ExternalOutput/allocated=false; the
    # driver (prepare_outputs) provides the address at runtime.
    # Scratch shared_hbm handling:
    #   lnc=1: ConvertFuncToNisaFunc converts to private hbm →
    #          LinearScanAllocation assigns address → internal_dram_allocated.
    #   lnc>1: stays as Internal/Shared/allocated=false →
    #          coloring_allocator_dram_shared (not skipped; runs because
    #          vncNcCount > 1 from --lnc=N) assigns addresses.
    # After all the above, no Internal/Local/allocated=false DRAM tensors
    # remain, so coloring_allocator_dram (Local addr space) has no candidates
    # and is a pure no-op.
    "coloring_allocator_dram",
    # NKI's DependencyAnalysis computes memory deps (RAW/WAW/WAR),
    # ACCPSUM deps, ORDER deps (alloc/release), and SSA deps (register
    # data flow via NisaMemDepOpInterface). build_fdeps adds similar deps
    # at the BIR level, but NKI already handles them in MLIR.
    # Known gaps vs build_fdeps (not yet triggered in tests):
    #   - PSUM bank conflicts: build_fdeps adds deps between different-engine
    #     ops accessing the same PSUM bank/quadrant (sync_psum_accesses).
    #   - Gen2 HW workaround: ACT/DVE/Pool PSUM sync regardless of bank.
    #   - Core V4 data corruption: transpose matmul patterns.
    "build_fdeps",
    # localize_shared_memory converts Shared+Internal tensors to Local when
    # not all cores in the LNC group access them. With the new design:
    # lnc=1 scratch shared_hbm is converted to private hbm in
    # ConvertFuncToNisaFunc, so no Shared+Internal tensors exist at lnc=1.
    # For lnc>1 scratch shared_hbm (e.g. core_barrier exchange buffers), every
    # core accesses the same named allocation, so localize_shared_memory
    # correctly leaves them as Shared. Those allocs are allocated by
    # coloring_allocator_dram_shared (not skipped, runs when vncNcCount > 1
    # via --lnc=N). Safe to skip here.
    "localize_shared_memory",
    # coalesce_multichannel_cc_ops merges consecutive per-channel
    # CollectiveCompute (PermuteImplicit/PermuteReduceImplicit) BIR instructions
    # into a single coalesced instruction with multiple inputs/outputs for more
    # efficient hardware dispatch. NKI's collective_permute_implicit emits one
    # MLIR op per channel_id which becomes separate BIR CollectiveCompute
    # instructions. This is a perf optimization, not required for correctness.
    "coalesce_multichannel_cc_ops",
    # lower_local_collectives lowers local CollectiveCompute (isLocal=true)
    # instructions into concrete DMA/TensorCopy/GPSIMDSB2SB ops bracketed by
    # CoreBarriers. It handles SendRecv (3 paths: self-copy, GPSIMDSB2SB for
    # small LNC swaps, or pull-DMA from remote core's SB), SendRecvCCE, and
    # local AllReduce (split/replicated CCE DMA patterns).
    # NKI's nisa.sendrecv emits CollectiveCompute{kind=SendRecv, isLocal=true}
    # which this pass lowers. NKI's nki.collectives APIs (all_reduce,
    # all_gather, etc.) emit isLocal=false (global) which this pass does NOT
    # lower — it only inserts CoreBarriers before them.
    # Currently skipped because NKI emits nisa.sendrecv as a high-level MLIR op
    # that the BIR emitter serializes directly — neuronxcc's lower_local_collectives
    # then handles the BIR-level lowering. If NKI were to lower sendrecv itself
    # (emitting DMACopy + CoreBarrier in BIR directly), this pass could be skipped
    # permanently.
    # TODO: Port lower_local_collectives functionality in.
    # "lower_local_collectives",
    # extend_shared_lifetimes extends Internal shared tensor lifetimes to
    # CoreBarrier synchronization points so the memory allocator doesn't reuse
    # memory while another core still needs it. NKI hoists all shared_hbm allocs
    # to function arguments (External tensors whose lifetime is managed by the
    # runtime), so there are no Internal shared tensors needing lifetime extension.
    "extend_shared_lifetimes",
    # sync_before_global_cc inserts a CoreBarrier immediately before every
    # global (isLocal=false) CollectiveCompute instruction to ensure all cores
    # have produced the CC's inputs before the inter-device collective fires.
    # This is a correctness requirement for global collectives (not perf).
    # It does NOT apply to local sendrecv (isLocal=true). For nki.collectives
    # APIs (global): must NOT be skipped. For nisa.sendrecv only: safe to skip.
    "sync_before_global_cc",
]


def compile_mlir_to_baremetal_ir(
    module,
    function_name,
    compile_opts: CompileOptions,
    work_dir: str,
    has_collectives: bool = False,
    mac_count: int = 0,
    hbm_bytes: int = 0,
) -> NirResult:
    """Compile MLIR module to BIR JSON using the baremetal pipeline."""
    work_dir = os.path.abspath(work_dir)
    os.makedirs(work_dir, exist_ok=True)

    if compile_opts.dump_mlir and compile_opts.artifacts_dir:
        mlir_path = os.path.join(work_dir, f"{function_name}.mlir")
        with open(mlir_path, "w") as f:
            module.operation.print(enable_debug_info=True, file=f)

    if compile_opts.verbose:
        print("Running MLIR pass pipeline to generate BIR (baremetal)...")

    if compile_opts.kernel_json_filename:
        kernel_json_filename = compile_opts.kernel_json_filename
    else:
        ext = "colz" if compile_opts.use_colz else "json"
        kernel_json_filename = f"{function_name}.{ext}"
    kernel_json_path = os.path.join(work_dir, kernel_json_filename)
    mlir_start_time = time.time()

    if compile_opts.pass_pipeline is not None:
        raise NotImplementedError("Custom pass pipelines are not supported yet")

    pipeline_options = _baremetal_pipeline_options(compile_opts)

    performance_metrics, bir_result = run_baremetal_compilation_pipeline(
        module,
        output_path=kernel_json_path,
        enable_timing=compile_opts.verbose,
        print_ir_after_all=compile_opts.print_ir_after_all,
        pipeline_options=pipeline_options,
        emit_reg_compute_as_affine_expr=compile_opts.emit_reg_compute_as_affine_expr,
        enable_verifier=not compile_opts.skip_verifier,
        enable_statistics=compile_opts.enable_statistics,
        emit_predicted_latency=compile_opts.emit_predicted_latency,
    )

    all_metrics = performance_metrics or {}
    func_metrics = all_metrics.get(function_name, {})
    estimated_utilization = func_metrics.get("estimated_utilization")
    total_time_ns = func_metrics.get("total_time_ns")

    mlir_time = time.time() - mlir_start_time

    if compile_opts.verbose:
        print(f"MLIR pass pipeline completed in {mlir_time:.2f}s")

    if compile_opts.dump_mlir and compile_opts.artifacts_dir:
        mlir_opt_path = os.path.join(work_dir, f"{function_name}_opt.mlir")
        with open(mlir_opt_path, "w") as f:
            module.operation.print(enable_debug_info=True, file=f)

    assert bir_result is not None, "BIR emission returned None"

    if bir_result.per_core_json_paths:
        kernel_json_path = bir_result.per_core_json_paths[0]
    else:
        kernel_json_path = bir_result.kernel_json_path
    assert os.path.isabs(kernel_json_path), (
        f"C++ pass returned relative kernel_json_path: {kernel_json_path}"
    )

    abs_sg00_dir = os.path.join(work_dir, compile_opts.sg00_dir)
    os.makedirs(abs_sg00_dir, exist_ok=True)
    _copy_npy_files(work_dir, abs_sg00_dir)
    if compile_opts.verbose:
        print(f"Kernel BIR written to: {kernel_json_path}")

    descriptor = _build_kernel_descriptor(bir_result, kernel_json_path, function_name)

    return NirResult(
        descriptor=descriptor,
        sg00_dir=abs_sg00_dir,
        work_dir=work_dir,
        mlir_time=mlir_time,
        estimated_utilization=estimated_utilization,
        total_time_ns=total_time_ns,
        sb_scratch_sizes=bir_result.sb_scratch_sizes,
        psum_scratch_sizes=bir_result.psum_scratch_sizes,
        backend_arch=bir_result.backend_arch,
        address_rotation=compile_opts.address_rotation,
        is_tuple_return=bir_result.is_tuple_return,
        none_positions=tuple(bir_result.none_positions),
        has_collectives=has_collectives,
        mac_count=mac_count,
        hbm_bytes=hbm_bytes,
        lnc=compile_opts.lnc,
        target=compile_opts.target,
        cache_hash=bir_result.cache_hash,
    )


def _serialize_birsim_inputs_from_descriptor(
    sg00_dir: str,
    input_specs: Iterable,
    user_arrays: Dict[str, np.ndarray],
    verbose: bool = False,
) -> None:
    """Serialize BIRSim input .npy files using the BIR descriptor as ground truth.

    user_arrays is keyed by spec.name (full name, including .must_alias_input
    suffix for must-alias inputs). For must-alias inputs where the full suffixed
    name is absent, the base name (without suffix) is tried as a fallback.
    """
    for spec in input_specs:
        name = spec.name
        if name not in user_arrays:
            # For must-alias inputs the spec name carries .must_alias_input;
            # the caller may have provided the base name (return_outputs=True).
            base_name = name.removesuffix(MUST_ALIAS_SUFFIX)
            if base_name in user_arrays:
                name = base_name
        arr = user_arrays[name]
        npy_path = os.path.join(sg00_dir, f"{spec.name}.npy")
        np.save(npy_path, arr)
        if verbose:
            print(f"  Serialized {spec.name} array to {npy_path}")


def compile_mlir_to_neff_baremetal(
    module,
    function_name: str,
    input_arrays: List[np.ndarray],
    argument_names: List[str],
    output_arg_names: List[str],
    compile_opts: CompileOptions,
    input_output_aliases: Optional[Dict[int, str]] = None,
    mac_count: int = 0,
    hbm_bytes: int = 0,
) -> CompiledKernel:
    """Compile MLIR module to NEFF using baremetal mode.

    Sets compilation mode to BAREMETAL (direct BIR emission, NKI schedule
    times emitted, no wrapper BIR) and adds extra ncc backend skip passes.

    In baremetal mode:
    - The MLIR pipeline emits schedule attrs (nki.scheduled_start/end)
    - BIR is emitted directly (no NKIKernel wrapper)
    - ncc backend post_sched and expand_scheduling_units are skipped
    - kernel.json is copied as bir.json in sg00/ (no wrapper generation)

    Dynamic loop kernels (scf.for/scf.while) are not supported in baremetal
    mode -- they require wrapper mode for inline_nki_kernel.

    Args:
        Same as ncc_driver.compile_mlir_to_neff.
    """
    output_path = compile_opts.output_path
    if not os.path.isabs(output_path):
        output_path = os.path.abspath(output_path)

    compile_opts = replace(compile_opts, output_path=output_path)

    if compile_opts.artifacts_dir:
        os.makedirs(compile_opts.artifacts_dir, exist_ok=True)
        work_dir = compile_opts.artifacts_dir
    else:
        # mkdtemp intentionally: the returned CompiledKernel.neff_path lives
        # inside work_dir and callers access it after this function returns,
        # so a TemporaryDirectory (auto-cleanup) would delete the neff too early.
        work_dir = tempfile.mkdtemp(prefix="nki_baremetal_")

    bir = compile_mlir_to_baremetal_ir(
        module,
        function_name,
        compile_opts,
        work_dir,
        mac_count=mac_count,
        hbm_bytes=hbm_bytes,
    )

    if input_output_aliases is None:
        input_output_aliases = bir.descriptor.to_input_output_aliases_dict() or None

    # Resolve output_path relative to work_dir (match ncc_driver convention)
    output_path = compile_opts.output_path
    if not os.path.isabs(output_path):
        output_path = os.path.join(bir.work_dir, output_path)
        compile_opts = replace(compile_opts, output_path=output_path)

    per_core_paths = list(bir.descriptor.per_core_json_paths or [])
    lnc_dirs = None

    if len(per_core_paths) > 1:
        # lnc>1: copy each per-core BIR as bir.json in nc{NN}/sg00/.
        for p in per_core_paths:
            setup_sg00_directory(p, os.path.dirname(p))
        lnc_dirs = [
            os.path.relpath(os.path.dirname(os.path.dirname(p)), bir.work_dir)
            for p in per_core_paths
        ]
        # neuronx-cc LNC mode reads info.json from the working directory.
        info_path = os.path.join(bir.work_dir, "info.json")
        if not os.path.exists(info_path):
            with open(info_path, "w") as f:
                json.dump({}, f)
    else:
        # lnc=1: copy kernel BIR as bir.{json,colz} in sg00/.
        kernel_path = (
            per_core_paths[0] if per_core_paths else bir.descriptor.kernel_json_path
        )
        setup_sg00_directory(kernel_path, bir.sg00_dir)

    # Filename neuronx-cc reads from the working directory. setup_sg00_directory
    # preserves the source extension, so colz kernels become "bir.colz".
    bir_filename = "bir" + os.path.splitext(bir.descriptor.kernel_json_path)[1]

    if compile_opts.enable_simulation:
        _serialize_birsim_inputs_from_descriptor(
            bir.sg00_dir,
            bir.descriptor.input_specs,
            user_arrays=dict(zip(argument_names, input_arrays)),
            verbose=compile_opts.verbose,
        )
        # For lnc>1, symlink shared .npy inputs into each per-core sg00/.
        if len(per_core_paths) > 1:
            for p in per_core_paths:
                sg00_dir = os.path.dirname(p)
                for npy_file in os.listdir(bir.sg00_dir):
                    if npy_file.endswith(".npy"):
                        dst = os.path.join(sg00_dir, npy_file)
                        if not os.path.exists(dst):
                            os.symlink(os.path.join(bir.sg00_dir, npy_file), dst)

    if compile_opts.verbose:
        print(f"Running neuronx-cc for target {compile_opts.target}...")

    neuronx_cc_start = time.time()
    try:
        neuronx_cc_args = _build_neuronx_cc_args(compile_opts, BAREMETAL_SKIP_PASSES)
        run_neuronx_cc(
            bir_filename,
            compile_opts.target,
            compile_opts.output_path,
            neuronx_cc_args,
            compile_opts.verbose_neuronx_cc,
            compile_opts.neuronx_cc_path,
            lnc_dirs=lnc_dirs,
            cwd=bir.work_dir,
        )
    finally:
        neuronx_cc_time = time.time() - neuronx_cc_start

    if compile_opts.verbose:
        print(f"neuronx-cc compilation completed in {neuronx_cc_time:.2f}s")

    birsim_outputs = None
    if compile_opts.enable_simulation and bir.descriptor.output_specs:
        birsim_output_dir = bir.sg00_dir
        if len(per_core_paths) > 1:
            birsim_output_dir = os.path.join(bir.work_dir, "nc00", "sg00")
            assert os.path.isdir(birsim_output_dir), (
                f"BIRSim output directory not found: {birsim_output_dir}"
            )
        # Use descriptor as ground truth for all outputs.
        all_output_names = [s.name for s in bir.descriptor.output_specs]
        birsim_outputs = _load_birsim_outputs(
            birsim_output_dir,
            output_arg_names=all_output_names,
            input_output_aliases=input_output_aliases,
            verbose=compile_opts.verbose,
        )

    return CompiledKernel(
        neff_path=compile_opts.output_path,
        target=compile_opts.target,
        lnc=compile_opts.lnc,
        artifacts_dir=compile_opts.artifacts_dir,
        mlir_time=bir.mlir_time,
        neuronx_cc_time=neuronx_cc_time,
        birsim_outputs=birsim_outputs,
        bir=bir,
    )
