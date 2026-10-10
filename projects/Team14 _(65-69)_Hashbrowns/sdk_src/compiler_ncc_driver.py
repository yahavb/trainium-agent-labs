"""[INTERNAL] NeuronX-CC Driver

INTERNAL MODULE - Not part of public API. May change without notice.

This module provides the interface for driving neuronx-cc backend compilation.
It handles MLIR → BIR → NEFF pipeline and BIRSim numerical simulation.

This module has no dependency on nki.lang.
"""

import base64
import ctypes
import glob
import json
import os
import shlex
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

from nki._version import __version__
from nki.compiler._errors import NCCError
from nki.compiler._internal import run_minimal_compilation_pipeline
from nki.compiler._ncc_error_translator import build_ncc_error_message
from nki.utils import get_binary_env_var


@dataclass(frozen=True)
class KernelDescriptor:
    """Metadata for native kernel integration.

    Public API for external callers (PyTorch/JAX/standalone/native codegen) that
    need to wrap NKI kernels as native operations in their computation graphs.

    Attributes:
        kernel_json_path: Absolute path to the kernel BIR JSON file
        input_specs: Tuple of input tensor specs (with .name and .dtype attributes)
        output_specs: Tuple of output tensor specs (with .name and .dtype attributes)
        version: NKI version string (for compatibility tracking)
        func_name: Kernel function name
        kernel_format: Format identifier (always "bir" for BIR kernels)
        aliases: Input-output aliases as (input_idx, output_idx) tuples
    """

    kernel_json_path: str
    input_specs: tuple[Any, ...]
    output_specs: tuple[Any, ...]
    version: str
    func_name: str
    kernel_format: str = "bir"
    aliases: tuple[tuple[int, int], ...] = ()
    per_core_json_paths: tuple[str, ...] = ()

    def to_input_output_aliases_dict(self) -> dict[int, str]:
        """Convert aliases to {output_idx: input_name} dict format.

        Returns dict mapping output index to the base input name (without
        .must_alias_input suffix).
        """
        result = {}
        for in_idx, out_idx in self.aliases:
            input_name = self.input_specs[in_idx].name.removesuffix(MUST_ALIAS_SUFFIX)
            result[out_idx] = input_name
        return result


# neuronx-cc BIRSim protocol: aliased inputs must use a distinct filename
# from their corresponding output to avoid collision during .npy serialization.
# BIR emitter adds this suffix to aliased input tensor names; BIRSim reads
# value_{name}.must_alias_input.npy for inputs and writes value_{name}-birsim.npy
# for outputs.
MUST_ALIAS_SUFFIX = ".must_alias_input"


class Device(Enum):
    """Execution device for running compiled kernels."""

    CPU = "cpu"  # BIRSim simulator (works anywhere)
    TRAINIUM = "trainium"  # Trainium hardware (requires spike runtime)


# Deprecated: kept for TorchNeuronEager backward compat, remove after CR-266331639 lands
class CompilationMode(Enum):
    INTEGRATION = "integration"
    STANDALONE = "standalone"


@dataclass
class ExecutionResult:
    """Result from kernel execution.

    Attributes:
        outputs: Output tensors keyed by name (e.g. {"y": array} or {"output_0": array})
        hfu: HFU (Hardware FLOPs Utilization) percentage, or None if profiling disabled
        mbu: MBU (Memory Bandwidth Utilization) percentage, or None if profiling disabled
        latency: Mean latency in seconds, or None if benchmarking disabled
        latency_min: Min latency in seconds across benchmark iterations
        latency_max: Max latency in seconds across benchmark iterations
        latency_std: Std dev of latency in seconds across benchmark iterations
        benchmark_iterations: Number of timed iterations
        benchmark_warmup: Number of warmup iterations
        input_output_aliases: Mapping of output index to input name for aliased outputs
    """

    outputs: Dict[str, np.ndarray]
    hfu: Optional[float]
    latency: Optional[float]
    mbu: Optional[float] = None
    latency_min: Optional[float] = None
    latency_max: Optional[float] = None
    latency_std: Optional[float] = None
    benchmark_iterations: Optional[int] = None
    benchmark_warmup: Optional[int] = None
    ntff_path: Optional[str] = None
    input_output_aliases: Dict[int, str] = field(default_factory=dict)

    def copy_to(
        self,
        output_specs: List,
        outputs: Dict[str, np.ndarray],
        input_output_aliases: Optional[Dict[int, str]] = None,
    ) -> None:
        """Copy execution outputs to pre-allocated output arrays.

        Resolves aliased outputs and copies via raw memmove.

        Args:
            output_specs: List of BackendTensorSpec defining output ordering.
            outputs: Pre-allocated output arrays to copy into, keyed by name.
            input_output_aliases: Override alias mapping. If None, uses self.input_output_aliases.
        """
        aliases = (
            input_output_aliases
            if input_output_aliases is not None
            else self.input_output_aliases
        )
        for i, spec in enumerate(output_specs):
            spike_key = aliases[i] if i in aliases else spec.name
            source = self.outputs[spike_key]
            target = outputs[spec.name]
            assert source.nbytes == target.nbytes, (
                f"Size mismatch in copy_to: source '{spike_key}' has {source.nbytes} bytes, "
                f"target '{spec.name}' has {target.nbytes} bytes"
            )
            ctypes.memmove(target.ctypes.data, source.ctypes.data, target.nbytes)


def copy_to_outputs(
    sources: List[np.ndarray], output_specs: List, outputs: Dict[str, np.ndarray]
) -> None:
    """Copy source arrays to output arrays using raw byte copy.

    Args:
        sources: Source arrays to copy from
        output_specs: List of BackendTensorSpec objects
        outputs: Pre-allocated output arrays keyed by name
    """
    for i, spec in enumerate(output_specs):
        source = sources[i]
        target = outputs[spec.name]
        assert source.nbytes == target.nbytes, (
            f"Size mismatch: source[{i}] has {source.nbytes} bytes, "
            f"target '{spec.name}' has {target.nbytes} bytes"
        )
        ctypes.memmove(target.ctypes.data, source.ctypes.data, target.nbytes)


def extract_perf_metrics(
    neff_path: str, ntff_path: str, verbose: bool = False
) -> Optional[Dict[str, Any]]:
    """Extract performance metrics from NTFF profile using neuron-profile CLI.

    Args:
        neff_path: Path to the NEFF file
        ntff_path: Path to the NTFF trace file
        verbose: If True, print warnings on neuron-profile failure or parse errors.

    Returns:
        Complete metrics dict from neuron-profile summary-json, or None on failure.
    """
    view_cmd = [
        "neuron-profile",
        "view",
        "-n",
        neff_path,
        "-s",
        ntff_path,
        "--output-format",
        "summary-json",
    ]

    try:
        result = subprocess.run(view_cmd, capture_output=True, text=True, check=True)
        profile_data = json.loads(result.stdout)
        model_key = next(iter(profile_data.keys()))
        return profile_data[model_key]
    except subprocess.CalledProcessError as e:
        if verbose:
            print(f"Warning: neuron-profile failed (exit {e.returncode})")
        return None
    except (json.JSONDecodeError, StopIteration) as e:
        if verbose:
            print(f"Warning: neuron-profile output parsing failed: {e}")
        return None


@dataclass
class CompiledKernel:
    """A compiled NKI kernel ready for execution.

    Returned by `compile_kernel()`. Can be executed on Trainium via `run()` or `execute()`.
    """

    neff_path: str  # Path to final .neff file
    target: str  # Compile target (trn1, trn2, trn3)
    bir: "NirResult"  # NirResult with KernelDescriptor
    lnc: int = 1  # Logical NeuronCore count (1 or 2)
    artifacts_dir: Optional[str] = None  # Directory for artifacts (None uses temp dir)
    compilation_time: float = 0.0  # Total compilation time
    frontend_time: float = 0.0  # Time spent in frontend (Python→MLIR)
    mlir_time: float = 0.0  # Time spent in MLIR pass pipeline
    neuronx_cc_time: float = 0.0  # Time spent in neuronx-cc backend compilation
    birsim_outputs: Optional[List[np.ndarray]] = (
        None  # BIRSim output arrays (if enable_simulation=True)
    )
    # Runtime state (excluded from repr/comparison).
    _models: dict[int, Any] = field(default_factory=dict, repr=False, compare=False)

    @property
    def input_specs(self):
        return list(self.bir.descriptor.input_specs)

    @property
    def output_specs(self):
        return list(self.bir.descriptor.output_specs)

    @property
    def is_tuple_return(self):
        return self.bir.is_tuple_return

    @property
    def input_output_aliases(self):
        return self.bir.descriptor.to_input_output_aliases_dict()

    @property
    def none_positions(self):
        return tuple(int(i) for i in self.bir.none_positions)

    @property
    def estimated_utilization(self):
        return self.bir.estimated_utilization

    @property
    def total_time_ns(self):
        return self.bir.total_time_ns

    @classmethod
    def from_frontend(
        cls, frontend_result, compile_opts, frontend_time=0.0, tensor_inputs=None
    ):
        """Compile a frontend CompilationResult (MLIR) to CompiledKernel (NEFF).

        Args:
            tensor_inputs: Optional dict of name→ndarray for BIRSim simulation.
                Required when compile_opts.enable_simulation is True.
        """
        if compile_opts.enable_simulation and tensor_inputs is None:
            raise ValueError("tensor_inputs is required when enable_simulation is True")
        result = compile_mlir_to_neff(
            frontend_result.module,
            frontend_result.function_name,
            compile_opts=compile_opts,
            tensor_inputs=tensor_inputs,
        )
        return replace(
            result,
            frontend_time=frontend_time,
            compilation_time=frontend_time + result.mlir_time + result.neuronx_cc_time,
        )

    def prepare_outputs(self):
        """Allocate output arrays from output_specs.

        Requires output_specs to contain spec objects with shape/dtype
        (set by from_frontend). Uses ml_dtypes for bf16/fp8 to preserve
        exact byte sizes — copy_to_outputs does raw memmove.
        """
        from nki.compiler import _dtypes

        result = {}
        for spec in self.output_specs:
            if spec is None:
                continue
            try:
                np_dtype = _dtypes.dtype.from_name(spec.dtype).to_numpy()
            except ValueError:
                # Dtype not in compiler registry (e.g. NRT "unknown"/"float32r").
                if not spec.element_size_bytes:
                    raise ValueError(
                        f"Cannot determine element size for output '{spec.name}' "
                        f"with dtype '{spec.dtype}'"
                    )
                np_dtype = np.dtype(f"V{spec.element_size_bytes}")
            result[spec.name] = np.empty(spec.shape, dtype=np_dtype)
        return result

    def prepare_inputs(self, inputs):
        """Rename aliased input keys per MUST_ALIAS_SUFFIX convention."""
        result = dict(inputs)
        for input_name in self.input_output_aliases.values():
            if input_name in result:
                result[f"{input_name}{MUST_ALIAS_SUFFIX}"] = result.pop(input_name)
        return result

    def _ensure_loaded(self, rank_id: int, world_size: int):
        """Lazy-load SpikeModel from NEFF."""

        if rank_id not in self._models:
            from nki.runtime import SpikeModel

            self._models[rank_id] = SpikeModel.load_from_neff(
                neff_path=self.neff_path,
                core_id=rank_id,
                cc_enabled=world_size > 1,
                rank_id=rank_id,
                world_size=world_size,
            )

        return self._models[rank_id]

    def run(
        self,
        profile: bool = False,
        rank_id: int = 0,
        world_size: int = 1,
        **inputs,
    ) -> ExecutionResult:
        """Execute kernel and return result.

        Args:
            profile: If True, save execution trace and extract HFU metrics
            rank_id: Rank/core this execution runs on (multi-rank)
            world_size: Total number of ranks (multi-rank)
            **inputs: Input arrays by name (e.g., a=array1, b=array2)

        Returns:
            ExecutionResult with outputs dict and HFU (None if profile=False)
        """
        return self._run(
            inputs, profile=profile, rank_id=rank_id, world_size=world_size
        )

    def _run(
        self,
        inputs: Dict[str, np.ndarray],
        *,
        profile: bool = False,
        rank_id: int = 0,
        world_size: int = 1,
    ) -> ExecutionResult:
        """Core execution path.

        Inputs are passed as an explicit dict (not ``**inputs``) so that kernel
        input names such as ``rank_id`` or ``world_size`` cannot collide with the
        control parameters when this is invoked from ``execute``.
        """
        from nki.runtime import SpikeTensor

        model = self._ensure_loaded(rank_id, world_size)

        # Convert inputs to SpikeTensors
        spike_inputs = {
            k: SpikeTensor.from_numpy(v, name=k, core_id=rank_id)
            for k, v in inputs.items()
        }

        # Allocate outputs from output_specs (correct sizes for packed x4
        # types) rather than from NRT metadata which underreports size for
        # dtypes it doesn't recognise.
        all_outputs = {
            name: SpikeTensor.from_numpy(arr, name=name, core_id=rank_id)
            for name, arr in self.prepare_outputs().items()
        }

        if profile:
            WARMUP_ITERATIONS = 5
            BENCHMARK_ITERATIONS = 10

            # Stable mean latency via device-side tracing (replaces noisy single-run NTFF total_time).
            bench = model.benchmark(
                spike_inputs,
                outputs=all_outputs,
                warmup_iter=WARMUP_ITERATIONS,
                benchmark_iter=BENCHMARK_ITERATIONS,
                mode="device",
            )
            mean_latency = bench.mean_ms / 1000.0

            # HFU/MBU from BIR compile-time metadata + mean latency.
            peak_ops = self._PEAK_OPS_PER_SEC[self.target]
            peak_bw = self._PEAK_HBM_BW[self.target]
            hfu = (self.bir.mac_count * 2) / (peak_ops * mean_latency * self.lnc)
            mbu = self.bir.hbm_bytes / (peak_bw * mean_latency * self.lnc)
            latency = mean_latency
            latency_min = bench.min_ms / 1000.0
            latency_max = bench.max_ms / 1000.0
            latency_std = bench.std_dev_ms / 1000.0
        else:
            model(spike_inputs, outputs=all_outputs, save_trace=False)
            hfu = None
            latency = None
            mbu = None
            latency_min = None
            latency_max = None
            latency_std = None

        # Convert outputs to numpy
        outputs = {name: tensor.numpy() for name, tensor in all_outputs.items()}

        return ExecutionResult(
            outputs=outputs,
            hfu=hfu,
            latency=latency,
            mbu=mbu,
            latency_min=latency_min,
            latency_max=latency_max,
            latency_std=latency_std,
            input_output_aliases=self.input_output_aliases,
        )

    def execute(
        self,
        inputs: Dict[str, np.ndarray],
        outputs: Dict[str, np.ndarray],
        *,
        rank_id: int = 0,
        world_size: int = 1,
        device: Device = Device.TRAINIUM,
        generate_perfetto_trace: bool = False,
    ) -> "ExecutionResult":
        """Execute kernel and write results to output arrays in place.

        Args:
            inputs: Dict mapping input names to numpy arrays
            outputs: Dict mapping output names to numpy arrays (mutated in place)
            device: Device.CPU (uses birsim_outputs) or Device.TRAINIUM (runs on hardware)
            generate_perfetto_trace: If True, generate perfetto trace and upload to S3

        Returns:
            ExecutionResult with outputs, HFU, MBU, and latency.
        """
        if device == Device.CPU:
            if self.birsim_outputs is None:
                raise RuntimeError("BIRSim simulation failed to produce outputs")
            copy_to_outputs(
                sources=self.birsim_outputs,
                output_specs=self.output_specs,
                outputs=outputs,
            )
            aliases = self.input_output_aliases or {}
            result_outputs = {}
            for i, spec in enumerate(self.output_specs):
                spike_key = aliases[i] if i in aliases else spec.name
                result_outputs[spike_key] = self.birsim_outputs[i]
            return ExecutionResult(
                outputs=result_outputs,
                hfu=None,
                latency=None,
                mbu=None,
                input_output_aliases=self.input_output_aliases,
            )
        else:
            # Execute on hardware
            result = self._run(
                inputs,
                profile=generate_perfetto_trace,
                rank_id=rank_id,
                world_size=world_size,
            )
            result.copy_to(self.output_specs, outputs)
            return result

    # Peak BF16 ops/sec per NeuronCore.
    _PEAK_OPS_PER_SEC = {
        "trn1": 128 * 128 * 2 * 2.8e9,
        "trn2": 128 * 128 * 2 * 2.4e9,
        "trn3pre": 128 * 128 * 2 * 2.4e9,
        "trn3": 128 * 128 * 2 * 2.4e9,
    }

    # Peak HBM bandwidth per NeuronCore (bytes/sec).
    _PEAK_HBM_BW = {
        "trn1": 410e9,
        "trn2": 716e9,
        "trn3pre": 1178e9,
        "trn3": 1178e9,
    }

    def benchmark(
        self,
        warmup: int = 5,
        iterations: int = 10,
        rank_id: int = 0,
        world_size: int = 1,
        **kwargs,
    ) -> "ExecutionResult":
        """Benchmark kernel execution and return outputs + latency.

        Runs warmup + timed iterations with device-side tracing.
        Returns ExecutionResult with outputs from the last iteration
        and latency stats from the benchmark.

        Args:
            warmup: Number of warmup iterations
            iterations: Number of benchmark iterations
            **kwargs: Input arrays by name

        Returns:
            ExecutionResult with outputs, latency (seconds), and latency stats.
        """
        from nki.runtime import SpikeTensor

        model = self._ensure_loaded(rank_id, world_size)

        spike_inputs = {
            k: SpikeTensor.from_numpy(v, name=k, core_id=rank_id)
            for k, v in kwargs.items()
        }
        all_outputs = {
            name: SpikeTensor.from_numpy(arr, name=name, core_id=rank_id)
            for name, arr in self.prepare_outputs().items()
        }

        bench = model.benchmark(
            spike_inputs,
            outputs=all_outputs,
            warmup_iter=warmup,
            benchmark_iter=iterations,
            mode="device",
        )

        outputs = {name: tensor.numpy() for name, tensor in all_outputs.items()}

        return ExecutionResult(
            outputs=outputs,
            hfu=None,
            latency=bench.mean_ms / 1000.0,
            latency_min=bench.min_ms / 1000.0,
            latency_max=bench.max_ms / 1000.0,
            latency_std=bench.std_dev_ms / 1000.0,
            mbu=None,
            benchmark_iterations=iterations,
            benchmark_warmup=warmup,
            input_output_aliases=self.input_output_aliases,
        )


@dataclass(frozen=True)
class CompileOptions:
    """
    Compilation configuration options for neuronx-cc compilation.

    This frozen dataclass encapsulates all compilation settings, providing a
    type-safe and reusable way to configure compilation behavior.

    Attributes:
        target: Target architecture ("trn1" or "trn2")
        lnc: Logical NeuronCore count (1 or 2) - number of cores for multi-core parallelism
        verbose: Whether to show compilation output. Defaults to False.
            Can be enabled via the NKI_VERBOSE_COMPILE environment variable
            (e.g., NKI_VERBOSE_COMPILE=1). Explicit values passed to the
            constructor take precedence over the environment variable.
        verbose_neuronx_cc: Whether to show verbose neuronx-cc output
        artifacts_dir: Optional directory to preserve intermediate files.
            Can also be set via the NKI_ARTIFACTS_DIR environment variable.
            Explicit values passed to the constructor take precedence over the
            environment variable. Note: neuronx-cc requires a clean directory;
            reusing the same directory will fail on subsequent compilations.
        neuronx_cc_args: Additional arguments for neuronx-cc backend
        neuronx_cc_backend_opts: Extra options merged into the neuronx-cc
            --internal-backend-options="..."
        neuronx_cc_path: Path to neuronx-cc executable
        pass_pipeline: Optional custom MLIR pass pipeline (not yet supported)
        output_path: Optional custom path for output .neff file
        enable_simulation: Whether to run BIRSim numerical simulation
        run_at_begin: Run BIRSim at the beginning of compilation (before optimizations)
        run_at_end: Run BIRSim at the end of compilation (after optimizations)
        nki_opt_pipeline_options: Tuple of options for nki-opt-pipeline
            (e.g., ("enable-instruction-scheduling=false", "enable-linear-scan-allocation=false"))
        print_ir_after_all: Print MLIR after every pass
        dump_mlir: Dump MLIR to artifacts_dir before and after pipeline

    BIRSim Debugging:
        Use run_at_begin/run_at_end to diagnose compilation issues:
        - If run_at_begin passes but run_at_end fails → bug in neuronx-cc lowering
        - If run_at_begin fails → bug in MLIR/BIR generation

    Example:
        opts = CompileOptions(
            target="trn1",
            verbose=True,
            artifacts_dir="./debug",
            enable_simulation=True,
        )
        result = nb.compile_kernel(kernel, a, b, compile_opts=opts)
        assert np.allclose(result.birsim_outputs[0], expected, rtol=1e-3)
    """

    target: str = "trn1"
    lnc: int = 1  # Logical NeuronCore count (1 or 2)
    verbose: bool = field(
        default_factory=lambda: get_binary_env_var("NKI_VERBOSE_COMPILE", False)
    )
    verbose_neuronx_cc: bool = False
    artifacts_dir: Optional[str] = None
    neuronx_cc_args: tuple[str, ...] = field(default_factory=tuple)
    neuronx_cc_backend_opts: tuple[str, ...] = field(default_factory=tuple)
    neuronx_cc_path: str = "neuronx-cc"
    pass_pipeline: Optional[str] = None
    output_path: str = "file.neff"

    # BIRSim configuration (merged from SimulationMode)
    enable_simulation: bool = False
    run_at_begin: bool = False
    run_at_end: bool = True

    # MLIR pass pipeline options
    nki_opt_pipeline_options: tuple[str, ...] = field(default_factory=tuple)

    # MLIR IR printing options
    print_ir_after_all: bool = False
    dump_mlir: bool = False  # Dump MLIR to artifacts_dir (before/after pipeline)

    # Skip MLIR verification — research mode for constraint probing.
    # When True, disables both module-level verification after tracing
    # and verify-after-each-pass during the MLIR pipeline.
    skip_verifier: bool = False

    # Print MLIR pass statistics (e.g. dep counts from InstructionScheduling).
    enable_statistics: bool = False

    # Required for latency calibration pipeline; adds per-instruction predicted_latency_ns to BIR.
    emit_predicted_latency: bool = False

    # Device dump: insert DevicePrint after every SBUF-writing ISA instruction (tracer only)
    enable_device_dump: bool = False

    # Address rotation: allow backend to rotate compiler-managed tensor addresses.
    # Controls whether neuronx-cc's ADDRESS_ROTATION_SB/PSUM passes may rotate
    # addresses for better memory utilization. Per-tensor optin_passes in the
    # kernel BIR control which memlocs are eligible. See NKI-1674.
    address_rotation: bool = True

    # Enable the LNC core barrier checker. On LNC=2, fails compilation when
    # one PNC writes a region that the other PNC reads or writes with no core
    # barrier on a common engine between the two accesses. See
    # `Kernel.barrier_check` for the user-facing kwarg that sets this.
    enable_barrier_checker: bool = False

    enable_nisa_func_multi_core: bool = False

    # BIR emission options (not pipeline options — consumed only by emitBIR)
    emit_reg_compute_as_affine_expr: bool = False

    # Lower dma_transpose to nc_transpose on TensorE (workaround for AXI deadlock
    # when DMA collectives overlap with DMA transpose).
    lower_dma_transpose: bool = False

    # Emit kernel debug info (*.debuginfo.json) for Neuron Explorer.
    # Can be enabled via NKI_DEBUG_INFO=1 environment variable.
    debug: bool = field(
        default_factory=lambda: get_binary_env_var("NKI_DEBUG_INFO", False)
    )

    # Emit BIR in the compressed columnar .colz format instead of JSON.
    # On by default; disable via NKI_USE_COLZ=0 environment variable.
    use_colz: bool = field(
        default_factory=lambda: get_binary_env_var("NKI_USE_COLZ", True)
    )

    # Internal options
    sg00_dir: str = "sg00"
    kernel_json_filename: Optional[str] = None

    # Deprecated: kept for TorchNeuronEager backward compat, remove after CR-266331639
    mode: Optional[Any] = None

    def __post_init__(self):
        """Validate configuration options."""
        from nki.compiler.target import _normalize_target

        canonical = _normalize_target(self.target)
        if canonical != self.target:
            object.__setattr__(self, "target", canonical)

        if self.target not in ["trn1", "trn2", "trn3pre", "trn3"]:
            raise ValueError(
                f"Invalid target: {self.target}. "
                "Must be 'trn1', 'inf2', 'trn2', or 'trn3'"
            )

        if self.lnc not in (1, 2):
            raise ValueError(f"lnc must be 1 or 2, got {self.lnc}")

        if self.pass_pipeline is not None:
            raise NotImplementedError("Custom pass pipelines are not supported yet")

        if not self.artifacts_dir:
            env_dir = os.environ.get("NKI_ARTIFACTS_DIR")
            if env_dir:
                object.__setattr__(self, "artifacts_dir", env_dir)

        env_lower = os.environ.get("NKI_DMA_TRANSPOSE_AS_PE_TRANSPOSE")
        if env_lower is not None:
            object.__setattr__(
                self,
                "lower_dma_transpose",
                env_lower.lower() in ("1", "true", "yes", "on"),
            )

    def set_pipeline_options(self, *options: str) -> "CompileOptions":
        """Return a new CompileOptions with additional pipeline options appended.

        Example:
            opts = base_opts.set_pipeline_options("enable-scheduling=true", "profile-path=/tmp")
        """
        return replace(
            self, nki_opt_pipeline_options=self.nki_opt_pipeline_options + options
        )

    def clear_pipeline_options(self, *option_prefixes: str) -> "CompileOptions":
        """Return a new CompileOptions with specified pipeline options removed.

        Options are matched by prefix (before '='), so "profile-path" matches "profile-path=/foo".

        Example:
            opts = base_opts.clear_pipeline_options("profile-path", "enable-scheduling")
        """

        def matches_any_prefix(opt: str) -> bool:
            opt_name = opt.split("=")[0]
            return opt_name in option_prefixes

        filtered = tuple(
            opt for opt in self.nki_opt_pipeline_options if not matches_any_prefix(opt)
        )
        return replace(self, nki_opt_pipeline_options=filtered)

    def disable_backend_optimizations(self) -> "CompileOptions":
        """Disable neuronx-cc scheduling and allocation (default for baremetal mode)."""
        opts = self.set_pipeline_options(
            "enable-linear-scan-allocation=false",
            "enable-instruction-scheduling=false",
        )
        return replace(opts, emit_reg_compute_as_affine_expr=True)


def setup_sg00_directory(
    bir_source_path: str,
    sg00_dir: str,
) -> str:
    """
    Create sg00 directory structure and copy BIR file.

    The destination filename is ``bir<ext>`` where ``<ext>`` is taken from
    ``bir_source_path`` (typically ``.json`` or ``.colz``). Preserves the
    extension so neuronx-cc can detect the format from the filename.

    Args:
        bir_source_path: Path to source BIR file (.json or .colz)
        sg00_dir: Directory path for sg00 structure

    Returns:
        Path to BIR file in sg00 directory.
    """
    os.makedirs(sg00_dir, exist_ok=True)
    ext = os.path.splitext(bir_source_path)[1] or ".json"
    bir_path = os.path.join(sg00_dir, f"bir{ext}")
    shutil.copyfile(bir_source_path, bir_path)
    # Copy constant .npy files (from shared_constant) alongside the BIR.
    # neuronx-cc needs them in sg00/ to embed in the NEFF.
    # kernel.{json,colz} stays in the top-level artifacts directory.
    bir_source_dir = os.path.dirname(bir_source_path) or "."
    _copy_npy_files(bir_source_dir, sg00_dir)
    return bir_path


def _copy_npy_files(src_dir: str, dst_dir: str) -> None:
    """Copy all .npy files from src_dir to dst_dir."""
    for npy_file in glob.glob(os.path.join(src_dir, "*.npy")):
        shutil.copyfile(npy_file, os.path.join(dst_dir, os.path.basename(npy_file)))


def strip_lnc_suffix(name: str) -> str:
    """Strip _lncN suffix added by frontend for multi-core kernels."""
    lnc_pos = name.rfind("_lnc")
    if lnc_pos != -1 and name[lnc_pos + 4 :].isdigit():
        return name[:lnc_pos]
    return name


def _build_kernel_descriptor(
    bir_result,
    kernel_json_path: str,
    function_name: str,
) -> "KernelDescriptor":
    """Build KernelDescriptor from BIR compilation result.

    Args:
        bir_result: Result from run_minimal_compilation_pipeline or
            run_baremetal_compilation_pipeline
        kernel_json_path: Absolute path to kernel JSON file
        function_name: Original function name (may have _lncN suffix)

    Returns:
        KernelDescriptor with input/output specs and aliases
    """
    per_core = bir_result.per_core_json_paths
    per_core_paths = tuple(per_core) if per_core else ()
    return KernelDescriptor(
        kernel_json_path=kernel_json_path,
        input_specs=tuple(bir_result.input_specs),
        output_specs=tuple(bir_result.output_specs),
        version=__version__,
        func_name=strip_lnc_suffix(function_name),
        kernel_format=bir_result.kernel_format,
        aliases=tuple(bir_result.input_output_aliases),
        per_core_json_paths=per_core_paths,
    )


def _load_birsim_outputs(
    sg00_dir: str,
    output_arg_names: List[str],
    input_output_aliases: Optional[Dict[int, str]] = None,
    verbose: bool = False,
) -> Optional[List[np.ndarray]]:
    """
    Load BIRSim output arrays from sg00 directory.

    BIRSim generates output files as: {tensor_name}-birsim.npy
    for each output tensor (internal allocations that are returned).
    For aliased outputs, the file is named {input_name}-birsim.npy.

    Args:
        sg00_dir: Directory containing BIRSim output files
        output_arg_names: List of output argument names (used for count only)
        input_output_aliases: Mapping of output index to input name for aliased outputs
        verbose: Print loading details

    Returns:
        List of numpy arrays in order, or None if any file is missing
    """
    aliases = input_output_aliases or {}
    outputs = []

    for i in range(len(output_arg_names)):
        if i in aliases:
            filename = f"{aliases[i]}-birsim.npy"
        else:
            filename = f"{output_arg_names[i]}-birsim.npy"
        filepath = os.path.join(sg00_dir, filename)

        if not os.path.exists(filepath):
            if verbose:
                print(f"Warning: Expected BIRSim output file not found: {filename}")
            return None

        arr = np.load(filepath)
        outputs.append(arr)

        if verbose:
            print(
                f"  Loaded BIRSim output {i}: {filename} shape={arr.shape} dtype={arr.dtype}"
            )

    if verbose:
        print(f"Loaded {len(outputs)} BIRSim output array(s)")

    return outputs


def _build_birsim_backend_options(compile_opts: CompileOptions) -> List[str]:
    """Build BIRSim backend options for neuronx-cc."""
    at_begin = "true" if compile_opts.run_at_begin else "false"
    at_end = "true" if compile_opts.run_at_end else "false"
    return [
        "--enable-birsim=true",
        f"--enable-birsim-at-begin={at_begin}",
        f"--enable-birsim-at-end={at_end}",
        "--enable-check-outputs=false",
    ]


@dataclass(frozen=True)
class FrameworkConfig:
    """Public API: framework integration config returned by NirResult.build_config().

    This is the sole interface between NKI and framework integrations
    (JAX, TorchXLA, TorchEager). Frameworks depend on these fields.
    """

    backend_config: dict
    backend_config_b64: bytes
    output_specs: tuple
    operand_output_aliases: Dict[int, int]
    has_collectives: bool


@dataclass(eq=False)
class NirResult:
    """Result of NKI compilation (MLIR to NIR).

    Semi-public: frameworks receive this from compile_kernel_to_nir() and
    call build_config() to get a FrameworkConfig. Internal fields (descriptor,
    sg00_dir, work_dir, etc.) are not part of the public API.
    """

    descriptor: "KernelDescriptor"  # Public API - kernel descriptor for wrappers

    # Internal compilation state (not exposed to external callers)
    sg00_dir: str  # Absolute path to sg00 directory
    work_dir: str  # Working directory containing compilation outputs
    mlir_time: float  # Time spent in MLIR pass pipeline
    estimated_utilization: Optional[Dict[str, float]] = None
    total_time_ns: Optional[int] = None

    # Time spent in the frontend (source → MLIR). Set by compile_to_bir, which
    # is the only caller that runs the frontend before compile_mlir_to_bir.
    # Stays 0.0 on paths that receive an already-built MLIR module.
    frontend_time: float = 0.0

    # Baremetal-specific fields (not used in neuronx-cc path)
    sb_scratch_sizes: Optional[List[int]] = None
    psum_scratch_sizes: Optional[List[int]] = None
    backend_arch: Optional[str] = None
    address_rotation: bool = True

    # Return type metadata for Python wrapper
    is_tuple_return: bool = False
    none_positions: tuple = ()

    # Frontend metadata (populated from CompilationResult via compile_to_bir)
    has_collectives: bool = False
    mac_count: int = 0
    hbm_bytes: int = 0
    cache_hash: str = ""  # Debug-free content hash from BIR emission

    # Compilation config (populated from CompileOptions)
    lnc: int = 1
    target: str = ""

    # ── Public API: convenience properties ──

    @property
    def output_specs(self):
        """Public API: output tensor specs (shape, dtype, name)."""
        return list(self.descriptor.output_specs)

    @property
    def input_specs(self):
        """Public API: input tensor specs (shape, dtype, name)."""
        return list(self.descriptor.input_specs)

    # ── Backward compat for TorchNeuronEager (remove after CR-266331639) ──

    @property
    def kernel_json_path(self):
        """Deprecated: use descriptor.kernel_json_path instead."""
        import warnings

        warnings.warn(
            "NirResult.kernel_json_path is deprecated. "
            "Use descriptor.kernel_json_path instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.descriptor.kernel_json_path

    @property
    def input_output_aliases(self):
        """Deprecated: {output_idx: input_name} with must_alias_input suffix.

        Kept for TorchNeuronEager mainline compat — names include suffix so
        they match input_specs[i].name for lookup.
        """
        import warnings

        warnings.warn(
            "NirResult.input_output_aliases is deprecated. "
            "Use build_config().operand_output_aliases instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        result = {}
        for in_idx, out_idx in self.descriptor.aliases:
            result[out_idx] = self.descriptor.input_specs[in_idx].name
        return result

    def __iter__(self):
        """Deprecated: compat for (bir, compilation_result) tuple unpack."""
        import warnings

        warnings.warn(
            "Unpacking NirResult as (bir, result) is deprecated. "
            "Use nir = compile_kernel_to_nir(...) directly.",
            DeprecationWarning,
            stacklevel=2,
        )
        return iter((self, self))

    def build_config(self) -> FrameworkConfig:
        """Public API: returns everything framework integrations need.

        Frameworks call compile_kernel_to_nir() then build_config().
        """
        from nki.compiler.target import target_to_nc_version

        aliases_dict = self.descriptor.to_input_output_aliases_dict()
        output_names = []
        for idx, spec in enumerate(self.descriptor.output_specs):
            if idx in aliases_dict:
                output_names.append(aliases_dict[idx])
            else:
                output_names.append(spec.name)

        backend_config = {
            "func_name": self.descriptor.func_name,
            "platform_target": target_to_nc_version(self.target),
            "kernel_format": self.descriptor.kernel_format,
            "kernel_version": 1,
            "klir_binary": {
                "binary": self.descriptor.kernel_json_path,
                "input_names": [s.name for s in self.descriptor.input_specs],
                "output_names": output_names,
                "version_identifier": "",
                "aliases": [list(pair) for pair in self.descriptor.aliases],
            },
            "grid": self.lnc,
            "has_collectives": self.has_collectives,
            "mac_count": self.mac_count,
        }

        backend_config_b64 = base64.b64encode(json.dumps(backend_config).encode())

        return FrameworkConfig(
            backend_config=backend_config,
            backend_config_b64=backend_config_b64,
            output_specs=tuple(self.descriptor.output_specs),
            operand_output_aliases={
                in_idx: out_idx for in_idx, out_idx in self.descriptor.aliases
            },
            has_collectives=self.has_collectives,
        )


def compile_mlir_to_bir(
    module,
    function_name,
    compile_opts: CompileOptions,
    work_dir: str,
    enable_cache: bool = True,
    has_collectives: bool = False,
    mac_count: int = 0,
    debug_asts_capsule=None,
) -> NirResult:
    """Compile MLIR module to BIR JSON using the integration pipeline.

    Includes disk caching for Jax/TorchXLA deterministic paths.
    """
    work_dir = os.path.abspath(work_dir)
    os.makedirs(work_dir, exist_ok=True)

    if compile_opts.dump_mlir and compile_opts.artifacts_dir:
        mlir_path = os.path.join(work_dir, f"{function_name}.mlir")
        with open(mlir_path, "w") as f:
            module.operation.print(enable_debug_info=True, file=f)

    if compile_opts.verbose:
        print("Running MLIR pass pipeline to generate BIR...")

    if compile_opts.kernel_json_filename:
        kernel_json_filename = compile_opts.kernel_json_filename
    else:
        ext = "colz" if compile_opts.use_colz else "json"
        kernel_json_filename = f"{function_name}.{ext}"
    kernel_json_path = os.path.join(work_dir, kernel_json_filename)
    mlir_start_time = time.time()

    if compile_opts.pass_pipeline is not None:
        raise NotImplementedError("Custom pass pipelines are not supported yet")

    pipeline_options = " ".join(compile_opts.nki_opt_pipeline_options) or None

    performance_metrics, bir_result = run_minimal_compilation_pipeline(
        module,
        output_path=kernel_json_path,
        enable_timing=compile_opts.verbose,
        print_ir_after_all=compile_opts.print_ir_after_all,
        pipeline_options=pipeline_options,
        emit_reg_compute_as_affine_expr=compile_opts.emit_reg_compute_as_affine_expr,
        enable_verifier=not compile_opts.skip_verifier,
        debug_asts_capsule=debug_asts_capsule,
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
    kernel_json_path = bir_result.kernel_json_path
    assert os.path.isabs(
        kernel_json_path
    ), f"C++ pass returned relative kernel_json_path: {kernel_json_path}"

    # Cache kernel BIR for Jax/TorchXLA deterministic paths.
    cache_hash = bir_result.cache_hash
    if enable_cache and cache_hash:
        from nki.compiler._disk_cache import cache_bir_artifacts

        kernel_json_path = cache_bir_artifacts(
            kernel_json_path,
            cache_hash,
            work_dir,
            function_name=function_name,
            verbose=compile_opts.verbose,
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
        lnc=compile_opts.lnc,
        target=compile_opts.target,
        cache_hash=bir_result.cache_hash,
    )


def _populate_per_core_birsim_inputs(work_dir: str, sg00_dir: str, lnc: int) -> None:
    """Pre-create nc{XX}/sg00/ directories with symlinks to input .npy files.

    BIRSim runs per-core and looks for input .npy files in each core's
    nc{XX}/sg00/ directory. Create those directories and symlink all .npy
    files from the top-level sg00/ so BIRSim can find them.
    """
    npy_files = [f for f in os.listdir(sg00_dir) if f.endswith(".npy")]
    for nc_idx in range(lnc):
        nc_sg00 = os.path.join(work_dir, f"nc{nc_idx:02d}", "sg00")
        os.makedirs(nc_sg00, exist_ok=True)
        for npy_file in npy_files:
            dst = os.path.join(nc_sg00, npy_file)
            if not os.path.exists(dst):
                src = os.path.join(sg00_dir, npy_file)
                os.symlink(src, dst)


def _find_birsim_output_dir(work_dir: str, sg00_dir: str) -> str:
    """Return the directory containing BIRSim output files.

    For LNC2+ kernels, neuronx-cc places per-core BIRSim outputs in
    nc00/sg00/ instead of the top-level sg00/. Check for that layout
    and fall back to sg00_dir for single-core kernels.
    """
    nc00_sg00 = os.path.join(work_dir, "nc00", "sg00")
    if os.path.isdir(nc00_sg00):
        return nc00_sg00
    return sg00_dir


def _symlink_birsim_file(
    sg00_dir: str, name: str, prefix: str, reverse: bool = False
) -> None:
    """Create a symlink for BIRSim file naming convention.

    Forward (reverse=False): value_{name}.npy -> {name}.npy
    Reverse (reverse=True): {name}.npy -> value_{name}.npy
    """
    original = os.path.join(sg00_dir, f"{name}.npy")
    prefixed = os.path.join(sg00_dir, f"{prefix}{name}.npy")
    if reverse:
        src, dst = prefixed, original
    else:
        src, dst = original, prefixed
    if os.path.exists(src) and not os.path.exists(dst):
        os.symlink(os.path.basename(src), dst)


def _compile_neff(
    compile_opts: CompileOptions,
    bir: "NirResult",
    input_output_aliases: Optional[Dict[int, str]] = None,
) -> None:
    """Compile kernel using neuronxcc's compile_nki_ir_kernel_to_neff API."""
    from neuronxcc.nki_standalone import NKI_IR_VERSION, compile_nki_ir_kernel_to_neff

    # Build KLIR binary dict for neuronxcc's compile_nki_ir_kernel_to_neff API
    kernel_dict = {
        "binary": bir.descriptor.kernel_json_path,
        "input_names": [s.name for s in bir.descriptor.input_specs],
        "output_names": [s.name for s in bir.descriptor.output_specs],
        "version_identifier": "0.2.0",
        "func_name": bir.descriptor.func_name,
        "kernel_format": bir.descriptor.kernel_format,
        "aliases": list(bir.descriptor.aliases),
    }
    kernel_inputs_dict = {s.name: s for s in bir.descriptor.input_specs}
    kernel_outputs = list(bir.descriptor.output_specs)

    neff_path = compile_opts.output_path
    if not os.path.isabs(neff_path):
        neff_path = os.path.join(bir.work_dir, neff_path)

    backend_opts = [
        *compile_opts.neuronx_cc_backend_opts,
    ]
    if compile_opts.enable_simulation:
        backend_opts.extend(_build_birsim_backend_options(compile_opts))

    if compile_opts.enable_barrier_checker:
        backend_opts.extend(
            ["--enable-barrier-checker=true", "--skip-barrier-checker=false"]
        )

    additional_args = (
        f'--output={neff_path} --internal-backend-options="{" ".join(backend_opts)}"'
    )
    if compile_opts.neuronx_cc_args:
        additional_args += " " + " ".join(
            shlex.quote(a) for a in compile_opts.neuronx_cc_args
        )
    neuronxcc_flags = os.environ.get("NEURON_CC_FLAGS")
    if neuronxcc_flags:
        additional_args += " " + neuronxcc_flags

    # Penguin prefixes tensor filenames with "value_" (CodeGenBase.py:218).
    # Create symlinks so BIRSim finds our .npy files under the Penguin names.
    if compile_opts.enable_simulation:
        for spec in bir.descriptor.input_specs or []:
            _symlink_birsim_file(bir.sg00_dir, spec.name, "value_")
        for alias_name in (input_output_aliases or {}).values():
            _symlink_birsim_file(
                bir.sg00_dir, f"{alias_name}{MUST_ALIAS_SUFFIX}", "value_"
            )
        # For LNC2+, BIRSim looks for input .npy files in nc{XX}/sg00/
        # (per-core directories). Pre-populate them with symlinks back to
        # the top-level sg00/ inputs so BIRSim can find them.
        if compile_opts.lnc > 1:
            _populate_per_core_birsim_inputs(
                bir.work_dir, bir.sg00_dir, compile_opts.lnc
            )

    # Use work_dir as output_directory — kernel BIR and .npy files are already
    # there, and BIRSim inputs/outputs in sg00/ stay accessible.
    try:
        compile_nki_ir_kernel_to_neff(
            kernel_func=kernel_dict,
            kernel_inputs_dict=kernel_inputs_dict,
            kernel_outputs=kernel_outputs,
            platform_target=compile_opts.target,
            logical_nc_config=compile_opts.lnc,
            output_directory=bir.work_dir,
            version=NKI_IR_VERSION.beta2,
            additional_compiler_args=additional_args,
            enable_device_dump=compile_opts.enable_device_dump,
        )
    except subprocess.CalledProcessError as e:
        # compile_nki_ir_kernel_to_neff invokes neuronx-cc via subprocess.run
        # with capture_output=True; on failure CalledProcessError surfaces here.
        # Translate the cryptic NCC_I... codes into actionable messages.
        stderr_text = e.stderr if hasattr(e, "stderr") and e.stderr else ""
        raise NCCError(
            build_ncc_error_message(e.returncode, stderr_text, bir.work_dir)
        ) from e

    # BIRSim writes outputs as value_{name}-birsim.npy. Create symlinks
    # with the original names so _load_birsim_outputs finds them.
    # For LNC2+, neuronx-cc places per-core outputs in nc00/sg00/ instead
    # of the top-level sg00/.
    if compile_opts.enable_simulation:
        birsim_output_dir = _find_birsim_output_dir(bir.work_dir, bir.sg00_dir)
        for spec in bir.descriptor.output_specs or []:
            _symlink_birsim_file(
                birsim_output_dir, f"{spec.name}-birsim", "value_", reverse=True
            )


def _serialize_birsim_inputs(
    sg00_dir: str,
    input_arrays: List[np.ndarray],
    argument_names: List[str],
    input_output_aliases: Optional[Dict[int, str]] = None,
    verbose: bool = False,
) -> None:
    """Serialize input arrays as .npy files for BIRSim numerical simulation."""
    if not input_arrays:
        return
    if len(input_arrays) != len(argument_names):
        raise RuntimeError(
            f"Mismatch between input arrays ({len(input_arrays)}) and "
            f"argument names ({len(argument_names)})"
        )

    # Rename aliased inputs per MUST_ALIAS_SUFFIX convention.
    aliased_inputs = set((input_output_aliases or {}).values())

    num_inputs = 0
    for arg_name, arr in zip(argument_names, input_arrays):
        if arr is None:
            continue
        file_name = (
            f"{arg_name}{MUST_ALIAS_SUFFIX}" if arg_name in aliased_inputs else arg_name
        )
        input_npy_path = os.path.join(sg00_dir, f"{file_name}.npy")
        np.save(input_npy_path, arr)
        num_inputs += 1
        if verbose:
            print(f"  Serialized {arg_name} array to {input_npy_path}")

    if verbose and num_inputs > 0:
        print(f"Serialized {num_inputs} BIRSim input(s) to {sg00_dir}/")


def compile_bir_to_neff(
    compile_opts: CompileOptions,
    bir: NirResult,
    input_arrays: List[np.ndarray],
    argument_names: List[str],
    output_arg_names: List[str],
) -> CompiledKernel:
    """
    Compile BIR to NEFF using the neuronxcc API.

    Serializes BIRSim inputs, invokes neuronxcc, and loads BIRSim outputs.
    For BAREMETAL compilation, use baremetal_driver.compile_mlir_to_neff_baremetal.
    """
    input_output_aliases = bir.descriptor.to_input_output_aliases_dict() or None

    output_path = compile_opts.output_path
    if not os.path.isabs(output_path):
        output_path = os.path.join(bir.work_dir, output_path)
        compile_opts = replace(compile_opts, output_path=output_path)

    if compile_opts.enable_simulation:
        _serialize_birsim_inputs(
            bir.sg00_dir,
            input_arrays,
            argument_names,
            input_output_aliases,
            compile_opts.verbose,
        )

    if compile_opts.verbose:
        print(f"Running neuronx-cc for target {compile_opts.target}...")

    neuronx_cc_start = time.time()
    try:
        _compile_neff(compile_opts, bir, input_output_aliases)
    finally:
        neuronx_cc_time = time.time() - neuronx_cc_start

    if compile_opts.verbose:
        print(f"neuronx-cc compilation completed in {neuronx_cc_time:.2f}s")

    birsim_outputs = None
    if compile_opts.enable_simulation and output_arg_names:
        birsim_output_dir = _find_birsim_output_dir(bir.work_dir, bir.sg00_dir)
        birsim_outputs = _load_birsim_outputs(
            birsim_output_dir,
            output_arg_names=output_arg_names,
            input_output_aliases=input_output_aliases,
            verbose=compile_opts.verbose,
        )

    return CompiledKernel(
        neff_path=compile_opts.output_path,
        target=compile_opts.target,
        lnc=compile_opts.lnc,
        artifacts_dir=compile_opts.artifacts_dir,
        compilation_time=bir.frontend_time + bir.mlir_time + neuronx_cc_time,
        frontend_time=bir.frontend_time,
        mlir_time=bir.mlir_time,
        neuronx_cc_time=neuronx_cc_time,
        birsim_outputs=birsim_outputs,
        bir=bir,
    )


def compile_mlir_to_neff(
    module,
    function_name: str,
    compile_opts: CompileOptions,
    tensor_inputs: Optional[Dict[str, np.ndarray]] = None,
) -> CompiledKernel:
    """
    Compile MLIR module to NEFF with temp directory and artifact handling.

    Derives argument names, output names, and input-output aliases from the
    BIR emission result. Returns a CompiledKernel with specs populated from BIR.

    Args:
        tensor_inputs: Optional dict of name→ndarray for BIRSim simulation.
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
        work_dir = tempfile.mkdtemp(prefix="nki_")

    bir = compile_mlir_to_bir(
        module,
        function_name,
        compile_opts,
        work_dir,
    )

    argument_names = [s.name for s in bir.descriptor.input_specs]
    output_arg_names = [s.name for s in bir.descriptor.output_specs]

    # Strip MUST_ALIAS_SUFFIX from BIR input names to match tensor_inputs dict keys.
    if tensor_inputs and compile_opts.enable_simulation:
        input_arrays = [
            tensor_inputs.get(name.removesuffix(MUST_ALIAS_SUFFIX))
            for name in argument_names
        ]
    else:
        input_arrays = []

    return compile_bir_to_neff(
        compile_opts,
        bir,
        input_arrays=input_arrays,
        argument_names=argument_names,
        output_arg_names=output_arg_names,
    )
