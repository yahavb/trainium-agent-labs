"""[INTERNAL] CompileKernel and derived classes for compilation on Neuron devices.

INTERNAL MODULE - Not part of public API. May change without notice.
"""

import hashlib
import inspect
import itertools
import os
import sys
import tempfile
from collections.abc import Hashable
from dataclasses import dataclass, replace
from typing import Any, Callable, Tuple

import numpy as np

from nki.compiler.driver import (
    _birsim_executor,
    compile_to_bir,
    multirank_run_from_bir,
    run_from_bir,
)
from nki.compiler.frontend import NKIFrontend, ParserFrontend, resolve_frontend_cls
from nki.compiler.ncc_driver import CompileOptions, FrameworkConfig, NirResult
from nki.compiler.target import resolve_target
from nki.framework.kernel import Kernel, _typenames
from nki.language._nki_base import NKIObject
from nki.utils import get_binary_env_var


class MultiRankExecutionError(RuntimeError):
    """Raised when one or more ranks fail during multi-rank execution.

    Python 3.10 (our build target) has no builtin ``ExceptionGroup`` (added in
    3.11), so per-rank failures are aggregated explicitly here instead.
    ``rank_failures`` maps each failing rank id to the exception it raised.
    """

    def __init__(self, rank_failures: dict):
        self.rank_failures = dict(rank_failures)
        detail = "; ".join(
            f"rank {rank_id}: {type(err).__name__}: {err}"
            for rank_id, err in sorted(self.rank_failures.items())
        )
        super().__init__(f"multi-rank execution failed ({detail})")


def compile_kernel_to_nir(
    kernel_func: Callable,
    *,
    inputs: dict[str, Any],
    compile_opts: CompileOptions,
    frontend: NKIFrontend | None = None,
    enable_cache: bool = True,
    trace_cache=None,
) -> NirResult:
    """Public API: compile NKI kernel to NirResult.

    Framework integrations (JAX, TorchXLA, TorchEager) use this as
    the single compilation entry point. Call build_config() on the
    returned NirResult to get framework config (FrameworkConfig).

    ``trace_cache`` (experimental) is an optional TraceCacheCoordinator that
    lets a cold process reuse persisted lowered IR and skip tracing. See
    nki.compiler._trace_cache.
    """
    if frontend is None:
        frontend = resolve_frontend_cls()()

    return compile_to_bir(
        kernel_func,
        frontend=frontend,
        inputs=inputs,
        compile_opts=compile_opts,
        enable_cache=enable_cache,
        trace_cache=trace_cache,
    )


# ==================================================
# Framework detection
# ==================================================


@dataclass(frozen=True)
class CompileKernel(Kernel):
    """A kernel configured for compilation and execution on a Neuron device."""

    artifacts_dir: str = ""
    """Optional path to directory for storing compiler artifacts."""

    target: str | None = None
    """Target NeuronDevice architecture. By default, infer from environment."""

    _enable_backend_opt: bool = False
    """Enable experimental MLIR optimization pipeline."""

    _executor: Callable | None = None
    """Kernel executor. By default, uses Spike executor."""

    _frontend_cls: type = ParserFrontend
    """Frontend class to use for compilation."""

    _enable_simulation: bool = False
    """Run BIRSim instead of hardware execution."""

    _enable_device_dump: bool = False
    """Insert DevicePrint after every SBUF-writing ISA instruction (tracer only)."""

    def __post_init__(self):
        super().__post_init__()
        # Enable compilation caching unless explicitly disabled via env var.
        # The cache lives on the function object so it persists across
        # Kernel instances. Deleting the attr also disables caching.
        if not get_binary_env_var("NKI_DISABLE_COMPILE_CACHE") and not hasattr(
            self.func, "_nki_compile_cache"
        ):
            self.func._nki_compile_cache = {}

    def _compile_opts(self):
        opts = CompileOptions(
            target=resolve_target(self.func, self.target),
            lnc=self.lnc,
            artifacts_dir=self.artifacts_dir,
            output_path="kernel.neff",
            enable_simulation=self._enable_simulation,
            enable_device_dump=self._enable_device_dump,
            address_rotation=self.address_rotation,
            enable_barrier_checker=self.barrier_check,
        )

        if not self._enable_backend_opt:
            opts = opts.disable_backend_optimizations()

        return opts

    @staticmethod
    def _hashable_arg(value, name):
        """Convert a kernel argument to something hashable, for use in a cache key.

        Tensors are represented by (shape, dtype).
        Lists and dictionaries are converted to tuples.
        Scalars and other hashable types pass through unchanged.

        Returns a tuple (value, is_hashable).
        (If is_hashable is False, then 'value' will be None.)
        """
        if isinstance(value, (tuple, list)):
            parts = []
            for i, e in enumerate(value):
                h, is_hashable = CompileKernel._hashable_arg(e, f"{name}[{i}]")
                if not is_hashable:
                    return None, False
                parts.append(h)
            return tuple(parts), True
        if isinstance(value, dict):
            parts = []
            for k, v in sorted(value.items()):
                h, is_hashable = CompileKernel._hashable_arg(v, f"{name}[{k}]")
                if not is_hashable:
                    return None, False
                parts.append((k, h))
            return tuple(parts), True
        if hasattr(value, "shape") and hasattr(value, "dtype"):
            shape = (
                tuple(value.shape) if hasattr(value.shape, "__iter__") else value.shape
            )
            return (shape, str(value.dtype)), True

        if isinstance(value, (NKIObject, Hashable)):
            # Assumes hashable types have suitable __repr__ defined.
            # We introduce a special case for NKIObjects, which are *ordinarily* Hashable, but might not be
            # if annotated with the @dataclass decorator. See test_compile_cache_key.py for more details.
            return value, True

        return None, False

    def _generate_cache_key(self, inputs, compile_opts):
        """Generate a cache key from kernel inputs and compilation options.

        Returns None if any argument is not hashable (caching is skipped).

        Includes argument shapes/dtypes, kernel config (LNC, schedule),
        and compilation options that affect output (target, mode, pipeline
        options, frontend class). Path-only fields (artifacts_dir,
        output_path) are excluded since they don't affect compiled output.
        """
        key_parts = []
        for name, v in inputs.items():
            h, is_hashable = self._hashable_arg(v, name)
            if not is_hashable:
                return None
            key_parts.append(h)
        key_parts.append(self.lnc)
        key_parts.append(self.schedule)
        key_parts.append(compile_opts.target)
        key_parts.append(compile_opts.nki_opt_pipeline_options)
        key_parts.append(self._frontend_cls.__name__)
        key_parts.append(self._enable_backend_opt)
        key_parts.append(compile_opts.address_rotation)
        return hashlib.sha256(repr(tuple(key_parts)).encode()).hexdigest()

    def _get_compile_cache(self):
        """Return the compilation cache, or None if caching is disabled.

        The cache is stored on ``self.func`` so it persists across Kernel
        instances wrapping the same function. If the attribute is absent,
        caching is disabled — we do not create it implicitly.
        """
        return getattr(self.func, "_nki_compile_cache", None)

    def _source_location(self):
        """Best-effort ``(abs_file, first_line)`` of the kernel's defining code.

        Unwraps decorator chains (``@nki.jit`` etc.) via ``inspect.unwrap`` so we
        read the real source, then takes ``co_filename`` / ``co_firstlineno``.
        Returns ``("", 0)`` when unavailable (REPL/exec'd/C callables).
        """
        fn = getattr(self.func, "func", self.func)
        try:
            fn = inspect.unwrap(fn)
        except ValueError:
            pass  # cyclic __wrapped__ chain, use the outermost we have
        code = getattr(fn, "__code__", None)
        if code is None:
            return "", 0
        try:
            return os.path.abspath(code.co_filename), int(code.co_firstlineno)
        except (OSError, TypeError, ValueError):
            return "", 0

    def _generate_trace_cache_key(self, inputs, compile_opts):
        """Cheap, trace-free key for the persistent (cross-process) trace cache.

        Same input/config terms as the in-memory key, but additionally scoped by:

          * a stable identity of the *kernel function*: its ``module.qualname``
            AND its defining file's absolute path. The qualname alone is not
            unique across environments: the same-named kernel installed at two
            paths (two venvs, two checkouts) would share a key, and because the
            dep-hash re-fingerprints the paths recorded at *trace* time, a cold
            process could validate the other environment's still-on-disk file
            and wrongly HIT, serving IR built from different source. Folding the
            file path in makes that a clean miss.
          * the Python version (``sys.version_info[:2]``): only 3.12+ has
            sys.monitoring, so a 3.11 (settrace) and a 3.12 (monitoring) process
            would otherwise share entries; keeping them separate is explicit
            rather than relying on the dep sets;
          * the user-declared dependency roots (``explicit_roots()``). These are
            an *input* that decides which files count as dependencies, so they
            belong in the key.
          * the BIR-affecting ``CompileOptions`` toggles the in-memory key omits
            (``bir_opts`` below). Several are env-derived (``lower_dma_transpose``
            ← ``NKI_DMA_TRANSPOSE_AS_PE_TRANSPOSE``, ``use_colz`` ← ``NKI_USE_COLZ``,
            ``debug`` ← ``NKI_DEBUG_INFO``) so two processes sharing one cache dir
            can differ on them while producing an otherwise-identical key.
            Folding them in makes a differing toggle a clean miss rather than a stale
            hit.

        Returns None if any argument is not hashable (skip caching).
        """
        from nki.compiler._trace_cache import explicit_roots

        base = self._generate_cache_key(inputs, compile_opts)
        if base is None:
            return None
        fn = self.func
        qualname = f"{getattr(fn, '__module__', '')}.{getattr(fn, '__qualname__', '')}"
        src_file, _src_line = self._source_location()
        pyver = f"{sys.version_info[0]}.{sys.version_info[1]}"
        roots = os.pathsep.join(sorted(explicit_roots()))
        # BIR-affecting fields not already covered by `base` (which folds in
        # target/pipeline-opts/lnc/schedule/frontend/backend-opt/addr-rotation).
        # `debug` gates side-artifact emission (*.debuginfo.json) and
        # `enable_device_dump` inserts DevicePrint ops into the traced IR, so a
        # hit from a populate with a different setting would otherwise silently
        # serve BIR built under the wrong one (missing/extra device prints, or
        # skipped debug output).
        bir_opts = (
            bool(compile_opts.lower_dma_transpose),
            bool(compile_opts.use_colz),
            bool(compile_opts.skip_verifier),
            bool(compile_opts.emit_predicted_latency),
            bool(compile_opts.emit_reg_compute_as_affine_expr),
            bool(compile_opts.debug),
            bool(compile_opts.enable_device_dump),
        )
        return hashlib.sha256(
            f"{qualname}\0{src_file}\0{pyver}\0{roots}\0{bir_opts!r}\0{base}".encode()
        ).hexdigest()

    def _make_trace_cache(self, inputs, compile_opts):
        """Build a TraceCacheCoordinator, or None if the trace cache is off.

        The dependency source is chosen by frontend: the parser resolves its
        dep files statically during specialize() (ParserDependencySource); the
        tracer observes them at runtime (TracerDependencySource). The frontend
        class is already part of the cache key (see _generate_cache_key), so a
        parser-written entry and a tracer-written entry never collide.
        """
        from nki.compiler._trace_cache import (
            ParserDependencySource,
            TraceCacheCoordinator,
            TracerDependencySource,
            default_roots,
            is_trace_cache_enabled,
        )

        if not is_trace_cache_enabled():
            return None
        key = self._generate_trace_cache_key(inputs, compile_opts)
        if key is None:
            return None
        roots = default_roots(self.func)
        if self._frontend_cls is ParserFrontend:
            dep_source = ParserDependencySource(roots)
        else:
            dep_source = TracerDependencySource(roots)
        return TraceCacheCoordinator(
            key=key, kernel_func=self.func, dep_source=dep_source
        )

    def _cached_compile_to_bir(self, *, frontend, inputs, compile_opts):
        """Compile with caching: skip recompilation when inputs match.

        Caching is skipped when: the cache is disabled (no attr on func),
        or any input argument is not hashable (cache key is None).

        Two tiers: the in-memory cache short-circuits repeated compiles within a
        process; on an in-memory miss, an experimental persistent trace cache
        (opt-in via NKI_ENABLE_TRACE_CACHE) can still let a cold process skip the
        expensive trace by reusing persisted lowered IR.
        """
        cache = self._get_compile_cache()
        cache_key = None
        if cache is not None:
            cache_key = self._generate_cache_key(inputs, compile_opts)
            if cache_key is not None:
                cached = cache.get(cache_key)
                if cached is not None:
                    return cached
        result = compile_kernel_to_nir(
            self,
            inputs=inputs,
            compile_opts=compile_opts,
            frontend=frontend,
            trace_cache=self._make_trace_cache(inputs, compile_opts),
        )
        if cache is not None and cache_key is not None:
            cache[cache_key] = result
        return result

    def compile(self, *args, **kwargs) -> Tuple[FrameworkConfig, str]:
        """Compile the kernel and return (FrameworkConfig, cache_key).

        Frameworks call this to get the compiled config for embedding
        in their computation graph. Input conversion is handled internally
        via _convert_input(), which subclasses override for their tensor types.

        Returns:
            Tuple of (FrameworkConfig, cache_key) where cache_key is the
            content-based disk cache hash from BIR emission.
        """
        inputs = self._convert_inputs(self._bind_args(args, kwargs))
        compile_opts = self._compile_opts()
        frontend = self._frontend_cls(enable_backend_opt=self._enable_backend_opt)
        nir = self._cached_compile_to_bir(
            frontend=frontend,
            inputs=inputs,
            compile_opts=compile_opts,
        )
        return nir.build_config(), nir.cache_hash

    def get_cache_key(self, *args, **kwargs) -> str:
        """Return the cache key for the given inputs for this kernel.

        Returns a hex string suitable for use in file paths or as part of a larger key.
        """

        # TODO: This currently just calls compile again to get the cache key,
        #       which is currently based on the BIR result. This should be a
        #       fast mechanism to check a kernel inputs/state's uniqueness
        _, cache_key = self.compile(*args, **kwargs)
        return cache_key

    def _convert_inputs(self, kwargs):
        """Convert framework-native inputs to NKI format.

        Calls _convert_input() for each argument. Subclasses override
        _convert_input() to handle their framework's tensor type.
        """
        return {
            name: self._convert_input(value, name) for name, value in kwargs.items()
        }

    def _convert_input(self, value, name):
        """Convert a single input value. Override in framework-specific subclasses."""
        return value

    def _build_backend_config(self, nir, _compilation_result=None):
        """Deprecated: use nir.build_config() instead. Kept for TorchNeuronEager compat."""
        import warnings

        warnings.warn(
            "_build_backend_config is deprecated. Use nir.build_config() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return nir.build_config().backend_config

    def _compute_mode(self, args, kwargs):
        # Detect appropriate subclass based on argument types.
        arg_list = list(args) + list(kwargs.values())
        arg_types = _typenames(*arg_list)

        if arg_list and any(t.startswith(("jaxlib.", "jax.")) for t in arg_types):
            from nki.framework.jax import JaxKernel

            return JaxKernel

        if arg_list and any(t.startswith("torch.") for t in arg_types):
            from nki.framework._torch_xla import TorchXlaKernel

            return TorchXlaKernel

        return StandaloneKernel

    def __call__(self, *args, **kwargs):
        # Detect framework based on argument types
        cls = self._compute_mode(args, kwargs)
        return self._to_subclass(cls)(*args, **kwargs)


@dataclass(frozen=True)
class StandaloneKernel(CompileKernel):
    def __call__(self, *args, **kwargs):
        return _execute_standalone_kernel(self, *args, **kwargs)


def _execute_standalone_kernel(kernel: StandaloneKernel, *args, **kwargs) -> Any:
    frontend = kernel._frontend_cls(enable_backend_opt=kernel._enable_backend_opt)
    inputs = kernel._bind_args(args, kwargs)

    # If using lists for multi-rank execution, determine the largest list
    # (the world_size). By default, the world_size is 1.
    world_size = max(
        map(
            lambda x: len(x) if isinstance(x, list) else 1,
            itertools.chain(args, kwargs.values()),
        ),
        default=1,
    )

    compile_opts = kernel._compile_opts()

    if kernel._executor is not None:
        executor = kernel._executor
    elif kernel._enable_simulation:
        executor = _birsim_executor
    else:
        executor = None

    executor_kwargs = {}
    if executor is not None:
        executor_kwargs["executor"] = executor

    def _partition_inputs_per_rank() -> list[dict]:
        per_rank_inputs = [{} for _ in range(world_size)]
        for rank_id in range(world_size):
            for name, input in inputs.items():
                # If the input is a list, ensure that the list has the same length as the world_size
                if isinstance(input, list):
                    if len(input) != world_size:
                        raise RuntimeError(
                            "Per-rank input does not match world size for multi-rank execution."
                        )
                    per_rank_inputs[rank_id][name] = input[rank_id]
                elif isinstance(input, np.ndarray):
                    raise RuntimeError(
                        "Tensors in multi-rank execution must be passed as lists whose length is equal to the world size."
                    )
                # Replicate non-tensor inputs to all ranks.
                else:
                    per_rank_inputs[rank_id][name] = input

        return per_rank_inputs

    def _compile_and_run(compile_opts: CompileOptions):
        # If the world_size is greater than 1, then partition the inputs across multiple ranks.
        if world_size > 1:
            per_rank_inputs = _partition_inputs_per_rank()
            nir = compile_kernel_to_nir(
                kernel,
                inputs=per_rank_inputs[0],
                compile_opts=compile_opts,
                frontend=frontend,
                enable_cache=False,
            )

            results, errors = multirank_run_from_bir(
                nir, compile_opts, per_rank_inputs, **executor_kwargs
            )

            rank_failures = {
                rank_id: err
                for rank_id, err in enumerate(errors)
                if err is not None
            }

            if rank_failures:
                raise MultiRankExecutionError(rank_failures)

            return results

        nir = compile_kernel_to_nir(
            kernel,
            inputs=inputs,
            compile_opts=compile_opts,
            frontend=frontend,
            enable_cache=False,
        )

        return run_from_bir(nir, compile_opts, inputs, **executor_kwargs)

    if compile_opts.artifacts_dir:
        return _compile_and_run(compile_opts)
    else:
        with tempfile.TemporaryDirectory(prefix="nki_") as work_dir:
            return _compile_and_run(
                replace(
                    compile_opts,
                    artifacts_dir=work_dir,
                    output_path=os.path.join(work_dir, "kernel.neff"),
                )
            )
