"""[INTERNAL] Kernel base class returned by @nki.jit.

INTERNAL MODULE - Not part of public API. May change without notice.
"""

import inspect
import sys
from dataclasses import dataclass, field, fields, replace
from types import FrameType
from typing import Callable

from nki.utils import get_binary_env_var

# ==================================================
# Kernel dataclasses
# ==================================================


def _typenames(*args):
    """Build a set of fully-qualified type names for the given arguments."""

    def _typename(a):
        t = type(a)
        return f"{getattr(t, '__module__', None)}.{getattr(t, '__qualname__', None)}"

    return {_typename(arg) for arg in args}


def has_torch_input(*args, **kwargs):
    arg_list = list(args)
    arg_list.extend(kwargs.values())

    arg_types = _typenames(*arg_list)

    if any(t.startswith("torch.") for t in arg_types):
        import torch

        # New torch_neuronx uses 'neuron' device instead of 'xla'.
        if any(
            torch.is_tensor(arg) and arg.device.type == "neuron" for arg in arg_list
        ):
            return True
    return False


def wrap_nki_to_torch_hop(kernel, **kwargs):
    """
    Wraps a NKI kernel to Pytorch HigherOrderOp (HOP) structure
    that can be understood by torch.compile (a.k.a Dynamo).
    """
    from torch_neuronx.nki_hop import NKIHOPCaller, register_kernel_to_torch

    sig = inspect.signature(kernel.func)
    arg_names = list(sig.parameters.keys())
    kernel_default_args = {
        name: param.default
        for name, param in sig.parameters.items()
        if param.default is not inspect.Parameter.empty
    }

    # We need to unwrap the fields of the kernel object and pass them
    # to register_kernel_to_torch, because torch.compile cannot represent
    # an user-defined class as a fx.Node
    k_fields = {
        f.name: getattr(kernel, f.name) for f in fields(kernel) if f.name != "func"
    }

    # register_kernel_to_torch generates the torch native kernel object
    # ,registers the kernel object to the registry, and returns the registry
    # index of the kernel. If the kernel is already registered, it returns the
    # existing registry index.
    kernel_idx = register_kernel_to_torch(
        kernel.func, k_fields, arg_names, kernel_default_args, **kwargs
    )

    # NKIHOPCaller creates NKI higher-order operator structure that can be
    # traced by torch.compile and compiled by the neuron backend.
    return NKIHOPCaller(kernel_idx, [kernel.lnc], arg_names, kernel_default_args)


@dataclass(frozen=True)
class Kernel:
    """Wrapper around a NKI kernel function returned by ``@nki.jit``.
    See the documentation for ``@nki.jit`` for more details.
    """

    func: Callable
    """The wrapped function."""

    lnc: int = 1
    """LNC degree."""

    schedule: tuple = ()
    """Schedule dependency edges as tuple of (from, [to, ...]) pairs."""

    address_rotation: bool = True
    """Enable address rotation optimization for multi-buffered tensors."""

    barrier_check: bool = False
    """Whether to compile with the LNC core barrier checker enabled.
    Set via :meth:`enable_core_barrier_checker` / :meth:`disable_core_barrier_checker`."""

    _invocation_frame: FrameType | None = field(default=None, compare=False, repr=False)
    """User frame that invoked the kernel at the @nki.jit boundary, captured in
    __call__ and carried to _compile so tracer diagnostics can show the call site.
    Excluded from equality/repr and never part of the compile cache key; set only
    on the transient per-call kernel, so the persistent @nki.jit kernel stays None."""

    def __post_init__(self):
        if self.lnc not in (1, 2):
            from nki.isa._validation import NkiValidationError

            raise NkiValidationError(
                f"NKI only supports LNC 1 or 2, but got {self.lnc}"
            )

        # Set __wrapped__ so inspect.signature() and parsing frontend work correctly.
        # Must be set here (not just in @nki.jit) so dataclasses.replace() preserves it.
        # Frozen dataclass requires object.__setattr__.
        object.__setattr__(self, "__wrapped__", self.func)

        # Set __doc__ here so wrapped function's docstring propagates to Kernel instance.
        # Set here instead of via @property so subclasses inherit func's docstring correctly.
        object.__setattr__(self, "__doc__", self.func.__doc__)

    # Setters

    def __getitem__(self, lnc):
        """Allows users to set LNC at the callsite using bracket syntax.

        .. code-block:
           @nki.jit
           def f(a, b):
               ...

           f(a,b) # run kernel with LNC1
           f[1](a,b) # run kernel with LNC1
           f[2](a, b) # run kernel with LNC2

        """
        return replace(self, lnc=lnc)

    def with_schedule(self, edges):
        """Attach dependency edges to control instruction scheduling.

        Args:
          edges: List of (from_name, to_name_or_list) tuples. Each tuple
            specifies that the instruction named ``from_name`` depends on
            the instruction(s) in ``to_name_or_list``. The second element
            must be a string or a list of strings (tuples are not accepted).

        Returns:
          A new Kernel with the schedule edges attached.
        """
        parsed = []
        for item in edges:
            if not isinstance(item, tuple) or len(item) != 2:
                raise TypeError("Each item must be a 2-tuple")

            key, value = item

            if not isinstance(key, str):
                raise TypeError("First element must be a string")

            if isinstance(value, str):
                value = [value]

            if not isinstance(value, list) or not all(
                isinstance(v, str) for v in value
            ):
                raise TypeError("Second element must be a string or list of strings")

            # Store as a tuple for frozen-dataclass immutability.
            parsed.append((key, tuple(value)))

        # Store as nested tuples for frozen-dataclass immutability.
        return replace(self, schedule=tuple(parsed))

    def enable_address_rotation(self):
        """Enable address rotation optimization for multi-buffered tensors.

        Returns:
          A new Kernel with address rotation enabled.
        """
        return replace(self, address_rotation=True)

    def disable_address_rotation(self):
        """Disable address rotation optimization for multi-buffered tensors.

        Returns:
          A new Kernel with address rotation disabled.
        """
        return replace(self, address_rotation=False)

    def enable_core_barrier_checker(self):
        """Compile this kernel with the LNC core barrier checker enabled.

        On LNC=2, the checker compares every pair of memory accesses across
        the two PNCs. Any pair where one PNC writes a region that the other
        PNC reads or writes, with no core barrier on a common engine between
        them, is reported as a race and compilation fails.

        Returns:
          A new Kernel that compiles with the core barrier checker enabled.
        """
        return replace(self, barrier_check=True)

    def disable_core_barrier_checker(self):
        """Compile this kernel without the LNC core barrier checker (default).

        Returns:
          A new Kernel that compiles with the core barrier checker disabled.
        """
        return replace(self, barrier_check=False)

    # Function wrapping

    @property
    def __name__(self):
        return self.func.__name__

    def __call__(self, *args, **kwargs):
        from nki._backends import has_active_backend

        if has_active_backend():
            return self.func(*args, **kwargs)

        if get_binary_env_var("NKI_SIMULATOR"):
            from nki.framework._simulate import simulate

            k = simulate(self)
        else:
            from nki.compiler.frontend import resolve_frontend_cls
            from nki.framework.compiled import CompileKernel

            fe = resolve_frontend_cls()

            # Native torch execution path should diverge from this top-level callsite
            # because class conversion between user-defined Python classes is not compatible
            # with torch.compile
            if has_torch_input(*args, **kwargs):
                torch_hop = wrap_nki_to_torch_hop(self, _frontend_cls=fe)
                return torch_hop(*args, **kwargs)
            # sys._getframe(1) is this call's caller -- the user's kernel(x) site.
            # Carried on the per-call kernel to _compile for tracer diagnostics.
            k = self._to_subclass(
                CompileKernel,
                _frontend_cls=fe,
                _invocation_frame=sys._getframe(1),
            )

        return k(*args, **kwargs)

    def _bind_args(self, args, kwargs):
        """Bind positional/keyword arguments to the kernel's signature.

        Returns a dict mapping parameter names to values.
        """
        return inspect.signature(self.func).bind(*args, **kwargs).arguments

    def _to_subclass(self, cls, **kwargs):
        """Convert this Kernel instance to a Kernel subclass, optionally setting
        additional fields. This method relies on the fact that derived dataclasses can
        only introduce optional fields."""

        k_fields = {
            f.name: getattr(self, f.name) for f in fields(self) if f.name != "func"
        }
        k_fields.update(kwargs)
        return cls(self.func, **k_fields)


def _check_kernel(k):
    """Wrap 'k' in a Kernel if it is not already decorated with nki.jit."""

    if not isinstance(k, Kernel):
        return Kernel(k)
    return k
