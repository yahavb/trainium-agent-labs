"""[INTERNAL] NKI Runtime — compatibility shim.

INTERNAL MODULE - Not part of public API. May change without notice.

Forwards to the standalone `nrtpy` package. `nrtpy` is only available on
platforms where the Neuron Runtime (libnrt) ships — currently amd64 Linux
only. On other platforms, this module fails to import.

New code should prefer `import nrtpy` directly.
"""

try:
    from nrtpy import (
        BenchmarkResult,
        ModelTensorInfo,
        NrtError,
        NrtModel,
        NrtTensor,
        SpikeError,
        TensorMetadata,
    )
    from nrtpy.profiler_adapter import SpikeProfiler
    from nrtpy.spike_model import SpikeModel
    from nrtpy.spike_singleton import configure, get_spike_singleton, reset
    from nrtpy.spike_tensor import SpikeTensor
except ImportError as e:
    raise ImportError(
        "nki.runtime is not available on this platform. The Neuron Runtime "
        "(libnrt) is only shipped for amd64 Linux. Compile NKI kernels on "
        "any platform, but execute them on an amd64 Linux host with a "
        "Neuron accelerator attached."
    ) from e

__all__ = [
    "SpikeModel",
    "SpikeTensor",
    "SpikeProfiler",
    "configure",
    "reset",
    "get_spike_singleton",
    "NrtError",
    "SpikeError",
    "BenchmarkResult",
    "ModelTensorInfo",
    "NrtModel",
    "NrtTensor",
    "TensorMetadata",
]
