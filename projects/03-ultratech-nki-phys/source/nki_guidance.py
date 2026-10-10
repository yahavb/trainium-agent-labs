"""Compact documentation-backed guidance, augmented by installed SDK signatures."""

import importlib
import inspect

SOURCES = (
    "https://awsdocs-neuron.readthedocs-hosted.com/en/latest/_modules/nki/isa.html",
    "https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/guides/architecture/trainium2_arch.html",
    "https://awsdocs-neuron.readthedocs-hosted.com/en/v2.26.0/general/nki/nki_perf_guide.html",
)

CARD = """NKI implementation guidance (API references, not a convergence guarantee):
- Load HBM inputs to SBUF with dma_copy before arithmetic; keep recurrent state on-chip.
- nc_matmul computes stationary.T @ moving. Both inputs use SBUF; destination uses PSUM. Our packed input is already A.T. Do not transpose again.
- tensor_tensor accepts SBUF/SBUF or SBUF/PSUM inputs, not HBM. Preserve 2D rate/history tiles and element counts.
- tensor_scalar is tile arithmetic with dst/data/op0/operand0, not scalar extraction. Copy old iterates before overwriting them.
- API repair, numerical convergence and hardware performance are distinct gates. Do not change precision to hide numerical errors.
- Performance guide: test reuse, fusion and tile utilization using device profiles. These tiny matrix-vector calls and serial-world loops suggest possible low utilization, but no profiler evidence exists yet. Do not claim memory-bound/compute-bound or speedup from simulator timing.
- Optional fusion/new APIs need verification on the installed SDK before use; documentation versions may differ from the seat.
"""


def sdk_context():
    metadata = dict(sources=list(SOURCES), installed_sdk_verified=False, signatures={})
    try:
        nki = importlib.import_module("nki")
        isa = importlib.import_module("nki.isa")
        metadata["nki_version"] = getattr(nki, "__version__", "unknown")
        for name in ("dma_copy", "nc_matmul", "tensor_tensor", "tensor_scalar", "tensor_copy"):
            function = getattr(isa, name)
            try:
                metadata["signatures"][name] = str(inspect.signature(function))
            except (ValueError, TypeError):
                metadata["signatures"][name] = "not introspectable; follow baseline and SDK simulation"
        metadata["installed_sdk_verified"] = True
    except (ImportError, AttributeError) as error:
        metadata["inspection_error"] = f"{type(error).__name__}: {error}"
    # Detailed signatures are preserved as artifacts; keep the model context small.
    version = metadata.get("nki_version", "unavailable locally; baseline API patterns only")
    signature_text = "\n".join(f"nisa.{name}{signature}" for name, signature in metadata["signatures"].items()
                               if name in ("tensor_scalar", "tensor_tensor") and len(signature) <= 450)
    return CARD + f"Installed nki: {version}.\n" + signature_text, metadata
