"""Supply the installed SDK's actual API signatures to the generation model."""
import inspect


def sdk_context():
    import nki.isa as nisa
    names = ("dma_copy", "tensor_tensor", "tensor_scalar", "tensor_reduce", "activation", "memset")
    signatures = [f"nisa.{name}{inspect.signature(getattr(nisa, name))}"
                  for name in names if hasattr(nisa, name)]
    return (
        "\nInstalled NKI SDK rules:\n"
        "NKI tensors do not support Python +, -, *, / or augmented assignments like +=. "
        "Use nisa instructions for all tensor arithmetic. Python arithmetic on shape integers is fine. "
        "There is no nl.memset; use nisa.memset with an explicit destination. "
        "Allocate all ISA destinations with nl.ndarray and pass dst explicitly. "
        "Do not invent APIs or reuse syntax from older Neuron SDK versions. "
        "Known-good syntax is demonstrated by the initial kernel.\n" + "\n".join(signatures) + "\n"
    )
