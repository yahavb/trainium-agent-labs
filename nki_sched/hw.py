"""Hardware description: tile caps, memory limits, and the ns-level instruction table.

Numbers marked [unverified] have not been confirmed against the installed NKI/hardware (PLAN §3.2)
and are overridable; tile caps come from nkibench / the nc_matmul documentation.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HardwareConfig:
    pmax: int = 128  # partition dim max (nl.tile_size.pmax)
    stationary_fmax: int = 128  # nl.tile_size.gemm_stationary_fmax
    moving_fmax: int = 512  # nl.tile_size.gemm_moving_fmax
    psum_bank_bytes: int = 2048  # per partition, one accumulation bank   [unverified]
    psum_banks: int = 8  # [unverified]
    sbuf_bytes_per_partition: int = 192 * 1024  # [unverified]


NC_DEFAULT = HardwareConfig()
# A tiny configuration so scalar-loop interpretation of whole schedules stays fast in tests.
NC_TINY = HardwareConfig(pmax=4, stationary_fmax=4, moving_fmax=8, psum_bank_bytes=8 * 4)

DTYPE_BYTES = {"f32": 4, "bf16": 2, "f16": 2}


@dataclass(frozen=True)
class Instr:
    name: str  # ns-level name: "ns.tensor.matmul"
    engine: str  # tensor | vector | scalar | gpsimd | sync
    nisa: str  # NKI function name: "nc_matmul"
    roles: tuple  # argument roles in call order
    mems: dict  # role -> required memory space (or tuple of allowed)


INSTRS = {
    "ns.tensor.matmul": Instr(
        "ns.tensor.matmul", "tensor", "nc_matmul", ("dst", "stationary", "moving"),
        {"dst": ("PSUM",), "stationary": ("SBUF",), "moving": ("SBUF",)},
    ),
    "ns.vector.tensor_copy": Instr(
        "ns.vector.tensor_copy", "vector", "tensor_copy", ("dst", "src"),
        {"dst": ("SBUF", "PSUM"), "src": ("SBUF", "PSUM")},
    ),
    "ns.sync.dma_copy": Instr(
        "ns.sync.dma_copy", "sync", "dma_copy", ("dst", "src"),
        {"dst": ("HBM", "SBUF"), "src": ("HBM", "SBUF")},
    ),
}

# engines a copy-like instruction may be moved to by set_engine (NKI: tensor_copy/dma_copy take engine=)
COPY_ENGINES = ("vector", "scalar", "gpsimd")
