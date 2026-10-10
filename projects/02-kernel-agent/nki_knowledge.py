"""Small version-aware knowledge cards; never reads benchmark reference source."""
import importlib
import inspect
import platform
from importlib import metadata

TOOLKIT = 'https://github.com/aws-neuron/neuron-agentic-development'
CARDS = {
    'api': 'Use installed API names and signatures. nl.ndarray allocates tensors; nl.affine_range loops; nl.sum(view, axis=[...]) reduces free axes. nisa.tensor_scalar applies scalar arithmetic; nisa.tensor_copy copies on-chip; nisa.dma_copy transfers HBM/SBUF.',
    'dimensions': 'SBUF/PSUM tensors need partition and free axes (at least 2 dimensions). Partition <=128. Preserve singleton axes when slicing.',
    'partition': 'Tile the partition axis in chunks <=128; match actual tail sizes rather than reading padded HBM.',
    'matmul_layout': 'nc_matmul consumes stationary [K,M] and moving [K,N] in SBUF; dst [M,N] in PSUM. K<=128, M<=128, N<=512 per instruction. Copy PSUM to SBUF before HBM DMA. Accumulate all K tiles before the final write; verify initialization/accumulation against the installed signature.',
    'memory': 'Allocate explicitly with nl.ndarray(shape, dtype=..., buffer=nl.sbuf/nl.psum/nl.shared_hbm). Memory regions are not callable.',
    'dma_shape': 'DMA source and destination views must have matching element counts and compatible layouts. Check each slice extent, including singleton and tail dimensions.',
    'bounds': 'Use ceiling tile counts and clamp tail extents. Do not access beyond actual HBM shape.',
    'reshape': 'Reshape preserves element count and does not transpose. Access-pattern strides are in elements; preserve the partition dimension.',
    'missing_writes': 'Allocate a fresh shared_hbm result. Every output tile must be written before return; do not modify input tensors.',
    'numerical': 'Use deterministic index/identity probes to separate indexing from arithmetic. For matmul verify contraction and accumulation over every K tile. For pooling divide by p*p. For attention use stable row-wise softmax.',
    'partial_shapes': 'Compare passing and failing shapes; check hardcoded extents, loop trip counts and tile boundaries. Preserve all passing shapes.',
    'traffic': 'Reuse operand tiles in SBUF across M/N loops. Keep partial sums on chip; bytes, not CPU simulator runtime, determine levels 5–7.',
}

def environment():
    info = {'platform': platform.platform(), 'python': platform.python_version(), 'nki_available': False, 'packages': {}}
    for package in ('nki', 'neuronx-cc', 'neuronxcc', 'numpy'):
        try:
            info['packages'][package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            pass
    try:
        mod = importlib.import_module('nki')
        info.update(nki_available=True, nki_version=str(getattr(mod, '__version__', 'unknown')), simulator=next((n for n in ('simulate', 'simulate_kernel') if hasattr(mod, n)), None))
    except ImportError:
        pass
    return info

def retrieve(categories):
    cards = [CARDS[c] for c in dict.fromkeys(categories) if c in CARDS]
    # Installed APIs take precedence over Beta 3 examples in the external toolkit.
    for name in ('dma_copy', 'nc_matmul', 'tensor_copy', 'tensor_scalar'):
        try:
            fn = getattr(importlib.import_module('nki.isa'), name)
            cards.append(f'Installed nisa.{name}{inspect.signature(fn)}')
        except (ImportError, AttributeError, TypeError, ValueError):
            pass
    return '\n'.join(cards)
