"""Small AWS-attributed, Trainium2/NKI-0.6 reference catalog and local retrieval.

No kernels, network access, embeddings, or model calls. Cards describe general
invariants, not benchmark solutions. Source/signature checks are not device proof.
"""
import ast
from dataclasses import dataclass, asdict
from functools import lru_cache
import inspect
import os
from pathlib import Path
import re

DOCS = 'https://awsdocs-neuron.readthedocs-hosted.com/en/v2.32.0/'
SDK_BUILD = '0.6.0+31049202112.g85070674'
AWS_COMMIT = 'ee25e45b9ace4aaefd8b8913bf5fa9b68adf6674'


@dataclass(frozen=True)
class KnowledgeCard:
    id: str
    concept: str
    categories: tuple
    api: str | None
    signature: str | None
    parameters: tuple
    buffers: str
    allowed: str
    invalid: str
    guidance: str
    source_url: str
    hardware: tuple = ('trn2',)
    sdk_versions: tuple = ('0.6.0',)
    verified_build: str = SDK_BUILD
    verification_status: str = 'installed signature/source checked; not device tested'


def _card(id, concept, categories, api, signature, parameters, buffers, allowed, invalid, guidance, page):
    return KnowledgeCard(id, concept, tuple(categories.split()), api, signature,
                         tuple(parameters.split()), buffers, allowed, invalid, guidance, DOCS + page)


CATALOG = (
    _card('dma', 'nisa.dma_copy', 'DMA_SHAPE_MISMATCH INVALID_BUFFER_PLACEMENT INVALID_API_ARGUMENT',
          'nki.isa.dma_copy', 'dma_copy(dst, src, priority=None, oob_mode=error, dge_mode=unknown, engine=unknown, name=None)',
          'dst src priority oob_mode dge_mode engine name', 'HBM/SBUF only; no PSUM',
          'dst=..., src=...; equal total element counts', 'broadcasting; whole padded destination with partial source',
          'Match src and dst slice element counts, including edge tiles. DMA neither broadcasts nor selects a smaller piece automatically. Leave priority unset on Trainium2.',
          'nki/api/generated/nki.isa.dma_copy.html'),
    _card('matmul', 'nisa.nc_matmul', 'INVALID_API_ARGUMENT INVALID_BUFFER_PLACEMENT INVALID_TENSOR_DIMENSIONS',
          'nki.isa.nc_matmul', 'nc_matmul(dst, stationary, moving, is_stationary_onezero=False, is_moving_onezero=False, is_transpose=False, accumulate=None, tile_position=(), tile_size=(), perf_mode=none, name=None)',
          'dst stationary moving is_stationary_onezero is_moving_onezero is_transpose accumulate tile_position tile_size perf_mode name',
          'stationary/moving SBUF; dst FP32 PSUM on trn2',
          'Normal FP32 mode: stationary[K,M], moving[K,N], result[M,N]; K<=128, M<=128, N<=512',
          'transpose_moving; SBUF dst; ordinary A[M,K] as stationary',
          'Computes stationary.T @ moving. Remove transpose_moving; do not substitute is_transpose without checking its specialized semantics. Recheck operand layout and legal tiles.',
          'nki/api/generated/nki.isa.nc_matmul.html'),
    _card('scalar', 'nisa.tensor_scalar', 'INVALID_API_FUNCTION INVALID_API_ARGUMENT INVALID_TENSOR_DIMENSIONS',
          'nki.isa.tensor_scalar', 'tensor_scalar(dst, data, op0, operand0, reverse0=False, op1=None, operand1=None, reverse1=False, engine=unknown, name=None)',
          'dst data op0 operand0 reverse0 op1 operand1 reverse1 engine name', 'data/dst SBUF or PSUM',
          'op0=nl.multiply, operand0=constant or FP32 (P,1) vector', 'nisa.multiply; nisa.scalar_mul; op=/operand= keywords',
          'For multiplication by a scalar or per-partition vector use tensor_scalar with op0/operand0 and explicit dst. nl.multiply is an operator specifier, not an ISA function.',
          'nki/api/generated/nki.isa.tensor_scalar.html'),
    _card('binary', 'nisa.tensor_tensor', 'INVALID_API_FUNCTION INVALID_API_ARGUMENT INVALID_TENSOR_DIMENSIONS',
          'nki.isa.tensor_tensor', 'tensor_tensor(dst, data1, data2, op, engine=unknown, name=None)',
          'dst data1 data2 op engine name', 'SBUF/PSUM; at least one input must be SBUF',
          'op=nl.multiply or nl.add; equal partition count and elements per partition', 'nisa.multiply; assuming automatic shape broadcasting',
          'For two same-layout tiles use tensor_tensor(dst=..., data1=..., data2=..., op=nl.multiply). Both inputs cannot be PSUM. Check partition and per-partition element counts; do not guess a broadcast.',
          'nki/api/generated/nki.isa.tensor_tensor.html'),
    _card('reduce', 'nisa.tensor_reduce', 'INVALID_API_ARGUMENT INVALID_TENSOR_DIMENSIONS NUMERICAL_MISMATCH',
          'nki.isa.tensor_reduce', 'tensor_reduce(dst, op, data, axis, negate=False, keepdims=False, name=None)',
          'dst op data axis negate keepdims name', 'data/dst SBUF or PSUM',
          'op=nl.add; trailing contiguous free axes ending at the last dimension', 'axis=0; wrong destination size; unsupported dtype= keyword',
          'Reduce only trailing contiguous free dimensions, never partition axis 0. Preserve P and allocate the reduced free shape. negate and keepdims are supported in this SDK.',
          'nki/api/generated/nki.isa.tensor_reduce.html'),
    _card('copy', 'nisa.tensor_copy', 'INVALID_BUFFER_PLACEMENT INVALID_TENSOR_DIMENSIONS INVALID_API_ARGUMENT',
          'nki.isa.tensor_copy', 'tensor_copy(dst, src, engine=unknown, name=None)',
          'dst src engine name', 'SBUF/PSUM on-chip only; not HBM',
          'same P and element count per partition; explicit dst/src', 'HBM destination; reshaping P during copy',
          'Use tensor_copy for PSUM to SBUF. Then dma_copy from SBUF to returned shared_hbm output; do not DMA PSUM directly.',
          'nki/api/generated/nki.isa.tensor_copy.html'),
    _card('allocation', 'nl.ndarray', 'INVALID_TENSOR_DIMENSIONS INVALID_BUFFER_PLACEMENT INVALID_API_FUNCTION',
          'nki.language.ndarray', "ndarray(shape, dtype, buffer=nl.sbuf, name='', address=None)",
          'shape dtype buffer name address', 'nl.sbuf / nl.psum / nl.shared_hbm region objects',
          'explicit shape and dtype; buffer=region', 'calling nl.sbuf(); buffer string; treating uninitialized allocation as zeros',
          'Allocate with ndarray(shape, dtype, buffer=...). Memory regions are objects, not functions. Allocation does not initialize values; write every element before reading.',
          'nki/api/generated/nki.language.ndarray.html'),
    _card('rank', 'On-chip tensor rank', 'INVALID_TENSOR_DIMENSIONS', None, None, '',
          'SBUF/PSUM', 'at least two dimensions: partition P followed by free F', 'one-dimensional on-chip allocations',
          'Keep SBUF/PSUM tensors at least 2D. A vector needs a legal (P,1) or (1,F) layout according to the consuming operation; these layouts are not interchangeable.',
          'nki/api/generated/nki.language.ndarray.html'),
    _card('memory', 'Memory regions and transfers', 'INVALID_BUFFER_PLACEMENT INVALID_API_FUNCTION', None, None, '',
          'HBM inputs/output; SBUF operands; PSUM matmul accumulation', 'region objects via buffer=; explicit transfers',
          'HBM compute operands; direct PSUM DMA',
          'Check the offending operation before changing a buffer. Compute on on-chip tiles; move PSUM via tensor_copy to SBUF before DMA to shared_hbm.',
          'nki/get-started/about/memory-hierarchy-overview.html'),
    _card('tiling', 'Partition/free dimensions and bounds', 'OUT_OF_BOUNDS INVALID_TENSOR_DIMENSIONS DMA_SHAPE_MISMATCH INCOMPLETE_OUTPUT',
          None, None, '', 'on-chip partition P <= 128; instruction-specific free limits',
          'bounds from tensor.shape; partial final slices', 'padding access to hardware maximum; treating instruction limit as minimum',
          'Hardware tile sizes are maxima. Clamp each boundary to the actual shape, use matching src/dst slices, preserve partition layout, and store each output tile at its own offset.',
          'nki/get-started/about/tiling-overview.html'),
    _card('accumulation', 'PSUM initialization and accumulation', 'HARDWARE_CORRECTNESS_HAZARD NUMERICAL_MISMATCH INCOMPLETE_OUTPUT',
          'nki.isa.nc_matmul', None, 'accumulate', 'FP32 PSUM destination on trn2',
          'first matmul accumulate=False; subsequent matmuls True; None auto-infers', 'uninitialized accumulate=True; memset then accumulate=True on trn2',
          'Initialize each output PSUM location with the first matmul overwrite, then accumulate subsequent contraction tiles. Preserve all output stores; do not replace initialization with a non-matmul write.',
          'nki/api/generated/nki.isa.nc_matmul.html'),
    _card('simulation', 'nki.simulate and limitations', 'HARDWARE_CORRECTNESS_HAZARD',
          'nki.simulate', 'simulate(kernel)', 'kernel', 'CPU simulation of on-chip/HBM tensors',
          'nki.simulate(kernel)(*args); trn2 target', 'assuming simulation proves compilation, physical allocation fit or device latency',
          'Respect simulator hardware warnings. Target trn2; passing CPU numerics does not prove device correctness or performance. Compiler and hardware validation remain separate.',
          'nki/guides/nki_simulator.html'),
)


@lru_cache(maxsize=1)
def installed_compatibility():
    """Local inspection only; fail closed when the SDK or expected parameters differ."""
    try:
        import nki
        import nki.isa as nisa
        import nki.language as nl
        signatures = {}
        for card in CATALOG:
            if card.api and card.api not in signatures:
                obj = nki if card.api == 'nki.simulate' else (nisa if '.isa.' in card.api else nl)
                fn = getattr(obj, card.api.rsplit('.', 1)[-1], None)
                signatures[card.api] = tuple(inspect.signature(fn).parameters) if callable(fn) else ()
        target = os.environ.get('NEURON_PLATFORM_TARGET_OVERRIDE', 'trn2')
        target = {'gen3':'trn2', 'gen4':'trn3', 'gen2':'trn1'}.get(target, target)
        return dict(sdk_version=nki.__version__, hardware=target, signatures=signatures)
    except (ImportError, AttributeError, TypeError, ValueError):
        return dict(sdk_version='unavailable', hardware='trn2', signatures={})


def compatible(card, compatibility):
    version = compatibility['sdk_version'].split('+')[0]
    if version not in card.sdk_versions or compatibility['hardware'] not in card.hardware:
        return False
    if compatibility['sdk_version'] != card.verified_build:
        return False
    if card.api:
        actual = compatibility['signatures'].get(card.api, ())
        if not set(card.parameters).issubset(actual):
            return False
        # For complete API cards, unexpected signature drift also fails closed.
        if card.signature and tuple(actual) != card.parameters:
            return False
    return card.verification_status.startswith('installed signature/source checked')


def referenced_operations(source):
    aliases = {'nl': 'nki.language', 'nisa': 'nki.isa', 'nki': 'nki'}
    calls = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return (), ()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names: aliases[item.asname or item.name] = item.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            for item in node.names: aliases[item.asname or item.name] = node.module + '.' + item.name
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            dotted = ast.unparse(node.func)
            head, _, tail = dotted.partition('.')
            canonical = aliases.get(head, head) + ('.' + tail if tail else '')
            if canonical.startswith(('nki.isa.', 'nki.language.')):
                calls.append(dict(api=canonical, line=node.lineno, expression=ast.unparse(node)[:200],
                                  keywords=tuple(k.arg for k in node.keywords if k.arg)))
    return tuple(dict.fromkeys(c['api'] for c in calls)), tuple(calls)


@lru_cache(maxsize=2)
def local_token_counter(model='Qwen/Qwen3-8B'):
    """Use a cached tokenizer file, never download or contact the model endpoint."""
    try:
        from tokenizers import Tokenizer
        cache = Path.home() / '.cache/huggingface/hub' / ('models--' + model.replace('/', '--'))
        paths = sorted(cache.glob('snapshots/*/tokenizer.json'))
        if paths:
            tokenizer = Tokenizer.from_file(str(paths[0]))
            return lambda text: len(tokenizer.encode(text, add_special_tokens=False).ids), 'local_tokenizer'
    except (ImportError, OSError, ValueError):
        pass
    # A byte-per-token ceiling is deliberately conservative for byte-level BPE.
    # This is a budgeting bound, NOT endpoint-reported token usage.
    return lambda text: len(text.encode('utf-8')), 'utf8_byte_ceiling'


def _render(card):
    signature = (card.signature + '\n') if card.signature else ''
    return f"[{card.id}] {card.concept}\n{signature}{card.buffers}. {card.allowed}.\n{card.guidance}\nAWS: {card.source_url}"


def retrieve(diagnostic, source, *, compatibility=None, token_budget=500, token_counter=None):
    compatibility = compatibility or installed_compatibility()
    counter, method = local_token_counter() if token_counter is None else (token_counter, 'provided_counter')
    apis, calls = referenced_operations(source)
    feedback = diagnostic.original_feedback
    category = diagnostic.failure_category
    ids = []
    # Feedback identifies the failing operation before unrelated source calls.
    known = ('dma_copy', 'nc_matmul', 'tensor_scalar', 'tensor_tensor', 'tensor_reduce', 'tensor_copy', 'ndarray')
    mentioned = {name for name in known if re.search(r'\b' + name + r'\b', feedback.split(' The real signature')[0])}
    if category == 'DMA_SHAPE_MISMATCH': ids = ['dma', 'tiling']
    elif category == 'INVALID_TENSOR_DIMENSIONS' and 'at least 2 dimensions' in feedback:
        ids = ['rank', 'allocation']
    elif category in ('OUT_OF_BOUNDS', 'INCOMPLETE_OUTPUT'): ids = ['tiling']
    elif category == 'INVALID_API_FUNCTION' and re.search(r"(?:no attribute|has no|no).*?(?:scalar_mul|multiply)", feedback):
        keywords = {k for c in calls if c['api'].endswith(('.multiply', '.scalar_mul')) for k in c['keywords']}
        ids = ['binary', 'scalar'] if {'data1', 'data2'} & keywords else ['scalar', 'binary']
    elif category == 'INVALID_API_FUNCTION' and 'MemoryRegion' in feedback: ids = ['allocation', 'memory']
    elif category == 'INVALID_BUFFER_PLACEMENT' and "dst must be in ['psum']" in feedback and 'nki.isa.nc_matmul' in apis:
        ids = ['matmul', 'memory']
    elif category == 'HARDWARE_CORRECTNESS_HAZARD':
        ids = ['accumulation', 'simulation'] if 'nki.isa.nc_matmul' in apis else ['simulation']
    elif category == 'NUMERICAL_MISMATCH' and 'nki.isa.nc_matmul' in apis and ('NaN' in feedback or 'PSUM' in feedback):
        ids = ['accumulation']
    mapping = {'dma_copy':'dma', 'nc_matmul':'matmul', 'tensor_scalar':'scalar', 'tensor_tensor':'binary',
               'tensor_reduce':'reduce', 'tensor_copy':'copy', 'ndarray':'allocation'}
    for name in known:
        if name in mentioned: ids.append(mapping[name])
    if not ids and category not in ('UNKNOWN', 'INVALID_API_FUNCTION'):
        relevant = [card.id for card in CATALOG if card.api in apis and category in card.categories]
        # If source has several plausible operations, avoid pretending to localize.
        if len(relevant) == 1: ids.extend(relevant)
    if category == 'INVALID_BUFFER_PLACEMENT' and ids: ids.append('memory')
    chosen, blocks = [], []
    catalog = {c.id:c for c in CATALOG}
    rejected = []
    for id in dict.fromkeys(ids):
        card = catalog[id]
        if category not in card.categories or not compatible(card, compatibility):
            rejected.append(id); continue
        block = _render(card)
        if counter('\n\n'.join(blocks + [block])) > max(0, token_budget):
            continue
        chosen.append(card); blocks.append(block)
        if len(chosen) == 2: break
    text = '\n\n'.join(blocks)
    return dict(text=text, cards=[asdict(card) for card in chosen], card_ids=[card.id for card in chosen],
                token_budget=token_budget, context_token_count=counter(text), counting_method=method,
                compatibility=compatibility, rejected_cards=rejected,
                referenced_apis=list(apis), source_calls=list(calls),
                reason='category and failing API/constraint; source aliases resolved; compatibility filtered',
                catalog_revision=AWS_COMMIT)


def ground_prompt(prompt, diagnostic, source, *, model='Qwen/Qwen3-8B', context=8192,
                  answer_budget=2500, compatibility=None):
    counter, method = local_token_counter(model)
    header = f"\n\nFailure category: {diagnostic.failure_category}. Verified local API/constraint guidance:\n"
    available = max(0, min(500, context - answer_budget - 128 - counter(prompt + header)))
    result = retrieve(diagnostic, source, compatibility=compatibility,
                      token_budget=available, token_counter=counter)
    result['counting_method'] = method
    if result['text']:
        prompt += header + result['text'] + '\nReturn one complete corrected kernel; preserve the original checker context.'
    result['applied'] = bool(result['text'])
    result['context_budget_scope'] = 'optional documentation only; original prompt is never truncated'
    return prompt, result
