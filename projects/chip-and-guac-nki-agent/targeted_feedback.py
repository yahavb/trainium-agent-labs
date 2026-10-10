"""Arm C: immutable B checker/selection, SDK-proven free-dimension guidance only."""

from contextlib import contextmanager
import functools
import inspect
import json
import os
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
import curriculum as base
from portable import verify_sources

TARGETS = {
    'stationary': re.compile(r'Matmul stationary free dimension (\d+) exceeds gemm_stationary_fmax=(\d+)'),
    'moving': re.compile(r'Matmul moving free dimension (\d+) exceeds max (\d+) for nc_version=nc_version\.(gen2|gen3)'),
}


def guard():
    base.process_guard()
    scripts = {'arm_c_agent.py', 'parity_arm_c.py', 'allocation_audit.py', 'test_arm_c.py'}
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            stat = (proc/'stat').read_text()
            if stat[stat.rfind(')')+2:].split()[0] == 'Z':
                continue
            args = (proc/'cmdline').read_bytes().decode(errors='replace').split('\0')
        except (FileNotFoundError, ProcessLookupError):
            continue
        if any(Path(arg).name in scripts for arg in args):
            raise RuntimeError('Another Arm C agent/auditor is active: '+proc.name)
    return {'active_agents_or_graders': []}


def verify_inputs():
    return verify_sources()


def sdk_constants():
    import nki.language as nl
    values = {name:int(getattr(nl.tile_size,name)) for name in
              ('pmax','gemm_stationary_fmax','gemm_moving_fmax','psum_bank_fmax')}
    expected = {'pmax':128,'gemm_stationary_fmax':128,'gemm_moving_fmax':512,'psum_bank_fmax':512}
    if values != expected:
        raise RuntimeError('SDK constants differ from inspected values')
    return values


def guidance(limits):
    p, m, n = (limits[name] for name in ('pmax','gemm_stationary_fmax','gemm_moving_fmax'))
    return (
        '\n\nFree-dimension repair (installed NKI SDK):\n'
        f'1. nl.tile_size.gemm_stationary_fmax={m}, gemm_moving_fmax={n}, and pmax={p}. '
        'A single nc_matmul uses stationary (K_chunk, M_tile), moving (K_chunk, N_tile), '
        f'and PSUM (M_tile, N_tile), with K_chunk<={p}, M_tile<={m}, N_tile<={n}.\n'
        '2. Add output-tile loops over BOTH M and N. For each output tile, allocate its own '
        'float32 nl.psum accumulator INSIDE the M/N loops but OUTSIDE its K loop. '
        'Do not use one full (M,N) PSUM tensor as the nc_matmul destination.\n'
        '3. Within that output tile, loop over K chunks and allocate only the current '
        'operand tiles in nl.sbuf, not full (K,M) or (K,N) backing tensors. lhsT already '
        'has K first: DMA lhsT[k0:k0+kt, m0:m0+mt] into (kt,mt), and '
        'rhs[k0:k0+kt, n0:n0+nt] into (kt,nt); local tile indices start at zero. '
        'Source and destination extents must match, including tails.\n'
        '4. Call nisa.nc_matmul into the SAME current PSUM tile for every K chunk. '
        'Use accumulate=False on its first matmul and accumulate=True thereafter; '
        'the installed default None also infers first-write overwrite then accumulation. '
        'Do not initialize it with a non-matmul operation or reset it between K chunks.\n'
        '5. Only after all K chunks, nisa.tensor_copy the completed PSUM tile to an '
        '(mt,nt) SBUF tile, then nisa.dma_copy it to '
        'out[m0:m0+mt, n0:n0+nt]. Write each output tile once, with both offsets; '
        'never DMA partial K sums. Derive mt, nt and kt from remaining extents, '
        'without unmasked padding.'
    )


@contextmanager
def sdk_provenance():
    """Observe unchanged SDK validators; forward exact calls/results/exceptions."""
    import nki.isa as nisa
    matmul = inspect.unwrap(nisa.nc_matmul)
    shapes = matmul.__globals__['validate_matmul_shapes']
    dst = shapes.__globals__['validate_matmul_dst_shape']
    events = []

    def observe(function, kind):
        @functools.wraps(function)
        def wrapped(*args, **kwargs):
            try:
                return function(*args, **kwargs)
            except AssertionError as exc:
                # Only a real assertion raised inside the pinned validator qualifies.
                cursor = exc.__traceback__
                proven = False
                while cursor:
                    if cursor.tb_frame.f_code is function.__code__:
                        proven = True
                    cursor = cursor.tb_next
                frame = inspect.currentframe()
                in_matmul = False
                while frame:
                    if frame.f_code is matmul.__code__:
                        in_matmul = True
                    frame = frame.f_back
                if proven and in_matmul and TARGETS[kind].fullmatch(str(exc)):
                    events.append({'kind':kind,'message':str(exc),
                                   'validator':function.__name__,
                                   'sdk_file':inspect.getsourcefile(function)})
                raise
        return wrapped

    matmul.__globals__['validate_matmul_shapes'] = observe(shapes,'stationary')
    shapes.__globals__['validate_matmul_dst_shape'] = observe(dst,'moving')
    try:
        yield events
    finally:
        matmul.__globals__['validate_matmul_shapes'] = shapes
        shapes.__globals__['validate_matmul_dst_shape'] = dst


def enrich_selected(result, details, level, limits, events, enabled):
    feedback = result[2]
    info = {'guidance_enabled':enabled,'guidance_applied':False,'target_family':None,
            'b_feedback':feedback,'sdk_provenance':events}
    if not enabled or level != 4:
        return result, info
    for label,message in details['failures']:
        if not feedback.endswith(f'On {label}: {message}'):
            continue
        for event in events:
            if message != 'raised AssertionError: '+event['message']:
                continue
            kind = event['kind']
            match = TARGETS[kind].fullmatch(event['message'])
            expected = limits['gemm_stationary_fmax' if kind=='stationary' else 'gemm_moving_fmax']
            if match and int(match[2]) == expected and int(match[1]) > expected:
                info.update(guidance_applied=True,target_family=kind)
                return (result[0],result[1],feedback+guidance(limits)),info
    return result,info


def evaluate(agent, source, level, enabled=False, policy='curriculum'):
    if not enabled or level != 4:
        result,details = base.evaluate(agent,source,level,policy)
        info = {'guidance_enabled':enabled,'guidance_applied':False,'target_family':None,
                'b_feedback':result[2],'sdk_provenance':[]}
    else:
        with sdk_provenance() as events:
            result,details = base.evaluate(agent,source,level,policy)
        result,info = enrich_selected(result,details,level,sdk_constants(),events,True)
    details.update(info)
    return result,details
