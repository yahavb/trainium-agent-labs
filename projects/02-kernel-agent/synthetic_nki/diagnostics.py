"""Bug-specific checks of diagnoses; not an LLM or proof of repairability."""
import json
import re
from pathlib import Path
import agent
from shape_repair import plan_repair


def identifies(cause,text):
    """Predeclared semantic checks independent of the classifier's category."""
    t=text.lower()
    rules={
        'boundary_slice_tail': lambda: ('partial' in t or 'final tile' in t) and ('clip' in t or 'min(' in t),
        'transpose_regions': lambda: 'nc_transpose' in t and ('sbuf' in t or 'on-chip' in t),
        'matching_dma_extents': lambda: 'dma' in t and ('slice' in t or 'source' in t) and ('allocation' in t or 'allocated' in t or 'destination' in t),
        'explicit_onchip_rank': lambda: ('rank 1' in t or '1d' in t or 'one-dimensional' in t) and ('2 dimensions' in t or '2d' in t or 'two-dimensional' in t),
        'reduction_rank': lambda: ('reduc' in t and ('keepdims' in t or 'free axis' in t)),
        'matmul_destination': lambda: 'nc_matmul' in t and ('produce' in t or 'result' in t) and ('dst' in t or 'destination' in t) and 'reshape' in t,
        'partition_limit': lambda: ('partition' in t or 'rows' in t) and ('tile' in t or 'chunk' in t) and ('128' in t or 'maximum' in t),
        'contraction_accumulation': lambda: 'accumulate=false' in t and 'overwrite' in t and ('first' in t or 'overwrite' in t) and ('contraction' in t or 'k tile' in t),
        'output_coverage': lambda: ('line ' in t and 'loop' in t and 'output' in t and ('cover' in t or 'missing' in t or 'every' in t)),
        'onchip_copy_regions': lambda: ('sbuf' in t or 'psum' in t) and ('region' in t or 'buffer' in t or 'on-chip' in t),
        'supported_scalar_api': lambda: 'scalar_mul' in t and ('signature' in t or 'supported' in t or 'use' in t or 'real' in t),
        'supported_binary_api': lambda: 'magic_multiply' in t and ('signature' in t or 'supported' in t or 'use' in t or 'real' in t),
        'unsupported_dma_api': lambda: 'nonexistent_dma' in t and ('signature' in t or 'supported' in t or 'real' in t),
        'valid_scalar_keyword': lambda: 'operand' in t and ('remove' in t or 'keyword' in t or 'signature' in t),
    }
    return bool(rules.get(cause,lambda:False)())


def evaluate(directory,output):
    rows=[]
    for split in ('train','heldout'):
        for line in (Path(directory)/(split+'.jsonl')).read_text().splitlines():
            record=json.loads(line);error=record['observed_error'];source=record['broken_kernel']
            shapes=dict(zip(('a','b'),record['input_shapes']))
            legacy=agent.enrich(error)
            plan=plan_repair(source,error,input_shapes=shapes)
            # Score added explanation, not the raw error containing the answer.
            legacy_added=legacy[len(error):] if legacy.startswith(error) else legacy
            targeted=plan['guidance']
            rows.append(dict(example_id=record['example_id'],split=split,category=record['failure_category'],injected_cause=record['root_cause'],actual_error=error,legacy_guidance=legacy_added,targeted_guidance=targeted,legacy_identifies_cause=identifies(record['root_cause'],legacy_added),targeted_identifies_cause=identifies(record['root_cause'],targeted)))
    categories={}
    for category in sorted({r['category'] for r in rows}):
        group=[r for r in rows if r['category']==category]
        categories[category]=dict(total=len(group),legacy_correct=sum(r['legacy_identifies_cause'] for r in group),targeted_correct=sum(r['targeted_identifies_cause'] for r in group))
    result=dict(total=len(rows),legacy_correct=sum(r['legacy_identifies_cause'] for r in rows),targeted_correct=sum(r['targeted_identifies_cause'] for r in rows),categories=categories,records=rows,limitations='Deterministic bug-specific semantic checks, manually reviewed sample; not general diagnostic accuracy or proof of model repair. Raw error excluded from scoring; input shapes independently specified.')
    with Path(output).open('x') as file:json.dump(result,file,indent=2)
    return result
