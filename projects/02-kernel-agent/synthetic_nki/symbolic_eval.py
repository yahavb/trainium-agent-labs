"""Offline constraints paired with actual simulator evidence; no model requests."""
import json
import time
from pathlib import Path
import sympy as sp
from symbolic_shapes import analyze
from synthetic_nki.diagnostics import identifies
from shape_repair import plan_repair
import agent

CAUSE_KINDS={'group_axis_element_conservation':{'dma_elements'},'access_pattern_bounds':{'slice_bounds'},'transpose_output_buffer':{'transpose_buffer'},'matching_dma_extents':{'dma_elements'},'explicit_onchip_rank':{'onchip_rank'},'reduction_rank':{'reduction_rank'},'matmul_destination':{'matmul_dimensions'},'partition_limit':{'partition_limit','matmul_tile_limit'},'onchip_copy_regions':{'matmul_buffer'},'transpose_regions':{'transpose_buffer'},'boundary_slice_tail':{'slice_bounds'}}


def evaluate(directory,output):
    records=[json.loads(line) for split in ('train','heldout') for line in (Path(directory)/(split+'.jsonl')).read_text().splitlines()]
    rows=[];clean_seen=set();clean_results=[]
    for r in records:
        shapes={name:tuple(shape) for name,shape in zip(('a','b'),r['input_shapes'])};source=r['broken_kernel'];error=r['observed_error']
        started=time.perf_counter();symbolic=analyze(source,shapes);elapsed=time.perf_counter()-started
        expected=CAUSE_KINDS.get(r['root_cause'],set());hits=[v for v in symbolic['violations'] if v['kind'] in expected]
        legacy=agent.enrich(error);legacy_added=legacy[len(error):] if legacy.startswith(error) else legacy
        targeted=plan_repair(source,error,input_shapes=shapes)['guidance']
        combined=targeted+'\n'+'\n'.join('line '+str(v['line'])+': '+v['message'] for v in hits)
        # Every reported location must refer to an actual mutation-relevant call
        # or allocation, compared against AST facts rather than error label.
        import ast
        nodes=list(ast.walk(ast.parse(source)))
        valid_lines={n.lineno for n in nodes if isinstance(n,(ast.Call,ast.Assign))}
        rows.append(dict(example_id=r['example_id'],cause=r['root_cause'],category=r['failure_category'],legacy_correct=identifies(r['root_cause'],legacy_added),targeted_correct=identifies(r['root_cause'],targeted),combined_correct=identifies(r['root_cause'],combined),symbolic_detected_cause=bool(hits),valid_source_locations=all(v['line'] in valid_lines for v in symbolic['violations']),symbolic=symbolic,analysis_seconds=elapsed))
        if r['structural_hash'] not in clean_seen:
            clean_seen.add(r['structural_hash']);clean_results.append(dict(example_id=r['example_id'],result=analyze(r['correct_kernel'],shapes)))
    summary=dict(sympy_version=sp.__version__,accepted_mutations=len(rows),legacy_correct=sum(r['legacy_correct'] for r in rows),targeted_correct=sum(r['targeted_correct'] for r in rows),combined_correct=sum(r['combined_correct'] for r in rows),symbolic_causes_detected=sum(r['symbolic_detected_cause'] for r in rows),unknown_count=sum(r['symbolic']['status']=='UNKNOWN' for r in rows),valid_source_location_records=sum(r['valid_source_locations'] for r in rows),total_analysis_seconds=sum(r['analysis_seconds'] for r in rows),clean_kernels=len(clean_results),clean_false_positive_kernels=sum(bool(r['result']['violations']) for r in clean_results),rows=rows,clean_results=clean_results,limits='Only directly supported shape constraints are scored. An API/numerical error can coexist with no static shape violation. Valid source locations are syntactic evidence, not proof of runtime causality. UNKNOWN is conservative; no physical device tested.')
    with Path(output).open('x') as file:json.dump(summary,file,indent=2)
    return summary

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--data',default='synthetic_nki/data_v2');p.add_argument('--output',required=True);a=p.parse_args();r=evaluate(a.data,a.output);print({k:v for k,v in r.items() if k not in ('rows','clean_results')})
