"""Verified train split only; render small changed statements, never full kernels."""
import difflib
import json
from pathlib import Path
from failure_selection import code_fingerprint
from nki_knowledge import installed_compatibility, local_token_counter, referenced_operations

DATA=Path(__file__).parent/'data_v4'/'train.jsonl'

def load_train(path=DATA):
    if Path(path).name!='train.jsonl':return []
    compatibility=installed_compatibility();records=[];seen=set()
    try:lines=Path(path).read_text().splitlines()
    except OSError:return []
    for line in lines:
        try:
            record=json.loads(line);results=record['verification_results']
            if record.get('split')!='train' or record.get('verification_status')!='SIMULATOR_VERIFIED':continue
            if record['sdk_version']!=compatibility['sdk_version'] or record['hardware_target']!=compatibility['hardware']:continue
            if not results['clean']['passed'] or results['broken']['passed'] or not results['repaired']['passed']:continue
            if record['corrected_kernel']!=record['correct_kernel']:continue
            fp=code_fingerprint(record['correct_kernel'])
            key=(fp,record['root_cause'])
            if fp!=record['structural_hash'] or key in seen:continue
            seen.add(key);records.append(record)
        except (KeyError,TypeError,ValueError):continue
    return records

def desired_cause(category,feedback,source):
    apis,_=referenced_operations(source)
    if category=='DMA_SHAPE_MISMATCH':return 'matching_dma_extents'
    if category=='OUT_OF_BOUNDS':return 'boundary_slice_tail'
    if category=='INVALID_BUFFER_PLACEMENT':return 'transpose_regions' if any(a.endswith('nc_transpose') for a in apis) else 'onchip_copy_regions'
    if category=='INVALID_API_ARGUMENT':return 'valid_scalar_keyword' if any(a.endswith('tensor_scalar') for a in apis) else None
    if category=='INVALID_API_FUNCTION':return 'supported_scalar_api' if 'scalar' in feedback or 'multiply' in feedback else 'supported_binary_api'
    if category=='INVALID_TENSOR_DIMENSIONS':
        if 'partition' in feedback and 'exceed' in feedback:return 'partition_limit'
        if any(a in ('nki.language.sum','nki.language.max') for a in apis):return 'reduction_rank'
        if any(a.endswith('nc_matmul') for a in apis):return 'matmul_destination'
        return 'explicit_onchip_rank'
    if category in ('NUMERICAL_MISMATCH','INCOMPLETE_OUTPUT'):
        return 'contraction_accumulation' if any(a.endswith('nc_matmul') for a in apis) else 'output_coverage'
    return None

def retrieve_examples(category,feedback,source,*,path=DATA,token_budget=350,token_counter=None):
    counter,_=local_token_counter() if token_counter is None else (token_counter,'provided')
    wanted=desired_cause(category,feedback,source)
    if wanted is None:return dict(text='',example_ids=[],reason='No safe cause match')
    ranked=sorted((r for r in load_train(path) if r['failure_category']==category and r['root_cause']==wanted),key=lambda r:r['example_id'])
    blocks=[];ids=[]
    for record in ranked:
        removed=[];added=[]
        for line in difflib.ndiff(record['broken_kernel'].splitlines(),record['corrected_kernel'].splitlines()):
            if line.startswith('- '):removed.append(line[2:].strip())
            elif line.startswith('+ '):added.append(line[2:].strip())
        block=(f"Verified tiny synthetic repair {record['example_id']} ({record['root_cause']}):\n"
               +'Before: '+'; '.join(removed[:2])+'\nAfter: '+'; '.join(added[:2])
               +'\nTransfer the invariant, not these illustrative dimensions or constants. Reconcile your actual allocations, slices and consumers. CPU verified; not device verified.')
        if counter('\n\n'.join(blocks+[block]))>token_budget:continue
        blocks.append(block);ids.append(record['example_id'])
        if len(ids)==2:break
    return dict(text='\n\n'.join(blocks),example_ids=ids,root_cause=wanted,context_token_count=counter('\n\n'.join(blocks)),reason='Exact failure category and inferred cause; train-only compatible verified records')

def example_prompt(prompt,diagnostic,source,*,model='Qwen/Qwen3-8B',context=8192,answer_budget=2500):
    counter,method=local_token_counter(model);budget=max(0,min(350,context-answer_budget-128-counter(prompt)))
    result=retrieve_examples(diagnostic.failure_category,diagnostic.original_feedback,source,token_budget=budget,token_counter=counter)
    result['counting_method']=method
    return prompt+('\n\nTransferable repair examples:\n'+result['text'] if result['text'] else ''),result
