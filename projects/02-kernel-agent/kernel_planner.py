"""Compact task plans generated from the benchmark specification and the installed SDK.

Nothing here is written per operation. Every sentence is computed from the level's
reference signature and reference outputs on the official shapes, the installed NKI tile
limits and instruction signatures (only instructions the task prompt itself names, plus
data movement), and the level's HBM traffic budget. No kernel code is generated.
"""
import inspect
import re
import sympy as sp
from nki_knowledge import installed_compatibility,local_token_counter
from symbolic_shapes import tiling


def tile_limits():
    import nki.language as nl
    t=nl.tile_size
    return dict(partition=int(t.pmax),psum_free=int(t.psum_fmax),stationary_free=int(t.gemm_stationary_fmax),moving_free=int(t.gemm_moving_fmax))


def plan(operation,input_shapes,*,output_shape=None,output_dtype=None,max_waste=None):
    compatibility=installed_compatibility()
    result=dict(operation=operation,input_shapes=[tuple(s) for s in input_shapes],output_shape=tuple(output_shape) if output_shape is not None else None,
                output_dtype=output_dtype,hardware='trn2',sdk_version=compatibility['sdk_version'],max_waste=max_waste,
                verification='Derived from the benchmark specification and installed SDK; simulator verification still required.')
    if not compatibility['sdk_version'].startswith('0.6.0'):return dict(result,status='UNKNOWN')
    limits=tile_limits()
    over=[dict(input=i,extent=int(shape[0]),tile_count=str(tiling(sp.Integer(int(shape[0])),sp.Integer(limits['partition']))['tile_count']))
          for i,shape in enumerate(input_shapes) if shape and int(shape[0])>limits['partition']]
    return dict(result,status='DERIVED',tile_limits=limits,partition_tiling=over,requires_tiling=bool(over))


def for_level(level):
    import nkibench
    spec=nkibench.LEVELS[level];plans=[];names=list(inspect.signature(spec['ref']).parameters)
    for case in spec['shapes']:
        args,_=nkibench.make_inputs(case,level)
        arrays=[(name,arg) for name,arg in zip(names,args) if hasattr(arg,'shape')]
        expected=spec['ref'](*args)
        plans.append(dict(plan(spec['op'],[tuple(a.shape) for _,a in arrays],output_shape=getattr(expected,'shape',None),
                               output_dtype=str(getattr(expected,'dtype','')),max_waste=spec.get('max_waste')),array_names=[n for n,_ in arrays]))
    return dict(operation=spec['op'],entry=spec['entry'],cases=plans)


def instruction_signatures(text):
    """Installed signatures of nisa/nl callables the task text names, plus data movement."""
    import nki.isa as nisa
    import nki.language as nl
    named=re.findall(r'\b(?:nisa|nl|nki\.isa|nki\.language)\.(\w+)',text)
    wanted=list(dict.fromkeys(named+['dma_copy','tensor_copy']))
    out=[]
    for name in wanted:
        module,prefix=(nisa,'nisa') if hasattr(nisa,name) else ((nl,'nl') if hasattr(nl,name) else (None,None))
        function=getattr(module,name,None) if module else None
        try:parameters=[p for p in inspect.signature(function).parameters.values() if p.default is inspect.Parameter.empty and p.kind not in (p.VAR_POSITIONAL,p.VAR_KEYWORD)]
        except (TypeError,ValueError):continue
        out.append((f"{prefix}.{name}({', '.join(p.name for p in parameters)})",any(p.name=='dst' for p in parameters)))
    return out


def guidance_sentences(result,task_text):
    cases=result['cases']
    if not cases or any(c['status']!='DERIVED' for c in cases):return []
    shapes='; '.join(', '.join(f"{n}{list(s)}" for n,s in zip(c['array_names'],c['input_shapes']))+f" -> output{list(c['output_shape'])}" for c in cases)
    limits=cases[0]['tile_limits'];sentences=[f"Derived task facts for {result['entry']}: official cases {shapes}; output dtype {cases[0]['output_dtype']}, allocated in nl.shared_hbm and returned."]
    sentences.append(f"Installed limits: on-chip tiles have partition dim <= {limits['partition']}; PSUM free dim <= {limits['psum_free']}; nc_matmul contracts over the partition dim with stationary free <= {limits['stationary_free']} and moving free <= {limits['moving_free']}.")
    tiled=sorted({(c['array_names'][o['input']],o['extent'],o['tile_count']) for c in cases for o in c['partition_tiling']})
    if tiled:sentences.append('Partition tiling needed before loading: '+', '.join(f"{n} first dim {e} -> {t} tiles" for n,e,t in tiled)+'.')
    if cases[0]['max_waste']:sentences.append(f"HBM traffic is checked: stay within {cases[0]['max_waste']}x of reading each input once and writing the output once.")
    signatures=instruction_signatures(task_text)
    if signatures:sentences.append('Installed signatures: '+'; '.join(s for s,_ in signatures)+('. Calls with a dst parameter write into dst; their return value is not a tensor.' if any(d for _,d in signatures) else '.'))
    return sentences


def generation_prompt(prompt,level,*,context=8192,answer_budget=2500,model='Qwen/Qwen3-8B'):
    result=for_level(level);counter,method=local_token_counter(model)
    budget=max(0,min(240,context-answer_budget-128-counter(prompt)))
    text=''
    for sentence in guidance_sentences(result,prompt):
        candidate=(text+' '+sentence).strip()
        if counter(candidate)<=budget:text=candidate
    result.update(guidance=text,applied=bool(text),counting_method=method,context_token_count=counter(text) if text else 0)
    return prompt+('\n\n'+text if text else ''),result


def analyze_semantics(source, operation):
    """Bounded source evidence, never a proof of mathematical kernel correctness."""
    import ast
    from symbolic_shapes import MAX_SOURCE,MAX_NODES
    result=dict(status='UNKNOWN',operation=operation,findings=[],reduction_calls=[],matmul_calls=[],verification='AST evidence only; independent numerical verification is required.')
    if len(source)>MAX_SOURCE:return result
    try:tree=ast.parse(source)
    except (SyntaxError,RecursionError):return result
    if sum(1 for _ in ast.walk(tree))>MAX_NODES:return result
    aliases={}
    for node in tree.body:
        if isinstance(node,ast.Import):
            for item in node.names:aliases[item.asname or item.name]=item.name
        elif isinstance(node,ast.ImportFrom) and node.module:
            for item in node.names:aliases[item.asname or item.name]=node.module+'.'+item.name
    def api(node):
        if isinstance(node,ast.Name):return aliases.get(node.id,node.id)
        if isinstance(node,ast.Attribute):return api(node.value)+'.'+node.attr
        return ''
    for call in (n for n in ast.walk(tree) if isinstance(n,ast.Call)):
        name=api(call.func)
        if name in ('nki.language.sum','nki.isa.tensor_reduce'):
            kwargs={k.arg:ast.unparse(k.value) for k in call.keywords if k.arg}
            result['reduction_calls'].append(dict(line=call.lineno,api=name,axes=kwargs.get('axis','UNKNOWN'),keepdims=kwargs.get('keepdims','default')))
        if name=='nki.isa.nc_matmul':
            kwargs={key:value for key,value in zip(('dst','stationary','moving'),call.args)}
            kwargs.update({k.arg:k.value for k in call.keywords if k.arg})
            same=all(k in kwargs for k in ('stationary','moving')) and ast.dump(kwargs['stationary'])==ast.dump(kwargs['moving'])
            result['matmul_calls'].append(dict(line=call.lineno,same_operand=same))
    result['status']='POSSIBLE_VIOLATION' if result['findings'] else 'UNKNOWN'
    result['operation_selection']='reduction_present' if result['reduction_calls'] else ('matmul_present' if result['matmul_calls'] else 'UNKNOWN')
    return result


def semantic_prompt(prompt,source,level,*,model='Qwen/Qwen3-8B',context=8192,answer_budget=2500):
    result=for_level(level)
    result['source_analysis']=semantic_gate(source,level) if source else None
    counter,method=local_token_counter(model)
    budget=max(0,min(180,context-answer_budget-128-counter(prompt)))
    findings=result['source_analysis']['findings'] if result['source_analysis'] else []
    text=''
    if findings:text=f"Semantic repair check at line {findings[0]['line']}: "+findings[0]['guidance']
    result.update(applied=bool(text) and counter(text)<=budget,counting_method=method)
    return prompt+('\n\n'+text if result['applied'] else ''),result


def semantic_gate(source,level):
    """Pre-simulation diagnostic stage; never changes checker rewards or tests.

    Only a side-effect-free identity function can be rejected by this limited
    semantic proof. Real kernels usually remain UNKNOWN/POSSIBLE_VIOLATION and
    still require the unchanged official checker. No API presence proves math.
    """
    import ast
    specification=for_level(level)
    result=analyze_semantics(source,specification['operation'])
    result['requirements']=[{k:p[k] for k in ('input_shapes','output_shape','requires_tiling','max_waste') if k in p} for p in specification['cases']]
    result['grading_policy']='Official checker remains authoritative; no reward or case changes.'
    try:tree=ast.parse(source)
    except (SyntaxError,RecursionError):return result
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef)]
    if len(functions)!=1:return result
    f=functions[0]
    if not any(isinstance(n,ast.Return) for n in ast.walk(f)):
        result['status']='PROVEN_INVALID'
        result['findings']=[dict(line=f.lineno,status='PROVEN_INVALID',kind='missing_returned_output',guidance='The entry function has no output return; normal completion yields None. Complete the required computation and HBM output writes, then return the output tensor. Comments describing a computation do not execute it. Repair the missing computation/output contract before layout optimization.')]
        return result
    body=[n for n in f.body if not isinstance(n,ast.Pass) and not (isinstance(n,ast.Expr) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str))]
    arguments=f.args.posonlyargs+f.args.args
    if len(body)!=1 or not isinstance(body[0],ast.Return) or not isinstance(body[0].value,ast.Name) or not arguments or body[0].value.id!=arguments[0].arg:return result
    nonidentity=any(p.get('output_shape') and tuple(p['output_shape'])!=tuple(p['input_shapes'][0]) for p in specification['cases'])
    if level==2:nonidentity=any(c['shape2D'][0]>1 and c['shape2D'][1]>1 for c in __import__('nkibench').LEVELS[2]['shapes'])
    if nonidentity:
        result['status']='PROVEN_INVALID'
        result['findings']=[dict(line=body[0].lineno,status='PROVEN_INVALID',kind='identity_incompatible_with_task',guidance='This side-effect-free function returns the unchanged input. It cannot implement the required output dimensions or nonidentity element permutation. Reconstruct the task computation before buffer repairs.')]
    return result
