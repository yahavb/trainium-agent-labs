"""Conservative AST evidence and bounded repair scopes; never executes source."""
import ast
import math
from failure_selection import classify_failure
from nki_knowledge import CATALOG, compatible, installed_compatibility, local_token_counter, referenced_operations


def inspect_shapes(source, input_shapes=None):
    """Literal arithmetic is evaluated; symbolic/unsupported expressions stay unknown.

    Loop values, branching assignments, views with advanced indices, and arbitrary
    function calls are not evaluated. Evidence is a static hypothesis, not proof.
    """
    try: tree=ast.parse(source)
    except SyntaxError as error:return dict(allocations=[],calls=[],issues=[],parse_error=str(error))
    env={};shapes=dict(input_shapes or {});allocations=[];issues=[]
    counts={}
    conditional_names=set()
    for parent in ast.walk(tree):
        if isinstance(parent,(ast.If,ast.While,ast.Try,ast.IfExp)):
            conditional_names.update(n.id for n in ast.walk(parent) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Store))
    if sum(isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) for n in ast.walk(tree)) != 1:
        conditional_names.update(n.id for n in ast.walk(tree) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Store))
    for node in ast.walk(tree):
        if isinstance(node,ast.Name) and isinstance(node.ctx,ast.Store):counts[node.id]=counts.get(node.id,0)+1
    # Rebinding an input invalidates its externally supplied initial shape.
    for name in counts:shapes.pop(name,None) if name in (input_shapes or {}) else None
    def value(node):
        if isinstance(node,ast.Constant) and type(node.value) is int:return node.value
        if isinstance(node,ast.Name):return env.get(node.id)
        if isinstance(node,(ast.Tuple,ast.List)):
            values=tuple(value(x) for x in node.elts)
            return values if all(x is not None for x in values) else None
        if isinstance(node,ast.Attribute) and node.attr=='shape':return shapes.get(ast.unparse(node.value))
        if isinstance(node,ast.Subscript):
            base=value(node.value)
            if isinstance(base,tuple):
                if isinstance(node.slice,ast.Slice):
                    def bound(x):return None if x is None else value(x)
                    if any(x is not None and value(x) is None for x in (node.slice.lower,node.slice.upper,node.slice.step)):return None
                    try:return base[slice(bound(node.slice.lower),bound(node.slice.upper),bound(node.slice.step))]
                    except (ValueError,TypeError):return None
                index=value(node.slice)
                if type(index) is int:
                    try:return base[index]
                    except IndexError:return None
        if isinstance(node,ast.BinOp):
            a,b=value(node.left),value(node.right)
            if type(a) is int and type(b) is int:
                try:
                    if isinstance(node.op,ast.Add):return a+b
                    if isinstance(node.op,ast.Sub):return a-b
                    if isinstance(node.op,ast.Mult):return a*b
                    if isinstance(node.op,ast.FloorDiv):return a//b
                except ZeroDivisionError:return None
        return None
    for node in sorted(ast.walk(tree),key=lambda x:getattr(x,'lineno',0)):
        if not isinstance(node,ast.Assign) or len(node.targets)!=1:continue
        target=node.targets[0]
        if isinstance(target,(ast.Tuple,ast.List)):
            result=value(node.value)
            if isinstance(result,tuple) and len(result)==len(target.elts):
                for name,item in zip(target.elts,result):
                    if isinstance(name,ast.Name) and counts[name.id]==1 and name.id not in conditional_names:env[name.id]=item
            continue
        if not isinstance(target,ast.Name) or counts[target.id]!=1 or target.id in conditional_names:continue
        result=value(node.value)
        if result is not None:env[target.id]=result
        call=node.value
        if isinstance(call,ast.Call) and ast.unparse(call.func).split('.')[-1]=='ndarray':
            keywords={k.arg:k.value for k in call.keywords};expr=keywords.get('shape',call.args[0] if call.args else None)
            shape=value(expr) if expr is not None else None
            # Only tuple shapes are inferred; scalar/unknown allocation shapes stay unknown.
            shape=shape if isinstance(shape,tuple) and all(type(n) is int and n>=0 for n in shape) else None
            buffer=ast.unparse(keywords['buffer']) if 'buffer' in keywords else 'default SBUF (verify API)'
            record=dict(name=target.id,line=node.lineno,shape_expression=ast.unparse(expr) if expr else 'unavailable',derived_shape=shape,elements=math.prod(shape) if shape else None,buffer=buffer)
            allocations.append(record)
            if shape:shapes[target.id]=shape
            if shape is not None and len(shape)<2 and ('sbuf' in buffer or 'psum' in buffer or buffer.startswith('default')):
                issues.append(f'line {node.lineno}: {target.id} has rank {len(shape)} in {buffer}; on-chip tensors require at least 2 dimensions.')
    _,calls=referenced_operations(source)
    def tensor_shape(node):
        if isinstance(node,ast.Name):return shapes.get(node.id)
        if isinstance(node,ast.Subscript):
            base=tensor_shape(node.value)
            if base is None:return None
            indices=node.slice.elts if isinstance(node.slice,ast.Tuple) else [node.slice]
            if len(indices)>len(base):return None
            result=[]
            for dim,index in zip(base,indices):
                if isinstance(index,ast.Slice):
                    bounds=[value(x) if x is not None else None for x in (index.lower,index.upper,index.step)]
                    if any(x is not None and value(x) is None for x in (index.lower,index.upper,index.step)):return None
                    try:result.append(len(range(*slice(*bounds).indices(dim))))
                    except (ValueError,TypeError):return None
                elif type(value(index)) is int:
                    if not -dim<=value(index)<dim:return None
                else:return None
            return tuple(result)+tuple(base[len(indices):])
        return None
    transfers=[]
    for node in ast.walk(tree):
        if not isinstance(node,ast.Call) or ast.unparse(node.func).split('.')[-1] not in ('dma_copy','tensor_copy','nc_matmul'):continue
        args={k.arg:k.value for k in node.keywords if k.arg}
        names=('dst','stationary','moving') if ast.unparse(node.func).endswith('nc_matmul') else ('dst','src')
        for name,arg in zip(names,node.args):args.setdefault(name,arg)
        record=dict(line=node.lineno,operation=ast.unparse(node.func),operands={name:dict(expression=ast.unparse(args[name]),derived_shape=tensor_shape(args[name])) for name in names if name in args})
        transfers.append(record)
        if 'dma_copy' in record['operation'] and set(record['operands'])=={'src','dst'}:
            src,dst=(record['operands'][name]['derived_shape'] for name in ('src','dst'))
            if src is not None and dst is not None and math.prod(src)!=math.prod(dst):issues.append(f'line {node.lineno}: DMA src shape {src} has {math.prod(src)} elements; dst shape {dst} has {math.prod(dst)}. Reconcile allocation, slice, and consumers together.')
    return dict(allocations=allocations,calls=list(calls),transfers=transfers,issues=issues,parse_error=None)


def failure_input_shapes(source, feedback, level):
    """Bind only an actually reported checker case to the entry's real argument order."""
    import nkibench
    try:
        spec=nkibench.LEVELS[level]
        tree=ast.parse(source)
        function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==spec['entry'])
        case=next(c for c in spec['shapes'] if 'On '+nkibench.label(c,level)+':' in feedback)
        args,_=nkibench.make_inputs(case,level)
        names=[n.arg for n in function.args.posonlyargs+function.args.args]
        if len(names)!=len(args):return {}
        return {name:tuple(arg.shape) for name,arg in zip(names,args) if hasattr(arg,'shape')}
    except (KeyError,SyntaxError,StopIteration):return {}


def input_lifetime_evidence(source,evidence):
    """Straight-line, whole-region DMA overwrite evidence; no alias speculation."""
    try:tree=ast.parse(source)
    except SyntaxError:return []
    if any(isinstance(n,(ast.For,ast.While,ast.If,ast.Try)) for n in ast.walk(tree)):return []
    def base(expression):
        try:node=ast.parse(expression,mode='eval').body
        except SyntaxError:return None
        while isinstance(node,ast.Subscript):node=node.value
        return node.id if isinstance(node,ast.Name) else None
    findings=[]
    transfers=evidence.get('transfers',[])
    for compute in transfers:
        if not compute['operation'].endswith('nc_matmul'):continue
        operands=compute['operands']
        stationary=operands.get('stationary',{}).get('expression','')
        moving=operands.get('moving',{}).get('expression','')
        shared=base(stationary)
        if not shared or shared!=base(moving):continue
        loads=[t for t in transfers if t['line']<compute['line'] and t['operation'].endswith('dma_copy') and base(t['operands'].get('dst',{}).get('expression',''))==shared]
        loads.sort(key=lambda t:t['line'])
        if len(loads)<2:continue
        first,last=loads[-2:]
        a,b=(t['operands'].get('src',{}).get('expression','') for t in (first,last))
        # A whole-region load followed by another nonempty load proves reuse;
        # whether both original values are required is a mathematical question.
        first_dst=first['operands']['dst']['expression'];last_shape=last['operands']['dst'].get('derived_shape')
        if first_dst!=shared or not last_shape or math.prod(last_shape)==0 or base(a)==base(b):continue
        findings.append(f"DMA lines {first['line']} and {last['line']} load {a} then {b} into shared allocation {shared}; nc_matmul line {compute['line']} uses that allocation for both stationary and moving. Once both loads are legal, the later load would overwrite the earlier input region. If the computation requires both original inputs, keep them in distinct live SBUF regions; reconcile each input's own load shape and the result-shaped PSUM/SBUF/output consumers together. Do not alternate resizing this single input buffer to satisfy one load at a time.")
    return findings


def root_cause(source,feedback,evidence):
    """Specific evidence first; no inferred runtime values beyond checker case inputs."""
    result=[];category=classify_failure(feedback).failure_category
    compatibility=installed_compatibility()
    verified={card.id for card in CATALOG if compatible(card,compatibility)}
    try:tree=ast.parse(source)
    except SyntaxError:return result
    lifetime=input_lifetime_evidence(source,evidence)
    lifetime_note=(' '+lifetime[0]) if lifetime else ''
    # Diagnose the reported transfer before later matmul consumers.
    if category=='DMA_SHAPE_MISMATCH':
        import re
        match=re.search(r'src=(\d+), dst=(\d+)',feedback)
        transfers=[t for t in evidence.get('transfers',[]) if t['operation'].endswith('dma_copy')]
        for transfer in transfers:
            operands=transfer['operands'];src=operands.get('src',{});dst=operands.get('dst',{})
            a,b=src.get('derived_shape'),dst.get('derived_shape')
            if a is not None and b is not None and math.prod(a)!=math.prod(b):
                if match and (math.prod(a),math.prod(b))!=tuple(map(int,match.groups())):continue
                return [f"line {transfer['line']}: dma_copy src {src['expression']} shape {a} ({math.prod(a)} elements) and dst {dst['expression']} shape {b} ({math.prod(b)} elements) disagree. Coordinate source slice, allocation and downstream consumers; do not just resize dst. Resolve this actual transfer before reasoning about later matmul operations."+lifetime_note]
        if transfers:
            return ['The checker reports a DMA element-count mismatch. Transfer call lines '+', '.join(str(t['line']) for t in transfers)+': operand shapes are unresolved or cannot be uniquely matched to this runtime error. Trace the failing source slice and destination allocation together, then check consumers and cover every element; do not infer a later matmul failure from this DMA error.'+lifetime_note]
    if category=='OUT_OF_BOUNDS':
        import re
        bounds=re.search(r'dimension (\d+): index range \[(\d+), (\d+)\] exceed dimension size of (\d+)',feedback)
        if bounds:
            dim,lo,hi,size=bounds.groups()
            lines=[str(t['line']) for t in evidence.get('transfers',[]) if t['operation'].endswith('dma_copy')]
            return [f'Runtime dimension {dim} has size {size}, but the reported slice reaches indices {lo} through {hi}. DMA call sites: '+(', '.join(lines) or 'unresolved')+'. Clip the final tile boundary to the actual dimension, derive its allocation extent from the clipped end minus start, and update both source/destination slices and consumers. Preserve accumulation and complete output coverage; do not pad the final partial tile past input bounds.']
    if category=='INVALID_API_ARGUMENT':
        import re
        import inspect
        import nki.isa as isa
        match=re.search(r"(\w+)\(\) got an unexpected keyword argument ['\"]([^'\"]+)",feedback)
        if match:
            api,keyword=match.groups();obj=getattr(isa,api,None)
            if obj is not None:
                for node in ast.walk(tree):
                    if isinstance(node,ast.Call) and ast.unparse(node.func).split('.')[-1]==api:
                        return [f'line {node.lineno}: {api} rejects keyword {keyword}. Remove this unsupported keyword and bind arguments using the installed signature {inspect.signature(obj)}. Preserve operand semantics, shapes and buffer placement.']
    if category=='INVALID_API_FUNCTION':
        import re
        missing=re.search(r"has no attribute ['\"]([^'\"]+)",feedback)
        if missing:
            name=missing.group(1)
            for node in ast.walk(tree):
                if isinstance(node,ast.Call) and ast.unparse(node.func).split('.')[-1]==name:
                    return [f'line {node.lineno}: unsupported API {ast.unparse(node.func)} is the reported failure. Verify a replacement against the installed signature; preserve operand shapes and buffers. nl.load/nl.store exist in this SDK and must not be rejected merely by name.']
    if category=='INVALID_TENSOR_DIMENSIONS' and 'at least 2 dimensions' in feedback:
        explicit=[issue for issue in evidence['issues'] if 'rank 1' in issue]
        if explicit:return [explicit[0]+' Correct this explicit allocation before blaming later reductions or matmul; keep its transfer and consumers consistent.']
    if category=='INVALID_BUFFER_PLACEMENT' and 'nc_transpose' in feedback:
        for node in ast.walk(tree):
            if not isinstance(node,ast.Call) or not ast.unparse(node.func).endswith('nc_transpose'):continue
            values={k.arg:k.value for k in node.keywords}
            dst=values.get('dst',node.args[0] if node.args else None)
            if dst is None:continue
            return [f'line {node.lineno}: nc_transpose destination {ast.unparse(dst)} must be on-chip. Vector engine reads/writes SBUF or PSUM with input tile <=32x32; tensor engine reads SBUF and writes PSUM with input tile <=128x128. This primitive swaps partition/free axes, not two free axes within each partition. Use matching on-chip shapes and legal final HBM transfers.']
    if category=='INVALID_BUFFER_PLACEMENT' and ('private_hbm' in feedback or 'shared_hbm' in feedback) and 'copy' in verified:
        parameters={argument.arg for function in tree.body if isinstance(function,ast.FunctionDef) for argument in function.args.args}
        for node in ast.walk(tree):
            if not isinstance(node,ast.Call) or not ast.unparse(node.func).endswith(('tensor_copy','.copy')):continue
            kwargs={k.arg:k.value for k in node.keywords}
            src=kwargs.get('src',node.args[1] if ast.unparse(node.func).endswith('tensor_copy') and len(node.args)>1 else node.args[0] if node.args else None)
            if isinstance(src,ast.Name) and src.id in parameters:
                result.append(f'line {node.lineno}: {ast.unparse(node.func)} reads input {src.id}, which the checker identifies as HBM. tensor_copy is on-chip SBUF/PSUM only: use dma_copy(dst=on_chip_tile, src=matching_input_slice) for this load. Apply the same correction to other HBM loads; keep result-shaped PSUM/SBUF tiles and shared_hbm output stores consistent.')
                break
    # These return-style reduction signatures/source were inspected on the installed SDK.
    for call in evidence['calls']:
        if call['api'] in ('nki.language.sum','nki.language.max') and category=='INVALID_TENSOR_DIMENSIONS' and 'at least 2 dimensions' in feedback and 'rank' in verified:
            node=next((n for n in ast.walk(tree) if isinstance(n,ast.Call) and n.lineno==call['line'] and ast.unparse(n).startswith(call['expression'].split('(')[0])),None)
            keep=next((k.value for k in node.keywords if k.arg=='keepdims'),None) if node else None
            if isinstance(keep,ast.Constant) and keep.value is True:continue
            result.append(f"line {call['line']}: {call['api']} may create the reported 1D on-chip result when every free axis is reduced. Inspect reduction axes and consumer shape; preserve a size-one free axis with keepdims=True only if that layout matches the consumer, or use a valid 2D reduction destination. Explicit allocations are not the only cause.")
    for transfer in evidence.get('transfers',[]):
        if not transfer['operation'].endswith('nc_matmul') or 'matmul' not in verified:continue
        op=transfer['operands'];shapes={name:data['derived_shape'] for name,data in op.items()}
        left,right,dst=(shapes.get(name) for name in ('stationary','moving','dst'))
        # Restrict dimensions to ordinary 2D matmul. Specialized modes remain unresolved.
        node=next((n for n in ast.walk(tree) if isinstance(n,ast.Call) and n.lineno==transfer['line'] and ast.unparse(n.func).endswith('nc_matmul')),None)
        if node and any(k.arg in ('is_transpose','perf_mode','tile_size','tile_position') and not (isinstance(k.value,ast.Constant) and k.value.value is False) for k in node.keywords):continue
        if left and right and len(left)==len(right)==2:
            K,M=left;Kr,N=right
            if K!=Kr:result.append(f"line {transfer['line']}: contraction extents differ: stationary {left}, moving {right}; align both K slices.")
            if dst is not None and tuple(dst)!=(M,N):
                explicit=[c['line'] for c in evidence['calls'] if c['api'].endswith('.reshape')]
                # Tensor methods are not NKI module calls; inspect actual AST separately.
                explicit=[n.lineno for n in ast.walk(tree) if isinstance(n,ast.Call) and ast.unparse(n.func).endswith('.reshape')]
                origin='explicit reshape calls also exist; inspect them separately' if explicit else 'no explicit reshape call exists; the simulator reshapes its matmul result to dst internally'
                result.append(f"line {transfer['line']}: nc_matmul stationary {left} and moving {right} produce ({M},{N}), but dst {op['dst']['expression']} is {dst}. {origin}. Fix PSUM destination and result-copy/store consumers, not an imaginary user reshape."+lifetime_note)
            excessive=[f'{name}={size} exceeds {limit}' for name,size,limit in [('K',K,128),('M',M,128),('N',N,512)] if size>limit]
            if excessive:
                result.append(f"line {transfer['line']}: normal FP32 Trainium2 matmul limits: "+'; '.join(excessive)+'. Tile all oversized M/N output dimensions and K contraction before DMA. Allocate one result-shaped FP32 PSUM tile per output region, overwrite on its first K tile, accumulate subsequent disjoint K tiles, then copy to a matching SBUF result tile and the matching HBM output slice. Cover every output region.')
    return result


def plan_repair(source, feedback, repeated=0, input_shapes=None):
    diagnostic=classify_failure(feedback);evidence=inspect_shapes(source,input_shapes)
    category=diagnostic.failure_category
    scope='localized_api_correction';guidance='Correct the named API or keyword using the installed signature; preserve unrelated operations.'
    if category in ('DMA_SHAPE_MISMATCH','INVALID_TENSOR_DIMENSIONS','OUT_OF_BOUNDS','INCOMPLETE_OUTPUT'):
        scope='coordinated_dataflow_repair' if evidence.get('transfers') else 'shape_allocation_correction'
        guidance=('Trace the offending allocation, source slice, transfer, compute consumers and output store together. '
                  'Match transferred element counts and legal partition/free extents; adjust dependent lines as a unit. '
                  'Hardware limits are maxima, not required sizes. Tile oversized dimensions before loading them, '
                  'not after a whole-tensor DMA. Preserve the full returned output shape and cover every element.')
    elif category=='INVALID_BUFFER_PLACEMENT':
        scope='buffer_placement_correction';guidance='Check each operation\'s required input/output region. Correct allocation plus dependent transfers; PSUM results need a separate result-shaped SBUF tile before DMA. Do not reuse an input tile with a different result shape.'
    elif category in ('NUMERICAL_MISMATCH','HARDWARE_CORRECTNESS_HAZARD'):
        scope='algorithm_redesign' if repeated>=3 else 'coordinated_dataflow_repair'
        guidance='Check mathematical operation, contraction coverage and accumulator lifetime; avoid overlapping contraction windows, overwrites of completed output, or missing output tiles. First matmul overwrites each PSUM tile; subsequent contraction tiles accumulate. Redesign only the failing dataflow if repeated repairs have not converged.'
    # Only cite real source lines. Unknown expressions are explicitly unresolved.
    excerpts=[]
    for record in evidence['allocations'][:5]:
        shape=str(record['derived_shape']) if record['derived_shape'] is not None else 'unresolved statically'
        excerpts.append(f"line {record['line']}: {record['name']} allocation {record['shape_expression']} -> {shape}; {record['buffer']}")
    for record in evidence.get('transfers',[])[:4]:
        excerpts.append(f"line {record['line']}: {record['operation']} "+'; '.join(name+'='+operand['expression'] for name,operand in record['operands'].items()))
    relevant_names={r['name'] for r in evidence['allocations']}
    dependencies=[c['line'] for c in evidence['calls'] if any(name in c['expression'] for name in relevant_names)]
    causes=root_cause(source,feedback,evidence)
    text=f'Repair scope: {scope}. '+('\n'.join(causes[:1]) if causes else guidance)+'\n'+'\n'.join(evidence['issues'][:1]+excerpts[:4])
    return dict(scope=scope,failure_category=category,original_feedback=feedback,evidence=evidence,dependent_call_lines=sorted(set(dependencies)),guidance=text,root_causes=causes, source_urls=[c.source_url for c in CATALOG if c.id in ('matmul','rank','dma')], verification='static evidence only; unknown expressions are not evaluated')


def constrained_prompt(prompt, operation, *, model='Qwen/Qwen3-8B', context=8192, answer_budget=2500):
    from agent import API_CARD
    compatibility=installed_compatibility();ids=['allocation','dma','tiling']
    if 'matmul' in operation:ids+=['matmul','copy','accumulation']
    else:ids+=['scalar','reduce']
    cards=[card for id in ids for card in CATALOG if card.id==id and compatible(card,compatibility)]
    if not cards:return prompt,dict(applied=False,card_ids=[],reason='SDK compatibility unavailable')
    # Replace the old whole-tensor copy example; keep reference math and entry point.
    base=prompt.replace(API_CARD,'')
    lines=['Verified NKI/Trainium2 constraints (general rules, not a kernel):',
           'Return an explicitly shaped shared_hbm output. Write every output element. Keep on-chip tiles at least 2D.',
           'Derive tile bounds from input shapes; never pad a small dimension to a hardware maximum.',
           'Allocation, source slice, compute operands, result tile and destination slice must agree.']
    if 'matmul' in operation:
        lines.append('Use dma_copy for input HBM->SBUF loads and output SBUF->HBM stores; tensor_copy only for on-chip transfers. Initialize each output PSUM tile with the first matmul overwrite, then accumulate later disjoint K tiles.')
    for card in cards:
        signature={'allocation':'ndarray(shape, dtype=, buffer=)', 'dma':'dma_copy(dst=, src=)', 'copy':'tensor_copy(dst=, src=)'}.get(card.id,card.signature or card.allowed)
        if card.id=='matmul':signature='nc_matmul(dst=, stationary=, moving=, accumulate=); ordinary matmul needs no perf_mode, tile_size, tile_position or is_transpose options'
        lines.append(f'{card.concept}: {signature}; {card.allowed}; {card.buffers}.')
    counter,method=local_token_counter(model);budget=max(0,min(450,context-answer_budget-128-counter(base)))
    chosen=[]
    for line in lines:
        if counter('\n'.join(chosen+[line]))<=budget:chosen.append(line)
    text='\n'.join(chosen)
    used=[card for card in cards if any(line.startswith(card.concept+':') for line in chosen)]
    return (base+'\n\n'+text if text else prompt),dict(applied=bool(text),card_ids=[c.id for c in used],source_urls=[c.source_url for c in used],context_token_count=counter(text),counting_method=method)


def shape_prompt(prompt, source, feedback, repeated=0, *, input_shapes=None, model='Qwen/Qwen3-8B', context=8192, answer_budget=2500):
    plan=plan_repair(source,feedback,repeated,input_shapes)
    base=prompt.replace('Change exactly what the checker names and keep everything else identical.',
                        'Apply the simplest scope described below, including dependent lines when necessary; preserve unrelated working code.')
    counter,method=local_token_counter(model);budget=max(0,min(450,context-answer_budget-128-counter(base)))
    lines=[]
    for line in plan['guidance'].splitlines():
        if counter('\n'.join(lines+[line]))<=budget:lines.append(line)
    plan['applied']=bool(lines);plan['context_token_count']=counter('\n'.join(lines));plan['counting_method']=method
    return base+('\n\nShape-aware repair plan:\n'+'\n'.join(lines) if lines else ''),plan
