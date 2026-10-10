"""Bounded host-only SymPy analysis. Unknown is not evidence of correctness."""
import ast
import math
import sympy as sp

EQUAL='PROVEN_EQUAL';MISMATCH='PROVEN_MISMATCH';UNKNOWN='UNKNOWN'
MAX_SOURCE=50000;MAX_NODES=2000;MAX_EXPR_NODES=80;MAX_DEPTH=24


def bounded(expr):
    return expr is not None and isinstance(expr,sp.Basic) and sp.count_ops(expr)<=MAX_EXPR_NODES


def equality(a,b):
    if not bounded(a) or not bounded(b):return UNKNOWN
    difference=sp.simplify(a-b)
    if not bounded(difference):return UNKNOWN
    if difference.is_zero is True:return EQUAL
    if difference.is_zero is False:return MISMATCH
    return UNKNOWN


def within_limit(value,limit):
    if not bounded(value):return UNKNOWN
    limit_value=sp.Integer(limit) if type(limit) is int else limit
    if not bounded(limit_value):return UNKNOWN
    difference=sp.simplify(value-limit_value)
    if difference.is_positive is True:return 'PROVEN_VIOLATION'
    if difference.is_nonpositive is True:return 'NO_STATIC_VIOLATION'
    return UNKNOWN


def tiling(dimension,tile,loop_count=None):
    if not bounded(dimension) or not bounded(tile) or tile.is_positive is not True:return dict(status=UNKNOWN)
    count=sp.ceiling(dimension/tile)
    tail=sp.simplify(dimension-(count-1)*tile)
    return dict(status='NO_STATIC_VIOLATION',tile_count=str(count),final_extent=str(tail),accumulation_required=bool(count>1) if (count>1) in (sp.true,sp.false) else UNKNOWN,
                output_coverage=equality(count,loop_count) if loop_count is not None else UNKNOWN,
                note='Count agreement alone does not establish correct slices, writes or accumulation.')


def broadcasting(left,right):
    if left is None or right is None:return UNKNOWN
    statuses=[]
    for a,b in zip(reversed(left),reversed(right)):
        if equality(a,b)==EQUAL or equality(a,sp.Integer(1))==EQUAL or equality(b,sp.Integer(1))==EQUAL:statuses.append(EQUAL)
        elif equality(a,b)==MISMATCH and equality(a,sp.Integer(1))==MISMATCH and equality(b,sp.Integer(1))==MISMATCH:return MISMATCH
        else:statuses.append(UNKNOWN)
    return EQUAL if all(s==EQUAL for s in statuses) else UNKNOWN


def analyze(source,input_shapes=None):
    result=dict(status=UNKNOWN,allocations=[],operations=[],violations=[],unknowns=[],loops=[],verification='Symbolic shape evidence only; simulator remains authoritative.')
    if len(source)>MAX_SOURCE:result['unknowns'].append('Source complexity budget exceeded');return result
    try:tree=ast.parse(source)
    except (SyntaxError,RecursionError,MemoryError):result['unknowns'].append('Source cannot be safely parsed');return result
    if sum(1 for _ in ast.walk(tree))>MAX_NODES:result['unknowns'].append('AST complexity budget exceeded');return result
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef)]
    if len(functions)!=1:result['unknowns'].append('Multiple or missing entry functions');return result
    env={};shapes={};buffers={};loop_values={};unknown=False
    for name,shape in (input_shapes or {}).items():
        converted=[]
        for dim in shape:
            if type(dim) is int and abs(dim)<2**31:converted.append(sp.Integer(dim))
            elif bounded(dim):converted.append(dim)
            else:converted=[];break
        if converted:shapes[name]=tuple(converted);buffers[name]='input_hbm'
    def record_violation(line,kind,message):
        value=dict(line=line,kind=kind,status='PROVEN_VIOLATION',message=message)
        if value not in result['violations']:result['violations'].append(value)
    def expr(node,depth=0):
        if node is None or depth>MAX_DEPTH:return None
        if sum(1 for _ in ast.walk(node))>MAX_EXPR_NODES:return None
        value=None
        if isinstance(node,ast.Constant) and type(node.value) is int and abs(node.value)<2**31:value=sp.Integer(node.value)
        elif isinstance(node,ast.Name):return env.get(node.id)
        elif isinstance(node,(ast.Tuple,ast.List)):
            values=tuple(expr(n,depth+1) for n in node.elts)
            return values if all(n is not None for n in values) else None
        elif isinstance(node,ast.Attribute) and node.attr=='shape':return tensor(node.value)
        elif isinstance(node,ast.Subscript):
            base=expr(node.value,depth+1);index=expr(node.slice,depth+1)
            if isinstance(base,tuple) and isinstance(index,sp.Integer):
                try:return base[int(index)]
                except IndexError:return None
        elif isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.USub,ast.UAdd)):
            a=expr(node.operand,depth+1)
            if bounded(a):value=-a if isinstance(node.op,ast.USub) else a
        elif isinstance(node,ast.BinOp):
            a,b=expr(node.left,depth+1),expr(node.right,depth+1)
            if bounded(a) and bounded(b):
                if isinstance(node.op,ast.Add):value=a+b
                elif isinstance(node.op,ast.Sub):value=a-b
                elif isinstance(node.op,ast.Mult):value=a*b
                elif isinstance(node.op,(ast.FloorDiv,ast.Div)) and b.is_zero is False:value=sp.floor(a/b) if isinstance(node.op,ast.FloorDiv) else a/b
        elif isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id in ('min','max') and node.func.id not in env and 1<len(node.args)<=4:
            args=[expr(n,depth+1) for n in node.args]
            if all(bounded(a) for a in args):value=(sp.Min if node.func.id=='min' else sp.Max)(*args)
        elif isinstance(node,ast.Call) and ast.unparse(node.func)=='math.ceil' and len(node.args)==1 and not node.keywords:
            arg=expr(node.args[0],depth+1)
            if bounded(arg):value=sp.ceiling(arg)
        # No eval, source-string sympify, parse_expr, arbitrary call or exponentiation.
        return value if bounded(value) else None
    def bound_proof(value,dimension):
        """Prove all known loop instantiations; otherwise use symbolic signs."""
        import itertools
        symbols=[symbol for symbol in loop_values if symbol in value.free_symbols or symbol in dimension.free_symbols]
        combinations=math.prod(len(loop_values[s]) for s in symbols)
        if symbols and combinations<=64:
            results=[]
            for choices in itertools.product(*(loop_values[s] for s in symbols)):
                substitutions=dict(zip(symbols,choices));difference=sp.simplify((value-dimension).subs(substitutions))
                if difference.is_positive is True:return 'PROVEN_VIOLATION'
                results.append(difference.is_nonpositive is True)
            return 'NO_STATIC_VIOLATION' if all(results) else UNKNOWN
        return within_limit(value,dimension)
    def tensor(node):
        if isinstance(node,ast.Name):return shapes.get(node.id)
        if isinstance(node,ast.Subscript):
            original=tensor(node.value)
            if original is None:return None
            indices=node.slice.elts if isinstance(node.slice,ast.Tuple) else [node.slice]
            if len(indices)>len(original):return None
            extent=[]
            for dim,index in zip(original,indices):
                if isinstance(index,ast.Slice):
                    step=sp.Integer(1) if index.step is None else expr(index.step)
                    if equality(step,sp.Integer(1))!=EQUAL:return None
                    start=sp.Integer(0) if index.lower is None else expr(index.lower)
                    stop=dim if index.upper is None else expr(index.upper)
                    if not bounded(start) or not bounded(stop):return None
                    if start.is_negative or stop.is_negative:return None
                    # NKI checks bounds; unlike NumPy it does not silently clip.
                    lower_status=bound_proof(start,dim);upper_status=bound_proof(stop,dim)
                    nonnegative_start=bound_proof(-start,sp.Integer(0));nonnegative_stop=bound_proof(-stop,sp.Integer(0))
                    if 'PROVEN_VIOLATION' in (nonnegative_start,nonnegative_stop):return None
                    if UNKNOWN in (nonnegative_start,nonnegative_stop) or bound_proof(start,stop)!='NO_STATIC_VIOLATION':return None
                    if 'PROVEN_VIOLATION' in (lower_status,upper_status):
                        record_violation(node.lineno,'slice_bounds',f'Slice start {start}, stop {stop} exceeds dimension {dim} in at least one established loop instance. Clip the boundary explicitly and coordinate the allocation and consumers.')
                        return None
                    if UNKNOWN in (lower_status,upper_status):return None
                    extent.append(stop-start)
                else:
                    index_value=expr(index)
                    if not isinstance(index_value,sp.Integer) or not isinstance(dim,sp.Integer) or not -dim<=index_value<dim:return None
            return tuple(extent)+tuple(original[len(indices):])
        return None
    def kw(call,names):
        values={n:arg for n,arg in zip(names,call.args)}
        values.update({k.arg:k.value for k in call.keywords if k.arg})
        return values
    def product(shape):return sp.prod(shape) if shape is not None else None
    def operation(call):
        nonlocal unknown
        api=ast.unparse(call.func).split('.')[-1];line=call.lineno
        if api=='dma_copy':
            values=kw(call,('dst','src'));a=tensor(values.get('src'));b=tensor(values.get('dst'))
            verdict=equality(product(a),product(b))
            result['operations'].append(dict(line=line,api=api,src_shape=list(map(str,a)) if a is not None else None,dst_shape=list(map(str,b)) if b is not None else None,src_elements=str(product(a)) if a is not None else None,dst_elements=str(product(b)) if b is not None else None,status=verdict))
            if verdict==MISMATCH:record_violation(line,'dma_elements',f'DMA source has {product(a)} elements, destination {product(b)}. Match actual slices and allocations; coordinate consumers. Equal counts alone do not establish legal layout.')
            unknown|=verdict==UNKNOWN
        elif api=='nc_matmul':
            values=kw(call,('dst','stationary','moving'))
            if any(k in values and not (isinstance(values[k],ast.Constant) and values[k].value is False) for k in ('is_transpose','perf_mode','tile_size','tile_position')):
                result['unknowns'].append(f'line {line}: specialized matmul mode');unknown=True;return
            a,b,d=(tensor(values.get(k)) for k in ('stationary','moving','dst'))
            if a is None or b is None or d is None or len(a)!=2 or len(b)!=2 or len(d)!=2:unknown=True;result['operations'].append(dict(line=line,api=api,status=UNKNOWN));return
            statuses=[equality(a[0],b[0]),equality(d[0],a[1]),equality(d[1],b[1])]
            result['operations'].append(dict(line=line,api=api,stationary=list(map(str,a)),moving=list(map(str,b)),dst=list(map(str,d)),axis_equalities=statuses,status=MISMATCH if MISMATCH in statuses else UNKNOWN if UNKNOWN in statuses else EQUAL))
            if MISMATCH in statuses:record_violation(line,'matmul_dimensions',f'nc_matmul stationary {a}, moving {b} require dst ({a[1]}, {b[1]}); actual dst {d}. Check K equality and each result axis, not just element count. Coordinate result copies and stores.')
            if within_limit(d[1],512)=='PROVEN_VIOLATION':record_violation(line,'psum_free_limit',f'Ordinary Trainium2 nc_matmul PSUM dst free extent {d[1]} exceeds the installed 512-element per-instruction limit. Tile the output extent and its consumers.')
            for name,dimension,limit in [('K',a[0],128),('M',a[1],128),('N',b[1],512)]:
                verdict=within_limit(dimension,limit)
                unknown|=verdict==UNKNOWN
                if verdict=='PROVEN_VIOLATION':record_violation(line,'matmul_tile_limit',f'Ordinary Trainium2 matmul {name}={dimension} exceeds {limit}; tile this extent before loading and preserve disjoint K accumulation and complete M/N writes.')
            for name,required in [('dst','psum'),('stationary','sbuf'),('moving','sbuf')]:
                operand=values.get(name);buffer=buffers.get(ast.unparse(operand)) if isinstance(operand,ast.Name) else None
                if buffer is not None and not buffer.endswith(required):record_violation(line,'matmul_buffer',f'nc_matmul {name} requires {required}; actual {buffer}. Correct placement and dependent transfers.')
            unknown|=UNKNOWN in statuses
        elif api=='nc_transpose':
            values=kw(call,('dst','data','engine'));a=tensor(values.get('data'));d=tensor(values.get('dst'))
            if a is None or d is None or len(a)!=2 or len(d)!=2:unknown=True;return
            statuses=[equality(d[0],a[1]),equality(d[1],a[0])]
            result['operations'].append(dict(line=line,api=api,data_shape=list(map(str,a)),dst_shape=list(map(str,d)),status=MISMATCH if MISMATCH in statuses else UNKNOWN if UNKNOWN in statuses else EQUAL))
            if MISMATCH in statuses:record_violation(line,'transpose_dimensions',f'nc_transpose swaps partition/free axes: input {a} requires dst ({a[1]}, {a[0]}), not {d}. Do not confuse this with preserving a partition while permuting two free axes.')
            engine=ast.unparse(values['engine']) if 'engine' in values else 'unknown'
            dst_buffer=buffers.get(ast.unparse(values['dst']))
            if engine.endswith('.vector'):
                for dimension in a:
                    if within_limit(dimension,32)=='PROVEN_VIOLATION':record_violation(line,'transpose_tile_limit',f'Vector transpose tile extent {dimension} exceeds 32.')
            for name in ('data','dst'):
                buffer=buffers.get(ast.unparse(values[name]))
                if buffer is not None and not buffer.endswith(('sbuf','psum')):record_violation(line,'transpose_buffer',f'nc_transpose {name} must be on-chip, not {buffer}. Use legal SBUF/PSUM then matching HBM DMA.')
            unknown|=UNKNOWN in statuses
        elif api=='tensor_tensor':
            values=kw(call,('dst','data1','data2','op'));status=broadcasting(tensor(values.get('data1')),tensor(values.get('data2')))
            result['operations'].append(dict(line=line,api=api,broadcast_status=status,nki_legality='UNKNOWN: generic broadcast is necessary but not sufficient'))
            if status==MISMATCH:record_violation(line,'broadcast', 'Elementwise operand axes cannot broadcast. Reconcile operands and result layout.')
    def assignment(node):
        nonlocal unknown
        if len(node.targets)!=1:return
        target=node.targets[0];value=node.value
        if isinstance(target,(ast.Tuple,ast.List)):
            v=expr(value)
            for i,name in enumerate(target.elts):
                if isinstance(name,ast.Name):env[name.id]=v[i] if isinstance(v,tuple) and len(v)==len(target.elts) else None
            return
        if not isinstance(target,ast.Name):return
        env[target.id]=expr(value);shapes.pop(target.id,None);buffers.pop(target.id,None)
        if isinstance(value,(ast.Name,ast.Subscript)):
            inferred=tensor(value)
            if inferred is not None:
                shapes[target.id]=inferred
                base=value
                while isinstance(base,ast.Subscript):base=base.value
                if isinstance(base,ast.Name) and base.id in buffers:buffers[target.id]=buffers[base.id]
            return
        if not isinstance(value,ast.Call):return
        api=ast.unparse(value.func).split('.')[-1]
        if api=='ndarray':
            values=kw(value,('shape','dtype','buffer'));shape=expr(values.get('shape'));buffer=ast.unparse(values['buffer']) if 'buffer' in values else 'nl.sbuf'
            if not isinstance(shape,tuple):unknown=True;result['unknowns'].append(f'line {node.lineno}: allocation shape unresolved');return
            shapes[target.id]=shape;buffers[target.id]=buffer
            result['allocations'].append(dict(line=node.lineno,name=target.id,shape=list(map(str,shape)),buffer=buffer))
            if buffer.endswith(('sbuf','psum')):
                if len(shape)<2:record_violation(node.lineno,'onchip_rank',f'{target.id} has rank {len(shape)} in {buffer}; on-chip tiles require at least 2 dimensions.')
                if shape and within_limit(shape[0],128)==UNKNOWN:unknown=True
                if shape and within_limit(shape[0],128)=='PROVEN_VIOLATION':record_violation(node.lineno,'partition_limit',f'{target.id} partition {shape[0]} exceeds 128. Coordinate tile slices and consumers.')
                # 512 is the FP32 per-bank/matmul limit, not a universal
                # PSUM allocation limit. Apply it at nc_matmul above.
        elif api in ('sum','max'):
            values=kw(value,('x','axis','dtype','keepdims'));shape=tensor(values.get('x'));axis=expr(values.get('axis'))
            axes=(int(axis),) if isinstance(axis,sp.Integer) else tuple(map(int,axis)) if isinstance(axis,tuple) and all(isinstance(i,sp.Integer) for i in axis) else None
            keep=values.get('keepdims');keepdims=keep.value if isinstance(keep,ast.Constant) and type(keep.value) is bool else False if keep is None else None
            if shape is None or axes is None or keepdims is None:unknown=True;return
            axes=tuple(i+len(shape) if i<0 else i for i in axes)
            if len(set(axes))!=len(axes) or any(i<0 or i>=len(shape) for i in axes):unknown=True;return
            out=tuple(sp.Integer(1) if i in axes else d for i,d in enumerate(shape)) if keepdims else tuple(d for i,d in enumerate(shape) if i not in axes)
            shapes[target.id]=out;buffers[target.id]='nl.sbuf'
            result['operations'].append(dict(line=value.lineno,api=api,input_shape=list(map(str,shape)),output_shape=list(map(str,out)),keepdims=keepdims))
            if len(out)<2:record_violation(value.lineno,'reduction_rank',f'{api} reduces {shape} on axes {axes} to rank {len(out)}. Preserve a free axis with keepdims=True only if required by the actual consumer.')
    def walk(body):
        nonlocal unknown
        for statement in body:
            if isinstance(statement,ast.Assign):assignment(statement)
            elif isinstance(statement,ast.Expr) and isinstance(statement.value,ast.Call):operation(statement.value)
            elif isinstance(statement,ast.For):
                iterator=statement.iter;bounds=iterator.args if isinstance(iterator,ast.Call) and ast.unparse(iterator.func) in ('range','nl.affine_range','nl.sequential_range') else []
                if len(bounds)==1 and isinstance(statement.target,ast.Name):
                    stop=expr(bounds[0]);symbol=sp.Symbol(statement.target.id,integer=True,nonnegative=True)
                    if bounded(stop) and stop.is_nonpositive is True:
                        result['loops'].append(dict(line=statement.lineno,index=statement.target.id,lower='0',upper_exclusive=str(stop),coverage=UNKNOWN,body_executes=False))
                        continue
                    if bounded(stop):
                        previous=env.get(statement.target.id);env[statement.target.id]=symbol
                        if isinstance(stop,sp.Integer) and 0<stop<=16:loop_values[symbol]=tuple(range(int(stop)))
                        result['loops'].append(dict(line=statement.lineno,index=str(symbol),lower='0',upper_exclusive=str(stop),coverage=UNKNOWN))
                        walk(statement.body);env[statement.target.id]=previous;loop_values.pop(symbol,None)
                    else:unknown=True
                else:unknown=True
            elif isinstance(statement,(ast.If,ast.While,ast.Try,ast.With)):
                # Do not merge conditional states or assume branches execute.
                for item in ast.walk(statement):
                    if isinstance(item,ast.Name) and isinstance(item.ctx,ast.Store):env[item.id]=None;shapes.pop(item.id,None);buffers.pop(item.id,None)
                unknown=True
    try:walk(functions[0].body)
    except (TypeError,ValueError,RecursionError,ZeroDivisionError):result['unknowns'].append('Unsupported symbolic expression');unknown=True
    result['status']='PROVEN_VIOLATION' if result['violations'] else UNKNOWN if unknown else 'NO_STATIC_VIOLATION'
    if not result['operations'] and not result['allocations']:result['status']=UNKNOWN
    return result


def repair_prompt(prompt,source,feedback,*,input_shapes=None,context=8192,answer_budget=2500,model='Qwen/Qwen3-8B'):
    from failure_selection import classify_failure
    from nki_knowledge import local_token_counter
    result=analyze(source,input_shapes);category=classify_failure(feedback).failure_category
    kinds={'DMA_SHAPE_MISMATCH':('dma_elements',),'INVALID_TENSOR_DIMENSIONS':('reduction_rank','onchip_rank','matmul_dimensions','partition_limit','psum_free_limit','matmul_tile_limit'),'OUT_OF_BOUNDS':('slice_bounds',),'INVALID_BUFFER_PLACEMENT':('matmul_buffer','transpose_buffer')}.get(category,())
    relevant=[v for v in result['violations'] if v['kind'] in kinds]
    # Original exception, not a later mathematical hypothesis, controls routing.
    if category=='INVALID_TENSOR_DIMENSIONS' and 'dma_copy' in feedback:relevant=[v for v in relevant if v['kind']=='partition_limit']
    counter,method=local_token_counter(model);budget=max(0,min(120,context-answer_budget-128-counter(prompt)))
    text=''
    if relevant:
        selected=min(relevant,key=lambda v:v['line']);text=f"Symbolic PROVEN_VIOLATION, line {selected['line']}: {selected['message']}"
        if counter(text)>budget:text=''
    elif result['status']==UNKNOWN:
        text='Symbolic UNKNOWN: unsupported expressions or missing shape bindings prevent a proof. Use the actual checker error; do not assume legality.'
        if counter(text)>budget:text=''
    result.update(applied=bool(text),guidance=text,context_token_count=counter(text),counting_method=method)
    return prompt+('\n\n'+text if text else ''),result
