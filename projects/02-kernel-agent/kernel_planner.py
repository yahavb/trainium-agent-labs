"""Task-derived, compact hardware plans; no kernel implementations or references."""
import sympy as sp
from nki_knowledge import installed_compatibility,local_token_counter
from symbolic_shapes import tiling


def plan(operation,input_shapes,*,pool_size=None,shape2D=None):
    compatibility=installed_compatibility()
    result=dict(operation=operation,input_shapes=input_shapes,hardware='trn2',sdk_version=compatibility['sdk_version'],verification='Task semantics and installed constraints; simulator verification still required.')
    if not compatibility['sdk_version'].startswith('0.6.0'):return dict(result,status='UNKNOWN',guidance='SDK constraints not verified.')
    name=operation.lower()
    if 'matmul' in name:
        if len(input_shapes)!=2 or len(input_shapes[0])!=2 or len(input_shapes[1])!=2:return dict(result,status='UNKNOWN')
        K,M=input_shapes[0];Kr,N=input_shapes[1]
        if K!=Kr:return dict(result,status='PROVEN_INVALID',guidance='Contraction dimensions disagree.')
        tm,tn,tk=min(M,128),min(N,512),min(K,128)
        counts={key:tiling(sp.Integer(d),sp.Integer(t)) for key,d,t in [('M',M,tm),('N',N,tn),('K',K,tk)]}
        result.update(status='PROVEN_VALID',input_layout='stationary [K,M], moving [K,N]',output_shape=(M,N),tile_constraints=dict(M=128,N=512,K=128),tile_counts=counts,requires_tiling=M>128 or N>512 or K>128,requires_accumulation=K>tk)
        # Load-once structure: simulator-checked at 1.00 with traffic passing on all L4-L7 shapes (max_waste 1.6/1.25/1.05)
        # using a hand-written kernel that is not part of any prompt; per-(m,n,k) reloads fail L5-L7 traffic bars.
        result['guidance']='Task plan: lhsT [K,M] stationary, rhs [K,N] moving, output [M,N] in nl.shared_hbm, returned. Read every input element from HBM exactly once (traffic is checked). TK=128, KT=K//TK, TM=min(M,128), TN=min(N,512). Allocate SBUF lhs_all (TK,KT,M) and rhs_all (TK,KT,N); for each k dma_copy rows k*TK:(k+1)*TK of lhsT and rhs into lhs_all[:,k,:] and rhs_all[:,k,:]. Then for each M tile and N tile (nl.affine_range): one FP32 PSUM tile (TM,TN); for k in nl.sequential_range(KT): nisa.nc_matmul(dst=psum, stationary=lhs_all[:,k,M-tile columns], moving=rhs_all[:,k,N-tile columns], accumulate=(k>0)). After the K loop nisa.tensor_copy PSUM to an SBUF tile of the output dtype, then dma_copy it to the matching output slice. No PSUM DMA or memset.'
    elif 'pool' in name:
        C,H,W=input_shapes[0];p=pool_size
        if type(p) is not int or p<=0:return dict(result,status='UNKNOWN')
        result.update(status='PROVEN_VALID',output_shape=(C,H//p,W//p),reduction_dimensions=('window_height','window_width'),divisor=p*p,normalization_factor=str(sp.Rational(1,p*p)),mathematical_operation='window sum divided by window element count',semantic_invariants=['Preserve channels and output positions', 'Reduce ONLY the two window axes', 'A reshape preserves element count'],appropriate_apis=['nki.language.sum','nki.isa.tensor_reduce','nki.isa.tensor_scalar'],memory_spaces={'input_tiles':'sbuf','output':'shared_hbm'},tile_constraints=dict(partition=128),requires_tiling=C>128)
        # Cold L1 runs given the earlier (H//p,p) grouping hint invented view APIs (APView, strided_view, arena)
        # in 20/24 candidates; plain window slicing uses only installed APIs.
        result['guidance']='Task plan: input [C,H,W], output [C,H//p,W//p], same dtype. Each output averages its own non-overlapping p-by-p window: sum the window, then multiply by 1.0/(p*p). Simplest legal structure: dma_copy x into one SBUF tile [C,H,W] (C<=128 partitions); loop over c, output row and output column with nl.affine_range; take the window with plain Python slices of that tile; nl.sum over ONLY the two window axes (keepdims=True); scale into a new SBUF tile with nki.isa.tensor_scalar(dst=..., data=..., op0=nl.multiply, operand0=1.0/(p*p)) (nisa, NOT nl; never Python tensor division); dma_copy that tile to the matching single output element; return the nl.shared_hbm output. Use only existing APIs: nl.ndarray, nl.affine_range, nl.sum, nisa.dma_copy, nisa.tensor_scalar. No reshape, view or access-pattern helpers. Give executable code, not commentary.'
    elif 'transpose' in name:
        P,F=input_shapes[0]
        if shape2D is None:return dict(result,status='UNKNOWN')
        F1,F2=shape2D
        if F1*F2!=F:return dict(result,status='PROVEN_INVALID',guidance='Flattened free extent disagrees with shape2D.')
        result.update(status='PROVEN_VALID',output_shape=(P,F),input_layout='[P,F1*F2]',output_layout='same P, flattened [F2,F1]',tile_constraints=dict(partition=128))
        result['guidance']='Task plan: each partition row contains a flattened F1-by-F2 matrix; preserve the P partition and output [P,F1*F2]. Free position r*F2+c maps to c*F1+r within the SAME partition. Load matching HBM/SBUF regions and permute free positions with legal on-chip copies, then store all rows. nl.transpose(x) and nc_transpose swap partition/free axes of a 2D tile; directly transposing the whole [P,F] input does not implement this task. tensor_copy accepts explicit dst/src on-chip views; keep copied slices at least 2D with equal extents.'
    elif 'attention' in name:
        if len(input_shapes)!=3 or any(len(s)!=2 for s in input_shapes) or len(set(input_shapes))!=1:return dict(result,status='UNKNOWN')
        S,D=input_shapes[0]
        if S>128 or D>128:return dict(result,status='UNKNOWN',guidance='Single-tile attention plan needs S,D<=128.')
        result.update(status='PROVEN_VALID',output_shape=(S,D),intermediate_shape=(S,S),mathematical_operation='softmax(q k^T / sqrt(D)) v with row-max subtraction',tile_constraints=dict(partition=128))
        # Structure simulator-checked at 1.00 on all L8 shapes with a hand-written kernel that is not part of any prompt.
        result['guidance']='Task plan: q, k, v [S,D], S,D<=128; output [S,D] in nl.shared_hbm; scores [S,S] stay on chip. Load q, k, v to SBUF. nc_matmul contracts over partitions: nc_transpose q and k (SBUF to FP32 PSUM [D,S]), copy to SBUF, nc_matmul stationary qT, moving kT into PSUM [S,S], copy to SBUF. Per row: nl.max axis 1 keepdims; bias = max times -1/sqrt(D) via tensor_scalar; activation op nl.exp, that bias, scale 1/sqrt(D); nl.sum axis 1 keepdims; reciprocal. nc_transpose exp; nc_matmul stationary expT, moving v into PSUM [S,D]; tensor_scalar multiply by row reciprocal into SBUF; dma_copy out. nisa calls write dst. No softmax/matmul/dot.'
    else:return dict(result,status='UNKNOWN',guidance='No verified operation semantics.')
    result['status_note']='PROVEN_VALID means this mathematical plan is consistent, not a verified generated kernel.'
    return result


def for_level(level):
    import nkibench
    spec=nkibench.LEVELS[level];plans=[]
    for case in spec['shapes']:
        args,_=nkibench.make_inputs(case,level)
        shapes=[tuple(arg.shape) for arg in args if hasattr(arg,'shape')]
        plans.append(plan(spec['op'],shapes,pool_size=case.get('pool_size'),shape2D=case.get('shape2D')))
    return dict(operation=spec['op'],cases=plans,guidance=next((p['guidance'] for p in plans if p.get('guidance') and p.get('status')=='PROVEN_VALID'),''))


def generation_prompt(prompt,level,*,context=8192,answer_budget=2500,model='Qwen/Qwen3-8B'):
    result=for_level(level);counter,method=local_token_counter(model)
    budget=max(0,min(240,context-answer_budget-128-counter(prompt)))
    text=result['guidance'];result.update(applied=bool(text) and counter(text)<=budget,counting_method=method,context_token_count=counter(text) if counter(text)<=budget else 0)
    return prompt+('\n\n'+text if result['applied'] else ''),result


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
            if 'pool' in operation.lower() and same:
                result['findings'].append(dict(line=call.lineno,status='POSSIBLE_VIOLATION',kind='pooling_self_product',guidance='This nc_matmul multiplies the same tile by itself. A self-product generally scales quadratically with input values; average pooling is linear and needs window sums divided by p*p. Check the result dataflow and replace a contributing self-product with the required window reduction. A matmul with separately established constant weights is a different case.'))
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
    result['requirements']=[{k:p[k] for k in ('input_shapes','output_shape','reduction_dimensions','divisor','requires_accumulation') if k in p} for p in specification['cases']]
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
