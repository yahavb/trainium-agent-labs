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
        result['guidance']='Task plan: stationary [K,M], moving [K,N], output [M,N]. Distinct live SBUF input tiles; result-shaped FP32 PSUM, separate result SBUF, shared_hbm output. Tile M<=128, N<=512, K<=128 BEFORE loading. Cover ceil(M/tM) by ceil(N/tN) output regions; include boundary extents. One PSUM lifetime per output region: first K contribution overwrites, later disjoint K contributions accumulate. Copy the final result to its matching output slice only after K completes. Avoid whole-input SBUF DMA.'
    elif 'pool' in name:
        C,H,W=input_shapes[0];p=pool_size
        if type(p) is not int or p<=0:return dict(result,status='UNKNOWN')
        result.update(status='PROVEN_VALID',output_shape=(C,H//p,W//p),reduction_dimensions=('window_height','window_width'),divisor=p*p,tile_constraints=dict(partition=128),requires_tiling=C>128)
        result['guidance']='Task plan: input [C,H,W], output [C,H//p,W//p], same dtype. Each output averages its own non-overlapping p-by-p input window; divide by p*p. Preserve channels and output positions; reduce ONLY the two window axes. Use SBUF for loads/reductions and shared_hbm for output; matmul/PSUM are unnecessary. Keep the channel partition <=128 and reduction results at least 2D; keepdims only where rank/consumer require it. Window source slice, reduction result and output destination must have matching extents; write every output position. Ignore incomplete trailing windows.'
    elif 'transpose' in name:
        P,F=input_shapes[0]
        if shape2D is None:return dict(result,status='UNKNOWN')
        F1,F2=shape2D
        if F1*F2!=F:return dict(result,status='PROVEN_INVALID',guidance='Flattened free extent disagrees with shape2D.')
        result.update(status='PROVEN_VALID',output_shape=(P,F),input_layout='[P,F1*F2]',output_layout='same P, flattened [F2,F1]',tile_constraints=dict(partition=128))
        result['guidance']='Task plan: each partition row contains a flattened F1-by-F2 matrix; preserve the P partition and output [P,F1*F2]. Free position r*F2+c maps to c*F1+r within the SAME partition. Load matching HBM/SBUF regions and permute free positions with legal on-chip copies, then store all rows. nl.transpose(x) and nc_transpose swap partition/free axes of a 2D tile; directly transposing the whole [P,F] input does not implement this task. tensor_copy accepts explicit dst/src on-chip views; keep copied slices at least 2D with equal extents.'
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
    return dict(operation=spec['op'],cases=plans,guidance=next((p['guidance'] for p in plans if p.get('guidance')),''))


def generation_prompt(prompt,level,*,context=8192,answer_budget=2500,model='Qwen/Qwen3-8B'):
    result=for_level(level);counter,method=local_token_counter(model)
    budget=max(0,min(240,context-answer_budget-128-counter(prompt)))
    text=result['guidance'];result.update(applied=bool(text) and counter(text)<=budget,counting_method=method,context_token_count=counter(text) if counter(text)<=budget else 0)
    return prompt+('\n\n'+text if result['applied'] else ''),result
