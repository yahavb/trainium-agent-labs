"""Optional conservative instruction lowering, without algorithm reconstruction.

Only tensor_scalar namespace/opcode and known HBM-result staging are supported.
No benchmark-specific names/shapes, source execution, or complete kernels. The
unchanged checker must verify every transformed candidate; no static correctness
or device-execution claim follows from these edits.
"""
import ast


def legalize(source):
    metadata={'policy':'legalize','applied':False,'changes':[],'verification':'Requires unchanged checker; preserves requested scalar operation, not proof of task semantics.'}
    if len(source)>50000:return source,metadata
    try:tree=ast.parse(source)
    except (SyntaxError,RecursionError):return source,metadata
    if sum(1 for _ in ast.walk(tree))>2000:return source,metadata
    import nki
    import nki.isa as isa
    import nki.language as nl
    if not nki.__version__.startswith('0.6.0') or not hasattr(isa,'tensor_scalar') or not hasattr(nl,'multiply'):return source,metadata
    modules={}
    for stmt in tree.body:
        if isinstance(stmt,ast.Import):
            for alias in stmt.names:modules[alias.asname or alias.name]=alias.name
    language=next((name for name,module in modules.items() if module=='nki.language'),None)
    instruction=next((name for name,module in modules.items() if module=='nki.isa'),None)
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef)]
    if not language or not instruction or len(functions)!=1:return source,metadata
    if any(a.arg in modules for a in functions[0].args.posonlyargs+functions[0].args.args+functions[0].args.kwonlyargs):return source,metadata
    function=functions[0]
    # Reject rebinding import aliases and repeated/conditional tensor definitions.
    if any(isinstance(n,ast.Name) and isinstance(n.ctx,ast.Store) and n.id in modules for n in ast.walk(tree)):return source,metadata
    if any(isinstance(n,(ast.Import,ast.ImportFrom,ast.ClassDef,ast.AsyncFunctionDef)) or isinstance(n,ast.FunctionDef) and n is not function for n in ast.walk(function)):return source,metadata
    if any(a is not None and a.arg in modules for a in (function.args.vararg,function.args.kwarg)):return source,metadata
    counts={}
    for node in ast.walk(function):
        if isinstance(node,ast.Name) and isinstance(node.ctx,ast.Store):counts[node.id]=counts.get(node.id,0)+1
    buffers={}
    for node in ast.walk(function):
        if not isinstance(node,ast.Assign) or len(node.targets)!=1 or not isinstance(node.targets[0],ast.Name) or counts[node.targets[0].id]!=1:continue
        value=node.value;name=node.targets[0].id
        if isinstance(value,ast.Call) and isinstance(value.func,ast.Attribute) and isinstance(value.func.value,ast.Name) and value.func.value.id==language:
            kwargs={k.arg:k.value for k in value.keywords if k.arg}
            if value.func.attr=='ndarray':
                buffer=kwargs.get('buffer',value.args[2] if len(value.args)>2 else ast.Attribute(ast.Name(language,ast.Load()),'sbuf',ast.Load()))
                if isinstance(buffer,ast.Attribute) and isinstance(buffer.value,ast.Name) and buffer.value.id==language:buffers[name]=buffer.attr
            elif value.func.attr in ('sum','max'):buffers[name]='sbuf'
    def root_name(node):
        while isinstance(node,ast.Subscript):node=node.value
        return node.id if isinstance(node,ast.Name) else None
    def buffer_of(node):return buffers.get(root_name(node))
    def attr(name,member):return ast.Attribute(ast.Name(name,ast.Load()),member,ast.Load())
    # The transformer visits existing statement lists; newly added statements do
    # not match scalar calls and do not induce recursion or evaluation.
    class Lower(ast.NodeTransformer):
        def visit_Expr(self,node):
            call=node.value
            if not isinstance(call,ast.Call) or not isinstance(call.func,ast.Attribute) or not isinstance(call.func.value,ast.Name):return self.generic_visit(node)
            namespace=call.func.value.id
            if call.func.attr!='tensor_scalar' or namespace not in (language,instruction):return self.generic_visit(node)
            if any(k.arg is None for k in call.keywords) or len(call.args)>4:return node
            if any(k.arg in ('dst','data','op0','operand0')[:len(call.args)] for k in call.keywords):return node
            args={key:value for key,value in zip(('dst','data','op0','operand0'),call.args)}
            args.update({k.arg:k.value for k in call.keywords})
            if not all(key in args for key in ('dst','data','op0','operand0')):return node
            if namespace==language:
                call.func=attr(instruction,'tensor_scalar');metadata['changes'].append({'line':node.lineno,'kind':'instruction_namespace','from':language+'.tensor_scalar','to':instruction+'.tensor_scalar'})
            for key in ('op0','op1'):
                op=args.get(key)
                if isinstance(op,ast.Attribute) and isinstance(op.value,ast.Name) and op.value.id==instruction and op.attr in ('multiply','add','subtract') and hasattr(nl,op.attr) and not hasattr(isa,op.attr):
                    op.value=ast.Name(language,ast.Load());metadata['changes'].append({'line':node.lineno,'kind':'opcode_namespace','opcode':op.attr})
            dst=args['dst'];data=args['data']
            if buffer_of(dst)!='shared_hbm' or buffer_of(data) not in ('sbuf','psum'):return node
            # Scalar operations require equal logical destination extents. This
            # change preserves data shape and the original HBM destination view;
            # incompatible extents remain errors for the authoritative checker.
            used={n.id for n in ast.walk(tree) if isinstance(n,ast.Name)}
            name=f'_nki_scalar_result_{node.lineno}'
            while name in used:name+='x'
            allocation=ast.Assign(targets=[ast.Name(name,ast.Store())],value=ast.Call(func=attr(language,'ndarray'),args=[ast.Attribute(data,'shape',ast.Load())],keywords=[ast.keyword('dtype',ast.Attribute(dst,'dtype',ast.Load())),ast.keyword('buffer',attr(language,'sbuf'))]))
            # Canonical keyword binding after validated positional roles.
            args['dst']=ast.Name(name,ast.Load());call.args=[];call.keywords=[ast.keyword(k,v) for k,v in args.items()]
            store=ast.Expr(ast.Call(func=attr(instruction,'dma_copy'),args=[],keywords=[ast.keyword('dst',dst),ast.keyword('src',ast.Name(name,ast.Load()))]))
            metadata['changes'].append({'line':node.lineno,'kind':'hbm_scalar_staging','data':ast.unparse(data),'original_dst':ast.unparse(dst),'temporary':name})
            return [ast.copy_location(allocation,node),node,ast.copy_location(store,node)]
    tree=Lower().visit(tree)
    if not metadata['changes']:return source,metadata
    ast.fix_missing_locations(tree)
    transformed=ast.unparse(tree)+'\n';compile(transformed,'<instruction-lowering>','exec')
    metadata['applied']=True
    return transformed,metadata
