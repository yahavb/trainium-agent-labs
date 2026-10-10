"""Actual held-out synthetic repairs: frozen inputs, real simulator/NumPy checks."""
import ast
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace


def safe_subset(source):
    """Limit generated code to NKI imports and bounded ordinary kernel syntax."""
    if len(source)>50000:return False
    try:tree=ast.parse(source)
    except SyntaxError:return False
    if sum(1 for _ in ast.walk(tree))>2000:return False
    if any(not isinstance(n,(ast.Import,ast.FunctionDef)) for n in tree.body):return False
    for n in ast.walk(tree):
        if isinstance(n,(ast.ImportFrom,ast.While,ast.With,ast.AsyncFunctionDef,ast.ClassDef)):return False
        if isinstance(n,ast.Import) and any(a.name not in ('nki','nki.language','nki.isa') for a in n.names):return False
        if isinstance(n,ast.Attribute) and n.attr.startswith('__'):return False
        if isinstance(n,ast.Constant) and type(n.value) is int and abs(n.value)>4096:return False
        if isinstance(n,ast.Pow):return False
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id not in ('range','min','max','len'):return False
    return True


def verify_generated(source,spec):
    if not safe_subset(source):return {'passed':False,'status':'REJECTED_STATIC_SUBSET','error':'Generated source exceeds the restricted synthetic-evaluation subset'}
    with tempfile.TemporaryDirectory(prefix='nki-heldout-private-') as temporary:
        p=Path(temporary)/'request.json';p.write_text(json.dumps({'source':source,'spec':spec}))
        code="import json,sys;from synthetic_nki.verify import validate;r=json.load(open(sys.argv[1]));print(json.dumps(validate(r['source'],r['spec'])))"
        try:
            done=subprocess.run([sys.executable,'-B','-c',code,str(p)],capture_output=True,text=True,timeout=30)
        except subprocess.TimeoutExpired:return {'passed':False,'status':'TIMEOUT','error':'Synthetic simulator exceeded 30 seconds'}
        if done.returncode:return {'passed':False,'status':'PROCESS_ERROR','error':done.stderr[-1000:]}
        try:return json.loads(done.stdout.splitlines()[-1])
        except (ValueError,IndexError):return {'passed':False,'status':'PROCESS_ERROR','error':done.stdout[-1000:]}


def evaluate(project,output):
    import agent
    from training.data import examples,load_verified
    from failure_selection import code_fingerprint
    from synthetic_nki.generate import templates
    from synthetic_nki.curriculum import specs
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    records=load_verified(Path(project)/'synthetic_nki/data_v4/heldout.jsonl','heldout')
    candidates=templates()+[s for s,_ in specs()]
    tasks={code_fingerprint(s['source']):s for s in candidates}
    summary=[]
    with (output/'attempts.jsonl').open('x') as log:
        for mode in ('base','lora'):
            for record in records:
                task=tasks[record['structural_hash']]
                prompt=next(e['messages'][0]['content'] for e in examples([record]) if e['kind']=='repair')
                options=SimpleNamespace(instrument=True,max_tokens=2500,context=8192,model='Qwen/Qwen3-8B',base='http://localhost:8001/v1',think=False,request_timeout=14400,adapter_mode=mode)
                replies=agent.ask_parallel(options,prompt,4)
                results=[]
                for index,reply in enumerate(replies):
                    source=agent.extract_code(reply);result=verify_generated(source,task)
                    row={'adapter_mode':mode,'example_id':record['example_id'],'candidate_index':index,'source':source,'original_failure':record['observed_error'],'verification':result,**getattr(reply,'metadata',{})}
                    log.write(json.dumps(row)+'\n');log.flush();results.append(result)
                summary.append({'adapter_mode':mode,'example_id':record['example_id'],'verified_repairs':sum(r['passed'] for r in results),'candidates':4,'operation_family':record['operation_family']})
    (output/'summary.json').write_text(json.dumps(summary,indent=2));return summary
