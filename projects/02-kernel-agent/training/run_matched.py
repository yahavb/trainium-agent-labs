"""Sequential same-CPU-backend model/planner 2x2 benchmark, after prior jobs finish."""
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import httpx

ARMS=(('A_original_legacy','base','off'),('B_original_planner','base','hardware'),('C_lora_legacy','lora','off'),('D_lora_planner','lora','hardware'))

def command(snapshot,level,directory,mode,planner,rounds,samples):
    return [sys.executable,'-B','-u',str(snapshot/'agent.py'),'--level',str(level),'--rounds',str(rounds),'--samples',str(samples),'--repeat','1','--context','8192','--max-tokens','2500','--base','http://localhost:8001/v1','--model','Qwen/Qwen3-8B','--adapter-mode',mode,'--candidate-policy','standard','--selection-policy','reward','--repair-policy','standard','--feedback-policy','legacy','--example-policy','off','--planner-policy',planner,'--instrument','--request-timeout','14400','--grade-dir',str(directory/'grade'),'--log',str(directory/'attempts.jsonl')]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prior-evaluation',required=True);parser.add_argument('--output',required=True);parser.add_argument('--rounds',type=int,default=8);parser.add_argument('--samples',type=int,default=4);parser.add_argument('--levels',type=int,nargs='+',default=[1,2,3,4])
    a=parser.parse_args();project=Path(__file__).resolve().parents[1];sys.path.insert(0,str(project))
    import run_controlled as guards
    from kernel_planner import analyze_semantics
    root=Path(a.output);root.mkdir(parents=True,exist_ok=False)
    (root/'status.json').write_text(json.dumps({'status':'waiting_for_prior_adapter_evaluation','prior':a.prior_evaluation},indent=2))
    prior=Path(a.prior_evaluation)
    while not (prior/'summary.json').exists():
        launch=json.loads((prior.parent/'launch.json').read_text())
        stat=Path('/proc')/str(launch['pid'])/'stat'
        if not stat.exists() or stat.read_text().rsplit(')',1)[1].strip().split()[0]=='Z':raise RuntimeError('Prior adapter evaluation ended without a final summary; inspect its runner.log')
        time.sleep(20)
    while guards.evaluation_processes() or guards.baseline_processes():time.sleep(20)
    response=httpx.get('http://localhost:8001/v1/models',timeout=10);response.raise_for_status()
    snapshot=root/'source';snapshot.mkdir();hashes={}
    paths=list(project.glob('*.py'))+[p for p in (project/'synthetic_nki').rglob('*') if p.is_file() and p.suffix in ('.py','.json','.jsonl')]
    for p in paths:
        dest=snapshot/p.relative_to(project);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes());hashes[str(p.relative_to(project))]=hashlib.sha256(p.read_bytes()).hexdigest()
    manifest={'arms':ARMS,'levels':a.levels,'rounds':a.rounds,'samples':a.samples,'repeat':1,'backend':'same CPU Transformers instance, adapter disabled/enabled per request','base_model':'Qwen/Qwen3-8B','endpoint_metadata':response.json(),'source_sha256':hashes,'temperature':.6,'top_p':.95,'thinking':False,'context':8192,'max_tokens':2500,'comparison_note':'A versus C and B versus D isolate weights; A versus B and C versus D isolate planner. Single repetition is not reliability evidence.'}
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2));outcomes=[]
    from training.heldout_eval import evaluate
    heldout=evaluate(project,root/'heldout-repair')
    print('Held-out repair summary:',heldout,flush=True)
    for level in a.levels:
        for name,mode,planner in ARMS:
            if guards.evaluation_processes() or guards.baseline_processes():raise RuntimeError('Another evaluation active; defer remaining arms')
            directory=root/f'level{level}'/name;directory.mkdir(parents=True)
            cmd=command(snapshot,level,directory,mode,planner,a.rounds,a.samples)
            (directory/'command.json').write_text(json.dumps(cmd,indent=2));start=time.perf_counter()
            print('Running',name,'level',level,flush=True)
            with (directory/'console.log').open('x') as out:done=subprocess.run(cmd,cwd=snapshot,env=dict(os.environ,NEURON_PLATFORM_TARGET_OVERRIDE='trn2'),stdout=out,stderr=subprocess.STDOUT)
            path=directory/'attempts.jsonl';rows=[json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
            operation={1:'average pooling',2:'transpose',3:'matmul',4:'matmul'}[level]
            semantic=[analyze_semantics(r['code'],operation) for r in rows]
            analysis={'possible_algorithm_flags':sum(bool(r['findings']) for r in semantic),'operation_selection':dict(collections.Counter(r['operation_selection'] for r in semantic)),'note':'AST flags are hypotheses; per-shape checker results determine correctness.'}
            result={'arm':name,'level':level,'adapter_mode':mode,'planner_policy':planner,'wall_seconds':time.perf_counter()-start,'returncode':done.returncode,'semantic_analysis':analysis,**guards.summarize(rows)}
            (directory/'summary.json').write_text(json.dumps(result,indent=2));outcomes.append(result)
            (root/f'partial-summary-{len(outcomes):02d}.json').write_text(json.dumps(outcomes,indent=2))
            if done.returncode:raise RuntimeError('Integration failure; inspect console.log')
    (root/'summary.json').write_text(json.dumps(outcomes,indent=2))
    (root/'final-status.json').write_text(json.dumps({'status':'completed'},indent=2))

if __name__=='__main__':main()
