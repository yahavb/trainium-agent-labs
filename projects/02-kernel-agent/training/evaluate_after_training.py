"""Wait for real training and active agent completion, then run all levels on CPU LoRA."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import httpx


def main():
    p=argparse.ArgumentParser();p.add_argument('--training-run',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();project=Path(__file__).resolve().parents[1];sys.path.insert(0,str(project))
    import run_controlled as guards
    root=Path(a.output);root.mkdir(parents=True,exist_ok=False)
    run=Path(a.training_run);launch=json.loads((run/'launch.json').read_text())
    while not (run/'training/result.json').exists():
        if not Path('/proc') .joinpath(str(launch['pid'])).exists():raise RuntimeError('Training ended without result.json; inspect its console.log')
        state=(Path('/proc')/str(launch['pid'])/'stat').read_text().rsplit(')',1)[1].strip().split()[0]
        if state=='Z':raise RuntimeError('Training process exited without a result')
        time.sleep(20)
    result=json.loads((run/'training/result.json').read_text())
    if result['tiny_test']:raise ValueError('Cannot benchmark a tiny pipeline-test adapter')
    while guards.baseline_processes() or guards.evaluation_processes():time.sleep(20)
    # Exclusive frozen source, excluding model weights and dependency packages.
    snapshot=root/'source';snapshot.mkdir()
    paths=list(project.glob('*.py'))+list((project/'synthetic_nki').rglob('*.py'))+list((project/'synthetic_nki').rglob('*.jsonl'))+list((project/'synthetic_nki').rglob('*.json'))
    hashes={}
    for path in paths:
        relative=path.relative_to(project);dest=snapshot/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
        hashes[str(relative)]=hashlib.sha256(path.read_bytes()).hexdigest()
    model=project/'training/base-local';adapter=run/'training/adapter'
    if httpx.get('http://localhost:8000/v1/models',timeout=10).status_code!=200:raise RuntimeError('Original endpoint unhealthy; defer evaluation')
    with (root/'endpoint.log').open('x') as out:
        server=subprocess.Popen([sys.executable,'-B','-u',str(project/'training/cpu_endpoint.py'),'--model',str(model),'--adapter',str(adapter),'--port','8001'],stdout=out,stderr=subprocess.STDOUT)
    (root/'server.json').write_text(json.dumps({'pid':server.pid,'port':8001},indent=2))
    for _ in range(180):
        if server.poll() is not None:raise RuntimeError('CPU endpoint failed; inspect endpoint.log')
        try:
            if httpx.get('http://localhost:8001/v1/models',timeout=3).status_code==200:break
        except httpx.HTTPError:pass
        time.sleep(5)
    else:raise RuntimeError('CPU endpoint loading timeout')
    manifest={'adapter':str(adapter),'model':'Qwen/Qwen3-8B','device':'CPU','sampling':{'temperature':.6,'top_p':.95,'thinking':False},'rounds':8,'samples':4,'repeat':1,'context':8192,'source_sha256':hashes,'comparison_caveat':'CPU Transformers generation differs from original Neuron vLLM; wall times are not comparable. No superiority claim from one repetition.'}
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2))
    summaries=[]
    for level in [1,2,3,4]:
        if guards.evaluation_processes() or guards.baseline_processes():raise RuntimeError('Another evaluation started; remaining levels deferred')
        directory=root/f'level{level}';directory.mkdir()
        cmd=[sys.executable,'-B','-u',str(snapshot/'agent.py'),'--level',str(level),'--rounds','8','--samples','4','--repeat','1','--context','8192','--max-tokens','2500','--base','http://localhost:8001/v1','--model','Qwen/Qwen3-8B','--candidate-policy','diverse','--selection-policy','diagnostic','--repair-policy','grounded','--feedback-policy','targeted','--example-policy','synthetic','--adaptive-repair','--instrument','--request-timeout','14400','--grade-dir',str(directory/'grade'),'--log',str(directory/'attempts.jsonl')]
        (directory/'command.json').write_text(json.dumps(cmd,indent=2));print('Running CPU LoRA level',level,flush=True)
        with (directory/'console.log').open('x') as out:completed=subprocess.run(cmd,cwd=snapshot,stdout=out,stderr=subprocess.STDOUT,env=dict(os.environ,NEURON_PLATFORM_TARGET_OVERRIDE='trn2'))
        rows=[json.loads(x) for x in (directory/'attempts.jsonl').read_text().splitlines()] if (directory/'attempts.jsonl').exists() else []
        summary={'level':level,'returncode':completed.returncode,**guards.summarize(rows)}
        (directory/'summary.json').write_text(json.dumps(summary,indent=2));summaries.append(summary)
        if completed.returncode:raise RuntimeError('Agent failure; remaining levels deferred')
    (root/'summary.json').write_text(json.dumps(summaries,indent=2))
    # Leave serving process untouched; record its PID for explicit lifecycle management.

if __name__=='__main__':main()
