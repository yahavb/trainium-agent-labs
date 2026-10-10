"""Prepare/run sequential A-D pilots with baseline guards and exclusive artifacts.

Default is preparation only. --run is refused while an original Project 2 agent
is active. No credentials/environment dumps are stored in the manifest.
"""
import argparse
import collections
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
import tempfile
import time

PROJECT = Path(__file__).resolve().parent
ORIGINAL = Path('/workspace/projects/02-kernel-agent')
ARMS = (
    ('A_baseline', 'standard', 'reward', 'standard'),
    ('B_diversity', 'diverse', 'reward', 'standard'),
    ('C_diversity_diagnostic', 'diverse', 'diagnostic', 'standard'),
    ('D_full', 'diverse', 'diagnostic', 'grounded'),
)


def baseline_processes(proc_root=Path('/proc'), original=ORIGINAL):
    found = []
    for path in proc_root.iterdir():
        if not path.name.isdigit(): continue
        try:
            state = (path/'stat').read_text().rsplit(')',1)[1].strip().split()[0]
            if state == 'Z': continue
            args = (path/'cmdline').read_bytes().decode(errors='replace').split('\0')
            cwd = (path/'cwd').resolve(strict=True)
            for arg in args:
                if Path(arg).name == 'agent.py':
                    target = (cwd/arg).resolve() if not Path(arg).is_absolute() else Path(arg).resolve()
                    if target == original/'agent.py':
                        found.append(dict(pid=int(path.name),cwd=str(cwd))); break
        except (OSError, IndexError): continue
    return sorted(found,key=lambda item:item['pid'])


def evaluation_processes():
    found=[]
    for path in Path('/proc').iterdir():
        if not path.name.isdigit(): continue
        try:
            state=(path/'stat').read_text().rsplit(')',1)[1].strip().split()[0]
            args=(path/'cmdline').read_bytes().decode(errors='replace').split('\0')
            if state!='Z' and int(path.name)!=os.getpid() and any(Path(arg).name in ('agent.py','run_controlled.py','try_level.py') for arg in args if arg):
                found.append(dict(pid=int(path.name),cwd=str((path/'cwd').resolve())))
        except (OSError,IndexError):pass
    return found


def endpoint_healthy():
    import httpx
    try:
        response=httpx.get('http://localhost:8000/v1/models',timeout=10)
        return response.status_code==200 and any(model.get('id')=='Qwen/Qwen3-8B' for model in response.json().get('data',[]))
    except (httpx.HTTPError,ValueError):return False


def build_command(options, arm, level, directory):
    _, candidate, selection, repair = arm
    command = [sys.executable, '-B', '-u', str(PROJECT/'agent.py'), '--level', str(level),
            '--rounds',str(options.rounds),'--samples',str(options.samples),'--repeat',str(options.repeat),
            '--context','8192','--max-tokens','2500','--model','Qwen/Qwen3-8B',
            '--base','http://localhost:8000/v1','--candidate-policy',candidate,
            '--selection-policy',selection,'--repair-policy',repair,'--instrument',
            '--grade-dir',str(directory/'grade'),'--log',str(directory/'attempts.jsonl')]
    if arm[0]=='B_shape_aware':command.extend(['--generation-policy','constrained'])
    if arm[0] in ('B_targeted','D_combined','E_combined_constrained','B_targeted_standard','C_targeted_synthetic','D_full_adaptive'):
        command.extend(['--feedback-policy','targeted'])
    if arm[0] in ('C_synthetic','D_combined','E_combined_constrained','C_targeted_synthetic','D_full_adaptive'):
        command.extend(['--example-policy','synthetic'])
    if arm[0] in ('D_combined','E_combined_constrained','D_full_adaptive'):command.append('--adaptive-repair')
    if arm[0]=='E_combined_constrained':command.extend(['--generation-policy','constrained'])
    return command


def summarize(rows):
    groups = collections.defaultdict(list)
    for row in rows: groups[(row.get('run_id'),row.get('repeat_index'),row['level'])].append(row)
    trials=[]
    for (run,repeat,level),records in groups.items():
        rounds=collections.defaultdict(list)
        for row in records:rounds[row['round']].append(row)
        selected=[row for row in records if row.get('selected')]
        success=next((row for row in selected if row['reward'] >= 1.-1e-9),None)
        token_fields=('prompt_tokens','completion_tokens','total_tokens')
        totals={key:sum(row[key] for row in records) if all(type(row.get(key)) is int for row in records) else None
                for key in token_fields}
        trial=dict(run_id=run,repeat_index=repeat,level=level,solved=success is not None,
                   best_reward=max(row['reward'] for row in records),
                   mean_candidate_reward=sum(row['reward'] for row in records)/len(records),
                   rounds_observed=len(rounds),candidates_observed=len(records),
                   rounds_to_success=success['round']+1 if success else None,
                   candidates_to_success=sum(len(batch) for rnd,batch in rounds.items() if rnd<=success['round']) if success else None,
                   mean_unique_ast=sum(batch[0]['diversity']['unique_ast_sources'] for batch in rounds.values())/len(rounds),
                   mean_duplicate_fraction=sum(batch[0]['diversity']['duplicate_fraction'] for batch in rounds.values())/len(rounds),
                   round_seconds=sum(batch[0]['round_seconds'] for batch in rounds.values()),
                   failure_categories=dict(collections.Counter(row['failure_category'] for row in records if row['reward']<1.-1e-9)),
                   repeated_selected_signatures=len(selected)-len({row['normalized_signature'] for row in selected}),
                   **totals)
        trials.append(trial)
    return dict(trials=trials,solved=sum(t['solved'] for t in trials),trial_count=len(trials),
                solve_rate=sum(t['solved'] for t in trials)/len(trials) if trials else None,
                interpretation='Pilot is an integration/diversity check, not evidence of superiority. Failures are censored; CPU simulation is not device performance.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--synthetic-pilot',choices=('smoke','ablation','iteration','diagnosis-smoke','diagnosis-ablation'))
    parser.add_argument('--correctness-pilot',action='store_true',help='existing full versus constrained generation/shape-aware repair')
    parser.add_argument('--rounds',type=int,default=2)
    parser.add_argument('--samples',type=int,default=4)
    parser.add_argument('--repeat',type=int,default=1)
    parser.add_argument('--levels',type=int,nargs='+',default=[1,3],choices=range(1,9))
    parser.add_argument('--output-root',type=Path,default=PROJECT/'runs')
    options=parser.parse_args()
    if min(options.rounds,options.samples,options.repeat)<1: parser.error('positive rounds/samples/repeat required')
    blocked=baseline_processes()
    if options.run and blocked:
        print('Inference refused: original Project 2 baseline is active:',blocked)
        return 2
    if options.run and (options.correctness_pilot or options.synthetic_pilot) and (evaluation_processes() or not endpoint_healthy()):
        print('Inference deferred: another evaluation is active or the model endpoint is unhealthy.');return 2
    options.output_root.mkdir(parents=True,exist_ok=True)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
    root=Path(tempfile.mkdtemp(prefix=f'controlled-{stamp}-',dir=options.output_root))
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT,text=True).strip()
    sources=sorted(PROJECT.glob('*.py'))+sorted(p for p in (PROJECT/'synthetic_nki').rglob('*') if p.is_file() and p.suffix in ('.py','.jsonl','.json'))
    file_hashes={}
    source_directory=root/'source'
    source_directory.mkdir()
    for path in sources:
        relative=path.relative_to(PROJECT);content=path.read_bytes()
        file_hashes[str(relative)]=hashlib.sha256(content).hexdigest()
        destination=source_directory/relative;destination.parent.mkdir(parents=True,exist_ok=True)
        with destination.open('xb') as output:output.write(content)
    plan=[]
    arms=(('A_full','diverse','diagnostic','grounded'),('B_shape_aware','diverse','diagnostic','shape-aware')) if options.correctness_pilot else ARMS
    if options.synthetic_pilot:
        arms=(('A_current','diverse','diagnostic','grounded'),('B_targeted','diverse','diagnostic','grounded'))
        if options.synthetic_pilot=='ablation':arms+=(('C_synthetic','diverse','diagnostic','grounded'),('D_combined','diverse','diagnostic','grounded'))
        if options.synthetic_pilot=='iteration':arms=(('D_combined','diverse','diagnostic','grounded'),('E_combined_constrained','diverse','diagnostic','grounded'))
    if options.synthetic_pilot in ('diagnosis-smoke','diagnosis-ablation'):
        arms=(('A_legacy_standard','standard','diagnostic','standard'),('B_targeted_standard','standard','diagnostic','standard'))
        if options.synthetic_pilot=='diagnosis-ablation':arms+=(('C_targeted_synthetic','standard','diagnostic','standard'),('D_full_adaptive','diverse','diagnostic','grounded'))
    for level in dict.fromkeys(options.levels):
        for arm in arms:
            directory=root/f'level{level}'/arm[0]
            command=build_command(options,arm,level,directory)
            command[3]=str(source_directory/'agent.py')
            plan.append(dict(arm=arm[0],level=level,directory=str(directory),command=command))
    import nki
    manifest=dict(synthetic_pilot=options.synthetic_pilot,correctness_pilot=options.correctness_pilot,preparation_only=not options.run,baseline_processes=blocked,revision=revision,
                  source_sha256=file_hashes,sdk_version=nki.__version__,target='trn2',
                  model='Qwen/Qwen3-8B',context=8192,max_tokens=2500,temperature=.6,top_p=.95,
                  thinking=False,rounds=options.rounds,samples=options.samples,repeat=options.repeat,
                  input_seed=0,numerical_tolerance='existing checker RMS-normalized 0.02',plan=plan)
    with (root/'manifest.json').open('x') as output:json.dump(manifest,output,indent=2)
    print('Artifacts:',root,flush=True)
    if not options.run:
        print('Prepared only; baseline guard:',blocked or 'clear')
        return 0
    # Only explicitly selected, nonsensitive environment settings go in the manifest.
    environment=dict(os.environ,NEURON_PLATFORM_TARGET_OVERRIDE='trn2')
    outcomes=[]
    for entry in plan:
        if baseline_processes() or ((options.correctness_pilot or options.synthetic_pilot) and (evaluation_processes() or not endpoint_healthy())):
            print('Original baseline started; remaining arms deferred.');return 2
        directory=Path(entry['directory']);directory.mkdir(parents=True,exist_ok=False)
        started=time.perf_counter()
        print('Running:',entry['arm'],'level',entry['level'],flush=True)
        with (directory/'console.log').open('x') as output:
            completed=subprocess.run(entry['command'],cwd=source_directory,env=environment,stdout=output,stderr=subprocess.STDOUT)
        rows=[]
        log=directory/'attempts.jsonl'
        if log.exists(): rows=[json.loads(line) for line in log.read_text().splitlines()]
        result=dict(arm=entry['arm'],level=entry['level'],returncode=completed.returncode,
                    wall_seconds=time.perf_counter()-started,**summarize(rows))
        outcomes.append(result)
        with (directory/'summary.json').open('x') as output:json.dump(result,output,indent=2)
        if completed.returncode:
            print('Integration failure; inspect',directory/'console.log');break
    with (root/'summary.json').open('x') as output:json.dump(outcomes,output,indent=2)
    return 0 if all(row['returncode']==0 for row in outcomes) else 1


if __name__=='__main__':sys.exit(main())
