import json
import tempfile
from pathlib import Path
import pytest
from training.data import export,examples,load_verified

SOURCE=Path(__file__).resolve().parents[1]/'synthetic_nki/data_v4'

def test_split_safe_export():
    with tempfile.TemporaryDirectory() as root:
        out=Path(root)/'dataset';result=export(SOURCE,out)
        assert result['train_repair_pairs']==26
        assert result['heldout_repair_pairs']==2
        assert result['benchmark_solutions_used'] is False
        assert result['train_examples']>26
        assert all('completion' in json.loads(x) for x in (out/'train.jsonl').read_text().splitlines())
        with pytest.raises(FileExistsError):export(SOURCE,out)

def test_generation_deduplication():
    r=load_verified(SOURCE/'train.jsonl','train')[0]
    rows=examples([r,r])
    assert sum(x['kind']=='generation' for x in rows)==1
    assert all('reference_level' not in x['messages'][0]['content'] for x in rows)

def test_unverified_rejected():
    r=load_verified(SOURCE/'train.jsonl','train')[0]
    r['verification_results']['repaired']['passed']=False
    with tempfile.TemporaryDirectory() as root:
        p=Path(root)/'bad.jsonl';p.write_text(json.dumps(r)+'\n')
        with pytest.raises(ValueError):load_verified(p,'train')

def test_actual_completion_token_count():
    from training.cpu_endpoint import completion_ids
    assert completion_ids([2,3,9,9],9)==[2,3,9]
    assert completion_ids([2,3],9)==[2,3]
    assert completion_ids([],9)==[]

def test_matched_comparison_changes_only_model_or_planner():
    from training.run_matched import command,ARMS
    p=Path('/private/source');d=Path('/private/run')
    cmds=[command(p,1,d,mode,planner,8,4) for _,mode,planner in ARMS]
    a,b,c,e=cmds
    def diff(left,right):return [(left[i-1],x,y) for i,(x,y) in enumerate(zip(left,right)) if x!=y]
    assert diff(a,b)==[('--planner-policy','off','hardware')]
    assert diff(a,c)==[('--adapter-mode','base','lora')]
    assert diff(b,e)==[('--adapter-mode','base','lora')]
    assert all('--adaptive-repair' not in cmd for cmd in cmds)

def test_heldout_restricted_execution():
    from training.heldout_eval import safe_subset
    assert safe_subset('import nki\n@nki.jit\ndef kernel(a):\n return a')
    assert not safe_subset("import os\nos.system('echo bad')")
    assert not safe_subset("import nki\ndef kernel(a):\n __import__('os').system('bad')")
    assert not safe_subset('import nki\ndef kernel(a):\n while True:pass')
