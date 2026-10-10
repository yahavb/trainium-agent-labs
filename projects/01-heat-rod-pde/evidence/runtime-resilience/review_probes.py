"""Review-only reproductions; no real model or network calls."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import compare_agents
import level0_heatrod
import pdecheck
import tool_calc
import improved_agent
from types import SimpleNamespace

results = {}
p = level0_heatrod.make(1)
answer = str(p['exact']) + ' + Rational(0,1)'
with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / 'attempts.jsonl'
    path.write_text(json.dumps(dict(answer=answer, round=0, trace=[]))+'\n')
    results['original_checker_parity'] = dict(
        answer=answer, direct_original=pdecheck.check(p, answer)['reward'],
        wrapper=improved_agent.grade_candidate(p, answer),
        comparison=compare_agents.summarize_attempts([path], p))
    path.write_text(json.dumps(dict(answer=str(p['exact']),round=0,trace=[]))+'\n'+ '{"answer":')
    try:
        result = compare_agents.summarize_attempts([path], p)
        results['partial_log'] = {'result': result}
    except Exception as exc:
        results['partial_log'] = {'raised': type(exc).__name__, 'message': str(exc)}

for text in ['(1, 2)', '[Integral(x,(x,0,1)), Integral(x,(x,0,2))]']:
    try:
        results['calculator:'+text] = {'result': tool_calc.compute(text)}
    except Exception as exc:
        results['calculator:'+text] = {'raised': type(exc).__name__, 'message': str(exc)}

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    cases = root / 'cases.json'
    cases.write_text('[[1,3,0]]')
    commands = []
    class FailedServiceProcess:
        def __init__(self, command, **kwargs):
            commands.append(command)
            folder = Path(command[command.index('--output')+1]) / 'failed-run'
            folder.mkdir()
            (folder/'summary.json').write_text(json.dumps({'results':[{'status':'infrastructure_error'}]}))
        def wait(self, timeout=None):
            return 0
    with patch.dict('os.environ', {'HEATROD_BASE_URL':'http://model.invalid/v1'}), \
         patch.object(compare_agents.subprocess, 'Popen', FailedServiceProcess), \
         contextlib.redirect_stdout(io.StringIO()):
        exit_code=compare_agents.main(['--cases',str(cases),'--variants','improved','decay',
                                      '--output-root',str(root/'runs')])
    document=json.loads(next((root/'runs').glob('*/comparison.json')).read_text())
    results['service_failure_batch'] = dict(exit_code=exit_code, started_cases=len(commands),
        batch_complete=document['batch_complete'], statuses=[r['status'] for r in document['results']])

with tempfile.TemporaryDirectory() as tmp:
    path=Path(tmp)/'attempts.jsonl'
    row=dict(answer='',round=0,error='HTTP 500 after 2 requests',trace=[
        {'type':'http_error','status':500},{'type':'http_error','status':500}])
    path.write_text(json.dumps(row)+'\n')
    results['request_count_on_http_errors']=compare_agents.summarize_attempts([path],p)

for bad in ('u(x,t) = [1,2]', 'u(x,t) = (1,2)'):
    log=io.StringIO()
    response=dict(answer=bad,tool_calls=0,trace=[],error=None)
    settings=SimpleNamespace(rounds=2,samples=1,workers=1,seed=0)
    try:
        with patch.object(improved_agent,'attempt',return_value=response), contextlib.redirect_stdout(io.StringIO()):
            outcome=improved_agent.solve(p,settings,log,'review')
        results['malformed_final:'+bad]={'outcome':outcome,'logged':bool(log.getvalue())}
    except Exception as exc:
        results['malformed_final:'+bad]={'raised':type(exc).__name__,'message':str(exc),'logged':bool(log.getvalue())}
print(json.dumps(results,indent=2,ensure_ascii=False))
