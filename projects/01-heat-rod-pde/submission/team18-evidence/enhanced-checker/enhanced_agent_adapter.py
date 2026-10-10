"""Test-only opt-in to existing validation.grade; frozen agent source is unchanged."""
from pathlib import Path
import sys,json,time,threading
root=Path(__file__).resolve().parent
sys.path.insert(0,str(root/'source/projects/01-heat-rod-pde'))
import algorithm_agent as agent
import validation
from checker_runtime import grade_candidate as original_grade
lock=threading.Lock()
policy='enhanced_validation_grade_v2_opt_in'
checks_path=Path(sys.argv[sys.argv.index('--output')+1])/'checker-calls.jsonl'
checks_path.parent.mkdir(parents=True,exist_ok=True)
def grade(problem,answer):
 started=time.perf_counter()
 official=original_grade(problem,answer)
 result=validation.grade(problem,answer)
 result['enhanced_protocol_original_reward']=result.get('original_reward')
 result['original_reward']=official.get('reward')
 result['public_original_grade']=official
 result['grading_policy']=policy
 result['evaluation_error']=official.get('evaluation_error')
 result['checker_seconds']=time.perf_counter()-started
 with lock:
  with checks_path.open('a') as out:
   out.write(json.dumps(dict(problem=problem.get('name'),answer=answer,result=result),allow_nan=False)+'\n')
 return result
agent.grade_candidate=grade
old_solve=agent.solve
def solve(raw,a,log,run_id):
 result=old_solve(raw,a,log,run_id)
 log.flush()
 rows=[json.loads(line) for line in Path(log.name).read_text().splitlines()]
 grades=[r['grade'] for r in rows if r.get('grade')]
 winning=next((g for g in grades if g.get('expr')==result.get('answer') and g.get('reward')==result.get('reward')),None)
 result.update(grading_policy=policy,additional_validation_status='run',test_adapter=Path(__file__).name)
 result['original_reward']=winning.get('original_reward') if winning else None
 result['enhanced_protocol_original_reward']=winning.get('enhanced_protocol_original_reward') if winning else None
 result['validation']=winning.get('validation') if winning else None
 result['public_original_best_reward']=max((g['original_reward'] for g in grades if g.get('original_reward') is not None),default=None)
 result['public_original_solved_candidates']=sum(g.get('original_reward')==1 for g in grades)
 result['enhanced_solved_candidates']=sum(g.get('solved') is True for g in grades)
 result['evaluation_errors']=sum(bool(g.get('evaluation_error')) for g in grades)
 return result
agent.solve=solve
raise SystemExit(agent.main())
