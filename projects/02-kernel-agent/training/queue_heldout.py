"""Run held-out repairs after preserved matched jobs, without blocking their guards."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--prior',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 root=Path(a.output);root.mkdir(parents=True,exist_ok=False)
 (root/'status.json').write_text(json.dumps({'status':'waiting_for_matched_study','prior':a.prior},indent=2))
 while not (Path(a.prior)/'summary.json').exists():time.sleep(20)
 if (Path(a.prior)/'heldout-repair/summary.json').exists():
  (root/'completed.json').write_text(json.dumps({'status':'already_evaluated_by_prior_study','artifact':str(Path(a.prior)/'heldout-repair/summary.json')},indent=2));return
 project=Path(__file__).resolve().parents[1]
 # Waiting filename is intentionally not agent.py. Only active model calls have
 # that name, so existing evaluation guards do not deadlock on queued work.
 code="from pathlib import Path\nimport sys,time\nPROJECT=Path("+repr(str(project))+ ")\nsys.path.insert(0,str(PROJECT))\nimport run_controlled\nwhile run_controlled.evaluation_processes() or run_controlled.baseline_processes():time.sleep(20)\nfrom training.heldout_eval import evaluate\nprint(evaluate(PROJECT,Path(__file__).parent/'results'),flush=True)\n"
 (root/'agent.py').write_text(code)
 os.execv(sys.executable,[sys.executable,'-B','-u',str(root/'agent.py')])
if __name__=='__main__':main()
