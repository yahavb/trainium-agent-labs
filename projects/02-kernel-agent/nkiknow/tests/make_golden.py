"""Regenerate golden_v0.json from a PRISTINE agent.py (the organizer's, before the hooks).
usage: python make_golden.py /path/to/pristine_agent.py   (run with cwd = projects/02-kernel-agent)"""
import importlib.util, json, os, sys
sys.path.insert(0, os.getcwd())
from golden_cases import cases
spec = importlib.util.spec_from_file_location("agent_pristine", sys.argv[1])
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
out = cases(m)
path = sys.argv[2] if len(sys.argv) > 2 else "golden_v0.json"
json.dump(out, open(path, "w"), indent=1, sort_keys=True)
print(len(out), "cases")
