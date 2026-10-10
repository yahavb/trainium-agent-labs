"""Verify outer-loop feedback on representative kernels; run on a free Trainium core."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--core', type=int, required=True)
parser.add_argument('--out', required=True)
options = parser.parse_args()
root = Path(__file__).resolve().parent
sys.path.insert(0, str(root))
os.environ['CHIPBOOST_CORE'] = str(options.core)
import speedcheck
import schema

cases = [
    ('baseline', root.parent/'02-kernel-agent/reference_level4.py', 'rhs slice depends on k and n'),
    ('n_outer', root/'tests/fixtures/h2_nm_order.py', 'rhs slice depends on k and n'),
    ('lhs_hoisted', root/'tests/fixtures/h3_hoist.py', 'existing lhsT reuse'),
    ('rhs_hoisted', root/'tests/fixtures/h4_rhs_hoist.py', 'lhsT slice depends on k and m'),
]
report = {'source_sha256': hashlib.sha256((root/'speedcheck.py').read_bytes()).hexdigest(),
          'core': options.core, 'cases': [], 'phase': 'running'}
out = Path(options.out)
with speedcheck.RefereeWorker(core=options.core, baseline=str(root.parent/'02-kernel-agent/reference_level4.py')) as w:
    for name, path, expected in cases:
        rec = w.check(str(path))
        row = {'case':name,'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'record':rec}
        report['cases'].append(row)
        out.write_text(json.dumps(report,indent=2)+'\n')
        assert rec is not None, w.last_error
        assert not schema.validate(rec), schema.validate(rec)
        assert rec['verdict'] in ('faster','no_gain','slower'), rec
        assert rec['sim_ok'] and rec['chip_ok'], rec
        assert expected in rec['instruction_given'], rec['instruction_given']
        assert 'innermost loop' not in rec['instruction_given']
        print(name, rec['verdict'], rec['speedup'], rec['instruction_given'], flush=True)
report.update(phase='complete',passed=True)
out.write_text(json.dumps(report,indent=2)+'\n')
print('FEEDBACK VERIFICATION PASSED',flush=True)
