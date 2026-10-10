"""Grade explicitly labelled research candidates with the unchanged correctness/timing gate."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', required=True)
    ap.add_argument('--core', type=int, required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--case', action='append', required=True, help='label=source_path')
    ap.add_argument('--wait-for-core', action='store_true')
    a = ap.parse_args()
    root = Path(a.root).resolve()/'projects/03-chipboost'
    out = Path(a.out).resolve()
    out.mkdir(exist_ok=False)
    import fcntl
    lock = open(f'/tmp/p1-comparison-core{a.core}.lock', 'w')
    fcntl.flock(lock, fcntl.LOCK_EX | (0 if a.wait_for_core else fcntl.LOCK_NB))
    os.environ.update(CHIPBOOST_CORE=str(a.core), CHIPBOOST_SEAT='100')
    sys.path.insert(0, str(root))
    import speedcheck as sc
    import schema
    baseline=root.parent/'02-kernel-agent/reference_level4.py'
    report=dict(phase='running', core=a.core, pid=os.getpid(),
                source_sha256=hashlib.sha256((root/'speedcheck.py').read_bytes()).hexdigest(),
                baseline_sha256=hashlib.sha256(baseline.read_bytes()).hexdigest(), cases=[])
    def save():
        (out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    save()
    try:
        with sc.RefereeWorker(core=a.core,baseline=str(baseline),max_checks=100) as worker:
            for item in ['baseline='+str(baseline)]+a.case:
                label, path=item.split('=',1)
                source=Path(path).read_bytes()
                candidate=out/f'candidate_{len(report["cases"]):02d}.py'
                candidate.write_bytes(source)
                record=worker.check(str(candidate))
                assert record is not None, worker.last_error
                assert not schema.validate(record), schema.validate(record)
                report['cases'].append(dict(label=label,sha256=hashlib.sha256(source).hexdigest(),record=record))
                save()
                print(label,record['verdict'],record.get('speedup'),record['instruction_given'],flush=True)
                if label=='baseline':
                    assert record['verdict']=='no_gain' and record['chip_ok'] and record['sim_ok']
            report['phase']='complete'
            save()
    except BaseException as e:
        report.update(phase='failed',error=f'{type(e).__name__}: {e}')
        save()
        raise


if __name__=='__main__':
    main()
