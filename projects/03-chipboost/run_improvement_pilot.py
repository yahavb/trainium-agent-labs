"""Separately labelled eight-evaluation pilot; never append to comparison logs."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--core', type=int, required=True)
    parser.add_argument('--think', action='store_true')
    parser.add_argument('--max-tokens', type=int, default=2500)
    parser.add_argument('--label', default='full_feedback_and_repair_controller_pilot')
    parser.add_argument('--agent-file', default='experimental_agent.py')
    parser.add_argument('--samples', type=int, default=1)
    parser.add_argument('--budget', type=int, default=8)
    a = parser.parse_args()
    root = Path(a.root).resolve() / 'projects/03-chipboost'
    out = Path(a.out).resolve()
    out.mkdir(exist_ok=False)
    import fcntl
    lock = open(f'/tmp/p1-comparison-core{a.core}.lock', 'w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.environ.update(CHIPBOOST_CORE=str(a.core), CHIPBOOST_SEAT='100')
    sys.path.insert(0, str(root))
    import speedcheck as sc
    import schema
    spec = importlib.util.spec_from_file_location('pilot_agent', root/a.agent_file)
    agent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(agent)
    baseline = root.parent/'02-kernel-agent/reference_level4.py'
    state = dict(phase='acceptance', pid=os.getpid(), core=a.core, budget=a.budget, samples=a.samples,
                 treatment=a.label, think=a.think, max_tokens=a.max_tokens,
                 warning='Exploratory multi-change treatment; not DMA-only v2 or original comparison',
                 model='Qwen/Qwen3-8B', sources={str(p.relative_to(root.parent)):hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in (root/'speedcheck.py', root/a.agent_file, baseline,
                           root.parent/'02-kernel-agent/agent.py', root.parent/'02-kernel-agent/nkibench.py',
                           root/'schema.py')})
    def save():
        (out/'state.json').write_text(json.dumps(state, indent=2)+'\n')
    save()
    try:
        with sc.RefereeWorker(core=a.core, baseline=str(baseline), max_checks=100) as worker:
            acceptance = worker.check(str(baseline))
            assert acceptance is not None, worker.last_error
            (out/'acceptance.json').write_text(json.dumps(acceptance, indent=2)+'\n')
            assert acceptance['verdict']=='no_gain' and acceptance['chip_ok'] and acceptance['sim_ok']
            class Bridge:
                pending_start = True
                def check_isolated(self, path, **kwargs):
                    if self.pending_start:
                        assert Path(path).read_text() == baseline.read_text()
                        self.pending_start = False
                        return dict(acceptance)
                    record = worker.check(path)
                    if record is None:
                        with (out/'infrastructure.jsonl').open('a') as f:
                            f.write(json.dumps(dict(timestamp=time.time(), error=worker.last_error,
                                                   counted_as_kernel_attempt=False))+'\n')
                    return record
            bridge = Bridge()
            agent.pick_referee = lambda: ('speedcheck exploratory pilot', bridge)
            state.update(phase='running', worker_pid=worker._p.pid)
            save()
            sys.argv = [a.agent_file, '--arm', 'referee', '--budget', str(a.budget), '--repeat', '1',
                        '--samples', str(a.samples), '--give-up-after', '0', '--start', str(baseline),
                        '--base', 'http://localhost:8000/v1', '--model', state['model'],
                        '--max-tokens', str(a.max_tokens), '--context', '8192', '--seat', '100',
                        '--log', str(out/'pilot.jsonl')]
            if a.think:
                sys.argv.append('--think')
            agent.main()
            rows=[json.loads(line) for line in (out/'pilot.jsonl').read_text().splitlines() if line.strip()]
            assert len(rows)==a.budget and all(not schema.validate(r) for r in rows)
            assert not bridge.pending_start
            assert [r['attempt_no'] for r in rows]==list(range(1,a.budget+1))
            assert all(r['arm']=='referee' for r in rows)
            state.update(phase='complete', attempts=len(rows),
                         faster=sum(r['verdict']=='faster' for r in rows),
                         best_valid_speedup=max((r['speedup'] for r in rows if r['verdict']=='faster'), default=None))
            save()
    except BaseException as exc:
        state.update(failed_phase=state['phase'], phase='failed', error=f'{type(exc).__name__}: {exc}')
        save()
        raise


if __name__ == '__main__':
    main()
