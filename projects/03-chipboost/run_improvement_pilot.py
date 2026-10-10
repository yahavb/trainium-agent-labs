"""Separately labelled eight-evaluation pilot; never append to comparison logs."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time


def evaluate_startup(worker, path, baseline_source, acceptance):
    """Reuse acceptance only for the exact normalized baseline source; never import a candidate."""
    if Path(path).read_text() == baseline_source:
        return dict(acceptance), True
    return worker.check(path), False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--p3', default=None,
                        help="P3 repo root for diagnose.py and redteam/stage12.py; default: --root, if it has them")
    parser.add_argument('--out', required=True)
    parser.add_argument('--start', default=None, help='Frozen candidate to continue; timing baseline stays reference_level4.py')
    parser.add_argument('--core', type=int, required=True)
    parser.add_argument('--think', action='store_true')
    parser.add_argument('--max-tokens', type=int, default=2500)
    parser.add_argument('--label', default='full_feedback_and_repair_controller_pilot')
    parser.add_argument('--agent-file', default='experimental_agent.py')
    parser.add_argument('--samples', type=int, default=1)
    parser.add_argument('--budget', type=int, default=8)
    parser.add_argument('--tag', default='')
    a = parser.parse_args()
    root = Path(a.root).resolve() / 'projects/03-chipboost'
    out = Path(a.out).resolve()
    import fcntl
    lock = open(f'/tmp/p1-comparison-core{a.core}.lock', 'w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.environ.update(CHIPBOOST_CORE=str(a.core), CHIPBOOST_SEAT='100')
    sys.path.insert(0, str(root))
    import speedcheck as sc
    import schema
    spec = importlib.util.spec_from_file_location('pilot_agent', root/a.agent_file)
    agent = importlib.util.module_from_spec(spec)
    os.environ['CHIPBOOST_P3'] = str(Path(a.p3).resolve() / 'projects/03-chipboost') if a.p3 else str(root)
    spec.loader.exec_module(agent)
    baseline = root.parent/'02-kernel-agent/reference_level4.py'
    start = Path(a.start).resolve() if a.start else baseline
    start_bytes, start_source = start.read_bytes(), start.read_text()
    baseline_source = baseline.read_text()
    out.mkdir(exist_ok=False)   # import and start validation must succeed before reserving --out
    state = dict(phase='acceptance', pid=os.getpid(), core=a.core, budget=a.budget, samples=a.samples,
                 treatment=a.label, tag=a.tag, think=a.think, max_tokens=a.max_tokens,
                 start_path=str(start), start_sha256=hashlib.sha256(start_bytes).hexdigest(),
                 start_text_sha256=hashlib.sha256(start_source.encode()).hexdigest(),
                 baseline_path=str(baseline),
                 startup_counted_as_kernel_attempt=False,
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
                        assert Path(path).read_text() == start_source, 'startup source changed after manifest'
                        record, reused = evaluate_startup(worker, path, baseline_source, acceptance)
                        if record is not None:
                            self.pending_start = False
                            (out/'startup.json').write_text(json.dumps(dict(
                                record=record, acceptance_reused=reused,
                                counted_as_kernel_attempt=False, start_path=str(start)), indent=2)+'\n')
                        else:
                            with (out/'infrastructure.jsonl').open('a') as f:
                                f.write(json.dumps(dict(timestamp=time.time(), stage='startup',
                                    error=worker.last_error, counted_as_kernel_attempt=False))+'\n')
                        return record
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
                        '--samples', str(a.samples), '--give-up-after', '0', '--start', str(start),
                        '--base', 'http://localhost:8000/v1', '--model', state['model'],
                        '--max-tokens', str(a.max_tokens), '--context', '8192', '--seat', '100',
                        '--log', str(out/'pilot.jsonl')]
            if a.think:
                sys.argv.append('--think')
            if a.tag:
                sys.argv.extend(['--tag', a.tag])
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
