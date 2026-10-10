"""Controlled real-Qwen comparisons, with durable request and calculator telemetry.

This driver never calls fake_model or reads exact/series answers. Existing agents
run without source changes. Per-case wall time includes Python/model/tool work.
"""
import argparse
import ast
import collections
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import runpy
import signal
import shutil
import subprocess
import sys
import threading
import time
import uuid


FORMAL_CASES = [(0, sub, 0) for sub in (1, 2, 3)] + [(1, sub, 0) for sub in (1, 2, 3)] + [(1, 3, seed) for seed in (1, 2)]
DIAGNOSTIC_CASES = [(1, 1, 0), (1, 3, 0)]
VARIANTS = {'baseline': None, 'previous': None, 'core': '', 'A': 'cache', 'B': 'final',
            'C': 'feedback', 'D': 'adaptive', 'E': 'cache,final,feedback,adaptive'}
HERE = Path(__file__).resolve().parent


def clean_json(value):
    """Replace legacy Infinity/NaN, keeping saved JSON portable and truthful."""
    import math
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    return value


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(clean_json(value), indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def read_jsonl(path):
    if not Path(path).exists():
        return []
    rows = []
    for number, line in enumerate(Path(path).read_bytes().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError('Expected a JSON object')
            rows.append(row)
        except (ValueError, UnicodeError) as exc:
            rows.append(dict(type='artifact_error', _log_error=True, path=str(path), line=number,
                             error=type(exc).__name__))
    return rows


def health(base, model):
    import httpx
    started = time.perf_counter()
    try:
        response = httpx.get(base.rstrip('/') + '/models', timeout=8)
        models = response.json().get('data', []) if response.status_code == 200 else []
        return {'available': response.status_code == 200 and any(m.get('id') == model for m in models),
                'status_code': response.status_code, 'models': models,
                'seconds': time.perf_counter() - started}
    except (httpx.HTTPError, ValueError) as exc:
        return {'available': False, 'error_type': type(exc).__name__,
                'seconds': time.perf_counter() - started}


def environment():
    versions = {}
    for name in ('httpx', 'sympy', 'numpy', 'torch', 'vllm', 'torch-neuronx'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    def git(*arguments):
        try:
            result = subprocess.run(['git', *arguments], cwd=HERE, capture_output=True, text=True, timeout=5)
            return result.stdout.strip() if result.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            return None
    return {'python': sys.version, 'platform': platform.platform(), 'machine': platform.machine(),
            'declared_source_commit': os.getenv('HEATROD_SOURCE_COMMIT'),
            'packages': versions, 'git_commit': git('rev-parse', 'HEAD'),
            'git_branch': git('branch', '--show-current'), 'git_status': git('status', '--short'),
            'file_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(HERE.glob('*.py'))},
            'runtime_environment': {k: os.environ[k] for k in ('NEURON_RT_VISIBLE_CORES', 'NEURON_LOGICAL_NC_CONFIG',
                'HEATROD_MODEL', 'HEATROD_MAX_TOKENS') if k in os.environ}}


class Recorder:
    def __init__(self, path):
        self.path, self.lock = Path(path), threading.Lock()
        self.start = time.perf_counter()

    def emit(self, **event):
        event['elapsed_seconds'] = time.perf_counter() - self.start
        event['utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
        with self.lock:
            with self.path.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(clean_json(event), ensure_ascii=False, allow_nan=False) + '\n')
                stream.flush()


def install_telemetry(recorder):
    """Observe calls without changing request payload, order, timeout or results."""
    import httpx
    import tool_calc
    original_post, original_compute = httpx.post, tool_calc.compute

    def post(*args, **kwargs):
        request_id = uuid.uuid4().hex
        started = time.perf_counter()
        body = kwargs.get('json') or {}
        recorder.emit(type='request_start', request_id=request_id, request=body,
                      thread=threading.current_thread().name,
                      timeout_seconds=kwargs.get('timeout'))
        try:
            response = original_post(*args, **kwargs)
        except Exception as exc:
            recorder.emit(type='request_error', request_id=request_id, error_type=type(exc).__name__,
                          seconds=time.perf_counter() - started)
            raise
        try:
            data = response.json()
        except ValueError:
            data = {}
        recorder.emit(type='request_end', request_id=request_id, status_code=response.status_code,
                      usage=data.get('usage'), choices=data.get('choices'),
                      seconds=time.perf_counter() - started)
        return response

    def compute(expression):
        calculation_id = uuid.uuid4().hex
        started = time.perf_counter()
        recorder.emit(type='calculator_start', calculation_id=calculation_id, expression=expression,
                      thread=threading.current_thread().name)
        try:
            result = original_compute(expression)
        except Exception as exc:
            recorder.emit(type='calculator_error', calculation_id=calculation_id,
                          error_type=type(exc).__name__, seconds=time.perf_counter() - started)
            raise
        recorder.emit(type='calculator_end', calculation_id=calculation_id, expression=expression,
                      result=result, seconds=time.perf_counter() - started)
        return result

    httpx.post, tool_calc.compute = post, compute


def common_arguments(a):
    return ['--level', str(a.level), '--sub', str(a.sub), '--seed', str(a.seed),
            '--samples', str(a.samples), '--rounds', str(a.rounds), '--max-tokens', str(a.max_tokens),
            '--tool-steps', str(a.tool_steps), '--model', a.model, '--base', a.base]


def worker(a):
    folder = Path(a.case_dir)
    recorder = Recorder(folder / 'telemetry.jsonl')
    install_telemetry(recorder)
    common = common_arguments(a)
    if a.worker == 'baseline':
        script, arguments = 'agent.py', ['--log', str(folder / 'attempts.jsonl')]
    elif a.worker == 'previous':
        script, arguments = 'improved_agent.py', ['--output', str(folder), '--workers', str(a.workers), '--retries', '0']
    else:
        script, arguments = 'algorithm_agent.py', ['--output', str(folder), '--workers', str(a.workers),
            '--deadline', str(max(1, a.deadline - 3)), '--features', a.combined_features if a.worker == 'E' else VARIANTS[a.worker]]
    recorder.emit(type='worker_start', variant=a.worker, script=script, arguments=common + arguments,
                  file_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in sorted(HERE.glob('*.py'))})
    sys.argv = [script] + common + arguments
    try:
        runpy.run_path(str(HERE / script), run_name='__main__')
    finally:
        recorder.emit(type='worker_end')


def expression_key(answer):
    import pdecheck
    expression = pdecheck.extract(answer or '')
    if not expression:
        return ''
    try:
        return ast.dump(ast.parse(expression, mode='eval'), include_attributes=False)
    except SyntaxError:
        return ' '.join(expression.split())


def summarize_case(folder, case, variant, outer_status):
    """Grade with the original checker; partial evidence never erases a timeout."""
    import agent
    from checker_runtime import grade_candidate
    folder = Path(folder)
    events = read_jsonl(folder / 'telemetry.jsonl')
    artifacts = [event for event in events if event.get('_log_error')]
    starts = [e for e in events if e.get('type') == 'request_start']
    ends = [e for e in events if e.get('type') == 'request_end']
    errors = [e for e in events if e.get('type') == 'request_error']
    ended_ids = {e['request_id'] for e in ends + errors}
    unfinished = sum(e['request_id'] not in ended_ids for e in starts)
    usages = [e.get('usage') or {} for e in ends]
    usage_missing = sum(not all(type(u.get(k)) is int for k in ('prompt_tokens','completion_tokens'))
                        for u in usages) + len(errors) + unfinished
    known_prompt = sum(u.get('prompt_tokens') or 0 for u in usages)
    known_completion = sum(u.get('completion_tokens') or 0 for u in usages)
    metrics = dict(requests=len(starts), responses=len(ends), request_errors=len(errors),
                   unfinished_requests=unfinished,
                   prompt_tokens=None if usage_missing else known_prompt,
                   completion_tokens=None if usage_missing else known_completion,
                   known_prompt_tokens=known_prompt, known_completion_tokens=known_completion,
                   unknown_usage_requests=usage_missing,
                   requested_completion_token_cap=sum((e.get('request') or {}).get('max_tokens',0) for e in starts),
                   tool_executions=sum(e.get('type')=='calculator_start' for e in events),
                   tool_calls=sum(e.get('type')=='calculator_start' for e in events),
                   cache_hits=0, status=outer_status, grading_policy='original_checker',
                   additional_validation_status='not_run',
                   truncated_responses=sum(c.get('finish_reason')=='length' for e in ends for c in (e.get('choices') or [])))
    saved = None
    for path in sorted(folder.glob('*/summary.json')):
        try:
            saved = json.loads(path.read_text())['results'][0]
            break
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            artifacts.append(dict(path=str(path), error=type(exc).__name__))
    if saved:
        metrics['agent_status'] = saved['status']
        if outer_status not in ('budget_timeout','interrupted'):
            if outer_status != 'worker_error' or saved['status'] in ('infrastructure_error','evaluation_error'):
                metrics['status'] = saved['status']
        for key in ('repair_requests','features','validation_failures','duplicate_candidates'):
            if key in saved:
                metrics[key] = saved[key]
    if variant not in ('baseline','previous'):
        native = [event for path in sorted(folder.glob('*/telemetry.jsonl')) for event in read_jsonl(path)]
        artifacts += [event for event in native if event.get('_log_error')]
        records = [e['record'] for e in native if e.get('type')=='calculator_result']
        metrics.update(tool_executions=sum(bool(r['executed']) for r in records),
                       tool_calls=sum(e.get('type')=='calculator_request' for e in native),
                       cache_hits=sum(bool(r['cached']) for r in records),
                       tool_metrics_complete=saved is not None)
    paths = sorted(folder.glob('**/attempts.jsonl'))
    rows = [r for path in paths for r in read_jsonl(path)]
    artifacts += [row for row in rows if row.get('_log_error')]
    rows = [row for row in rows if not row.get('_log_error')]
    problem = agent.LEVELS[case[0]].make(case[1],case[2])
    grades, expressions, best = [], [], None
    for row in rows:
        answer = row.get('answer') or ''
        key = expression_key(answer)
        if key:
            expressions.append(key)
        grade = grade_candidate(problem,answer) if not row.get('error') else None
        grades.append(dict(round=row.get('round'),sample=row.get('sample'),answer=answer,grade=grade))
        if grade and grade['reward'] is not None:
            if best is None or grade['reward']>best['reward']:
                best=grade
    write_json(folder/'original-checker-audit.json',grades)
    metrics.update(reward=best['reward'] if best else None,
                   original_reward=best['reward'] if best else None,
                   answer=best['expr'] if best else None,
                   candidate_original_solved=bool(best and best['reward']==1.0),
                   rounds=max((r.get('round',-1) for r in rows),default=-1)+1,
                   attempts_logged=len(rows), expressions_logged=len(expressions),
                   distinct_expressions=len(set(expressions)),
                   duplicate_expressions=len(expressions)-len(set(expressions)),
                   evaluation_errors=sum(bool(r['grade'] and r['grade']['evaluation_error']) for r in grades),
                   format_failures=sum(bool(r['grade'] and not r['grade']['parts']) for r in grades),
                   math_failures=sum(bool(r['grade'] and r['grade']['parts'] and r['grade']['reward']<1) for r in grades),
                   log_errors=artifacts, artifacts_complete=bool(paths) and not artifacts)
    if metrics['status'] in ('solved','unsolved'):
        metrics['status']='solved' if metrics['candidate_original_solved'] else 'unsolved'
    if artifacts and metrics['status'] in ('solved','unsolved'):
        metrics['status']='artifact_error'
    metrics['http_error_statuses']=dict(collections.Counter(str(e['status_code']) for e in ends if e['status_code']!=200))
    metrics['transport_error_types']=dict(collections.Counter(e['error_type'] for e in errors))
    return metrics



def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite', choices=('diagnostic', 'formal'), default='diagnostic')
    p.add_argument('--variants', default='core,A,B,C,D,E')
    p.add_argument('--combined-features', default=VARIANTS['E'])
    p.add_argument('--repeat', type=int, default=1)
    p.add_argument('--samples', type=int, default=2)
    p.add_argument('--rounds', type=int, default=3)
    p.add_argument('--max-tokens', type=int, default=512)
    p.add_argument('--tool-steps', type=int, default=1)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--deadline', type=float, default=180)
    p.add_argument('--model', default=os.getenv('HEATROD_MODEL', 'Qwen/Qwen3-8B'))
    p.add_argument('--base', default=os.getenv('HEATROD_BASE_URL', 'http://localhost:8000/v1'))
    p.add_argument('--output')
    p.add_argument('--worker', choices=tuple(VARIANTS), help=argparse.SUPPRESS)
    p.add_argument('--case-dir', help=argparse.SUPPRESS)
    p.add_argument('--level', type=int, help=argparse.SUPPRESS)
    p.add_argument('--sub', type=int, help=argparse.SUPPRESS)
    p.add_argument('--seed', type=int, help=argparse.SUPPRESS)
    return p


def main():
    p = parser()
    a = p.parse_args()
    if min(a.repeat, a.samples, a.rounds, a.max_tokens, a.workers, a.deadline) <= 0 or a.tool_steps < 0:
        p.error('Budgets must be positive; tool-steps may be zero')
    if a.worker:
        worker(a)
        return 0
    variants = a.variants.split(',')
    if any(v not in VARIANTS for v in variants) or len(set(variants)) != len(variants):
        p.error('Provide unique variants from ' + ','.join(VARIANTS))
    if any(f and f not in {'cache', 'final', 'feedback', 'adaptive'} for f in a.combined_features.split(',')):
        p.error('Unknown combined feature')
    cases = FORMAL_CASES if a.suite == 'formal' else DIAGNOSTIC_CASES
    root = Path(a.output) if a.output else HERE / 'algorithm-results' / (dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + a.suite)
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    source=root/'source'
    source.mkdir()
    for path in HERE.glob('*.py'):
        shutil.copy2(path,source/path.name)
    settings = {k: v for k, v in vars(a).items() if k not in {'worker', 'case_dir', 'level', 'sub', 'seed'}}
    settings.update(offline=False, problem_cases=cases, sampling_seed_sent_to_model=False,
                    grading_policy='original_checker', additional_validation_status='not_run',
                    transport_retries=0,
                    order='left-rotate variant list by (case_index + repeat_index) modulo variants',
                    budget_notes='All variants share the same outer wall deadline. Baseline/previous request timeouts remain unmodified. New controller has a 3-second reserve to save summaries before the outer deadline; adaptive requests never exceed the common per-request token cap.')
    document = {'settings': settings, 'environment': environment(), 'health_before': health(a.base, a.model),
                'started_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'results': [], 'status': 'running'}
    write_json(root / 'comparison.json', document)
    if not document['health_before']['available']:
        document['status'] = 'stopped_service_unavailable'
        write_json(root / 'comparison.json', document)
        print('STOP: requested real model is unavailable; no experiments were run.', flush=True)
        return 1
    for repetition in range(a.repeat):
        for index, case in enumerate(cases):
            offset = (index + repetition) % len(variants)
            for variant in variants[offset:] + variants[:offset]:
                folder = root / f'repeat{repetition + 1}-{variant}-level{case[0]}.{case[1]}-seed{case[2]}'
                folder.mkdir()
                command = [sys.executable, str(source/'algorithm_benchmark.py'), '--worker', variant, '--case-dir', str(folder),
                    '--level', str(case[0]), '--sub', str(case[1]), '--seed', str(case[2]),
                    '--samples', str(a.samples), '--rounds', str(a.rounds), '--max-tokens', str(a.max_tokens),
                    '--tool-steps', str(a.tool_steps), '--workers', str(a.workers), '--deadline', str(a.deadline),
                    '--model', a.model, '--base', a.base, '--combined-features', a.combined_features]
                write_json(folder / 'invocation.json', {'command': command, 'cwd': str(source), 'outer_timeout': a.deadline})
                print('START ' + folder.name, flush=True)
                started = time.perf_counter()
                exit_code, status = None, 'unsolved'
                with (folder / 'console.log').open('w', encoding='utf-8') as console:
                    process = subprocess.Popen(command, cwd=source, stdout=console, stderr=subprocess.STDOUT,
                                               start_new_session=True)
                    try:
                        exit_code = process.wait(timeout=a.deadline)
                        if exit_code:
                            status = 'worker_error'
                    except subprocess.TimeoutExpired:
                        status = 'budget_timeout'
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    except KeyboardInterrupt:
                        status = 'interrupted'
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                elapsed = time.perf_counter() - started
                entry = {'variant': variant, 'level': case[0], 'sub': case[1], 'seed': case[2],
                    'repeat': repetition + 1, 'case_path': folder.name, 'seconds': elapsed,
                    'features': a.combined_features if variant == 'E' else VARIANTS[variant],
                    'exit_code': exit_code, 'outer_status': status,
                    **summarize_case(folder, case, variant, status)}
                document['results'].append(entry)
                print(f"DONE {folder.name}: {entry['status']} ({elapsed:.1f}s)", flush=True)
                if status == 'interrupted':
                    document['status'] = 'stopped_user_interrupt'
                unhealthy_events = bool(entry['http_error_statuses']) or any(error not in {'ReadTimeout', 'WriteTimeout', 'PoolTimeout'} for error in entry['transport_error_types'])
                if unhealthy_events or status == 'worker_error' or entry['status'] == 'infrastructure_error':
                    entry['health_after_error'] = health(a.base, a.model)
                    # Authentication/model errors and unavailable inference stop the suite.
                    bad_http = any(int(code) in (401, 403, 404, 429) or int(code) >= 500 for code in entry['http_error_statuses'])
                    if unhealthy_events or not entry['health_after_error']['available'] or entry['status']=='infrastructure_error':
                        document['status'] = 'stopped_service_unavailable'
                    elif status == 'worker_error':
                        document['status'] = 'stopped_worker_error'
                if entry['status']=='artifact_error':
                    document['status']='stopped_artifact_error'
                write_json(root / 'comparison.json', document)
                if document['status'].startswith('stopped'):
                    print('STOP ' + document['status'] + '; partial evidence retained at ' + str(root), flush=True)
                    return 130 if status=='interrupted' else 1
    document.update(status='complete', finished_utc=dt.datetime.now(dt.timezone.utc).isoformat(), health_after=health(a.base, a.model))
    write_json(root / 'comparison.json', document)
    print('COMPLETE ' + str(root), flush=True)
    return 0


if __name__ == '__main__':
    def stop(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    sys.exit(main())
