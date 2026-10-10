"""Generic Markdown-to-Verilog iteration with bounded, dynamic taxonomy retrieval."""
import argparse
import hashlib
import json
import os
import pathlib
import re
import time
import urllib.request
from datetime import datetime, timezone

import checker

ROOT = pathlib.Path(__file__).resolve().parent
VERSION = 'md_generic_v4_dynamic_taxonomy'
BASE = os.environ.get('KERNEL_AGENT_BASE_URL', 'http://localhost:8000/v1').rstrip('/')
BASE = BASE if BASE.endswith('/v1') else BASE + '/v1'


def http(path, body=None):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode() if body else None,
                                 headers={'Content-Type': 'application/json'})
    return json.load(urllib.request.urlopen(req, timeout=600))


def ask(messages, model, temperature=.7, max_tokens=1500):
    out = http('/chat/completions', dict(model=model, messages=messages,
               temperature=temperature, max_tokens=max_tokens,
               chat_template_kwargs={'enable_thinking': False}))
    text = out['choices'][0]['message']['content'] or ''
    return re.sub(r'<think>.*?</think>', '', text, flags=re.S)


def failing_checks(feedback):
    labels = re.findall(r'^(\w+): requirement:', feedback, re.M)
    return ', '.join(labels) if labels else feedback.splitlines()[0][:70]


def run_once(model, spec, rounds, tag, exp, show_code, explain, temperature,
             problem='lfsr', *, taxonomy_path=None, knowledge_path=None, category_overrides=(),
             exclude_categories=(), rules_limit=2, show_detect_text=False,
             rule_bytes=1200, context_tokens=8000, max_tokens=1500, log_sink=None,
             output_dir=None, **grade_options):
    history, records = [], []
    data = checker._problem(problem)
    name = data['module']
    prompt_problem = checker.prompt_problem(problem, spec=spec, explain=explain, show_code=show_code,
        taxonomy_path=taxonomy_path, knowledge_path=knowledge_path,
        category_overrides=category_overrides, exclude_categories=exclude_categories,
        rules_limit=rules_limit, show_detect_text=show_detect_text, rule_bytes=rule_bytes, context_tokens=context_tokens,
        max_tokens=max_tokens, **grade_options)
    outdir = pathlib.Path(output_dir or 'candidates/gen') / exp / name
    outdir.mkdir(parents=True, exist_ok=True)
    checker_options = checker.feedback_options(prompt_problem, grade_options)
    for rnd in range(rounds):
        messages = checker.build_prompt(prompt_problem, history)
        audit = checker.prompt_audit(prompt_problem)
        applicable, signatures, selected = audit['categories'], audit['signatures'], audit['selected_rules']
        taxonomy_digest, knowledge_digest = audit['taxonomy_sha256'], audit['knowledge_sha256']
        taxonomy_path, knowledge_path = audit['taxonomy_path'], audit['knowledge_path']
        input_budget = audit['prompt_byte_budget']
        if log_sink:
            log_sink(dict(event='request', run=tag, round=rnd, model=model,
                          messages=messages, categories=applicable, signatures=signatures,
                          selected_rules=selected, taxonomy_sha256=taxonomy_digest, knowledge_sha256=knowledge_digest,
                          prompt_byte_budget=input_budget))
        start = time.monotonic()
        reply = ask(messages, model, temperature, max_tokens=max_tokens)
        elapsed = time.monotonic() - start
        code = checker.extract(reply)
        if log_sink:
            log_sink(dict(event='reply', run=tag, round=rnd, reply=reply, code=code,
                          elapsed_seconds=elapsed))
        candidate, verdict, diagnostics = None, None, {}
        if code is None:
            code = ''
            score, feedback = checker.no_module_result(diagnostics)
        else:
            path = outdir / f'{name}_{tag}_{rnd}.v'
            path.write_text(code)
            candidate = str(path)
            score, feedback = checker.grade(problem, path, diagnostics=diagnostics, **checker_options)
            verdict = True if score == 1. and data.get('official') else checker.official(problem, path)
        same = bool(history) and code.strip() == history[-1]['code'].strip()
        record = dict(event='round', version=VERSION, exp=exp, run=tag, round=rnd, score=score,
            feedback=feedback, code=code, reply=reply, same_as_previous=same,
            problem=name, model=model, official_pass=verdict, candidate=candidate,
            messages=messages, elapsed_seconds=elapsed, temperature=temperature,
            max_tokens=max_tokens, thinking=False, context_tokens=context_tokens,
            prompt_utf8_bytes=checker.prompt_size(messages),
            prompt_byte_budget=input_budget, categories=applicable,
            retrieval_signatures=signatures, selected_rule_keys=[entry['key'] for entry in selected if entry['key']],
            kb_hits=[entry['key'] for entry in selected if entry['key']],
            selected_rules=selected, diagnostics=diagnostics, checker_options=grade_options,
            spec_path=data.get('spec'), reference_path=data.get('ref_verilog'),
            seed=data.get('seed'), comparison_contracts=data.get('comparisons', {}),
            taxonomy_path=str(taxonomy_path), taxonomy_sha256=taxonomy_digest,
            knowledge_path=str(knowledge_path), knowledge_sha256=knowledge_digest,
            excluded_categories=list(exclude_categories),
            checker_sha256=hashlib.sha256(pathlib.Path(checker.__file__).read_bytes()).hexdigest())
        records.append(record)
        if log_sink:
            log_sink(record)
        history.append(record)
        print(f'run {tag} round {rnd}: {score:.2f}; checks={failing_checks(feedback)}; '
              f'rules={record["selected_rule_keys"]}; input_bytes={record["prompt_utf8_bytes"]}/{input_budget}', flush=True)
        print(feedback, flush=True)
        if score == 1.:
            break
    return records


def _main():
    global BASE
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--problem', help='convenience name for a same-stem Markdown specification')
    ap.add_argument('--spec', help='Markdown file; mine/bench remain aliases for the default problem')
    ap.add_argument('--tb', help='optional official testbench; same-stem testbenches are discovered')
    ap.add_argument('--ref', help='independent reference; same-stem references are discovered')
    ap.add_argument('--rounds', type=int, default=6)
    ap.add_argument('--repeat', type=int, default=5)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--cycles', type=int, default=256)
    ap.add_argument('--random-cycles', type=int, default=128)
    ap.add_argument('--facts', action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument('--detectors', action='store_true')
    ap.add_argument('--principles', action='store_true', help='retrieve taxonomy rules for current failures')
    ap.add_argument('--principles-v2', action='store_true', help='enable pattern detectors and dynamic taxonomy retrieval')
    ap.add_argument('--explain', action='store_true')
    ap.add_argument('--hint', action='store_true')
    ap.add_argument('--no-code', action='store_true')
    ap.add_argument('--interface-from-tb', action='store_true')
    reset = ap.add_mutually_exclusive_group()
    reset.add_argument('--async-reset', dest='async_reset', action='store_true')
    reset.add_argument('--no-async-reset', dest='async_reset', action='store_false')
    ap.set_defaults(async_reset=None)
    ap.add_argument('--taxonomy', default=str(ROOT / 'taxonomy.json'))
    ap.add_argument('--knowledge', default=str(ROOT / 'knowledge.json'))
    ap.add_argument('--show-detect-text', action='store_true')
    ap.add_argument('--exclude-categories', default='', help='comma-separated categories excluded from retrieval')
    ap.add_argument('--category', action='append', default=[], help='override or supplement inferred categories')
    ap.add_argument('--rules-limit', type=int, default=2)
    ap.add_argument('--rule-bytes', type=int, default=1200)
    ap.add_argument('--context-tokens', type=int, default=8000)
    ap.add_argument('--max-tokens', type=int, default=1500)
    ap.add_argument('--temperature', type=float, default=.7)
    ap.add_argument('--model')
    ap.add_argument('--base-url')
    ap.add_argument('--log-dir', default=str(ROOT / 'log'))
    ap.add_argument('--no-console-log', action='store_true')
    ap.add_argument('--run-name', default=VERSION)
    ap.add_argument('--output-dir', default='candidates/gen')
    a = ap.parse_args()
    if a.rounds < 1 or a.repeat < 0 or a.rules_limit < 1 or a.cycles < 1 or a.random_cycles < 1:
        ap.error('rounds, rules-limit, and cycle counts must be positive; repeat must be nonnegative')
    if a.problem and a.spec and a.spec not in ('mine', 'bench'):
        ap.error('use --problem or --spec FILE')
    style = a.spec if a.spec in ('mine', 'bench') else 'mine'
    spec_file = checker._find_spec(a.problem or 'lfsr') if not a.spec or a.spec in ('mine', 'bench') else pathlib.Path(a.spec)
    try:
        problem = checker.load_problem(spec_file, a.tb, a.ref, seed=a.seed,
                    async_reset=a.async_reset, cycles=a.cycles, random_cycles=a.random_cycles)
        checker.validate_knowledge(a.taxonomy, a.knowledge)
        spec = checker.public_spec(problem, hint=a.hint or a.async_reset is True,
                                   interface_from_tb=a.interface_from_tb)
    except (OSError, ValueError) as error:
        ap.error(str(error))
    a.problem = problem['module']
    a.principles = a.principles or a.principles_v2
    a.detectors = a.detectors or a.principles_v2
    grade_options = dict(facts=a.facts, detectors=a.detectors, principles=a.principles,
                         async_reset=a.async_reset)
    exp = style + ('_hint' if a.hint else '') + ('_nocode' if a.no_code else '') \
        + ('_explain' if a.explain else '') + ('_principles' if a.principles else '') \
        + ('_v2' if a.principles_v2 else '') + (f'_t{a.temperature}' if a.temperature != .7 else '')
    if a.base_url:
        BASE = a.base_url.rstrip('/')
        BASE = BASE if BASE.endswith('/v1') else BASE + '/v1'
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + str(time.time_ns())
    run_id = re.sub(r'[^\w.-]+', '_', a.run_name) + '_' + a.problem + '_' + stamp
    directory = pathlib.Path(a.log_dir) / 'runs' / run_id
    directory.mkdir(parents=True, exist_ok=True)
    log_path = directory / 'events.jsonl'
    solved, final_scores = 0, []
    with log_path.open('w') as logfile:
        def write(record):
            logfile.write(json.dumps(record) + '\n')
            logfile.flush()
            if record.get('event') == 'request':
                checker.save_prompt_event(record, log_path)
        logged_arguments = dict(vars(a))
        if not a.interface_from_tb:
            logged_arguments.pop('interface_from_tb')
        if not a.show_detect_text:
            logged_arguments.pop('show_detect_text')
        if not a.no_console_log:
            logged_arguments.pop('no_console_log')
        write(dict(event='start', version=VERSION, run_id=run_id, arguments=logged_arguments,
                   spec_path=str(spec_file), problem=problem['module'], base_url=BASE))
        try:
            model = a.model or http('/models')['data'][0]['id']
            for run in range(a.repeat):
                records = run_once(model, spec, a.rounds, run, run_id, not a.no_code,
                    a.explain, a.temperature, problem, taxonomy_path=a.taxonomy, knowledge_path=a.knowledge,
                    category_overrides=a.category,
                    exclude_categories=[value.strip() for value in a.exclude_categories.split(',') if value.strip()],
                    rules_limit=a.rules_limit, show_detect_text=a.show_detect_text,
                    rule_bytes=a.rule_bytes, context_tokens=a.context_tokens,
                    max_tokens=a.max_tokens, log_sink=write, output_dir=a.output_dir, **grade_options)
                final_scores.append(records[-1]['score'])
                solved += records[-1]['score'] == 1.
            write(dict(event='summary', solved=solved, repeat=a.repeat, final_scores=final_scores))
        except Exception as error:
            write(dict(event='error', type=type(error).__name__, message=str(error)))
            raise
    print(f'SOLVED {solved}/{a.repeat}; final_scores={final_scores}; log={log_path}', flush=True)
    return dict(a=a, exp=exp, spec=spec, problem=problem, grade_options=grade_options)


def main():
    return checker.run_logged(_main)


if __name__ == '__main__':
    globals().update(main())
