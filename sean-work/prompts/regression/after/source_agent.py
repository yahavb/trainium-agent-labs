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
import taxonomy
from checker import grade

ROOT = pathlib.Path(__file__).resolve().parent
VERSION = 'md_generic_v4_dynamic_taxonomy'
BASE = os.environ.get('KERNEL_AGENT_BASE_URL', 'http://localhost:8000/v1').rstrip('/')
BASE = BASE if BASE.endswith('/v1') else BASE + '/v1'
CODE_ONLY = '\nReturn only the complete Verilog module in one code block.'
EXTRA_HINT = ' Reset is asynchronous and takes effect without a clock edge.'


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


def extract(text):
    blocks = re.findall(r'```(?:verilog|systemverilog|v)?\s*\n(.*?)```', text, re.S)
    blocks = [block for block in blocks if re.search(r'\bmodule\b', block)]
    if blocks:
        return blocks[-1]
    return text if re.search(r'\bmodule\b', text) and 'endmodule' in text else None


def _bytes(text):
    return len(text.encode('utf-8'))


def _fit(text, budget):
    return text.encode('utf-8')[:max(0, budget)].decode('utf-8', errors='ignore')


def build_prompt(spec, history, show_code=True, explain=False, *, rules=(), max_input_bytes=None):
    """Only the latest attempt enters context; full history remains in the log.

    UTF-8 bytes conservatively bound byte-token input length. Reserve generation
    tokens and chat overhead outside this budget; never truncate the specification.
    """
    if not history:
        content = spec + CODE_ONLY
    else:
        code, score, feedback = history[-1]
        instruction = ('First, for each failure, say in one sentence which part of the design causes it. '
                       'Then end with the complete corrected Verilog module in one code block.' if explain else
                       'Fix the failures and return the complete corrected Verilog module in one code block.')
        content = spec + '\n\n' + instruction
        rule_text = '\n'.join(taxonomy.entry_text(entry) for entry in rules)
        if rule_text:
            content += '\n\nRules selected for the current failures:\n' + rule_text
        header = f'\n\nLatest checker result (score {score:.2f}):\n'
        available = max_input_bytes - _bytes(content + header) if max_input_bytes is not None else 4000
        if available < 160:
            raise ValueError('specification and selected rules exceed the prompt budget; increase --context-tokens or shorten the specification')
        feedback_budget = min(3000, available)
        visible = _fit(feedback, feedback_budget)
        if visible != feedback:
            visible = _fit(visible, feedback_budget - 60) + '\n[More feedback is recorded in the log.]'
        content += header + visible
        if show_code and code:
            header = '\n\nPrevious attempt:\n```verilog\n'
            end = '\n```'
            remaining = max_input_bytes - _bytes(content + header + end) if max_input_bytes is not None else _bytes(code)
            if remaining > 200:
                preview = code if _bytes(code) <= remaining else _fit(code, remaining - 80) + '\n// Excerpt: the full previous attempt is saved in the log.'
                content += header + preview + end
        if len(history) > 1 and history[-1][0].strip() == history[-2][0].strip():
            note = '\nThe last two attempts were identical; revise the logic.'
            if max_input_bytes is None or _bytes(content + note) <= max_input_bytes:
                content += note
    if max_input_bytes is not None and _bytes(content) > max_input_bytes:
        raise ValueError('the Markdown specification exceeds the input budget; increase --context-tokens or shorten it')
    return [{'role': 'user', 'content': content}]


def failing_checks(feedback):
    labels = re.findall(r'^(\w+): requirement:', feedback, re.M)
    return ', '.join(labels) if labels else feedback.splitlines()[0][:70]


def public_spec(problem):
    text = pathlib.Path(problem['spec']).read_text()
    interface = ', '.join(f'{direction} {name}' + (f' [{width - 1}:0]' if width > 1 else '')
                          for name, direction, width in problem['ports'])
    text += f"\nUse module name {problem['module']} with ports: {interface}."
    if problem.get('spec_addendum'):
        text += '\n' + problem['spec_addendum']
    return text


def _find_spec(name):
    return next((p for p in ROOT.glob('*.md') if p.stem.lower() == name.lower()), ROOT / (name + '.md'))


def problem_spec(problem, style='mine', hint=False):
    # Compatibility helper; specifications and clarifications now come from files.
    data = checker.load_problem(_find_spec(problem))
    return public_spec(data) + (EXTRA_HINT if hint else '')


def run_once(model, spec, rounds, tag, exp, show_code, explain, temperature,
             problem='lfsr', *, taxonomy_path=None, knowledge_path=None, category_overrides=(),
             exclude_categories=(), rules_limit=2,
             rule_bytes=1200, context_tokens=8000, max_tokens=1500, log_sink=None,
             output_dir=None, **grade_options):
    history, records, last_diagnostics = [], [], {}
    data = checker._problem(problem)
    name = data['module']
    taxonomy_path = pathlib.Path(taxonomy_path or ROOT / 'taxonomy.json')
    knowledge_path = pathlib.Path(knowledge_path or ROOT / 'knowledge.json')
    input_budget = context_tokens - max_tokens - 512
    if input_budget < 512:
        raise ValueError('context budget must exceed output budget by at least 1024 tokens')
    outdir = pathlib.Path(output_dir or 'candidates/gen') / exp / name
    outdir.mkdir(parents=True, exist_ok=True)
    inject_rules = grade_options.get('principles', False)
    # The agent retrieves taxonomy itself; checker feedback contains observations.
    checker_options = dict(grade_options, principles=False)
    for rnd in range(rounds):
        definitions, taxonomy_digest = taxonomy.snapshot(taxonomy_path)
        knowledge, knowledge_digest = taxonomy.snapshot(knowledge_path)
        applicable = taxonomy.categories(spec, data['ports'], taxonomy_path, category_overrides,
                                         data=definitions, exclude=exclude_categories)
        signatures = last_diagnostics.get('signatures', [])
        selected = taxonomy.select(taxonomy_path, applicable, signatures,
                    limit=rules_limit, data=definitions, knowledge=knowledge,
                    max_bytes=min(rule_bytes, max(0, input_budget - _bytes(spec) - 700))) if inject_rules else []
        messages = build_prompt(spec, history, show_code, explain, rules=selected,
                                max_input_bytes=input_budget)
        if log_sink:
            log_sink(dict(event='request', run=tag, round=rnd, model=model,
                          messages=messages, categories=applicable, signatures=signatures,
                          selected_rules=selected, taxonomy_sha256=taxonomy_digest, knowledge_sha256=knowledge_digest,
                          prompt_byte_budget=input_budget))
        start = time.monotonic()
        reply = ask(messages, model, temperature, max_tokens=max_tokens)
        elapsed = time.monotonic() - start
        code = extract(reply)
        if log_sink:
            log_sink(dict(event='reply', run=tag, round=rnd, reply=reply, code=code,
                          elapsed_seconds=elapsed))
        candidate, verdict, diagnostics = None, None, {}
        if code is None:
            code = ''
            score, feedback = 0., 'format: requirement: return a complete module. Observed: no module found.'
            diagnostics['signatures'] = []
        else:
            path = outdir / f'{name}_{tag}_{rnd}.v'
            path.write_text(code)
            candidate = str(path)
            score, feedback = grade(problem, path, diagnostics=diagnostics, **checker_options)
            verdict = True if score == 1. and data.get('official') else checker.official(problem, path)
        same = bool(history) and code.strip() == history[-1][0].strip()
        record = dict(event='round', version=VERSION, exp=exp, run=tag, round=rnd, score=score,
            feedback=feedback, code=code, reply=reply, same_as_previous=same,
            problem=name, model=model, official_pass=verdict, candidate=candidate,
            messages=messages, elapsed_seconds=elapsed, temperature=temperature,
            max_tokens=max_tokens, thinking=False, context_tokens=context_tokens,
            prompt_utf8_bytes=sum(_bytes(message['content']) for message in messages),
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
        history.append((code, score, feedback))
        last_diagnostics = diagnostics
        print(f'run {tag} round {rnd}: {score:.2f}; checks={failing_checks(feedback)}; '
              f'rules={record["selected_rule_keys"]}; input_bytes={record["prompt_utf8_bytes"]}/{input_budget}', flush=True)
        print(feedback, flush=True)
        if score == 1.:
            break
    return records


def main():
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
    reset = ap.add_mutually_exclusive_group()
    reset.add_argument('--async-reset', dest='async_reset', action='store_true')
    reset.add_argument('--no-async-reset', dest='async_reset', action='store_false')
    ap.set_defaults(async_reset=None)
    ap.add_argument('--taxonomy', default=str(ROOT / 'taxonomy.json'))
    ap.add_argument('--knowledge', default=str(ROOT / 'knowledge.json'))
    ap.add_argument('--exclude-categories', default='', help='comma-separated categories excluded from retrieval')
    ap.add_argument('--category', action='append', default=[], help='override or supplement inferred categories')
    ap.add_argument('--rules-limit', type=int, default=2)
    ap.add_argument('--rule-bytes', type=int, default=1200)
    ap.add_argument('--context-tokens', type=int, default=8000)
    ap.add_argument('--max-tokens', type=int, default=1500)
    ap.add_argument('--temperature', type=float, default=.7)
    ap.add_argument('--model')
    ap.add_argument('--base-url')
    ap.add_argument('--log-dir', default='logs')
    ap.add_argument('--run-name', default=VERSION)
    ap.add_argument('--output-dir', default='candidates/gen')
    a = ap.parse_args()
    if a.rounds < 1 or a.repeat < 0 or a.rules_limit < 1 or a.cycles < 1 or a.random_cycles < 1:
        ap.error('rounds, rules-limit, and cycle counts must be positive; repeat must be nonnegative')
    if a.problem and a.spec and a.spec not in ('mine', 'bench'):
        ap.error('use --problem or --spec FILE')
    style = a.spec if a.spec in ('mine', 'bench') else 'mine'
    spec_file = _find_spec(a.problem or 'lfsr') if not a.spec or a.spec in ('mine', 'bench') else pathlib.Path(a.spec)
    try:
        problem = checker.load_problem(spec_file, a.tb, a.ref, seed=a.seed,
                    async_reset=a.async_reset, cycles=a.cycles, random_cycles=a.random_cycles)
        taxonomy.read(a.taxonomy)
        taxonomy.read(a.knowledge)
        spec = public_spec(problem)
    except (OSError, ValueError) as error:
        ap.error(str(error))
    a.problem = problem['module']
    a.principles = a.principles or a.principles_v2
    a.detectors = a.detectors or a.principles_v2
    if a.hint or a.async_reset is True:
        spec += EXTRA_HINT
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
    directory = pathlib.Path(a.log_dir)
    directory.mkdir(parents=True, exist_ok=True)
    log_path = directory / (run_id + '.jsonl')
    solved, final_scores = 0, []
    with log_path.open('w') as logfile:
        def write(record):
            logfile.write(json.dumps(record) + '\n')
            logfile.flush()
        write(dict(event='start', version=VERSION, run_id=run_id, arguments=vars(a),
                   spec_path=str(spec_file), problem=problem['module'], base_url=BASE))
        try:
            model = a.model or http('/models')['data'][0]['id']
            print(f'problem={a.problem}; model={model}; log={log_path}', flush=True)
            for run in range(a.repeat):
                records = run_once(model, spec, a.rounds, run, run_id, not a.no_code,
                    a.explain, a.temperature, problem, taxonomy_path=a.taxonomy, knowledge_path=a.knowledge,
                    category_overrides=a.category,
                    exclude_categories=[value.strip() for value in a.exclude_categories.split(',') if value.strip()],
                    rules_limit=a.rules_limit,
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


if __name__ == '__main__':
    globals().update(main())
