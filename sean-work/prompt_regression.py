"""Capture/replay prompts using real JSONL attempts; never makes model requests."""
import argparse
import collections
import importlib.util
import json
import pathlib
import shutil
import tempfile
import sys
sys.dont_write_bytecode = True
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent
BEFORE = ROOT / 'prompts/regression/before'
AFTER = ROOT / 'prompts/regression/after'


def load_cases():
    cases = json.loads((BEFORE / 'inputs.json').read_text())
    for case in cases:
        for option in ('taxonomy_path', 'knowledge_path'):
            value = case['options'].get(option)
            if value:
                case['options'][option] = str(BEFORE / pathlib.Path(value).name)
    return cases


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def serialize(messages):
    return json.dumps(messages, ensure_ascii=False, indent=2) + '\n'


def write_messages(directory, case, messages):
    (directory / (case['id'] + '.json')).write_text(serialize(messages))
    (directory / (case['id'] + '.txt')).write_text(messages[0]['content'])


def capture():
    if (BEFORE / 'inputs.json').exists():
        raise RuntimeError('before fixtures already exist; refusing to overwrite regression evidence')
    current = module(BEFORE / 'source_agent.py', 'prompt_before_current')
    legacy = module(BEFORE / 'source_agent_legacy.py', 'prompt_before_legacy')
    v1 = module(BEFORE / 'source_agent_v1.py', 'prompt_before_v1')
    import checker
    import taxonomy
    cases, representatives, coverage = [], {}, collections.Counter()
    for source in sorted((ROOT / 'archive').rglob('*.jsonl')):
        records = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
        records = [r for r in records if 'run' in r and 'round' in r and 'code' in r and 'score' in r]
        if not records:
            continue
        groups = collections.defaultdict(list)
        for record in records:
            groups[record['run']].append(record)
        sequence = source.name.startswith('sequence_generator_')
        exp = records[0].get('exp', '')
        condition = 'explain_principles_v2' if '_v2' in exp else 'explain_principles' if 'principles' in exp else 'explain' if 'explain' in exp else 'no_flags'
        explain = 'explain' in condition
        style = 'legacy' if exp or sequence else 'legacy_v1'
        if sequence:
            zero = next(r for r in records if r['round'] == 0)
            spec = zero['messages'][0]['content']
            assert spec.endswith(legacy.CODE_ONLY)
            spec = spec[:-len(legacy.CODE_ONLY)]
        else:
            spec = legacy.SPECS['mine'].format(extra='') if exp else v1.SPEC.format(extra='')
        for run, rows in sorted(groups.items()):
            rows.sort(key=lambda row: row['round'])
            history = []
            for row in rows:
                # Required round 0/1 cases, plus real ledger/repeat cases after that.
                include = row['round'] <= 1 or (row['round'] == 2 and run == min(groups))
                if include:
                    case = dict(id=f'historical_{source.stem}_run{run}_round{row["round"]}',
                        layout=style, source=str(source.relative_to(ROOT)), run=run,
                        round=row['round'], condition=condition, spec=spec,
                        problem={'module': 'sequence_generator' if sequence else 'lfsr', 'ports': []},
                        history=list(history), options=dict(explain=explain, show_code=True))
                    messages = (legacy.build_prompt(spec, history, True, explain) if style == 'legacy'
                                else v1.build_prompt(spec, history))
                    if 'messages' in row:
                        assert messages == row['messages'], f'logged prompt reconstruction differs: {case["id"]}'
                    write_messages(BEFORE, case, messages)
                    cases.append(case)
                    coverage[condition] += 1
                if row['round'] == 0 and any(r['round'] == 1 for r in rows):
                    representatives[(condition, sequence)] = (source, row)
                history.append([row['code'], row['score'], row['feedback']])
    required = {'no_flags', 'explain', 'explain_principles', 'explain_principles_v2'}
    assert required <= set(coverage), coverage
    # The current bounded layout is also captured under all four flag conditions.
    # Its current diagnostics/feedback are regenerated from real logged Verilog.
    with tempfile.TemporaryDirectory(prefix='prompt_capture_') as temporary:
        directory = pathlib.Path(temporary)
        with mock.patch.object(checker, 'BUILD', directory):
            for (condition, sequence), (source, row) in representatives.items():
                p = checker.load_problem(ROOT / ('sequence_generator.md' if sequence else 'LFSR.md'))
                p['spec'] = str(ROOT / ('sequence_generator.md' if sequence else 'LFSR.md'))
                spec = current.public_spec(p)
                candidate = directory / ('sequence_generator.v' if sequence else 'lfsr.v')
                candidate.write_text(row['code'])
                diagnostics = {}
                detectors = condition == 'explain_principles_v2'
                score, feedback = checker.grade(p, candidate, diagnostics=diagnostics, detectors=detectors)
                applicable = taxonomy.categories(spec, p['ports'], BEFORE / 'taxonomy.json')
                definitions = taxonomy.read(BEFORE / 'taxonomy.json')
                knowledge = taxonomy.read(BEFORE / 'knowledge.json')
                for rnd in (0, 1):
                    history = [] if rnd == 0 else [dict(code=row['code'], score=score, feedback=feedback, diagnostics=diagnostics)]
                    signatures = diagnostics['signatures'] if rnd else []
                    rules = taxonomy.select(BEFORE / 'taxonomy.json', applicable, signatures,
                            data=definitions, knowledge=knowledge, limit=2,
                            max_bytes=min(1200, max(0, 5988 - current._bytes(spec) - 700))) if 'principles' in condition else []
                    case = dict(id=f'current_{"sequence_generator" if sequence else "lfsr"}_{condition}_round{rnd}',
                        layout='bounded', source=str(source.relative_to(ROOT)), run=row['run'], round=rnd,
                        condition=condition, spec=spec, problem=p, history=history,
                        options=dict(explain=condition != 'no_flags', show_code=True,
                            principles='principles' in condition, detectors=detectors,
                            taxonomy_path=str(BEFORE / 'taxonomy.json'), knowledge_path=str(BEFORE / 'knowledge.json'),
                            context_tokens=8000, max_tokens=1500))
                    messages = current.build_prompt(spec, [(r['code'], r['score'], r['feedback']) for r in history],
                               True, condition != 'no_flags', rules=rules, max_input_bytes=5988)
                    write_messages(BEFORE, case, messages)
                    cases.append(case)
    (BEFORE / 'inputs.json').write_text(json.dumps(cases, indent=2) + '\n')
    (BEFORE / 'manifest.json').write_text(json.dumps(dict(cases=len(cases), historical_coverage=coverage,
        note='Historical prompts reconstructed using archived source and original feedback; current-layout prompts use current checker feedback for real archived attempts.'), indent=2) + '\n')
    print(f'Saved {len(cases)} prompt cases before source changes; coverage={dict(coverage)}')


def replay():
    import checker
    cases = load_cases()
    AFTER.mkdir(exist_ok=True)
    generated = {case['id'] + suffix for case in cases for suffix in ('.json', '.txt')}
    for source in BEFORE.iterdir():
        if source.is_file() and source.name not in generated:
            shutil.copy2(source, AFTER / source.name)
    differences = []
    for case in cases:
        problem = checker.prompt_problem(case['problem'], spec=case['spec'],
                    layout=case['layout'], show_detect_text=True, **case['options'])
        messages = checker.build_prompt(problem, case['history'])
        write_messages(AFTER, case, messages)
        for suffix in ('.json', '.txt'):
            name = case['id'] + suffix
            if (BEFORE / name).read_bytes() != (AFTER / name).read_bytes():
                differences.append(name)
    report = dict(cases=len(cases), compared_files=2*len(cases), differences=differences,
                  result='byte-identical' if not differences else 'FAILED')
    report_path = ROOT / 'log/history/prompt_refactor_regression.json'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))
    return bool(differences)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('before', 'after'))
    args = parser.parse_args()
    if args.stage == 'before':
        capture()
    else:
        raise SystemExit(replay())
