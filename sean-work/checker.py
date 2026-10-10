"""
checker.py - grades a candidate Verilog design and explains what is wrong.

How it grades: it generates its own probe testbench, drives the design, and compares
every output against a Python model or a Verilog reference driven by the same inputs.
A 1.0 also requires the official testbench to pass when one is available.

What the feedback says: for each failed check, the REQUIREMENT that check enforces and
what was OBSERVED (signal, time or cycle, which bits are wrong). It never prints the
expected value and never suggests code. The reference design and reference model are
used only to grade and to self-test the checker; the model never sees them.

Problem definitions supply ports, reset behavior, reference models, stimulus
segments, feedback requirements, and weights. The probe and grader interpret this
data without dispatching to a problem-specific checker.
"""

import copy, json, random, re, subprocess, pathlib, sys, time, hashlib, contextlib, shutil
from datetime import datetime, timezone
import taxonomy

# ---- model-facing prompt construction; the agent only transports messages ----


ROOT = pathlib.Path(__file__).resolve().parent


CODE_ONLY = '\nReturn only the complete Verilog module in one code block.'


EXTRA_HINT = ' Reset is asynchronous and takes effect without a clock edge.'


LEGACY_SPECS = {
    "mine": """Write a Verilog module named `lfsr` for an 8-bit linear feedback shift register.
Ports: input clk, input reset_n (active low), output reg [7:0] data.
On reset_n low, data is set to 8'b10001010.{extra}
On each rising clock edge with reset_n high, shift data left by one and set the new
bit 0 to the XOR of the old bits 0, 3, 5 and 6.""",
    "bench": """I am trying to create a Verilog model for an LFSR. It must meet the following specifications:
- Inputs: Clock, Active-low reset
- Outputs: Data (8-bits)
The initial state should be 10001010, and the taps should be at locations 1, 4, 6, and 7.{extra}
Use module name `lfsr` with ports: input clk, input reset_n, output reg [7:0] data.""",
}


LEGACY_CODE_ONLY = "\nReturn only the Verilog module in one code block."


LEGACY_V1_SPEC = """Write a Verilog module named `lfsr` for an 8-bit linear feedback shift register.
Ports: input clk, input reset_n (active low), output reg [7:0] data.
On reset_n low, data is set to 8'b10001010.{extra}
On each rising clock edge with reset_n high, shift data left by one and set the new
bit 0 to the XOR of the old bits 0, 3, 5 and 6.
Return only the Verilog module in one code block."""


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


def _bounded_prompt(spec, history, show_code=True, explain=False, *, rules=(), max_input_bytes=None,
                    show_detect_text=False):
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
        rule_text = '\n'.join(taxonomy.entry_text(entry, show_detect_text=show_detect_text) for entry in rules)
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


def _legacy_prompt(spec, history, show_code=True, explain=False):
    if not history:                      # round 0 is identical in every condition
        return [{"role": "user", "content": spec + LEGACY_CODE_ONLY}]
    code, score, fb = history[-1]
    msg = spec
    if show_code:
        msg += f"\n\nYour previous attempt:\n```verilog\n{code}\n```"
    msg += f"\n\nA checker tested your previous attempt (score {score:.2f}):\n{fb}\n"
    if len(history) > 1:
        msg += "\nEarlier feedback:\n" + "\n".join(
            f"- {h[2].splitlines()[0]}" for h in history[:-1])
    if len(history) > 1 and history[-1][0].strip() == history[-2][0].strip():
        msg += ("\nYour last two attempts were identical and both failed. "
                "Change the logic, not just the formatting.")
    if explain:
        msg += ("\nFirst, for each failure above, say in one sentence which part of the design "
                "causes it. Then end with the complete corrected Verilog module in one code block.")
    else:
        msg += "\nFix the problems and return only the complete corrected Verilog module in one code block."
    return [{"role": "user", "content": msg}]


def _legacy_v1_prompt(spec, history):
    msg = spec
    if history:
        code, score, fb = history[-1]
        msg += f"\n\nYour previous attempt:\n```verilog\n{code}\n```\nResult (score {score:.2f}):\n{fb}\n"
        if len(history) > 1:
            msg += "\nEarlier feedback you already received:\n" + "\n".join(
                f"- {h[2].splitlines()[0]}" for h in history[:-1])
        msg += "\nFix the problem and return the complete corrected module."
        if len(history) > 1 and history[-1][0].strip() == history[-2][0].strip():
            msg += ("\nYour last two attempts were identical and both failed. "
                    "Change the logic, not just the formatting.\n")
    return [{"role": "user", "content": msg}]


def spec_path(path):
    """Resolve explicit specs and preserve aliases for the bundled specs we moved."""
    candidate = pathlib.Path(path)
    if candidate.is_file():
        return candidate.resolve()
    if not candidate.is_absolute() and candidate.parent != pathlib.Path('.'):
        return candidate.resolve()
    if candidate.is_absolute() and candidate.parent != ROOT:
        return candidate
    return next((p for p in (ROOT / 'prompts/specs').glob('*.md')
                 if p.name.lower() == candidate.name.lower()), candidate.resolve())


def public_spec(problem, *, hint=False, interface_from_tb=False):
    text = spec_path(problem['spec']).read_text()
    text += '\n' + interface_line(problem, generated=interface_from_tb)
    if problem.get('spec_addendum'):
        text += '\n' + problem['spec_addendum']
    return text + (EXTRA_HINT if hint else '')


def _find_spec(name):
    return next((p for directory in (ROOT / 'prompts/specs', ROOT)
                 for p in directory.glob('*.md') if p.stem.lower() == name.lower()), ROOT / (name + '.md'))


def problem_spec(problem, style='mine', hint=False):
    # Compatibility helper; specifications and clarifications now come from files.
    data = load_problem(_find_spec(problem))
    return public_spec(data) + (EXTRA_HINT if hint else '')

def interface_line(problem, *, generated=False):
    """The default formatting is unchanged; TB-derived formatting is opt-in."""
    p = _problem(problem)
    ports, module = p['ports'], p['module']
    if generated:
        if not p.get('official'):
            raise ValueError('--interface-from-tb requires a testbench')
        ref = p.get('ref_verilog') or next((path for path in p.get('references', ())
                                           if pathlib.Path(path).exists()), None)
        if not ref:
            raise ValueError('--interface-from-tb requires an independent reference')
        parsed = file_problem(p['spec'], p['official'], ref)
        ports, module = parsed['ports'], parsed['module']
    interface = ', '.join(f'{direction} {name}' + (f' [{width - 1}:0]' if width > 1 else '')
                          for name, direction, width in ports)
    return f"Use module name {module} with ports: {interface}."


def prompt_problem(problem, *, spec=None, layout='bounded', **options):
    """Attach checker-owned prompt options without changing the grading data."""
    p = dict(_problem(problem))
    p['_prompt_options'] = dict(options, layout=layout)
    p['_prompt_spec'] = spec if spec is not None else public_spec(p,
        hint=options.get('hint', False), interface_from_tb=options.get('interface_from_tb', False))
    return p


def build_prompt(problem, history, explain=False, show_code=True):
    """Build checker-owned messages, including live KB lookup.

    Historical layouts preserve archived experiments verbatim. New Markdown runs
    use bounded context and show principles alone unless detect text is requested.
    """
    p = _problem(problem)
    options = p.get('_prompt_options', {})
    spec = p.get('_prompt_spec')
    if spec is None:
        spec = public_spec(p)
    explain = explain or options.get('explain', False)
    show_code = show_code and options.get('show_code', True)
    tuples = [(entry['code'], entry['score'], entry['feedback']) if isinstance(entry, dict)
              else tuple(entry) for entry in history]
    layout = options.get('layout', 'bounded')
    if layout == 'legacy':
        return _legacy_prompt(spec, tuples, show_code, explain)
    if layout == 'legacy_v1':
        return _legacy_v1_prompt(spec, tuples)
    input_budget = options.get('context_tokens', 8000) - options.get('max_tokens', 1500) - 512
    if input_budget < 512:
        raise ValueError('context budget must exceed output budget by at least 1024 tokens')
    taxonomy_path = pathlib.Path(options.get('taxonomy_path') or ROOT / 'taxonomy.json')
    knowledge_path = pathlib.Path(options.get('knowledge_path') or ROOT / 'knowledge.json')
    definitions, taxonomy_digest = taxonomy.snapshot(taxonomy_path)
    knowledge, knowledge_digest = taxonomy.snapshot(knowledge_path)
    excluded = options.get('exclude_categories', ())
    applicable = taxonomy.categories(spec, p['ports'], taxonomy_path,
        options.get('category_overrides', ()), data=definitions, exclude=excluded)
    diagnostics = history[-1].get('diagnostics', {}) if history and isinstance(history[-1], dict) else {}
    signatures = diagnostics.get('signatures', [])
    selected = taxonomy.select(taxonomy_path, applicable, signatures,
        limit=options.get('rules_limit', 2), data=definitions, knowledge=knowledge,
        show_detect_text=options.get('show_detect_text', False),
        max_bytes=min(options.get('rule_bytes', 1200), max(0, input_budget - _bytes(spec) - 700))) \
        if options.get('principles', False) else []
    messages = _bounded_prompt(spec, tuples, show_code, explain, rules=selected,
                               max_input_bytes=input_budget,
                               show_detect_text=options.get('show_detect_text', False))
    p['_prompt_audit'] = dict(categories=applicable, signatures=signatures, selected_rules=selected,
        taxonomy_sha256=taxonomy_digest, knowledge_sha256=knowledge_digest,
        taxonomy_path=str(taxonomy_path), knowledge_path=str(knowledge_path),
        prompt_byte_budget=input_budget, prompt_utf8_bytes=prompt_size(messages),
        excluded_categories=list(excluded))
    return messages


def prompt_audit(problem):
    return dict(problem['_prompt_audit'])


def prompt_size(messages):
    return sum(_bytes(message['content']) for message in messages)


def feedback_options(problem, supplied):
    # Principles belong to the next prompt; grade() continues to report observations.
    return dict(supplied, principles=False)


def no_module_result(diagnostics):
    diagnostics['signatures'] = []
    return 0., 'format: requirement: return a complete module. Observed: no module found.'


def validate_knowledge(taxonomy_path, knowledge_path):
    taxonomy.read(taxonomy_path)
    taxonomy.read(knowledge_path)


def save_prompt_event(record, source_log):
    """Store exact logged messages separately without changing the JSONL schema."""
    if not record.get('messages') or 'round' not in record:
        return
    source = pathlib.Path(source_log)
    run_name = source.parent.name if source.stem == 'events' else source.stem
    directory = ROOT / 'prompts/runs' / run_name
    directory.mkdir(parents=True, exist_ok=True)
    name = f"run_{record.get('run', 0)}_round_{record['round']}"
    (directory / (name + '.json')).write_text(json.dumps(record['messages'], ensure_ascii=False, indent=2) + '\n')
    (directory / (name + '.txt')).write_text('\n\n'.join(message['content'] for message in record['messages']))
    (directory / 'source.json').write_text(json.dumps(dict(source_log=str(source)), indent=2) + '\n')


def run_logged(callback, argv=None):
    """Mirror plain console output to log/NAME.log; preserve original stdout."""
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    for option in ('problem', 'spec', 'tb', 'log-dir'):
        parser.add_argument('--' + option)
    parser.add_argument('--run-all', action='store_true')
    parser.add_argument('--knowledge-report', action='store_true')
    parser.add_argument('--no-console-log', action='store_true')
    options, _ = parser.parse_known_args(argv)
    if options.run_all or options.no_console_log or any(flag in (sys.argv[1:] if argv is None else argv) for flag in ('-h', '--help')):
        return callback()
    name = options.problem or pathlib.Path(options.spec or options.tb or 'lfsr').stem.lower()
    if options.tb and not options.problem and not options.spec:
        name = name.removesuffix('_tb')
    if options.knowledge_report:
        name = 'knowledge_report'
    name = re.sub(r'[^\w.-]+', '_', name)
    directory = pathlib.Path(options.log_dir or ROOT / 'log')
    directory.mkdir(parents=True, exist_ok=True)
    class Tee:
        def __init__(self, console, stream):
            self.console, self.stream = console, stream
        def write(self, text):
            self.stream.write(text)
            return self.console.write(text)
        def flush(self):
            self.stream.flush()
            self.console.flush()
    with (directory / (name + '.log')).open('w') as stream, \
         contextlib.redirect_stdout(Tee(sys.stdout, stream)), \
         contextlib.redirect_stderr(Tee(sys.stderr, stream)):
        return callback()


BUILD = pathlib.Path("build"); BUILD.mkdir(exist_ok=True)


def _principle_lines(v1_ids, v2_ids=(), *, principles=False, detectors=False,
                     taxonomy_path=None, knowledge_path=None, categories=(), max_rules=2):
    """Share one two-line budget, with v1 taking priority over v2."""
    if not principles:
        return []
    ids = list(v1_ids) + (list(v2_ids) if detectors else [])
    if not ids:
        return []
    if taxonomy_path:
        import taxonomy
        definitions = taxonomy.read(taxonomy_path)
        knowledge = taxonomy.read(knowledge_path or pathlib.Path(__file__).with_name('knowledge.json'))
        return [taxonomy.entry_text(entry) for entry in
                taxonomy.select(taxonomy_path, categories, ids, limit=max_rules,
                                data=definitions, knowledge=knowledge)]
    entries = json.loads(pathlib.Path(__file__).with_name("principles.json").read_text())
    principles = {entry["id"]: entry["principle"] for entry in entries}
    lines = []
    for signature in dict.fromkeys(ids):
        line = "Principle: " + principles[signature]
        if line not in lines:
            lines.append(line)
        if len(lines) == 2:
            break
    return lines


def _with_principles(feedback, v1_ids, v2_ids=(), **options):
    return "\n".join([feedback] + _principle_lines(v1_ids, v2_ids, **options))


# ---- problem data: the only per-problem behavior ----

def _lfsr_step(s):
    # Reference values are used only to grade, never included in feedback.
    return ((s << 1) & 0xFF) | (((s >> 0) ^ (s >> 3) ^ (s >> 5) ^ (s >> 6)) & 1)


def _sequence_next(state, inputs):
    position, value = state
    if not inputs["enable"]:
        return state
    return ((position + 1) % len(SEQUENCE), SEQUENCE[position])


SEQUENCE = (0xAF, 0xBC, 0xE2, 0x78, 0xFF, 0xE2, 0x0B, 0x8D)
RESET_REQUIREMENTS = {
    "reset_value": "while {rst} is {act}, {out} must equal the initial state from the spec",
    "reset_immediate": "while {rst} is {act}, {out} must equal the initial state at every moment, including before the first clock edge",
    "reset_midrun": "reset during a run must restore the initial state {reset_timing}, hold it across a clock edge, and restart the specified sequence after release",
}

PROBLEMS = {
    "lfsr": dict(
        module="lfsr",
        ports=(("clk", "input", 1), ("reset_n", "input", 1), ("data", "output", 8)),
        clock="clk", reset="reset_n", reset_active=0, async_reset=True,
        model=dict(reset_state=0x8A, next_state=lambda state, inputs: _lfsr_step(state),
                   output=lambda state: {"data": state}),
        initial_inputs={}, reset_hold_cycles=1, restart_cycles=4,
        segments=[dict(label="sequence", cycles=256, inputs={},
                       first_check="first_step",
                       requirement="every clock edge must advance {out} by one step of the LFSR in the spec",
                       observed="at cycle {cycle} after release, {out} is {description}.")],
        requirements=dict(RESET_REQUIREMENTS, first_step=
            "on each clock edge after {rst} is released, {out} must advance by one step of the LFSR in the spec"),
        observations=dict(first_step="on the first edge after release, {out} is {description}."),
        weights=dict(compiles=.1, reset_value=.15, reset_immediate=.15,
                     reset_midrun=.2, first_step=.15, sequence=.25),
        official="TestBench/lfsr_tb.v", spec="LFSR.md",
        references=("reference/lfsr.v", "candidates/lfsr.v"),
    ),
    "sequence_generator": dict(
        module="sequence_generator",
        ports=(("clk", "input", 1), ("reset_n", "input", 1),
               ("enable", "input", 1), ("data", "output", 8)),
        clock="clk", reset="reset_n", reset_active=0, async_reset=True,
        # Behavior A: reset shows the first item; each enabled edge advances.
        model=dict(reset_state=(1, SEQUENCE[0]), next_state=_sequence_next,
                   output=lambda state: {"data": state[1]}),
        initial_inputs={"enable": 0}, reset_hold_cycles=2, restart_cycles=11,
        segments=[
            dict(label="sequence", cycles=8, inputs={"enable": 1}, first_check="first_step",
                 requirement="enabled rising edges must advance through the specified sequence in order"),
            dict(label="repetition", cycles=16, inputs={"enable": 1},
                 requirement="the sequence must repeat in order across its boundary"),
            dict(label="disable", cycles=5, inputs={"enable": 0},
                 requirement="data and the sequence position must hold while enable is low"),
            dict(label="resume", cycles=16,
                 inputs=[{"enable": 1}] * 3 + [{"enable": 0}] * 5 + [{"enable": 1}] * 8,
                 requirement="enabling again must continue from the paused sequence position"),
            dict(label="reset_restart", cycles=11, inputs={"enable": 1},
                 reset_before={"enable": 1},
                 requirement="reset while enabled or disabled must restore the initial state and restart the sequence after release"),
            dict(label="reset_restart", cycles=11, inputs={"enable": 1},
                 reset_before={"enable": 0},
                 requirement="reset while enabled or disabled must restore the initial state and restart the sequence after release"),
        ],
        requirements=dict(RESET_REQUIREMENTS, first_step=
            "the first enabled rising edge after reset must advance from the initial sequence item"),
        observations=dict(first_step="on the first enabled edge after release, {out} is {description}."),
        weights=dict(compiles=.1, reset_value=.05, reset_immediate=.05, reset_midrun=.1,
                     first_step=.1, sequence=.15, repetition=.1, disable=.1,
                     resume=.1, reset_restart=.15),
        official="TestBench/sequence_generator_tb.v", spec="sequence_generator.md",
        references=("reference/seq/seq_a.v",),
    ),
}

FAIL_RE = re.compile(r"^\s*Error|completed with\s+\d+\s+errors", re.I)
PASS_RE = re.compile(r"All test cases passed|Simulation successful|completed successfully", re.I)


# ---- one probe generator and reference timeline for every problem ----

def _problem(problem_name):
    if isinstance(problem_name, dict):
        return problem_name
    # Accept historical testbench paths without keeping a second grading path.
    name = pathlib.Path(problem_name).stem
    if name.endswith("_tb"):
        name = name[:-3]
    return PROBLEMS[name]


def _probe(p):
    """Generate a probe and a private reference sample for every observation.

    All input assignments happen with the clock low. Every rising edge is sampled
    one ns later, including edges while reset is held. Reset assertions also get
    a sample before the next edge for asynchronous designs only.
    """
    clk, rst, act = p["clock"] or "_checker_clock", p["reset"], p["reset_active"]
    outputs = [(name, width) for name, direction, width in p["ports"] if direction == "output"]
    inputs = dict(p["initial_inputs"])
    model = p.get("model") if not p.get("ref_verilog") else None
    state = model["reset_state"] if model else None
    reference = bool(p.get("ref_verilog"))
    lines = ['`timescale 1ns/1ps', 'module probe;']
    if not p['clock']:
        lines.append('  reg _checker_clock = 0;')
    for name, direction, width in p["ports"]:
        size = f" [{width - 1}:0]" if width > 1 else ""
        if direction == "input":
            value = 0 if name == clk else 1 - act if name == rst else inputs[name]
            lines.append(f"  reg{size} {name} = {value};")
        else:
            lines.append(f"  wire{size} {name};")
            if reference:
                lines.append(f"  wire{size} _checker_ref_{name};")
    connections = ', '.join(f'.{name}({name})' for name, _, _ in p["ports"])
    lines.append(f'  {p["module"]} dut ({connections});')
    if reference:
        ref_connections = ', '.join(f'.{name}({name if direction == "input" else "_checker_ref_" + name})'
                                    for name, direction, _ in p["ports"])
        lines.append(f'  _checker_reference reference ({ref_connections});')
    lines.append('  initial begin')
    samples, now, edge_count = [], 0, 0
    transitions = {}

    def emit(text, delay=0):
        nonlocal now
        now += delay
        lines.append(f"    #{delay} {text}" if delay else f"    {text}")

    def drive(values):
        for name, value in values.items():
            if name not in inputs:
                raise ValueError(f"not a stimulus input: {name}")
            if inputs[name] != value:
                transitions[name] = (edge_count, value)
            inputs[name] = value
            emit(f"{name} = {value};")

    def sample(checks, when, *, kind="segment", cycle=0, subcheck=None,
               requirements=None, observed=None, previous=None, following=None):
        number = len(samples)
        names = [name for name, _ in outputs]
        if reference:
            names += ['_checker_ref_' + name for name, _ in outputs]
        fmt = " ".join("%b" for _ in names)
        args = ', '.join(names)
        emit(f'$display("P {number} {fmt}", {args});')
        context = [f"{edge_count - since} cycle{'s' if edge_count - since != 1 else ''} after {name} "
                   + ("went low" if value == 0 else "went high" if value == 1 else "changed")
                   for name, (since, value) in transitions.items()]
        samples.append(dict(checks=checks, want=model["output"](state) if model else {}, when=when,
                            time=now, kind=kind, cycle=cycle, subcheck=subcheck,
                            edge=edge_count, context=context, inputs=dict(inputs),
                            requirements=requirements or {}, observed=observed,
                            previous=model["output"](state if previous is None else previous) if model else {},
                            following=model["output"](state if following is None else following) if model else {}))

    def edge(checks, when, *, held=False, rise_delay=4, **metadata):
        nonlocal state, edge_count
        previous = state
        emit(f"{clk} = 1;", rise_delay)
        edge_count += 1
        if model:
            state = model["reset_state"] if held else model["next_state"](state, inputs)
        emit("", 1)
        following = (model["reset_state"] if held else model["next_state"](state, inputs)) if model else None
        sample(checks, when, previous=previous, following=following, **metadata)
        emit(f"{clk} = 0;", 4)

    def reset(check=None, initial=False, values=None):
        nonlocal state
        if values is not None:
            drive(values)
        if not rst:
            return
        emit(f"{rst} = {act};", 1)
        emit("", 1)
        if p["async_reset"]:
            if model:
                state = model["reset_state"]
            sample(("reset_immediate",) if initial else (check,),
                   "before the first clock edge" if initial else
                   f"at {now} ns, one ns after reset assertion with {clk} low and no clock edge",
                   kind="initial_immediate" if initial else "reset", subcheck=0)
        for i in range(p["reset_hold_cycles"]):
            # The first held edge is three ns after the pre-edge sample.
            edge(("reset_value",) if initial else (check,),
                 f"at {now + (3 if i == 0 else 4) + 1} ns, after a clock edge with reset held",
                 held=True, rise_delay=3 if i == 0 else 4,
                 kind="initial_value" if initial else "reset", subcheck=1)
        emit(f"{rst} = {1 - act};", 1)

    reset(initial=True)
    for segment in p["segments"]:
        label = segment["label"]
        requirement = {label: segment["requirement"]}
        if "reset_before" in segment:
            reset(label, values=segment["reset_before"])
        for i in range(segment["cycles"]):
            values = segment["inputs"]
            drive(values[i] if isinstance(values, (list, tuple)) else values)
            checks = ((segment["first_check"], label)
                      if i == 0 and "first_check" in segment else (label,))
            edge(checks, f"at probe cycle {i + 1}", cycle=i + 1,
                 requirements=requirement, observed=segment.get("observed"))
            emit("", 1)

    if not rst:
        lines.extend(['    $finish;', '  end', 'endmodule'])
        return '\n'.join(lines) + '\n', samples

    # Reset away from the reset state, even if the run ends exactly at a wrap.
    # Advancing inputs are data too; the engine does not know about enable.
    drive(p.get("midrun_inputs", p["segments"][-1]["inputs"]
                if isinstance(p["segments"][-1]["inputs"], dict) else inputs))
    if model:
        for _ in range(p.get("seek_limit", 256)):
            if state != model["reset_state"]:
                break
            edge((), "before midrun reset", kind="setup")
            emit("", 1)
        else:
            raise ValueError("reference never leaves the reset state")
    reset("reset_midrun")
    for i in range(1, p["restart_cycles"] + 1):
        edge(("reset_midrun",), f"at restart cycle {i} after reset release",
             kind="restart", cycle=i, subcheck=i + 1)
        emit("", 1)
    lines.extend(['    $finish;', '  end', 'endmodule'])
    return '\n'.join(lines) + '\n', samples


def _harness(p):
    return _probe(p)[0]


def _without_comments(source):
    return re.sub(r'//[^\n]*|/\*[\s\S]*?\*/', '', source)


def file_problem(spec, tb, ref, *, seed=0, async_reset=True, cycles=256, random_cycles=128):
    """Infer the interface from a named testbench instantiation, then add stimulus.

    Benchmark testbenches declare driven signals as reg/logic and DUT outputs as
    wire. A logic signal is an input only if the testbench assigns it. Parameter
    widths may use integer localparams; unsupported connections fail explicitly.
    """
    if cycles < 1 or random_cycles < 1:
        raise ValueError('stimulus cycle counts must be positive')
    source = _without_comments(pathlib.Path(tb).read_text())
    spec = spec_path(spec)
    spec_text = spec.read_text()
    constants = {name: int(value) for name, value in re.findall(
        r'\b(?:localparam|parameter)\s+(?:integer\s+)?(\w+)\s*=\s*(\d+)\s*;', source)}

    def bound(expression):
        expression = expression.strip()
        if expression.isdecimal():
            return int(expression)
        if expression in constants:
            return constants[expression]
        match = re.fullmatch(r'(\w+)\s*([+-])\s*(\d+)', expression)
        if match and match[1] in constants:
            return constants[match[1]] + (1 if match[2] == '+' else -1) * int(match[3])
        raise ValueError(f"unsupported testbench port width: {expression}")

    signals = {}
    for kind, size, declarations in re.findall(
            r'\b(reg|wire|logic)\s*(?:signed\s*)?(\[[^\]]+\])?\s*([^;]+);', source):
        width = 1
        if size:
            msb, lsb = size[1:-1].split(':')
            width = abs(bound(msb) - bound(lsb)) + 1
        for declaration in declarations.split(','):
            match = re.match(r'\s*(\w+)\b', declaration)
            if match:
                signals[match[1]] = (kind, width)

    candidates = []
    for module, instance, connections in re.findall(
            r'\b(\w+)\s+(?:#\s*\([^;]*?\)\s*)?(\w+)\s*\(([^;]*?)\)\s*;', source):
        if module in ('module', 'if', 'for', 'while', 'task', 'function'):
            continue
        named = re.findall(r'\.(\w+)\s*\(\s*([^()]*)\s*\)', connections)
        if named:
            candidates.append((module, instance, named))
    # Prefer the module exposed by the reference if the TB also instantiates helpers.
    ref_source = _without_comments(pathlib.Path(ref).read_text())
    ref_modules = re.findall(r'\bmodule\s+(\w+)', ref_source)
    matching = [candidate for candidate in candidates if candidate[0] in ref_modules]
    candidates = matching or candidates
    if len(candidates) != 1:
        raise ValueError("testbench must identify one DUT with named, signal-connected ports")
    module, _, connections = candidates[0]
    ports = []
    for port, signal in connections:
        signal = signal.strip()
        if signal not in signals:
            raise ValueError(f"testbench port {port} must connect to a declared signal")
        kind, width = signals[signal]
        driven = kind == 'reg' or (kind == 'logic' and re.search(
            r'\b' + re.escape(signal) + r'\s*(?:<=|=(?!=))', source))
        ports.append((port, 'input' if driven else 'output', width))
    input_names = [name for name, direction, _ in ports if direction == 'input']
    clocks = [name for name in input_names if re.search(r'clk|clock', name, re.I)]
    resets = [name for name in input_names if re.search(r'reset|rst|clear', name, re.I)]
    if len(clocks) > 1 or len(resets) > 1:
        raise ValueError("multiple clock/reset inputs need an explicit interface")
    clock, reset = clocks[0] if clocks else None, resets[0] if resets else None
    async_reset = async_reset and reset is not None
    active = 0 if re.search(r'active[ -]*low', spec_text, re.I) or re.search(r'_n$|^n_?reset|^n_?rst', reset or '', re.I) else 1
    stimulus_ports = [(name, width) for name, direction, width in ports
                      if direction == 'input' and name not in (clock, reset)]
    high = {name: (1 << width) - 1 for name, width in stimulus_ports}
    rng = random.Random(seed)
    segments = [
        dict(label='sequence', cycles=cycles, inputs=high, first_check='first_step',
             requirement='every output must follow the specified behavior on each rising edge'),
        dict(label='random', cycles=random_cycles,
             inputs=[{name: rng.getrandbits(width) for name, width in stimulus_ports} for _ in range(random_cycles)],
             requirement='every output must follow the specified behavior for the sampled inputs'),
    ]
    for name, _ in stimulus_ports:
        for value, cycles in ((0, 5), (high[name], 12)):
            segments.append(dict(label='toggles', cycles=cycles, inputs=dict(high, **{name: value}),
                                 requirement='input changes and holds must follow the specified behavior'))
    for enabled in ((high, dict.fromkeys(high, 0)) if reset else ()):
        segments.append(dict(label='reset_restart', cycles=11, inputs=high, reset_before=enabled,
                             requirement='reset with either input level must restore and restart the specified behavior'))
    weights = dict(compiles=.1, reset_value=.1)
    if async_reset:
        weights['reset_immediate'] = .1
    else:
        weights['reset_value'] += .1
    weights.update(reset_midrun=.1, first_step=.1, sequence=.15, random=.1,
                   toggles=.15, reset_restart=.1)
    if not reset:
        for key in ('reset_value', 'reset_immediate', 'reset_midrun', 'reset_restart'):
            weights['sequence'] += weights.pop(key, 0)
    # With no stimulus ports, there are no toggle checks to weight.
    if not stimulus_ports:
        weights['random'] += weights.pop('toggles')
    return dict(module=module, ports=tuple(ports), clock=clock, reset=reset,
                reset_active=active, async_reset=async_reset,
                ref_verilog=str(ref), spec=str(spec), official=str(tb),
                initial_inputs=dict.fromkeys(high, 0), midrun_inputs=high,
                reset_hold_cycles=2, restart_cycles=11,
                segments=segments, requirements=dict(RESET_REQUIREMENTS,
                    first_step='the first rising edge after reset must follow the specified behavior'),
                observations={}, weights=weights, include_context=True, seed=seed)


def _reference_copy(p, tag):
    """Give the reference top a private name so identical DUT module names coexist."""
    source = _without_comments(pathlib.Path(p['ref_verilog']).read_text())
    modules = re.findall(r'\bmodule\s+(\w+)', source)
    if not modules:
        raise ValueError("reference must declare a module")
    name = p.get('ref_module', p['module'] if p['module'] in modules else modules[0])
    source = re.sub(r'(\bmodule\s+)' + re.escape(name) + r'\b',
                    r'\1_checker_reference', source, count=1)
    path = BUILD / f'{tag}_reference.v'
    path.write_text(source)
    return path


def _interface_testbench(ref):
    """Read simple ANSI or classic module ports when no official TB is supplied."""
    source = _without_comments(pathlib.Path(ref).read_text())
    match = re.search(r'\bmodule\s+(\w+)\s*\((.*?)\)\s*;', source, re.S)
    if not match:
        raise ValueError('provide --tb for a reference with a parameterized or complex interface')
    module, header = match.groups()
    ports = []
    direction, width = None, ''
    if re.search(r'\b(input|output)\b', header):
        for item in header.split(','):
            declared = re.match(r'\s*(input|output)\s+(?:(?:wire|reg|logic|signed)\s+)*(\[[^]]+\])?\s*(\w+)\s*$', item)
            if declared:
                direction, width, name = declared.groups()
                width = width or ''
            else:
                name = item.strip()
                if not direction or not re.fullmatch(r'\w+', name):
                    raise ValueError('provide --tb for this reference interface')
            ports.append((name, direction, width))
    else:
        for direction, width, names in re.findall(
                r'\b(input|output)\s+(?:(?:wire|reg|logic|signed)\s+)*(\[[^]]+\])?\s*([^;]+);', source):
            ports.extend((name.strip(), direction, width or '') for name in names.split(','))
    if not ports:
        raise ValueError('no reference ports found; provide --tb')
    lines = ['module interface_description;']
    lines += [f'{"reg" if direction == "input" else "wire"} {width} {name};'
              for name, direction, width in ports]
    lines += [f'{module} dut (' + ', '.join(f'.{name}({name})' for name, _, _ in ports) + ');', 'endmodule']
    path = BUILD / f'{module}_interface.v'
    path.write_text('\n'.join(lines) + '\n')
    return path


def load_problem(spec, tb=None, ref=None, *, seed=0, async_reset=None,
                 cycles=256, random_cycles=128):
    """Load an arbitrary Markdown specification with its independent oracle files.

    Same-stem files are discovered automatically; explicit paths always win.
    Adjacent reference JSON holds public clarifications and optional contracts.
    No problem-specific Python branch or registry entry is required.
    """
    spec = spec_path(spec)
    if not spec.is_file():
        raise ValueError(f'specification does not exist: {spec}')
    stems = list(dict.fromkeys((spec.stem, spec.stem.lower())))
    root = pathlib.Path(__file__).resolve().parent
    if not ref:
        candidates = [directory / (stem + '.v') for directory in
                      (spec.parent, spec.parent / 'reference', root / 'verilog/reference', root / 'reference')
                      for stem in stems]
        ref = next((path for path in candidates if path.is_file()), None)
    if not ref:
        raise ValueError('a Markdown spec needs an independent reference: provide --ref or a same-stem reference file')
    ref = pathlib.Path(ref).resolve()
    metadata_path = ref.with_suffix('.json')
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    if not tb:
        candidates = [directory / (stem + '_tb.v') for directory in
                      (spec.parent / 'TestBench', root / 'TestBench') for stem in stems]
        tb = next((path for path in candidates if path.is_file()), None)
    official = str(pathlib.Path(tb).resolve()) if tb else None
    interface = tb or _interface_testbench(ref)
    timing = metadata.get('async_reset', True) if async_reset is None else async_reset
    p = file_problem(spec, interface, ref, seed=seed, async_reset=timing,
                     cycles=cycles, random_cycles=random_cycles)
    if async_reset is True and not p['reset']:
        raise ValueError('--async-reset requires a reset port')
    p['official'] = official
    p['spec_addendum'] = metadata.get('spec_addendum', '')
    p['comparisons'] = metadata.get('comparisons', {})
    return p


def _contract_description(contract, got, width, sample, previous, previous_sample):
    """Output contracts allow valid nondeterministic results without comparing PRNGs."""
    if len(got) != width:
        return 'not observed (the simulation ended early)'
    if set(got.lower()) - set('01'):
        return 'undefined (X)'
    in_reset = sample['kind'] in ('initial_value', 'initial_immediate', 'reset')
    if in_reset and contract.get('reset') == 'defined':
        return None
    trigger = contract.get('trigger')
    triggered = True
    if trigger:
        name = trigger['input']
        old = (previous_sample['inputs'].get(name, 0) if previous_sample and
               previous_sample['kind'] not in ('initial_value', 'initial_immediate', 'reset') else 0)
        new = sample['inputs'].get(name, 0)
        triggered = bool(new) and (trigger.get('edge') != 'rising' or not old)
    if not triggered and contract.get('hold_outside_trigger'):
        if len(previous) == width and not (set(previous) - set('01')):
            return _describe(got, int(previous, 2), width)
        return 'not observed (the previous sample is unavailable)'
    if contract['kind'] == 'range':
        selector = str(sample['inputs'][contract['selector']])
        low, high = contract['ranges'][selector]
        if low <= int(got, 2) <= high:
            return None
        return 'outside the permitted range for the sampled inputs'
    raise ValueError(f"unknown output contract: {contract['kind']}")


def _describe(got, want, w):
    """None if equal; otherwise what is wrong, never the expected value."""
    if len(got) != w:
        return "not observed (the simulation ended early)"
    if set(got.lower()) & set("xz"):
        return "undefined (X)"
    exp = format(want, f"0{w}b")
    bad = [w - 1 - i for i in range(w) if got[i] != exp[i]]
    if not bad:
        return None
    return f"wrong in bit(s) {bad}; the other {w - len(bad)} bits are correct"


# Order is significant: report every match, and use this order for v2 principles.
V2_OBSERVATIONS = {
    "PARTIAL_X": "some output bits are known while others are X or Z.",
    "OUTPUT_STUCK": "the output is unchanged from the previous observed state.",
    "OUTPUT_INVERTED": "every output bit is the opposite of the corresponding specified bit.",
    "BIT_ORDER_REVERSED": "the output matches the specified state with its bit order reversed.",
    "SHIFTED_TOWARD_LSB": "compared with the previous state, the bits moved one position toward bit 0.",
    "SHIFTED_TOWARD_MSB": "compared with the previous state, the bits moved one position toward the highest bit.",
    "OFF_BY_ONE_VALUE": "the unsigned output value differs from the specified value by exactly one.",
    "ONE_CYCLE_LATE": "the output matches the specified state from one cycle earlier.",
    "ONE_CYCLE_EARLY": "the output matches the specified state from one cycle later.",
    "UPPER_BITS_ZERO": "the upper {upper} output bits are zero while the lower {lower} bits match the specified state.",
}


def _v2_signatures(got, want, previous, width, previous_expected, next_expected, *, detectors=True):
    """Compare only a failing sample; reference values never enter feedback.

    O = observed unsigned value, E = expected value, P = previous observed value.
    Partial X requires both known and X/Z bits; all other matches require known O.
    Stuck: O == P. Inverted: O == (~E & mask). Reversed: O == reverse_bits(E).
    Shifts compare all but the newly entered bit of O with the corresponding P bits.
    Off by one: abs(O - E) == 1 (no wrap). Late/early compare adjacent reference states.
    Upper zero: O's upper half is zero, E's upper half is nonzero, lower halves match.
    For odd widths, the lower half has width // 2 bits. Shifts require width > 1.
    """
    if not detectors or width < 1:
        return []
    got, previous = got.lower(), previous.lower()
    if len(got) != width or set(got) - set("01xz"):
        return []
    if set(got) & set("xz"):
        return ["PARTIAL_X"] if set(got) & set("01") else []
    observed = int(got, 2)
    if observed == want:
        return []
    previous_known = len(previous) == width and not (set(previous) - set("01"))
    mask = (1 << width) - 1
    lower = width // 2
    lower_mask = (1 << lower) - 1
    matches = {
        "OUTPUT_STUCK": previous_known and got == previous,
        "OUTPUT_INVERTED": observed == (~want & mask),
        "BIT_ORDER_REVERSED": got == format(want, f"0{width}b")[::-1],
        "SHIFTED_TOWARD_LSB": width > 1 and previous_known and got[1:] == previous[:-1],
        "SHIFTED_TOWARD_MSB": width > 1 and previous_known and got[:-1] == previous[1:],
        "OFF_BY_ONE_VALUE": abs(observed - want) == 1,
        "ONE_CYCLE_LATE": observed == previous_expected,
        "ONE_CYCLE_EARLY": observed == next_expected,
        "UPPER_BITS_ZERO": (lower > 0 and observed >> lower == 0 and want >> lower != 0
                            and observed & lower_mask == want & lower_mask),
    }
    return [signature for signature in V2_OBSERVATIONS if matches.get(signature, False)]


def _v2_observation(signature, width):
    return "Observed pattern: " + V2_OBSERVATIONS[signature].format(
        upper=width - width // 2, lower=width // 2)


def _official_ok(tb_path, dut_path, tag, timeout):
    """True/False from the benchmark's own testbench; None if it is not present."""
    if not tb_path or not pathlib.Path(tb_path).exists():
        return None
    exe = BUILD / f"{tag}_official.vvp"
    c = subprocess.run(["iverilog", "-g2012", "-I", str(pathlib.Path(tb_path).resolve().parent),
                       "-I", str(pathlib.Path(dut_path).resolve().parent),
                       "-o", str(exe), str(tb_path), str(dut_path)],
                       capture_output=True, text=True)
    if c.returncode != 0:
        return False
    try:
        r = subprocess.run(["vvp", str(exe.resolve())], cwd=BUILD,
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False
    lines = r.stdout.splitlines()
    if any(FAIL_RE.search(l) for l in lines):
        return False
    return any(PASS_RE.search(l) for l in lines)


def official(problem_name, dut_path, timeout=30):
    p = _problem(problem_name)
    return _official_ok(p["official"], dut_path, pathlib.Path(dut_path).stem, timeout)


# ---- grading ----

def code_facts(dut_path, out):
    """Observe procedural drivers in comment-free source, preserving source lines."""
    source = pathlib.Path(dut_path).read_text()
    source = re.sub(r'//[^\n]*|/\*[\s\S]*?\*/',
                    lambda m: re.sub(r'[^\n]', ' ', m.group()), source)
    # Token positions let us delimit nested statements without changing source text.
    tokens = list(re.finditer(r'"(?:\\.|[^"\\])*"|[A-Za-z_$][\w$]*|\S', source))

    def statement(i):
        if i >= len(tokens):
            return i
        word = tokens[i].group()
        if word == 'begin':
            i += 1
            if i < len(tokens) and tokens[i].group() == ':':
                i += 2
            while i < len(tokens) and tokens[i].group() != 'end':
                i = statement(i)
            return min(i + 1, len(tokens))
        if word in ('if', 'for', 'while', 'repeat'):
            i += 1
            depth = 0
            while i < len(tokens):
                t = tokens[i].group()
                depth += (t == '(') - (t == ')')
                i += 1
                if t == ')' and depth == 0:
                    break
            i = statement(i)
            if word == 'if' and i < len(tokens) and tokens[i].group() == 'else':
                i = statement(i + 1)
            return i
        if word in ('case', 'casex', 'casez'):
            depth = 1
            i += 1
            while i < len(tokens) and depth:
                t = tokens[i].group()
                depth += (t in ('case', 'casex', 'casez')) - (t == 'endcase')
                i += 1
            return i
        while i < len(tokens):
            t = tokens[i].group()
            i += 1
            if t == ';':
                break
        return i

    assignment = re.compile(r'(?<![\w$])' + re.escape(out) +
                            r'(?![\w$])\s*(?:\[[^\]]+\]\s*)?(<=|=(?!=))')
    blocks = []
    for match in re.finditer(r'\balways\s*@\s*\(([^)]*)\)|\binitial\b', source):
        i = next((j for j, t in enumerate(tokens) if t.start() >= match.end()), len(tokens))
        end = statement(i)
        body_end = tokens[end - 1].end() if end > i else match.end()
        assignments = list(assignment.finditer(source, match.end(), body_end))
        if assignments:
            blocks.append(dict(line=source.count('\n', 0, match.start()) + 1,
                               kind='always' if match.group(1) is not None else 'initial',
                               sensitivity=match.group(1),
                               blocking_lines=[source.count('\n', 0, a.start()) + 1
                                               for a in assignments if a.group(1) == '=']))
    return dict(blocks=blocks, multiple_blocks=len(blocks) > 1,
                blocking_lines=sorted({line for b in blocks for line in b['blocking_lines']}))


def _facts_line(out, facts):
    places = [f"line {b['line']}, " +
              (f"a block triggered by {b['sensitivity']}" if b['kind'] == 'always'
               else 'an initial block') for b in facts['blocks']]
    line = f"In your design, {out} is assigned in {len(places)} place(s): " + '; '.join(places) + '.'
    if facts['multiple_blocks']:
        line += f" ({out} is driven from more than one block)"
    for number in facts['blocking_lines']:
        line += f" ({out} uses a blocking assignment on line {number})"
    return line


def _v1_signatures(parts, rows, facts, midrun_failure):
    """Select existing principles from failed checks and observed source facts."""
    before_edge_x = bool(set(rows.get("R0", "").lower()) & set("xz"))
    clocked_blocking = any(b['blocking_lines'] and b['kind'] == 'always'
                          and re.search(r'\b(?:posedge|negedge)\b', b['sensitivity'])
                          for b in facts['blocks'])
    matches = {
        "X_BEFORE_FIRST_EDGE": not parts["reset_immediate"] and before_edge_x,
        "WRONG_BETWEEN_EDGES_IN_RESET": (
            not parts["reset_immediate"] and not before_edge_x) or midrun_failure == 0,
        "WRONG_AFTER_EDGE_IN_RESET": not parts["reset_value"] or midrun_failure == 1,
        "MULTIPLE_DRIVERS": facts['multiple_blocks'],
        "BLOCKING_IN_CLOCKED_LOGIC": clocked_blocking,
        "INIT_ONLY_VALUE": midrun_failure == 0 and any(
            b['kind'] == 'initial' for b in facts['blocks']),
        "NEXT_STATE_WRONG_FIRST_EDGE": not parts["first_step"],
        "NEXT_STATE_DIVERGES_LATER": parts["first_step"] and not parts["sequence"],
    }
    return [signature for signature, matched in matches.items() if matched]


def grade(problem_name, dut_path, timeout=30, *, facts=True, detectors=False,
          principles=False, async_reset=None, diagnostics=None, taxonomy_path=None, knowledge_path=None,
          categories=(), max_rules=2):
    p = copy.deepcopy(_problem(problem_name))
    if async_reset is not None and async_reset != p["async_reset"]:
        p["async_reset"] = async_reset
        if not async_reset:
            p["weights"]["reset_value"] += p["weights"].pop("reset_immediate", 0)
        else:
            share = p["weights"]["reset_value"] / 2
            p["weights"]["reset_value"] -= share
            p["weights"]["reset_immediate"] = share
    options = dict(principles=principles, detectors=detectors, taxonomy_path=taxonomy_path,
                   knowledge_path=knowledge_path,
                   categories=categories, max_rules=max_rules)
    if principles and taxonomy_path and diagnostics is None:
        diagnostics = {}
    if diagnostics is not None:
        diagnostics.clear()
        diagnostics.update(signatures=[], checks={}, mismatches=[], categories=list(categories))
    weights = p["weights"]
    if abs(sum(weights.values()) - 1.0) > 1e-12:
        raise ValueError("problem weights must sum to 1.0")
    if not p["async_reset"] and "reset_immediate" in weights:
        raise ValueError("synchronous reset must not weight reset_immediate")
    tag = pathlib.Path(dut_path).stem
    source, samples = _probe(p)
    probe = BUILD / f"{tag}_probe.v"
    probe.write_text(source)
    exe = BUILD / f"{tag}_probe.vvp"
    sources = [str(probe), str(dut_path)]
    if p.get("ref_verilog"):
        sources.append(str(_reference_copy(p, tag)))
    include_dirs = ['-I', str(pathlib.Path(dut_path).resolve().parent)]
    if p.get('ref_verilog'):
        include_dirs += ['-I', str(pathlib.Path(p['ref_verilog']).resolve().parent)]
    c = subprocess.run(["iverilog", "-g2012", *include_dirs, "-s", "probe", "-o", str(exe),
                        *sources], capture_output=True, text=True)
    outputs = [(name, width) for name, direction, width in p["ports"] if direction == "output"]
    if c.returncode != 0:
        ports = [name + (f"[{width - 1}:0]" if width > 1 else "")
                 for name, _, width in p["ports"]]
        port_text = ', '.join(ports[:-1]) + ' and ' + ports[-1]
        feedback = (f"compiles: requirement: a module named `{p['module']}` with ports "
                    f"{port_text} that compiles. Observed: compiler output:\n" + c.stderr.strip()[:1200])
        if diagnostics is not None:
            diagnostics['signatures'] = ['COMPILE_ERROR']
        return 0.0, _with_principles(feedback, ["COMPILE_ERROR"], **options)
    try:
        r = subprocess.run(["vvp", str(exe)], capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        feedback = ("timeout: requirement: the simulation must finish. "
                    "Observed: it never finished.")
        if diagnostics is not None:
            diagnostics['signatures'] = ['TIMEOUT']
        return weights["compiles"], _with_principles(feedback, ["TIMEOUT"], **options)

    rows, ref_rows = {}, {}
    reference = bool(p.get("ref_verilog"))
    count = len(outputs)
    for line in r.stdout.splitlines():
        fields = line.split()
        if len(fields) == count * (2 if reference else 1) + 2 and fields[0] == "P" and fields[1].isdigit():
            number = int(fields[1])
            rows[number] = dict(zip((name for name, _ in outputs), fields[2:2 + count]))
            if reference:
                values = fields[2 + count:]
                if any(len(value) != width or set(value) - set("01")
                       for value, (_, width) in zip(values, outputs)):
                    return 0.0, "reference: requirement: reference outputs must be defined. Observed: the reference produced an undefined or invalid sample."
                ref_rows[number] = dict(zip((name for name, _ in outputs), (int(value, 2) for value in values)))

    parts = dict.fromkeys(weights, True)
    failures = {}
    fact_data = {name: code_facts(dut_path, name) for name, _ in outputs}
    v2_ids, pattern_lines = [], []
    midrun_failure = None
    for number, sample in enumerate(samples):
        for out, width in outputs:
            got = rows.get(number, {}).get(out, "")
            want = ref_rows.get(number, {}).get(out, 0) if reference else sample["want"][out]
            comparison = p.get('comparisons', {}).get(out)
            description = (_contract_description(comparison, got, width, sample,
                           rows.get(number - 1, {}).get(out, ''),
                           samples[number - 1] if number else None)
                           if comparison else _describe(got, want, width))
            if description is None:
                continue
            new_checks = [check for check in sample["checks"] if check not in failures]
            for check in new_checks:
                parts[check] = False
                context = dict(rst=p["reset"], act=p["reset_active"], out=out,
                               cycle=sample["cycle"], description=description,
                               reset_timing="immediately" if p["async_reset"] else "across a clock edge")
                requirement = sample["requirements"].get(check, p["requirements"].get(check))
                if requirement is None:
                    requirement = next(s["requirement"] for s in p["segments"] if s["label"] == check)
                if sample["kind"] == "initial_value":
                    observed = (f"with {p['reset']} held at {p['reset_active']}, after "
                                f"a clock edge, {out} is {description}.")
                elif sample["kind"] == "initial_immediate":
                    observed = (f"at {sample['time']} ns, with {p['reset']} at {p['reset_active']} "
                                f"and no clock edge yet, {out} is {description}.")
                elif sample["kind"] == "restart":
                    observed = ("reset restored and held the initial state correctly; "
                                f"the sequence after restart fails {sample['when']}, {out} is {description}.")
                else:
                    template = p["observations"].get(check, sample["observed"])
                    observed = (template.format(**context) if template else
                                f"{sample['when']}, {out} is {description}.")
                if p.get("include_context"):
                    observed = f"at cycle {sample['edge']}, {out} is {description}."
                    if sample["context"]:
                        observed += " Context: " + "; ".join(sample["context"]) + "."
                message = f"{check}: requirement: {requirement.format(**context)}. Observed: {observed}"
                if facts and sample["kind"] in ("initial_value", "initial_immediate", "reset"):
                    message += '\n' + _facts_line(out, fact_data[out])
                failures[check] = message
                if diagnostics is not None:
                    diagnostics['mismatches'].append(dict(check=check, output=out, width=width,
                        cycle=sample['edge'], description=description, context=sample['context'],
                        inputs=sample['inputs'], kind=sample['kind']))
                if check == "reset_midrun":
                    midrun_failure = sample["subcheck"]
            if new_checks and sample["kind"] == "segment" and (detectors or diagnostics is not None) and not comparison:
                previous_expected = (ref_rows.get(number - 1, {}).get(out) if reference
                                     else sample["previous"][out])
                next_edge = next((i for i in range(number + 1, len(samples))
                                  if samples[i]["edge"] > sample["edge"]), None)
                next_expected = (ref_rows.get(next_edge, {}).get(out) if reference
                                 else sample["following"][out])
                ids = _v2_signatures(got, want,
                                     rows.get(number - 1, {}).get(out, ""), width,
                                     previous_expected, next_expected)
                v2_ids.extend(ids)
                if detectors:
                    pattern_lines.extend(_v2_observation(signature, width) for signature in ids)

    score = round(sum(weight for check, weight in weights.items() if parts[check]), 3)
    if diagnostics is not None:
        diagnostics['checks'] = parts.copy()
    if all(parts.values()):
        if _official_ok(p["official"], dut_path, tag, timeout) is False:
            return 0.9, ("official: requirement: the benchmark testbench must pass. "
                         "Observed: the benchmark testbench failed.")
        return 1.0, "correct"
    v1_ids = []
    if principles or diagnostics is not None:
        # Normalize check outcomes for the unchanged, problem-independent detectors.
        signature_parts = dict(parts)
        signature_parts.setdefault("reset_immediate", True)
        signature_parts.setdefault("reset_value", True)
        signature_parts.setdefault("first_step", True)
        signature_parts["sequence"] = all(parts[s["label"]] for s in p["segments"])
        initial = next((i for i, sample in enumerate(samples)
                        if sample["kind"] == "initial_immediate"), None)
        for out, _ in outputs:
            initial_rows = {"R0": rows.get(initial, {}).get(out, "")}
            v1_ids.extend(_v1_signatures(signature_parts, initial_rows, fact_data[out], midrun_failure))
    semantic_ids = []
    if diagnostics is not None:
        for mismatch in diagnostics['mismatches']:
            if mismatch['description'] == 'outside the permitted range for the sampled inputs':
                semantic_ids.append('VALUE_OUT_OF_RANGE')
            if mismatch['kind'] == 'segment' and mismatch['inputs'].get('enable') == 0:
                semantic_ids.append('WRONG_WHILE_DISABLED')
            elif mismatch['kind'] == 'segment' and any('enable went high' in context for context in mismatch['context']) and any(s['kind'] == 'segment' and s['edge'] < mismatch['cycle'] and s['inputs'].get('enable') == 0 for s in samples):
                semantic_ids.append('WRONG_AFTER_REENABLE')
            if mismatch['check'] == 'repetition':
                semantic_ids.append('WRONG_AFTER_WRAP')
        diagnostics['signatures'] = list(dict.fromkeys(v1_ids + semantic_ids + v2_ids))
    messages = [failures[check] for check in weights if check in failures]
    return score, _with_principles("\n".join(messages + pattern_lines), v1_ids + semantic_ids, v2_ids, **options)


# ---- individual and batch execution; all runners stay in Python ----

def run_suite(args):
    """Run the manifest in Python and preserve the existing suite log format."""
    manifest_path = pathlib.Path(args.manifest)
    manifest = json.loads(manifest_path.read_text())
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + str(time.time_ns())
    run_name = manifest.get('version', 'md_generic_dynamic_taxonomy') + '_' + stamp
    directory = args.log_dir.resolve() / 'runs' / run_name
    directory.mkdir(parents=True, exist_ok=False)
    tasks = [task for task in manifest['tasks'] if not args.only or task['name'] in args.only]
    if not tasks:
        raise ValueError('no tasks selected')
    if args.only and set(args.only) - {task['name'] for task in tasks}:
        raise ValueError('unknown task name in --only')
    (directory / 'run.json').write_text(json.dumps(dict(version=manifest.get('version'),
        mode=args.mode, seed=args.seed, manifest=manifest,
        manifest_path=str(manifest_path.resolve()),
        manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        arguments={key: str(value) if isinstance(value, pathlib.Path) else value for key, value in vars(args).items()}), indent=2) + '\n')
    results = []
    for task in tasks:
        def path(value):
            candidate = pathlib.Path(value)
            return str(candidate if candidate.is_absolute() else ROOT / candidate)
        command = [sys.executable, str(ROOT / ('agent.py' if args.mode == 'agent' else 'checker.py')),
                   '--no-console-log',
                   '--spec', path(task['spec']), '--seed', str(args.seed),
                   '--cycles', str(manifest.get('cycles', 256)),
                   '--random-cycles', str(manifest.get('random_cycles', 128)),
                   '--taxonomy', path(manifest.get('taxonomy', 'taxonomy.json')),
                   '--rules-limit', str(manifest.get('rules_limit', 2)), '--facts', '--detectors']
        command += ['--knowledge', path(manifest.get('knowledge', 'knowledge.json'))]
        if args.show_detect_text:
            command += ['--show-detect-text']
        if args.exclude_categories:
            command += ['--exclude-categories', args.exclude_categories]
        for flag in ('tb', 'ref'):
            if task.get(flag):
                command += ['--' + flag, path(task[flag])]
        if not args.no_principles:
            command += ['--principles']
        if args.async_reset is not None:
            command += ['--async-reset' if args.async_reset else '--no-async-reset']
        for category in task.get('categories', []):
            command += ['--category', category]
        if args.mode == 'agent':
            command += ['--repeat', str(args.repeat), '--rounds', str(args.rounds),
                        '--context-tokens', str(args.context_tokens if args.context_tokens is not None else manifest.get('context_tokens', 8000)),
                        '--max-tokens', str(args.max_tokens if args.max_tokens is not None else manifest.get('max_tokens', 1500)),
                        '--rule-bytes', str(args.rule_bytes if args.rule_bytes is not None else manifest.get('rule_bytes', 1200)),
                        '--log-dir', str(directory), '--run-name', run_name,
                        '--output-dir', str(ROOT / 'candidates/gen')]
            if not args.no_explain:
                command += ['--explain']
            for flag in ('interface_from_tb', 'no_code', 'hint'):
                if getattr(args, flag, False):
                    command += ['--' + flag.replace('_', '-')]
            if args.model:
                command += ['--model', args.model]
            if args.base_url:
                command += ['--base-url', args.base_url]
        else:
            candidate = args.dut_dir.resolve() / (task['name'] + '.v') if args.dut_dir else pathlib.Path(path(task['ref']))
            command += ['--dut', str(candidate), '--diagnostics-json', str(directory / (task['name'] + '.json'))]
        logfile = args.log_dir.resolve() / (task['name'] + '.log')
        previous_records = set(directory.rglob('*.jsonl'))
        print(f"{task['name']}: {logfile}", flush=True)
        with logfile.open('w') as stream:
            start = time.monotonic()
            completed = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        shutil.copyfile(logfile, directory / logfile.name)
        result = dict(task=task['name'], exit_code=completed.returncode,
                      elapsed_seconds=time.monotonic() - start, log=str(logfile), command=command)
        record_files = sorted(set(directory.rglob('*.jsonl')) - previous_records)
        result['records'] = [str(path) for path in record_files]
        for record_file in record_files:
            for line in record_file.read_text().splitlines():
                record = json.loads(line)
                if record.get('event') == 'summary':
                    result['scores'] = record
        report = directory / (task['name'] + '.json')
        if args.mode == 'checker' and report.exists():
            result['score'] = json.loads(report.read_text())['score']
        results.append(result)
        # Persist the partial summary even if a later task is interrupted.
        (directory / 'summary.json').write_text(json.dumps(results, indent=2) + '\n')
    print(f'Suite logs: {directory}', flush=True)
    return int(any(result['exit_code'] != 0 for result in results))

def interface_comparison():
    rows = []
    for name in ('lfsr', 'sequence_generator'):
        problem = load_problem(_find_spec(name))
        rows.append(dict(problem=name, current=interface_line(problem),
                         generated=interface_line(problem, generated=True)))
    return rows


def run_testbench(tb_path, dut_path, timeout=30):
    """Compile and run the provided benchmark without requiring a spec/reference."""
    if not pathlib.Path(tb_path).is_file() or not pathlib.Path(dut_path).is_file():
        raise ValueError('testbench and DUT must both be existing files')
    return _official_ok(tb_path, dut_path, pathlib.Path(dut_path).stem, timeout)


def _main(argv=None):
    import argparse, glob, sys
    parser = argparse.ArgumentParser(description="Grade a DUT or run checker selftests")
    parser.add_argument("--problem", choices=PROBLEMS)
    parser.add_argument("--dut")
    parser.add_argument("--spec")
    parser.add_argument("--tb")
    parser.add_argument("--ref")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cycles", type=int, default=256)
    parser.add_argument("--random-cycles", type=int, default=128)
    parser.add_argument("--taxonomy", default=str(pathlib.Path(__file__).with_name('taxonomy.json')))
    parser.add_argument("--knowledge", default=str(pathlib.Path(__file__).with_name('knowledge.json')))
    parser.add_argument('--show-detect-text', action='store_true')
    parser.add_argument('--knowledge-report', action='store_true', help='report every knowledge key difference without editing inputs')
    parser.add_argument('--knowledge-original', default=str(ROOT / 'knowledge_original.json'))
    parser.add_argument("--exclude-categories", default='')
    parser.add_argument("--category", action="append", default=[])
    parser.add_argument("--rules-limit", type=int, default=2)
    parser.add_argument("--diagnostics-json", help="write signatures and observations as JSON")
    parser.add_argument("--facts", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--detectors", action="store_true")
    parser.add_argument("--principles", action="store_true")
    reset = parser.add_mutually_exclusive_group()
    reset.add_argument("--async-reset", dest="async_reset", action="store_true")
    reset.add_argument("--no-async-reset", dest="async_reset", action="store_false")
    parser.set_defaults(async_reset=None)
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument('--run-all', action='store_true', help='run all tasks in the manifest using Python')
    parser.add_argument('--manifest', type=pathlib.Path, default=ROOT / 'experiments/md_generic_v4_dynamic_taxonomy.json')
    parser.add_argument('--mode', choices=('agent', 'checker'), default='checker')
    parser.add_argument('--only', nargs='+')
    parser.add_argument('--dut-dir', type=pathlib.Path)
    parser.add_argument('--log-dir', type=pathlib.Path, default=ROOT / 'log')
    parser.add_argument('--no-console-log', action='store_true')
    parser.add_argument('--rounds', type=int, default=6)
    parser.add_argument('--repeat', type=int, default=5)
    parser.add_argument('--model')
    parser.add_argument('--base-url')
    parser.add_argument('--no-principles', action='store_true')
    parser.add_argument('--no-explain', action='store_true')
    parser.add_argument('--explain', action='store_true')
    parser.add_argument('--principles-v2', action='store_true')
    parser.add_argument('--hint', action='store_true')
    parser.add_argument('--no-code', action='store_true')
    parser.add_argument('--interface-from-tb', action='store_true')
    parser.add_argument('--show-interface-lines', action='store_true')
    parser.add_argument('--context-tokens', type=int)
    parser.add_argument('--max-tokens', type=int)
    parser.add_argument('--rule-bytes', type=int)
    parser.add_argument('--run-tb', action='store_true', help='compile and run --tb with --dut, without a spec/reference')
    parser.add_argument('--timeout', type=float, default=30)
    args = parser.parse_args(argv)
    if args.knowledge_report:
        report = taxonomy.knowledge_report(args.knowledge, args.knowledge_original)
        print(json.dumps(report, indent=2))
        return 0 if report['status'] == 'compared' else 1
    if args.run_all:
        if args.dut or args.spec or args.problem:
            parser.error('--run-all uses the manifest; use --only or --dut-dir to select tasks/candidates')
        try:
            return run_suite(args)
        except (ValueError, OSError) as error:
            parser.error(str(error))
    if args.show_interface_lines:
        rows = interface_comparison()
        print(json.dumps(rows, indent=2))
        return 0
    if args.run_tb or (args.tb and args.dut and not args.spec and not args.problem):
        if not args.tb or not args.dut:
            parser.error('--run-tb requires --tb and --dut')
        try:
            passed = run_testbench(args.tb, args.dut, args.timeout)
        except (ValueError, OSError) as error:
            parser.error(str(error))
        print('correct' if passed else 'official: requirement: the benchmark testbench must pass. Observed: the benchmark testbench failed.')
        return 0 if passed else 1
    args.principles = args.principles or args.principles_v2
    args.detectors = args.detectors or args.principles_v2
    grade_options = dict(facts=args.facts, detectors=args.detectors,
                         principles=args.principles, async_reset=args.async_reset)
    file_based = any((args.spec, args.tb, args.ref))
    if file_based:
        if args.problem or not args.spec:
            parser.error("use --spec FILE with optional --tb and --ref, or --problem")
        try:
            selected = load_problem(args.spec, args.tb, args.ref, seed=args.seed,
                                    async_reset=args.async_reset, cycles=args.cycles,
                                    random_cycles=args.random_cycles)
            if args.principles:
                applicable = taxonomy.categories(pathlib.Path(selected['spec']).read_text(),
                                                selected['ports'], args.taxonomy, args.category,
                                                exclude=[value.strip() for value in args.exclude_categories.split(',') if value.strip()])
                grade_options.update(taxonomy_path=args.taxonomy, knowledge_path=args.knowledge, categories=applicable,
                                     max_rules=args.rules_limit)
        except (ValueError, OSError) as error:
            parser.error(str(error))
    else:
        selected = args.problem or "lfsr"
    if args.dut:
        diagnostic_data = {} if args.diagnostics_json or (args.show_detect_text and args.principles) else None
        direct_options = dict(grade_options)
        if args.show_detect_text and args.principles:
            direct_options['principles'] = False
        score, feedback = grade(selected, args.dut, diagnostics=diagnostic_data, **direct_options)
        if args.show_detect_text and args.principles:
            p = _problem(selected)
            applicable = taxonomy.categories(public_spec(p), p['ports'], args.taxonomy, args.category,
                exclude=[value.strip() for value in args.exclude_categories.split(',') if value.strip()])
            entries = taxonomy.select(args.taxonomy, applicable, diagnostic_data['signatures'],
                knowledge=taxonomy.read(args.knowledge), limit=args.rules_limit, show_detect_text=True)
            if entries:
                feedback += '\n' + '\n'.join(taxonomy.entry_text(entry, show_detect_text=True) for entry in entries)
        print(f"score: {score}\n{feedback}")
        if args.diagnostics_json:
            report = pathlib.Path(args.diagnostics_json)
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text(json.dumps(dict(score=score, feedback=feedback,
                                              diagnostics=diagnostic_data), indent=2) + '\n')
        if not args.selftest:
            sys.exit(0 if score == 1.0 else 1)
    if args.selftest and not args.problem and not file_based:
        cases = [(name, p, p["references"]) for name, p in PROBLEMS.items()]
        file_sequence = file_problem("sequence_generator.md", "TestBench/sequence_generator_tb.v",
                                     "reference/seq/seq_a.v", seed=args.seed)
        cases.append(("sequence_generator (files)", file_sequence, ("reference/seq/seq_a.v",)))
        rc = 0
        for name, p, references in cases:
            refs = [next(f for f in references if pathlib.Path(f).exists())]
            pattern = "candidates/bad/*.v" if p["module"] == "lfsr" else f"candidates/bad/{p['module']}/*.v"
            for f in refs + sorted(glob.glob(pattern)):
                score, feedback = grade(p, f, **grade_options)
                print(f"{name}: {f}: {score}\n    " + feedback.replace("\n", "\n    "))
                rc |= (score != 1.0) if f in refs else (score >= 1.0)
        import unittest
        suite = unittest.defaultTestLoader.discover(str(pathlib.Path(__file__).with_name("tests")))
        rc |= not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful()
        print("\nSELFTEST " + ("PASSED" if rc == 0 else "FAILED"))
        sys.exit(rc)
    args.problem = args.problem or "lfsr"
    if args.problem != "lfsr" or file_based:
        p = selected if file_based else PROBLEMS[args.problem]
        references = (p['ref_verilog'],) if file_based else p["references"]
        rc = 0
        for f in [*references, *sorted(glob.glob(f"candidates/bad/{p['module']}/*.v"))]:
            score, feedback = grade(p, f, **grade_options)
            print(f"{f}: {score}\n    " + feedback.replace("\n", "\n    "))
            reference = f in references
            if reference:
                verdict = official(p, f)
                print(f"    Official testbench: {verdict}")
                rc |= score != 1.0 or verdict is not True
            else:
                rc |= score >= 1.0
        import unittest
        suite = unittest.defaultTestLoader.discover(
            str(pathlib.Path(__file__).with_name("tests")), pattern="test_engine.py")
        if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
            rc = 1
        print("\nSELFTEST " + ("PASSED" if rc == 0 else "FAILED"))
        sys.exit(rc)
    ref = next((f for f in ["reference/lfsr.v", "candidates/lfsr.v"]
                if pathlib.Path(f).exists()), None)
    if ref is None:
        print("no reference design found (reference/lfsr.v or candidates/lfsr.v)")
        sys.exit(1)
    files = [ref] + sorted(glob.glob("candidates/bad/*.v"))
    expected_facts = {
        'lfsr_syncreset.v': 'In your design, data is assigned in 1 place(s): line 6, '
                            'a block triggered by posedge clk.',
        'lfsr_initial_hack.v': 'In your design, data is assigned in 2 place(s): line 6, '
                               'an initial block; line 8, a block triggered by posedge clk. '
                               '(data is driven from more than one block) '
                               '(data uses a blocking assignment on line 6)',
        'lfsr_two_drivers.v': 'In your design, data is assigned in 2 place(s): line 6, '
                              'a block triggered by posedge clk; line 10, '
                              'a block triggered by reset_n. '
                              '(data is driven from more than one block)',
    }
    expected_scores = {'lfsr_badtap.v': 0.55, 'lfsr_initial_hack.v': 0.8,
                       'lfsr_syncreset.v': 0.65, 'lfsr_two_drivers.v': 0.65,
                       'shift_wrong_direction.v': 0.4, 'inverted_feedback.v': 0.4}
    rc = 0
    for f in files:
        s, fb = grade("lfsr", f, **grade_options)
        print(f"{f}: {s}\n    " + fb.replace("\n", "\n    "))
        if (f == ref and s != 1.0) or (f != ref and not s < 1.0):
            rc = 1
        if f != ref:
            name = pathlib.Path(f).name
            lines = fb.splitlines()
            reset_lines = [i for i, line in enumerate(lines)
                           if line.startswith(('reset_value:', 'reset_immediate:', 'reset_midrun:'))]
            restart_lines = [i for i in reset_lines
                             if 'reset restored and held the initial state correctly;' in lines[i]]
            facts_lines = [i for i in reset_lines if i not in restart_lines]
            expected = expected_facts.get(name, _facts_line('data', code_facts(f, 'data')))
            valid = bool(reset_lines) and all(
                i + 1 < len(lines) and lines[i + 1] == expected for i in facts_lines)
            valid = valid and sum(line.startswith('In your design,') for line in lines) == len(facts_lines)
            if name == 'lfsr_badtap.v':
                valid = valid and len(restart_lines) == 1 and not facts_lines
            checks = [line.split(':', 1)[0] for line in lines if ': requirement:' in line]
            valid = valid and len(checks) == len(set(checks))
            valid = valid and (name not in expected_scores or s == expected_scores[name])
            print('    Facts and score ' + ('verified' if valid else 'FAILED'))
            if not valid:
                rc = 1
    import unittest
    suite = unittest.defaultTestLoader.discover(
        str(pathlib.Path(__file__).with_name("tests")), pattern="test_checker.py")
    if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
        rc = 1
    print("\nSELFTEST " + ("PASSED" if rc == 0 else "FAILED"))
    sys.exit(rc)


def main(argv=None):
    return run_logged(lambda: _main(argv), argv)


if __name__ == "__main__":
    raise SystemExit(main())
