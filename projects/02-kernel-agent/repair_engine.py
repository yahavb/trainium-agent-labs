"""Evidence-based failure classification and targeted repair prompts."""
import re
from nki_knowledge import retrieve
from agent import enrich

RULES = [
    ('dimensions', r'at least (?:two|2)|minimum.*dimension', r'ndarray|\[[^]]+\]'),
    ('partition', r'partition.*(?:exceed|maximum|128)|256.partition', r'ndarray|dma_copy'),
    ('dma_shape', r'dma.*(?:shape|elements)|src=.*dst=', r'dma_copy'),
    ('bounds', r'out.of.bound|index.*(?:range|size)|indexerror', r'\[[^]]+\]|range'),
    ('api', r'no attribute|no module|unexpected keyword|not callable', r'nl\.|nisa\.|import '),
    ('matmul_layout', r'stationary|moving|nc_matmul|contraction', r'nc_matmul|psum'),
    ('memory', r'psum|sbuf|memoryregion|allocation', r'ndarray|buffer='),
    ('reshape', r'reshape|broadcast|slicing', r'reshape|\.ap\(|\[[^]]+\]'),
    ('output_shape', r'wrong shape', r'ndarray|return'),
    ('missing_writes', r'uninitialized|unwritten|modified its input', r'dma_copy|return'),
    ('traffic', r'traffic|too much hbm', r'dma_copy|range'),
    ('syntax', r'syntaxerror|does not parse|no code', r'.'),
    ('rules', r'rule violation|no function named|not decorated', r'def |import |@'),
    ('numerical', r'mismatch|wrong|off by|nan|inf|index \(', r'nc_matmul|tensor_scalar|sum|dma_copy'),
]

def diagnose(source, result):
    diagnostics = []
    for case in result.get('cases', []):
        evidence = case.get('feedback', '')
        if case.get('passed'):
            continue
        category, lines_pattern = next(((c, p) for c, rx, p in RULES if re.search(rx, evidence, re.I)), ('unknown', r'.'))
        explicit = [int(n) for n in re.findall(r'line (\d+)', evidence)]
        lines = [{'line': i, 'expression': line.strip(), 'certainty': 'reported' if i in explicit else 'possible location'} for i, line in enumerate(source.splitlines(), 1) if i in explicit or re.search(lines_pattern, line)][:8]
        diagnostics.append({'category': category, 'case': case.get('label'), 'evidence': evidence, 'certainty': 'checker-confirmed failure; root cause hypothesis', 'likely_root_cause': retrieve([category]) or 'Cause not yet localized; inspect the reported exception and input shape.', 'source_lines': lines, 'recommended_repair': (enrich(evidence) if category == 'api' else retrieve([category])) or 'Resolve the specific checker failure with minimal edits.', 'constraints': ['Keep function signature and mathematical semantics', 'Preserve previously passing cases', 'Use installed NKI APIs'], 'validation': ['Re-run every official shape', 'Compare passing case sets', 'Run signature-correct deterministic probes']})
    if result.get('passed_cases') and diagnostics:
        diagnostics.append({'category': 'partial_shapes', 'evidence': 'Some official shapes passed and others failed', 'certainty': 'confirmed partial pass', 'likely_root_cause': 'Shape-specific indexing or allocation is a hypothesis.', 'recommended_repair': retrieve(['partial_shapes']), 'source_lines': [], 'constraints': ['Preserve passing shapes'], 'validation': ['Run all official cases']})
    return diagnostics

def repair_prompt(specification, source, result, history, fallback=False):
    import json
    ds = diagnose(source, result)
    return (specification + '\nRepair the following candidate. Change the smallest relevant portion; return the ENTIRE kernel in one python block.\n'
            + f'```python\n{source}\n```\n'
            + 'Checker diagnostics:\n' + json.dumps(ds) + '\nPreviously passing cases: ' + str(result.get('passed_cases', []))
            + '\nFailed approaches (do not repeat): ' + json.dumps(history[-8:])
            + ('\nRepeated failure fallback: simplify the failing data flow or use a different tiling strategy. Keep correct semantics and passing cases.' if fallback else '')
            + '\nDiagnostic probes (advisory, official cases determine success): ' + json.dumps(result.get('probe_results', []))
            + '\nRelevant knowledge:\n' + retrieve([d['category'] for d in ds]))
