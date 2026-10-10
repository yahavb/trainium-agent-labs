"""Export simulator-verified synthetic tasks to leakage-safe completion-only SFT."""
import ast
import hashlib
import json
from pathlib import Path


def load_verified(path, split):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    for row in rows:
        checks = row['verification_results']
        if row['split'] != split or row['verification_status'] != 'SIMULATOR_VERIFIED':
            raise ValueError('Unverified record or split mismatch')
        if not checks['clean']['passed'] or checks['broken']['passed'] or not checks['repaired']['passed']:
            raise ValueError('Repair pair has not passed verification gates')
        if row['correct_kernel'] != row['corrected_kernel']:
            raise ValueError('Restored source differs from verified clean source')
        ast.parse(row['corrected_kernel'])
    return rows


def examples(rows):
    result = []; seen_generation = set()
    for r in rows:
        common = f"NKI SDK {r['sdk_version']}; target AWS Trainium2. Task: {r['task_description']}\nInput shapes: {r['input_shapes']}; dtypes: {r['input_dtypes']}."
        key = ast.dump(ast.parse(r['correct_kernel']), include_attributes=False)
        if key not in seen_generation:
            seen_generation.add(key)
            result.append(dict(id=r['example_id']+'-generation', kind='generation', source_id=r['example_id'], messages=[{'role':'user','content':common+'\nWrite a complete NKI kernel. Return Python code only.'}], completion=r['correct_kernel']))
        result.append(dict(id=r['example_id']+'-repair',kind='repair',source_id=r['example_id'],messages=[{'role':'user','content':common+'\nRepair this kernel. Preserve the mathematical operation. Return complete Python code only.\n```python\n'+r['broken_kernel']+'```\nActual verification failure:\n'+r['observed_error']}],completion=r['corrected_kernel']))
    return result


def export(source, output):
    source = Path(source); output = Path(output)
    train = load_verified(source/'train.jsonl','train')
    heldout = load_verified(source/'heldout.jsonl','heldout')
    hashes = lambda rows: {ast.dump(ast.parse(r['correct_kernel']),include_attributes=False) for r in rows}
    if hashes(train) & hashes(heldout): raise ValueError('Structural leakage across splits')
    if {r['operation_family'] for r in train} & {r['operation_family'] for r in heldout}: raise ValueError('Operation-family leakage')
    output.mkdir(parents=True,exist_ok=False)
    summary = {'source':str(source.resolve()),'source_hashes':{},'benchmark_solutions_used':False,'device_verified':0}
    for split, rows in [('train',train),('heldout',heldout)]:
        samples = examples(rows)
        with (output/(split+'.jsonl')).open('x') as f:
            for sample in samples: f.write(json.dumps(sample)+'\n')
        summary[split+'_examples'] = len(samples)
        summary[split+'_repair_pairs'] = len(rows)
        summary['source_hashes'][split]=hashlib.sha256((source/(split+'.jsonl')).read_bytes()).hexdigest()
    (output/'manifest.json').write_text(json.dumps(summary,indent=2))
    return summary

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--source',default='synthetic_nki/data_v4');p.add_argument('--output',required=True)
    a=p.parse_args();print(json.dumps(export(a.source,a.output),indent=2))
