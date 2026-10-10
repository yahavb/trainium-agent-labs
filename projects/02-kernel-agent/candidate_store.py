"""Content-addressed candidates and append-only experiment records."""
import ast
import hashlib
import json
from pathlib import Path

def candidate_hash(source):
    try:
        canonical = ast.dump(ast.parse(source), include_attributes=False)
    except SyntaxError:
        canonical = source.strip()
    return hashlib.sha256(canonical.encode()).hexdigest()

class CandidateStore:
    def __init__(self, directory):
        self.root = Path(directory)
        self.root.mkdir(parents=True, exist_ok=True)
        self.seen = {}
        self.best = None

    def record(self, source, result, **extra):
        digest = candidate_hash(source)
        (self.root / f'{digest}.py').write_text(source)
        record = dict(hash=digest, result=result, **extra)
        with (self.root / 'attempts.jsonl').open('a') as f:
            f.write(json.dumps(record) + '\n')
        self.seen[digest] = result
        if self.best is None or (not extra.get('regression') and result['reward'] > self.best['result']['reward']):
            self.best = record
            (self.root / 'best.py').write_text(source)
            self.write('best.json', record)
        if result['solved']:
            (self.root / 'solved.py').write_text(source)
        return record

    def write(self, name, value):
        path = self.root / name
        temporary = path.with_suffix(path.suffix + '.tmp')
        temporary.write_text(json.dumps(value, indent=2))
        temporary.replace(path)
