"""Print numbered source sentence candidates; does not fabricate annotations."""
from pathlib import Path
import json
from conditions import sentences

root = Path(__file__).resolve().parent
items = [json.loads(line) for line in (root / 'data/abstracts.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
for item in items:
    print(f"\n{item['id']} — {item.get('title', '')}")
    for n, s in enumerate(sentences(item['text'])):
        print(f"  {n}: {s}")
    print("  CHECK boundaries against the unchanged abstract before copying exact sentences.")

path = root / 'annotations.json'
if not path.exists():
    template = {item['id']: {
        'key_sentences': [],
        'main_findings': [],
        'numerical_results': [],
        'uncertainty_or_limitations': []
    } for item in items}
    path.write_text(json.dumps(template, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f"\nCreated {path.name}. Fill exactly two verbatim key_sentences for each abstract.")
else:
    print(f"\nExisting {path.name} left unchanged.")
