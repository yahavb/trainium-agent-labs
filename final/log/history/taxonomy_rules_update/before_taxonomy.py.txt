"""Dynamic category classification, signature descriptions, and bounded rule retrieval."""
import hashlib
import json
import pathlib
import re


def read(path):
    return snapshot(path)[0]


def snapshot(path):
    raw = pathlib.Path(path).read_bytes()
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError('taxonomy and knowledge files must be JSON objects')
    structured = 'categories' in data or 'signatures' in data
    if structured:
        if not isinstance(data.get('categories', []), list) or not isinstance(data.get('signatures', {}), dict):
            raise ValueError('taxonomy needs a category list and a signature dictionary')
    elif any(not isinstance(key, str) or ':' not in key or not isinstance(rule, str)
             for key, rule in data.items()):
        raise ValueError('knowledge must map category:signature to a rule sentence')
    return data, hashlib.sha256(raw).hexdigest()


def categories(text, ports=(), path='taxonomy.json', overrides=(), *, data=None, exclude=()):
    """Category keywords, port hints, and priority come from the live taxonomy JSON."""
    data = read(path) if data is None else data
    text = text.lower().replace('_', ' ')
    names = [name.lower() for name, _, _ in ports]
    found = list(overrides)
    if 'categories' in data:
        for category in data['categories']:
            keywords = category.get('spec_keywords', [])
            hints = category.get('port_hints', [])
            matched = any(keyword.lower() in text for keyword in keywords)
            matched |= any(hint.lower() == name or hint.lower() in name.split('_')
                           for hint in hints for name in names)
            if category['name'] == 'combinational' and category.get('rule'):
                matched |= not any(re.search(r'clk|clock', name) for name, direction, _ in ports if direction == 'input')
            if matched:
                found.append(category['name'])
    else:
        # A flat knowledge dictionary remains usable; match its category labels.
        available = dict.fromkeys(key.split(':', 1)[0] for key in data)
        for category in available:
            words = category.replace('_', ' ').split()
            if category not in ('*', 'generic') and all(re.search(r'\b' + re.escape(word) + r'\w*', text) for word in words):
                found.append(category)
    return [name for name in dict.fromkeys(found + ['generic']) if name not in exclude]


def select(path, applicable, signatures, *, limit=2, max_bytes=1200, data=None, knowledge=None):
    """Select current fired signatures only; never send the entire JSON to a model."""
    data = read(path) if data is None else data
    definitions = data.get('signatures', {})
    if 'categories' in data or 'signatures' in data:
        rules = data.get('knowledge', data.get('rules', knowledge or {}))
        lookup = data.get('lookup', {})
        cap = lookup.get('max_rules')
        if cap is None:
            match = re.search(r'at most\s+(\d+)', lookup.get('rule', ''), re.I)
            cap = int(match[1]) if match else limit
        limit = min(limit, cap)
    else:
        rules = data
    selected, used, budget = [], set(), 0
    for signature in dict.fromkeys(signatures):
        keys = [f'{category}:{signature}' for category in applicable] + [f'*:{signature}']
        key = next((key for key in keys if key in rules), None)
        rule = rules[key] if key else None
        definition = definitions.get(signature, {})
        description = definition.get('detect', '')
        if rule in used and rule is not None:
            continue
        if not rule and not description:
            continue
        entry = dict(key=key, signature=signature, rule=rule, description=description,
                     source=definition.get('source'))
        size = len(entry_text(entry).encode()) + 1
        if budget + size > max_bytes:
            continue
        selected.append(entry)
        if rule:
            used.add(rule)
        budget += size
        if len(selected) >= limit:
            break
    return selected


def entry_text(entry):
    lines = []
    if entry.get('description'):
        lines.append(f"Detected signature {entry['signature']}: {entry['description']}.")
    if entry.get('rule'):
        lines.append('Principle: ' + entry['rule'])
    return '\n'.join(lines)
