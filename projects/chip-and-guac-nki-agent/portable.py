"""Discover and verify the organizers' unmodified checker from this checkout."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent / '02-kernel-agent'


def verify_sources():
    expected = json.loads((HERE / 'source_manifest.json').read_text())
    for name, digest in expected.items():
        data = (PROJECT / name).read_bytes().replace(b'\r\n', b'\n')
        if hashlib.sha256(data).hexdigest() != digest:
            raise RuntimeError(f'Canonical source differs: {name}. Use the documented upstream revision.')
    return {'checked_files': len(expected), 'unchanged': True}


def load_agent():
    verify_sources()
    sys.path.insert(0, str(PROJECT))
    spec = importlib.util.spec_from_file_location('_chip_guac_canonical_agent', PROJECT / 'agent.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
