"""Print a pipeline result JSON compactly: python g3_show.py 4core"""
import json
import sys

from block_port import ART_DIR

d = json.load(open(ART_DIR / f"g3_pipeline_{sys.argv[1]}.json"))
for p in d["parity"]:
    print({k: (round(v, 6) if isinstance(v, float) else v) for k, v in p.items()})
print({k: v for k, v in d.items() if k != "parity"})
