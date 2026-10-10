"""E27 probe: are samples of one prompt different? 4 concurrent + 2 sequential identical requests
(temperature 0.6, no seed, cache off) on a seat pod: kubectl exec -i $POD -c app -- python3 - < scripts/sampling_probe.py"""

import concurrent.futures as cf, hashlib, sys, time
sys.path.insert(0, "/workspace/stageA")
from kagent import client
from kagent.agent import first_prompt
from kagent.levels import LEVELS

prompt = first_prompt(LEVELS[4], 0, 0)[0][1]          # L4 first prompt, wording 0
msgs = [{"role": "user", "content": prompt}]
call = lambda s: client.chat(msgs, max_tokens=1500, temperature=0.6, sample=f"probe:{s}", use_cache=False)
h = lambda r: hashlib.sha1(r.content.encode()).hexdigest()[:10]

t0 = time.time()
with cf.ThreadPoolExecutor(max_workers=4) as ex:
    conc = list(ex.map(call, range(4)))
print(f"concurrent x4 ({time.time() - t0:.0f}s):", [(h(r), r.completion_tokens) for r in conc])
seq = [call(10 + i) for i in range(2)]
print("sequential x2:", [(h(r), r.completion_tokens) for r in seq])
allh = [h(r) for r in conc + seq]
print(f"distinct outputs: {len(set(allh))} of {len(allh)}")
