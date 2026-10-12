"""Summarise a neuron-monitor JSON-lines capture: NeuronCore utilisation and memory while Flux ran.

    python neuron_summary.py out/neuron-monitor.jsonl
"""

import json
import statistics
import sys
from collections import defaultdict


def main(path):
    util = defaultdict(list)       # core -> utilisation % samples
    dev_mem, host_mem, lat = [], [], []
    for line in open(path):
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        for rt in rec.get("neuron_runtime_data") or []:
            rep = rt.get("report") or {}
            counters = (rep.get("neuroncore_counters") or {}).get("neuroncores_in_use") or {}
            for core, v in counters.items():
                util[int(core)].append((v or {}).get("neuroncore_utilization") or 0.0)
            used = (rep.get("memory_used") or {}).get("neuron_runtime_used_bytes") or {}
            if used.get("neuron_device") is not None:
                dev_mem.append(used["neuron_device"])
            if used.get("host") is not None:
                host_mem.append(used["host"])
            stats = ((rep.get("execution_stats") or {}).get("latency_stats") or {}).get("total_latency") or {}
            for k in ("p50", "p99"):
                if stats.get(k) is not None:
                    lat.append((k, stats[k]))
    if not util:
        sys.exit("No NeuronCore samples found: was the model running while neuron-monitor ran?")

    print("NeuronCore utilisation (%)       mean    peak")
    for core in sorted(util):
        s = util[core]
        print(f"  core {core:<2}                   {statistics.mean(s):7.1f} {max(s):7.1f}")
    allv = [x for s in util.values() for x in s]
    print(f"  all cores                  {statistics.mean(allv):7.1f} {max(allv):7.1f}")
    busy = [x for x in allv if x > 0]
    if busy:
        print(f"  while busy (>0%)           {statistics.mean(busy):7.1f}")
    if dev_mem:
        print(f"Device memory used: peak {max(dev_mem) / 2**30:.1f} GiB")
    if host_mem:
        print(f"Host memory used by runtime: peak {max(host_mem) / 2**30:.1f} GiB")
    for k in ("p50", "p99"):
        v = [x for kk, x in lat if kk == k]
        if v:
            print(f"Per-execution latency {k}: mean {statistics.mean(v) * 1000:.1f} ms")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
