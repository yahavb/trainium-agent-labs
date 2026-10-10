"""Child process for hostile-state probes: call quote() on crafted extreme states and check every
rule on each one. The episodes only visit states the market happens to produce; this visits the
corners on purpose (inventory at and next to the limit, a one-entry history, spreads from 1 to 15
ticks, one-sided books). Same idea as the kernel challenge's hostile shapes and values.

    python probe_worker.py CODEFILE        (prints JSON; checker.probe() is the caller)
"""
import itertools
import json
import sys
import traceback

import mmsim

INVENTORIES = (-10, -9, -5, 0, 5, 9, 10)
SPREADS = (1, 2, 3, 15)
HISTORIES = (1, 2, 100)
BOOKS = ((1, 1), (1, 60), (60, 1))


def main():
    code = open(sys.argv[1]).read()
    fn = mmsim.load_strategy(code)
    findings, n = [], 0
    for inv, spread, hist, (b0, a0) in itertools.product(INVENTORIES, SPREADS, HISTORIES, BOOKS):
        n += 1
        bb, ba = 1000, 1000 + spread
        mid = (bb + ba) / 2
        state = {"t": hist - 1, "T": mmsim.T_STEPS, "best_bid": bb, "best_ask": ba,
                 "bid_sizes": [b0] + [6] * (mmsim.DEPTH - 1), "ask_sizes": [a0] + [6] * (mmsim.DEPTH - 1),
                 "mid_history": [mid] * hist, "inventory": inv, "cash": 0.0,
                 "max_inventory": mmsim.MAX_INV, "max_order_size": mmsim.MAX_SIZE, "memory": {}}
        where = dict(inventory=inv, spread=spread, history=hist, bid0=b0, ask0=a0)
        try:
            q = fn(state)
        except Exception as e:  # noqa: BLE001
            tb = [f for f in traceback.extract_tb(e.__traceback__) if f.filename == "<strategy>"]
            findings.append(dict(where, rule="EXCEPTION",
                                 detail=f"{type(e).__name__}: {str(e)[:120]}"
                                        + (f" (line {tb[-1].lineno})" if tb else "")))
            continue
        for _, rule, detail in mmsim.validate(q, 0, inv, bb, ba)[4]:
            findings.append(dict(where, rule=rule, detail=detail))
    sys.stdout.write(json.dumps(dict(probes=n, findings=findings)))


if __name__ == "__main__":
    main()
