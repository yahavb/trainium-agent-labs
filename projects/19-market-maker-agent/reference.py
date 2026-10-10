# One strategy that passes every calibrated level (1-3). It exists to prove each level is passable, the way the
# kernel ladder ships reference kernels. Never shown to the model.
import math
import numpy as np

USE_IMB, USE_TREND, USE_VOL, USE_INV = True, True, True, True


def quote(state):
    bb, ba = state["best_bid"], state["best_ask"]
    inv, lim = state["inventory"], state["max_inventory"]
    mids = state["mid_history"]
    mid = (bb + ba) / 2
    moves = np.diff(mids[-21:])
    vol = float(np.std(moves)) if len(moves) > 1 else 0.0
    if USE_VOL and vol > 1.0:
        # step back two ticks in a storm: still quoting, rarely filled, only swept by real jumps
        return {"bid": bb - 2, "bid_size": 1 if inv < lim else 0,
                "ask": ba + 2, "ask_size": 1 if inv > -lim else 0}
    trend = (mids[-1] - mids[-41]) / 40 if len(mids) > 40 else 0.0
    imb = math.log(state["bid_sizes"][0] / state["ask_sizes"][0])
    fair = mid + USE_IMB * 0.7 * imb + USE_TREND * 5 * trend - USE_INV * 0.15 * inv
    bid = min(math.floor(fair - 1 + 0.5), ba - 1)
    ask = max(math.floor(fair + 1 + 0.5), bb + 1)
    if bid >= ask:
        bid = ask - 1
    return {"bid": bid, "bid_size": 1 if inv < lim else 0,
            "ask": ask, "ask_size": 1 if inv > -lim else 0}
