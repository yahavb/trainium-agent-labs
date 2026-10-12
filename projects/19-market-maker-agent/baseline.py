# The naive market maker: quote the touch on both sides, one share, and stop a side only at the
# inventory limit. Earns the spread on level 1; each later level is built to break it.
def quote(state):
    inv, lim = state["inventory"], state["max_inventory"]
    return {"bid": state["best_bid"], "bid_size": 1 if inv < lim else 0,
            "ask": state["best_ask"], "ask_size": 1 if inv > -lim else 0}
