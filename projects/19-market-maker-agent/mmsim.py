"""A deterministic market-making simulator. This is the checker's ground truth.

Every market is generated from (level, seed) before the strategy runs, so two strategies on the
same seed face the same prices and the same incoming orders; only their own fills differ. That
makes comparisons between attempts fair (common random numbers) and every number reproducible.

Prices are integers in ticks. The state handed to quote() at step t is built from data up to t
only, so look-ahead is impossible by construction rather than something we hope to catch.

Each level hides one mechanism a market maker has to handle. The strategy is never told which:

  1 calm      uninformed two-sided flow. Quote the touch and you earn the spread.
  2 trend     the price drifts and the flow leans with it. Ignore inventory and you sit at the
              limit, short a rising stock.
  3 informed  a hidden pressure moves the price, tilts the flow and shows up in the displayed
              top-of-book sizes. Quote blind and every fill is followed by a move against you.
  4 volatile  calm spells and storms. In a storm the spread widens a little, the price jumps,
              and the flow is toxic: it trades ahead of the next move.
"""
import json
import math
import sys
import traceback

import numpy as np

T_STEPS = 1500     # steps per episode
MAX_INV = 10       # hard limit on |inventory|
MAX_SIZE = 5       # largest quote size
QUEUE_SHARE = 0.5  # chance an incoming order fills us when we join (not improve) the touch
DEPTH = 5          # displayed price levels per side
HIST = 100         # mids the strategy sees
KMAX = 12          # most market orders per side per step
MARKOUT_H = 10     # steps after a fill at which it is judged
VOL_WIN = 20       # window for realised volatility in diagnostics
OBLIGATION = 0.5   # share of steps that must quote both sides near the touch
NEAR = 2           # "near the touch" means within this many ticks of it

BASE = dict(sigma=0.15, lam=0.4, spread=2,
            mu=0.0, tilt=0.0,                                      # level 2
            alpha=0.0, phi=0.95, kappa=0.0, depth_signal=0.0,      # level 3
            p_storm=0.0, p_calm=0.05, sigma_storm=1.0, spread_storm=3,
            toxic=0.0, p_jump=0.0, jump=6.0)                        # level 4

LEVELS = {
    1: dict(BASE, name="calm"),
    2: dict(BASE, name="trend", mu=0.06, tilt=0.15),
    3: dict(BASE, name="informed", alpha=0.35, kappa=0.9, phi=0.85, depth_signal=0.8),
    # EXPERIMENTAL, not part of the claim: no parameter set we tried (calibrate.py, 2026-10-10)
    # made the naive quoter fail while a simple reference passed, so it is not a fair level yet.
    4: dict(BASE, name="volatile", sigma_storm=1.0, toxic=2.0, p_jump=0.15, jump=4.0,
            p_storm=0.04, p_calm=0.05, spread_storm=2),
}
CALIBRATED = (1, 2, 3)

DEV_SEEDS = list(range(16))


def heldout_seeds():
    """Frozen in eval/heldout_seeds.txt. The agent never sees these; only the final check does."""
    import os
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval", "heldout_seeds.txt")
    return [int(x) for x in open(path).read().split()]


# ---------------------------------------------------------------- the market

def make_market(level, seed, T=T_STEPS, params=None):
    p = params or LEVELS[level]
    rng = np.random.default_rng([level, seed])
    sign = 1.0 if rng.random() < 0.5 else -1.0          # trend direction, level 2
    eps = rng.standard_normal(T)
    eta = rng.standard_normal(T)
    u_storm = rng.random(T)
    u_jump = rng.random(T)
    jsign = np.where(rng.random(T) < 0.5, -1.0, 1.0)
    z = np.empty(T + 1)
    z[0] = rng.standard_normal()
    storm = np.zeros(T + 1, bool)
    V = np.empty(T + 1)
    V[0] = 1000.0 + rng.random()
    dV = np.empty(T)
    for t in range(T):
        sig = p["sigma_storm"] if storm[t] else p["sigma"]
        d = p["mu"] * sign + p["alpha"] * z[t] + sig * eps[t]
        if storm[t] and u_jump[t] < p["p_jump"]:
            d += jsign[t] * p["jump"]
        dV[t] = d
        V[t + 1] = V[t] + d
        z[t + 1] = p["phi"] * z[t] + math.sqrt(1 - p["phi"] ** 2) * eta[t]
        storm[t + 1] = (u_storm[t] >= p["p_calm"]) if storm[t] else (u_storm[t] < p["p_storm"])
    spread = np.where(storm, p["spread_storm"], p["spread"]).astype(np.int64)
    bb = np.floor(V - spread / 2 + 0.5).astype(np.int64)
    ba = bb + spread
    # Incoming market orders during step t. Buyers lean with the trend (level 2), with the hidden
    # pressure z (level 3) and, in a storm, with the move that is about to happen (level 4).
    tox = np.where(storm[:T], np.clip(p["toxic"] * dV / p["sigma_storm"], -3, 3), 0.0)
    lam_buy = p["lam"] * (1 + p["tilt"] * sign) * np.exp(p["kappa"] * z[:T] + tox)
    lam_sell = p["lam"] * (1 - p["tilt"] * sign) * np.exp(-p["kappa"] * z[:T] - tox)
    n_buy = np.minimum(rng.poisson(lam_buy), KMAX)
    n_sell = np.minimum(rng.poisson(lam_sell), KMAX)
    u_buy = rng.random((T, KMAX))
    u_sell = rng.random((T, KMAX))
    # Displayed depth. Random everywhere; on level 3 the top levels also carry the pressure z.
    w = np.array([1.0, 0.5, 0.25, 0.0, 0.0])
    raw = rng.lognormal(math.log(6), 0.4, size=(T + 1, DEPTH, 2))
    sd = p["depth_signal"] * z
    bid_sz = np.maximum(1, np.rint(raw[:, :, 0] * np.exp(+sd[:, None] * w))).astype(np.int64)
    ask_sz = np.maximum(1, np.rint(raw[:, :, 1] * np.exp(-sd[:, None] * w))).astype(np.int64)
    return dict(level=level, seed=seed, T=T, V=V, bb=bb, ba=ba, spread=spread, z=z, storm=storm,
                sign=sign, n_buy=n_buy, n_sell=n_sell, u_buy=u_buy, u_sell=u_sell,
                bid_sz=bid_sz, ask_sz=ask_sz)


# ---------------------------------------------------------------- the rules

QUOTE_KEYS = ("bid", "bid_size", "ask", "ask_size")


def _num(x):
    if isinstance(x, (bool, np.bool_)):
        raise TypeError("a bool is not a number")
    return float(x)


def validate(q, t, inv, bbt, bat):
    """Return (bid, bid_size, ask, ask_size, violations). A side that breaks a rule is withdrawn,
    so the episode keeps running and the checker can still report everything else."""
    bad = []
    if not isinstance(q, dict) or not set(QUOTE_KEYS) <= set(q):
        got = type(q).__name__ if not isinstance(q, dict) else f"keys {sorted(q)}"
        return None, 0, None, 0, [(t, "FORMAT", f"quote() must return a dict with keys "
                                                f"bid, bid_size, ask, ask_size; got {got}")]

    def price(name):
        x = q[name]
        if x is None:
            return None
        try:
            xf = _num(x)
        except (TypeError, ValueError):
            bad.append((t, "FORMAT", f"{name}={x!r} is not a number or None"))
            return None
        if not math.isfinite(xf) or xf != int(xf):
            bad.append((t, "OFF_TICK", f"{name}={x!r} is not a whole number of ticks"))
            return None
        return int(xf)

    def size(name):
        x = q[name]
        try:
            xf = _num(x)
        except (TypeError, ValueError):
            bad.append((t, "FORMAT", f"{name}={x!r} is not a number"))
            return 0
        if not math.isfinite(xf) or xf != int(xf) or xf < 0 or xf > MAX_SIZE:
            bad.append((t, "BAD_SIZE", f"{name}={x!r}; sizes are whole numbers from 0 to {MAX_SIZE}"))
            return 0
        return int(xf)

    bid, ask = price("bid"), price("ask")
    bsz, asz = size("bid_size"), size("ask_size")
    if bid is None:
        bsz = 0
    if ask is None:
        asz = 0
    if bid is not None and ask is not None and bid >= ask:
        bad.append((t, "CROSSED", f"bid {bid} >= ask {ask}"))
        bid, ask, bsz, asz = None, None, 0, 0
    if bid is not None and bid >= bat:
        bad.append((t, "TAKES", f"bid {bid} >= market best ask {bat}: that takes liquidity; "
                                f"a market maker only posts"))
        bid, bsz = None, 0
    if ask is not None and ask <= bbt:
        bad.append((t, "TAKES", f"ask {ask} <= market best bid {bbt}: that takes liquidity; "
                                f"a market maker only posts"))
        ask, asz = None, 0
    if bsz and inv + bsz > MAX_INV:
        bad.append((t, "INVENTORY_LIMIT", f"inventory {inv} with bid_size {bsz} could reach "
                                          f"{inv + bsz}, over the limit {MAX_INV}"))
        bid, bsz = None, 0
    if asz and inv - asz < -MAX_INV:
        bad.append((t, "INVENTORY_LIMIT", f"inventory {inv} with ask_size {asz} could reach "
                                          f"{inv - asz}, under the limit -{MAX_INV}"))
        ask, asz = None, 0
    return bid, bsz, ask, asz, bad


# ---------------------------------------------------------------- one episode

def run_episode(quote_fn, mkt):
    T, bb, ba = mkt["T"], mkt["bb"], mkt["ba"]
    mid = (bb + ba) / 2.0
    inv, cash = 0, 0.0
    memory = {}
    L = dict(bid=np.full(T, np.nan), ask=np.full(T, np.nan),
             bsz=np.zeros(T, np.int64), asz=np.zeros(T, np.int64),
             inv=np.zeros(T + 1, np.int64), cash=np.zeros(T + 1))
    fills = []       # (t, side, price, swept): side +1 we bought, -1 we sold
    violations = []
    for t in range(T):
        lo = max(0, t - HIST + 1)
        state = {
            "t": t, "T": T,
            "best_bid": int(bb[t]), "best_ask": int(ba[t]),
            "bid_sizes": mkt["bid_sz"][t].tolist(), "ask_sizes": mkt["ask_sz"][t].tolist(),
            "mid_history": mid[lo:t + 1].tolist(),
            "inventory": int(inv), "cash": float(cash),
            "max_inventory": MAX_INV, "max_order_size": MAX_SIZE,
            "memory": memory,
        }
        try:
            q = quote_fn(state)
        except Exception as e:  # noqa: BLE001 -- any strategy error is a finding, not a crash
            tb = traceback.extract_tb(e.__traceback__)
            frames = [f for f in tb if f.filename == "<strategy>"]
            line = frames[-1].lineno if frames else None
            return dict(error=dict(step=t, type=type(e).__name__, message=str(e)[:300],
                                   line=line, inventory=int(inv),
                                   best_bid=int(bb[t]), best_ask=int(ba[t]),
                                   history_len=len(state["mid_history"])))
        bid, bsz, ask, asz, bad = validate(q, t, inv, int(bb[t]), int(ba[t]))
        violations.extend(bad)
        if bid is not None:
            L["bid"][t], L["bsz"][t] = bid, bsz
        if ask is not None:
            L["ask"][t], L["asz"][t] = ask, asz
        # incoming market sells hit the best bid; market buys lift the best ask
        rem_b = bsz
        for k in range(int(mkt["n_sell"][t])):
            if rem_b == 0:
                break
            if bid > bb[t] or (bid == bb[t] and mkt["u_sell"][t, k] < QUEUE_SHARE):
                inv += 1
                cash -= bid
                rem_b -= 1
                fills.append((t, 1, bid, 0))
        rem_a = asz
        for k in range(int(mkt["n_buy"][t])):
            if rem_a == 0:
                break
            if ask < ba[t] or (ask == ba[t] and mkt["u_buy"][t, k] < QUEUE_SHARE):
                inv -= 1
                cash += ask
                rem_a -= 1
                fills.append((t, -1, ask, 0))
        # the price moves; a resting quote the market moved through is taken at its price
        if rem_a and bb[t + 1] > ask:
            inv -= rem_a
            cash += ask * rem_a
            fills.extend([(t, -1, ask, 1)] * rem_a)
        if rem_b and ba[t + 1] < bid:
            inv += rem_b
            cash -= bid * rem_b
            fills.extend([(t, 1, bid, 1)] * rem_b)
        L["inv"][t + 1], L["cash"][t + 1] = inv, cash
    liq = abs(inv) * mkt["spread"][T] / 2.0     # flatten at the end by crossing half the spread
    pnl = cash + inv * mid[T] - liq
    return dict(error=None, pnl=float(pnl), liquidation=float(liq), fills=fills,
                violations=violations, log=L, mid=mid)


# ---------------------------------------------------------------- what an episode shows

def episode_metrics(ep, mkt):
    """Everything the checker reports about one episode, plus the loss attribution the located
    feedback is built from. All in ticks."""
    T, mid, L = mkt["T"], ep["mid"], ep["log"]
    bb, ba = mkt["bb"], mkt["ba"]
    inv = L["inv"]
    mtm = L["cash"] + inv * mid
    dd = float(np.max(np.maximum.accumulate(mtm) - mtm))
    both = ~np.isnan(L["bid"]) & ~np.isnan(L["ask"])
    near = both & (L["bid"] >= bb[:T] - NEAR) & (L["ask"] <= ba[:T] + NEAR)
    dmid = np.diff(mid)
    # PnL from holding inventory while the price moves, step by step
    hold = inv[:T] * dmid
    # realised volatility the strategy could have seen at each step
    rv = np.zeros(T)
    for t in range(1, T):
        w = dmid[max(0, t - VOL_WIN):t]
        rv[t] = float(np.std(w)) if len(w) > 1 else 0.0
    hot = rv > 1.0
    f = np.array(ep["fills"], dtype=float).reshape(-1, 4)
    edge = markout = np.zeros(0)
    against = np.zeros(0, bool)
    table_n = [[0, 0, 0], [0, 0, 0]]
    table_sum = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    worst = []
    if len(f):
        ft, side, px = f[:, 0].astype(int), f[:, 1], f[:, 2]
        edge = side * (mid[ft] - px)                          # captured at the fill
        markout = side * (mid[np.minimum(ft + 1 + MARKOUT_H, T)] - px)
        b0 = mkt["bid_sz"][ft, 0].astype(float)
        a0 = mkt["ask_sz"][ft, 0].astype(float)
        imb = (b0 - a0) / (b0 + a0)
        # trading against the book: selling while bids outweigh asks, or buying the reverse
        against = side * imb < -0.2
        # counterexample material for run C5: fills by side x imbalance bucket, and the worst ones
        bucket = np.digitize(imb, [-0.3, 0.3])                # 0 asks heavy, 1 balanced, 2 bids heavy
        for si, sv in ((0, 1), (1, -1)):                      # 0 = you bought, 1 = you sold
            for b in range(3):
                sel = (side == sv) & (bucket == b)
                table_n[si][b] = int(sel.sum())
                table_sum[si][b] = float(markout[sel].sum())
        seen_steps = set()
        for i in np.argsort(markout):
            if len(worst) == 2:
                break
            if (int(ft[i]), int(side[i])) in seen_steps:
                continue
            seen_steps.add((int(ft[i]), int(side[i])))
            worst.append(dict(step=int(ft[i]), side="sold" if side[i] < 0 else "bought",
                              price=int(px[i]), bid0=int(b0[i]), ask0=int(a0[i]),
                              mid=float(mid[ft[i]]),
                              mid_later=float(mid[min(ft[i] + 1 + MARKOUT_H, T)])))
    # quote placement relative to mid, where both sides were posted
    centre = np.where(both, (L["bid"] + L["ask"]) / 2 - mid[:T], np.nan)
    half = np.where(both, (L["ask"] - L["bid"]) / 2, np.nan)

    def corr(x, y):
        m = ~np.isnan(x) & ~np.isnan(y)
        if m.sum() < 30 or np.std(x[m]) == 0 or np.std(y[m]) == 0:
            return 0.0
        return float(np.corrcoef(x[m], y[m])[0, 1])

    b0s = mkt["bid_sz"][:T, 0].astype(float)
    a0s = mkt["ask_sz"][:T, 0].astype(float)
    imb_t = (b0s - a0s) / (b0s + a0s)
    return dict(
        pnl=ep["pnl"],
        fills=int(len(f)), swept=int(f[:, 3].sum()) if len(f) else 0,
        swept_loss=float(-np.sum(np.minimum(markout[f[:, 3] == 1], 0))) if len(f) else 0.0,
        edge_pnl=float(edge.sum()), hold_pnl=float(hold.sum()), liquidation=ep["liquidation"],
        markout_mean=float(markout.mean()) if len(markout) else 0.0,
        against_n=int(against.sum()),
        against_markout=float(markout[against].mean()) if against.any() else 0.0,
        with_markout=float(markout[~against].mean()) if (~against).any() else 0.0,
        against_pnl=float(markout[against].sum()) if against.any() else 0.0,
        hot_steps=int(hot.sum()),
        hot_pnl=float(np.sum(np.diff(mtm)[hot])),
        calm_pnl=float(np.sum(np.diff(mtm)[~hot])),
        hot_half=float(np.nanmean(half[hot])) if np.any(hot & both) else float("nan"),
        calm_half=float(np.nanmean(half[~hot])) if np.any(~hot & both) else float("nan"),
        at_limit=float(np.mean(np.abs(inv[1:]) >= MAX_INV)),
        heavy_pnl=float(hold[np.abs(inv[:T]) >= 0.7 * MAX_INV].sum()),
        mean_abs_inv=float(np.mean(np.abs(inv))),
        obligation=float(near.mean()),
        both_sides=float(both.mean()),
        skew_inv=corr(centre, inv[:T].astype(float)),
        skew_imb=corr(centre, imb_t),
        widen_vol=corr(half, rv),
        max_drawdown=dd,
        table_n=table_n, table_sum=table_sum, worst=worst,
    )


# ---------------------------------------------------------------- loading a strategy

def load_strategy(code):
    ns = {"__name__": "strategy"}
    exec(compile(code, "<strategy>", "exec"), ns)
    fn = ns.get("quote")
    if not callable(fn):
        raise ValueError("the module does not define a function quote(state)")
    return fn


def evaluate(code, level, seeds, keep=False):
    """Run one strategy on every seed of a level. Plain data out, so it can cross a process."""
    fn = load_strategy(code)
    per, viol = [], []
    for s in seeds:
        mkt = make_market(level, s)
        ep = run_episode(fn, mkt)
        if ep["error"]:
            return dict(error=dict(ep["error"], seed=s), episodes=per, violations=viol)
        m = episode_metrics(ep, mkt)
        m["seed"] = s
        per.append(m)
        viol.extend([(s,) + v for v in ep["violations"]])
    return dict(error=None, episodes=per, violations=viol)


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "--worker":
    # Child-process entry used by checker.py: python mmsim.py --worker CODEFILE LEVEL SEEDS
    _code = open(sys.argv[2]).read()
    _level = int(sys.argv[3])
    _seeds = [int(x) for x in sys.argv[4].split(",")]
    try:
        out = evaluate(_code, _level, _seeds)
    except Exception as e:  # noqa: BLE001
        tb = traceback.extract_tb(e.__traceback__)
        frames = [f for f in tb if f.filename == "<strategy>"]
        out = dict(error=dict(step=None, type=type(e).__name__, message=str(e)[:300],
                              line=frames[-1].lineno if frames else None, seed=None),
                   episodes=[], violations=[])
    sys.stdout.write(json.dumps(out))
