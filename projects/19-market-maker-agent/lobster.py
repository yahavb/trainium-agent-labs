"""Replay a real trading day (LOBSTER sample files, Nasdaq, 2012-06-21) through the same simulator.

A sanity check, not part of the pass/fail claim. The synthetic levels let the checker know the true
mechanism; this asks whether strategies written against them do anything sensible on a real book.

The day becomes the same arrays mmsim.make_market() returns, so run_episode() and episode_metrics()
are reused unchanged:
  * one step = one second, 09:45 to 15:45 (the first and last 15 minutes are skipped), cut into
    episodes of 1,500 steps
  * the book at step t is the LOBSTER book after the last event at or before that second; prices
    are in cents (one tick = $0.01)
  * market orders during step t are the executions (types 4 and 5) in the following second. All
    executions sharing one timestamp and side count as one order (a market order walking the book
    prints several). An executed buy limit order is a seller-initiated trade, so it hits bids.

Limits: the replay cannot react to our quotes (no impact, no queue position beyond the same 50%
rule), and our size is 1-5 shares against real orders of hundreds.

    python lobster.py --build                         # CSVs -> real/<TICKER>.npz (run once)
    python lobster.py baseline.py reference.py        # score strategy files on every ticker
    python lobster.py --from-runs                     # score each run's final strategy
"""
import argparse
import glob
import json
import os

import numpy as np

import mmsim

HERE = os.path.dirname(os.path.abspath(__file__))
REAL = os.path.join(HERE, "real")
DATA = os.environ.get("LOBSTER_DIR",
                      os.path.join(HERE, "..", "..", "..", "data", "lobster"))
TICKERS = ("MSFT", "INTC")   # seen: the replay that motivated run C4
UNSEEN = ("GOOG", "AAPL")    # opened only after C4 finished: the out-of-sample check
START, END = 34200 + 900, 57600 - 900   # seconds after midnight
EMPTY = 9_999_999_999                   # LOBSTER's dummy price for an empty level


def build(ticker):
    msg_f = glob.glob(os.path.join(DATA, ticker, f"{ticker}_*_message_5.csv"))[0]
    book_f = glob.glob(os.path.join(DATA, ticker, f"{ticker}_*_orderbook_5.csv"))[0]
    msg = np.loadtxt(msg_f, delimiter=",")
    book = np.loadtxt(book_f, delimiter=",", dtype=np.int64)
    t_msg, typ, size, direction = msg[:, 0], msg[:, 1].astype(int), msg[:, 3], msg[:, 5].astype(int)
    grid = np.arange(START, END + 1)                      # N + 1 book snapshots
    idx = np.searchsorted(t_msg, grid, side="right") - 1
    b = book[idx]
    ask_px, ask_sz = b[:, 0::4], b[:, 1::4]
    bid_px, bid_sz = b[:, 2::4], b[:, 3::4]
    empty_a, empty_b = np.abs(ask_px) >= EMPTY, np.abs(bid_px) >= EMPTY
    ask_sz = np.where(empty_a, 0, ask_sz)
    bid_sz = np.where(empty_b, 0, bid_sz)
    bb, ba = bid_px[:, 0] // 100, ask_px[:, 0] // 100     # price x 10000 -> cents
    ex = (typ == 4) | (typ == 5)
    n = len(grid) - 1
    counts = {}
    for side, d in (("sell", 1), ("buy", -1)):
        m = ex & (direction == d)
        stamps = np.unique(t_msg[m])                       # one market order per timestamp
        step = np.floor(stamps - START).astype(int)        # executions in (t, t+1] -> step t
        step = step[(step >= 0) & (step < n)]
        counts[side] = np.bincount(step, minlength=n)
    os.makedirs(REAL, exist_ok=True)
    out = os.path.join(REAL, f"{ticker}.npz")
    np.savez_compressed(out, bb=bb, ba=ba, bid_sz=bid_sz[:, :mmsim.DEPTH],
                        ask_sz=ask_sz[:, :mmsim.DEPTH], n_buy=counts["buy"], n_sell=counts["sell"])
    sp = ba - bb
    print(f"{ticker}: {n} steps, mid ${(bb[0] + ba[0]) / 200:.2f} -> ${(bb[-1] + ba[-1]) / 200:.2f}, "
          f"spread 1 tick {np.mean(sp == 1):.0%} / 2 {np.mean(sp == 2):.0%} / more "
          f"{np.mean(sp > 2):.0%}; market buys {counts['buy'].sum()}, sells {counts['sell'].sum()} "
          f"-> {out}")


def episodes(ticker):
    d = np.load(os.path.join(REAL, f"{ticker}.npz"))
    T = mmsim.T_STEPS
    n = len(d["n_buy"])
    rng = np.random.default_rng(list(map(ord, ticker)))
    out = []
    for e in range(n // T):
        a, z = e * T, e * T + T
        out.append(dict(level=f"real-{ticker}", seed=e, T=T,
                        bb=d["bb"][a:z + 1], ba=d["ba"][a:z + 1],
                        spread=(d["ba"] - d["bb"])[a:z + 1],
                        bid_sz=np.maximum(1, d["bid_sz"][a:z + 1]),
                        ask_sz=np.maximum(1, d["ask_sz"][a:z + 1]),
                        n_buy=np.minimum(d["n_buy"][a:z], mmsim.KMAX),
                        n_sell=np.minimum(d["n_sell"][a:z], mmsim.KMAX),
                        u_buy=rng.random((T, mmsim.KMAX)), u_sell=rng.random((T, mmsim.KMAX))))
    return out


def score(code, ticker):
    import checker
    bad = checker.static_check(code)        # same gate as the checker: math/numpy only, no I/O
    if bad:
        return dict(error=f"static check: {bad[1]}")
    fn = mmsim.load_strategy(code)
    pnl, viol, per = [], {}, []
    for mkt in episodes(ticker):
        ep = mmsim.run_episode(fn, mkt)
        if ep["error"]:
            return dict(error=f"{ep['error']['type']}: {ep['error']['message']} (episode "
                              f"{mkt['seed']}, step {ep['error']['step']})")
        m = mmsim.episode_metrics(ep, mkt)
        per.append(m)
        pnl.append(m["pnl"])
        for v in ep["violations"]:
            viol[v[1]] = viol.get(v[1], 0) + 1
    p = np.array(pnl)
    se = p.std(ddof=1) / np.sqrt(len(p))
    g = lambda k: float(np.mean([m[k] for m in per]))  # noqa: E731
    return dict(error=None, n=len(p), pnl=float(p.mean()), se=float(se), lcb=float(p.mean() - 2 * se),
                win=float(np.mean(p > 0)), fills=g("fills"), markout=g("markout_mean"),
                at_limit=g("at_limit"), obligation=g("obligation"), skew_inv=g("skew_inv"),
                skew_imb=g("skew_imb"), violations=viol)


def row(name, r):
    if r.get("error"):
        return f"| {name} | error: {r['error'][:90]} |||||||"
    v = ", ".join(f"{k} {c}" for k, c in sorted(r["violations"].items())) or "0"
    return (f"| {name} | {r['pnl']:+.0f} ± {r['se']:.0f} | {r['lcb']:+.0f} | {r['win']:.0%} | "
            f"{r['fills']:.0f} | {r['markout']:+.2f} | {r['obligation']:.0%} | {v} |")


HEAD = ("| strategy | PnL per episode (ticks) | lcb | episodes won | fills | markout (10 s) | "
        "obligation | rule violations (side withdrawn) |\n|---|---|---|---|---|---|---|---|")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--from-runs", action="store_true")
    ap.add_argument("--tickers", nargs="+", default=list(TICKERS))
    ap.add_argument("--out", default="real_data.md")
    ap.add_argument("files", nargs="*")
    a = ap.parse_args()
    if a.build:
        for t in a.tickers:
            build(t)
        return
    entries = [(os.path.basename(f), open(f).read()) for f in a.files]
    if a.from_runs:
        for path in sorted(glob.glob(os.path.join(HERE, "runs", "*.jsonl"))):
            for line in open(path):
                r = json.loads(line)
                if r.get("final") and r.get("code"):
                    tag = "solved" if r["dev_solved"] else "unsolved"
                    entries.append((f"{r['run']} rep {r['rep']} level {r['level']} ({tag})", r["code"]))
    lines = []
    for t in a.tickers:
        lines += [f"\n### {t}, 2012-06-21, {len(episodes(t))} episodes of {mmsim.T_STEPS} s\n", HEAD]
        for name, code in entries:
            lines.append(row(name, score(code, t)))
    text = "\n".join(lines)
    print(text)
    if a.from_runs:
        os.makedirs(os.path.join(HERE, "runs"), exist_ok=True)
        open(os.path.join(HERE, "runs", a.out), "w").write(
            "# Real-data replay (sanity check, not part of the claim)\n" + text + "\n")


if __name__ == "__main__":
    main()
