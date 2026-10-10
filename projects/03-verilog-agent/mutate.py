#!/usr/bin/env python3
"""Mutation testing: how good is a testbench?

Plant one fake bug at a time in a CORRECT design (flip + to -, & to |, change a constant, ...),
run the testbench on each broken copy ("mutant"), and count how many it catches ("kills").

  mutation score = killed / (mutants that compile)

  python mutate.py --problem all                      # hand-written testbenches (no AI needed)
  python mutate.py --problem all --ai --repeat 3      # AI-written testbench vs hand-written, per problem
  python mutate.py --design result_1.v --tb result_1_tb.v   # score any design + testbench

A surviving mutant means either the testbench missed a bug, or the change didn't really change
behaviour (an "equivalent mutant"). Survivors are printed so a human can tell which.
"""
import re, sys, json, argparse
import vagent
from vagent import P, simulate, auto_tb, header

# (name, regex, replacement). One mutant per match site, so each mutant has exactly one bug.
OPS = [
    ("+ to -",        r"(?<![+\-])\+(?![+=:])",   "-"),
    ("- to +",        r"(?<![-+<])-(?![-=>:])",   "+"),
    ("drop '+ term'", r"\s*\+\s*\w+(?:\[[^\]]*\])?", ""),
    ("* to +",        r"(?<![*/])\*(?![*)])",     "+"),
    ("& to |",        r"(?<!&)&(?!&)",            "|"),
    ("| to &",        r"(?<!\|)\|(?!\|)",         "&"),
    ("^ to |",        r"(?<!~)\^(?!~)",           "|"),
    ("&& to ||",      r"&&",                      "||"),
    ("|| to &&",      r"\|\|",                    "&&"),
    ("== to !=",      r"(?<![=!])==(?!=)",        "!="),
    ("!= to ==",      r"!=(?!=)",                 "=="),
    ("<< to >>",      r"<<",                      ">>"),
    (">> to <<",      r">>",                      "<<"),
    ("negate if",     r"\bif\s*\(",               "if (!"),
    ("drop ~",        r"~(?![\^&|])",             ""),
    ("drop !",        r"!(?!=)",                  ""),
    ("1'b0 to 1'b1",  r"1'b0",                    "1'b1"),
    ("1'b1 to 1'b0",  r"1'b1",                    "1'b0"),
    ("constant +1",   r"(\d+)'([dDhHbB])([0-9a-fA-F_]+)",
                      lambda m: f"{m[1]}'d{int(m[3].replace('_',''), {'d':10,'h':16,'b':2}[m[2].lower()]) + 1}"),
    ("number +1",     r"(?<![\w'\[:.$#])(\d+)(?![\w'\]:.])", lambda m: str(int(m[1]) + 1)),
]

def strip_comments(code):
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.S)
    return re.sub(r"//[^\n]*", "", code)

def mutants(code, limit=60):
    """Return [(description, mutated_code)]. The module header (ports) is never touched."""
    code = strip_comments(code)
    hdr = header(code)
    start = code.find(hdr[:-1]) + len(hdr) - 1 if hdr else 0   # mutate only after the header
    head, body = code[:start], code[start:]
    out, seen = [], {code}
    for name, rx, rp in OPS:
        for m in re.finditer(rx, body):
            new = m.expand(rp) if isinstance(rp, str) else rp(m)
            mb = body[:m.start()] + new + body[m.end():]
            full = head + mb
            if full in seen:
                continue
            seen.add(full)
            line_no = head.count("\n") + body[:m.start()].count("\n") + 1
            line = mb.splitlines()[body[:m.start()].count("\n")].strip()
            out.append((f"line {line_no}: {name:13s} -> {line}", full))
    return out[:limit]

def score(code, tb, show=5, quiet=False):
    """Run the testbench on every mutant. Returns dict with killed/total/survivors."""
    base = simulate(code, tb)[0]
    if base != 1.0:
        if not quiet:
            print("  ! the ORIGINAL design fails this testbench, so mutation scoring is meaningless")
        return dict(killed=0, total=0, stillborn=0, survivors=[], valid=False)
    killed, stillborn, survivors = 0, 0, []
    for desc, mut in mutants(code):
        s, det, _ = simulate(mut, tb)
        if det.startswith("COMPILE ERROR"):
            stillborn += 1          # didn't compile: not a real test of the testbench, skip it
        elif s < 1.0:
            killed += 1
        else:
            survivors.append(desc)
    total = killed + len(survivors)
    if not quiet:
        pct = f"{100 * killed / total:.0f}%" if total else "-"
        print(f"  killed {killed}/{total} mutants ({pct})   [{stillborn} didn't compile, skipped]")
        for d in survivors[:show]:
            print(f"    survived  {d}")
        if len(survivors) > show:
            print(f"    ... {len(survivors) - show} more survivors")
    return dict(killed=killed, total=total, stillborn=stillborn, survivors=survivors, valid=True)

def pct(r):
    return f"{100 * r['killed'] / r['total']:.0f}%" if r.get("total") else "-"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", help="built-in problem name, or 'all'")
    ap.add_argument("--design", help="a correct design file (.v)")
    ap.add_argument("--tb", help="testbench file to score (with --design)")
    ap.add_argument("--ai", action="store_true", help="also score an AI-written testbench (needs the model)")
    ap.add_argument("--repeat", type=int, default=1, help="AI testbenches per problem (rates, not luck)")
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--show", type=int, default=5, help="how many survivors to print")
    ap.add_argument("--log", default="mutation.jsonl")
    a = ap.parse_args()
    vagent.SAMPLES = max(1, a.samples)

    if a.design:
        code = open(a.design).read()
        tb = open(a.tb).read() if a.tb else None
        if not tb:
            sys.exit("need --tb")
        print(f"=== {a.design} with {a.tb} ===")
        score(code, tb, a.show)
        return

    if not a.problem:
        ap.print_help()
        return
    names = list(P) if a.problem == "all" else a.problem.split(",")
    table = []
    with open(a.log, "a") as log:
        for n in names:
            ref, spec = P[n]["ref"], P[n]["spec"]
            print(f"\n=== {n}: hand-written testbench ===")
            hand = score(ref, P[n]["tb"], a.show)
            row = dict(problem=n, hand=hand, ai=[])
            if a.ai:
                for rep in range(1, a.repeat + 1):
                    print(f"=== {n}: AI-written testbench, rep {rep}/{a.repeat} ===")
                    tb = auto_tb(spec, ref, log, f"mut-{n}")
                    r = score(ref, tb, a.show) if tb else dict(killed=0, total=0, survivors=[], valid=False)
                    row["ai"].append(r)
                    log.write(json.dumps(dict(problem=n, kind="mutation", rep=rep, tb=tb,
                                              killed=r["killed"], total=r["total"],
                                              survivors=r["survivors"])) + "\n")
                    log.flush()
            table.append(row)

    print("\n===== MUTATION SCORE SUMMARY =====")
    print(f"{'problem':10s} {'hand tb':>10s}   AI tb (each rep)")
    for row in table:
        ai = "  ".join(pct(r) if r.get("valid") else "no-tb" for r in row["ai"]) or "(not run)"
        print(f"{row['problem']:10s} {pct(row['hand']):>10s}   {ai}")

if __name__ == "__main__":
    main()
