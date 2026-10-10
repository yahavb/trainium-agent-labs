"""Prompt-leak scan: does any line of a forbidden source appear in a prompt the agent sent, or in a
string constant that can be put into one? Writes /tmp/leak_scan.json."""
import ast, glob, json, re, sys

ROOT = "/h/tal-deliv/projects/02-kernel-agent"
FORBIDDEN = {
    "tutorial kernels": sorted(glob.glob("/h/neuron-agentic-development/skills/*/references/downloads/*_nki_kernels.py")),
    "organizers' reference kernels": [f"{ROOT}/reference_level{i}.py" for i in (1, 2, 3, 4)],
    "our answers (levels 9-14)": [f"{ROOT}/answers/ans_level{i}.py" for i in range(9, 15)],
}
CODE = ["agent.py", "feedback_v2.py", "feedback_v3.py", "feedback_v4.py", "feedback_v5.py",
        "feedback_v6.py", "feedback_v7.py", "gate_nki.py", "verdict_nki.py", "hidden_eval.py",
        "ops07.py", "ops08.py"]


def norm(line):
    return re.sub(r"\s+", " ", line.strip())


def keep(line):
    s = norm(line)
    return (len(s) >= 40 and not s.startswith(("import ", "from ")) and len(s.split()) > 2)


forb = {}   # normalized line -> [(group, file, lineno)]
n_forb = 0
for group, files in FORBIDDEN.items():
    for f in files:
        for i, line in enumerate(open(f), 1):
            if keep(line):
                forb.setdefault(norm(line), []).append((group, f.replace("/h/", ""), i))
                n_forb += 1

reqs = [json.loads(l) for l in open("/tmp/leak_prompts.jsonl")]
prompts = {}
for r in reqs:
    p = r["body"]["messages"][0]["content"]
    prompts.setdefault(p, dict(level=r["level"], n=0, first=not p.startswith("This NKI kernel for")))
    prompts[p]["n"] += 1

consts = []   # (file, lineno, text)
for f in CODE:
    tree = ast.parse(open(f"{ROOT}/{f}").read())
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            consts.append((f, node.lineno, node.value))
        elif isinstance(node, ast.JoinedStr):
            for v in node.values:
                if isinstance(v, ast.Constant) and isinstance(v.value, str):
                    consts.append((f, node.lineno, v.value))

hits = []
for p, meta in prompts.items():
    for line in p.splitlines():
        if keep(line) and norm(line) in forb:
            hits.append(dict(where="prompt", level=meta["level"], first=meta["first"], line=norm(line),
                             sources=forb[norm(line)]))
for f, ln, text in consts:
    for line in text.splitlines():
        if keep(line) and norm(line) in forb:
            hits.append(dict(where=f"{f}:{ln}", line=norm(line), sources=forb[norm(line)]))

ap = dict(prompts=sum(1 for p in prompts if ".ap([" in p),
          constants=[f"{f}:{ln}" for f, ln, t in consts if ".ap([" in t],
          forbidden=[f.replace("/h/", "") for fs in FORBIDDEN.values() for f in fs if ".ap([" in open(f).read()])
levels = sorted({m["level"] for m in prompts.values()})
json.dump(dict(requests=len(reqs), prompts=len(prompts),
               first=sum(m["first"] for m in prompts.values()),
               repair=sum(not m["first"] for m in prompts.values()), levels=levels,
               per_level={lv: [sum(1 for m in prompts.values() if m["level"] == lv and m["first"]),
                               sum(1 for m in prompts.values() if m["level"] == lv and not m["first"])]
                          for lv in levels},
               forbidden_files={g: [f.replace("/h/", "") for f in fs] for g, fs in FORBIDDEN.items()},
               forbidden_lines=n_forb, distinct_forbidden=len(forb), constants=len(consts),
               code_files=CODE, hits=hits, ap=ap), open("/tmp/leak_scan.json", "w"), indent=1)
print(f"{len(reqs)} requests, {len(prompts)} distinct prompts, {len(consts)} string constants, "
      f"{len(forb)} distinct forbidden lines; hits: {len(hits)}; .ap([ in prompts: {ap['prompts']}, "
      f"in constants: {ap['constants']}, in forbidden: {ap['forbidden']}")
for h in hits[:20]:
    print(" ", h["where"], "|", h["line"][:100], "|", h["sources"][:2])
