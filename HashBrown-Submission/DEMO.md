# DEMO.md — 2-minute live demo script

Every command below was run once against seat-66 to confirm it works, and timed. Total
command execution time: **~7.2 seconds**; the rest of the 2 minutes is narration. All
commands are read-only `--check` calls — none of them touch the running baseline or
regression agent jobs.

Run these from your laptop shell (they `kubectl exec` into the pod); swap `seat-66` for your
own seat if different.

---

## (a) Cheat C3, against the ORIGINAL checker — passes at 1.0

`cheats/c3_hardcoded_shape.py` is a matmul kernel that hardcodes `assert K==128, M==64,
N==512` — it only works at exactly one shape. The ORIGINAL checker (frozen in
`/workspace/baseline-copy` before any hardening) declared only that one shape for level 3, so
this cheat passes it completely.

```
kubectl exec seat-66 -- bash -c "cd /workspace/baseline-copy && python nkibench.py --level 3 --check /workspace/projects/02-kernel-agent/cheats/c3_hardcoded_shape.py"
```

**Measured: 1.9s.** Say while it prints: *"One shape, one assert, `numerics 1/1 shapes
passed` — the original checker has nothing to disagree with."*

Expected last line before the roofline note: `numerics   1/1 shapes passed`

---

## (b) The SAME cheat, against the HARDENED checker — rejected, with the exact reason

No file was copied anywhere — this is the identical cheat file, now checked by the current,
hardened `nkibench.py`, which has a SECOND declared shape for level 3
(`K=64, M=32, N=256`) added specifically because this cheat's hardcoded assert can't
survive it.

```
kubectl exec seat-66 -- bash -c "cd /workspace/projects/02-kernel-agent && python nkibench.py --level 3 --check cheats/c3_hardcoded_shape.py"
```

**Measured: 1.8s, exit code 1.** Say while it prints: *"Same file, same cheat — now it fails
on the shape that wasn't there before, with the exact assertion it tripped."*

Expected output includes:
```
numerics   1/2 checks passed

  case K=64 M=32 N=256:
    RAISED during simulation: AssertionError: expected K=128, got 64
```

---

## (c) C9, a SMARTER cheat — passes the hardened fixed tests, caught by `--augment`

`cheats/c9_memorise_all_shapes.py` memorized BOTH of level 3's current declared shapes (not
just one), so step (b)'s fix alone doesn't catch it. First, show it passing the hardened
checker with augmentation OFF:

```
kubectl exec seat-66 -- bash -c "cd /workspace/projects/02-kernel-agent && python nkibench.py --level 3 --check cheats/c9_memorise_all_shapes.py"
```

**Measured: 1.7s, exit code 0.** Expected last line: `numerics   2/2 checks passed` — a
clean pass, both declared shapes recognized.

Now turn augmentation on — same cheat, same checker, one flag added:

```
kubectl exec seat-66 -- bash -c "cd /workspace/projects/02-kernel-agent && python nkibench.py --level 3 --check cheats/c9_memorise_all_shapes.py --augment"
```

**Measured: 1.9s, exit code 1.** Say while it prints: *"`--augment` adds one more shape in
the same class — not memorized, so it falls back to its zero-output branch and gets caught
immediately."*

Expected output includes:
```
augment    tiers=['a1', 'a2a', 'a2b', 'a3', 'a4']  +1 A2a shape(s), +1 A2b ragged shape(s), seed=...
numerics   16/18 checks passed

  case K=96 M=48 N=384:
    NON-FINITE OUTPUT: 18432 NaN and 0 Inf, first at (0, 0). ...
```

---

## Closing line for the demo

*"Three structural fixes — a second shape, a measured tolerance, a tamper check — closed
every cheat we could build by hand without needing augmentation at all. The one exception,
a cheat that memorized the whole test set instead of one shape, needed exactly one more
shape to fall over. That's the story: harden the fixed checks first; augmentation is for the
smarter attacker, not the first one."*
