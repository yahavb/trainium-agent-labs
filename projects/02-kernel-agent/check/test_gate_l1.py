"""Check gate_nki's two l1fix rules (partition_fixes, dst_shape_fixes) against kernels whose trn2 verdict was
MEASURED on seat-115 (compile_solves7.py: full neuronx-cc build + birsim, shapes (4,8,8)/2 and (8,12,12)/3).
Run like test_v7.py, with this folder first on PYTHONPATH so the patched gate_nki.py is the one imported:

    PYTHONPATH=<this folder>:<their 02-kernel-agent>:<kit> MESSAGES=v5 CARD=category PROMPT1=v2 \\
        NEURON_PLATFORM_TARGET_OVERRIDE=trn2 python test_gate_l1.py

Fixtures (fixtures/): 16 of the 17 distinct level-1 kernels that scored 1.0 in the 4090 rehearsal logs (the 17th, a verbatim copy
of the organizers' reference, is left out), 4 hand
variants (l1_hand_*), and the gate's own rewrites of the failing ones (l1_rew_*; l1_rewP_3fc74c23 is the
partition rewrite alone, before the dst fix). Also: every rewrite passes the simulator (agent.grade 1.0).
"""
import glob
import os
import sys

import gate_nki

HERE = os.path.dirname(os.path.abspath(__file__))
# file stem -> measured: True = builds for trn2 and birsim matches, False = the build fails
COMPILES = {
    "l1_32c697c5": True, "l1_3fbcebcf": True, "l1_41729d4f": True, "l1_942b52d9": True,
    "l1_b749b8b5": True, "l1_d2336a02": True, "l1_e845b12e": True,
    "l1_387db587": False, "l1_3fc74c23": False, "l1_43b0b66a": False, "l1_a13041c4": False,
    "l1_d83c07ee": False, "l1_dd7a9648": False, "l1_fd6dae25": False,
    "l1_81207765": False, "l1_d4522ec9": False,      # task 05: nisa.tensor_reduce dst (1, p); not caught
    "l1_hand_neg_dma_channel": True, "l1_hand_neg_hbm_channel": True, "l1_hand_neg_rows_loop": True,
    "l1_hand_pos_slice_channel": False,
    "l1_rew_387db587": True, "l1_rew_3fc74c23_D": True, "l1_rew_43b0b66a": True, "l1_rew_a13041c4": True,
    "l1_rew_d83c07ee": True, "l1_rew_fd6dae25": True, "l1_rewP_3fc74c23": False,
    "l1_rew_hand_pos_slice_channel": True,
}
KNOWN_MISS = {"l1_81207765", "l1_d4522ec9"}

tp = fp = tn = fn = 0
bad = []
for p in sorted(glob.glob(os.path.join(HERE, "fixtures", "*.py"))):
    stem = os.path.basename(p)[:-3]
    code = open(p).read()
    new = gate_nki.partition_fixes(code) + gate_nki.dst_shape_fixes(code)
    flagged, ok = bool(new), COMPILES[stem]
    tp += flagged and not ok
    fp += flagged and ok
    tn += not flagged and ok
    fn += not flagged and not ok
    if flagged == ok and stem not in KNOWN_MISS:
        bad.append(stem)
    if flagged:                                    # the code the message gives must clear the partition rule
        lines, coded = code.splitlines(), False
        for a0, a1, _, rep, _ in sorted(gate_nki.fixes(code), reverse=True):
            if rep:
                lines[a0 - 1:a1], coded = rep, True
        if coded and gate_nki.partition_fixes("\n".join(lines) + "\n"):
            bad.append(stem + " (rewrite still flagged)")
    print(f"{'FLAG' if flagged else '    '} {'builds' if ok else 'FAILS '} {stem}"
          + (f"  <- {new[0][4][:60]}" if new else ""))

print(f"\nflagged & fails {tp}, flagged & builds {fp}, not flagged & builds {tn}, not flagged & fails {fn} "
      f"(known misses: {len(KNOWN_MISS)})")
print("FAIL: " + ", ".join(bad) if bad else "0 failed")
sys.exit(1 if bad else 0)
