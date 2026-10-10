"""
optimize_agent.py  (Member 5)
Controller for the optimization loop:

  Member 2's JSON -> Vprev.v + testbench.v
  Vprev -> Qwen -> Vnew -> Yosys compare
     same/more cells -> stop, return Vprev
     fewer cells     -> Member 3 checker(Vprev, Vnew, testbench)
         approved    -> Vprev = Vnew, repeat
         rejected    -> stop, return None (Member 3 handles the output)

Vprev is ALWAYS a verified circuit. Unverified Qwen output never replaces it.

Use from the team's main program:
    from optimize_agent import run
    from member3_file import check          # Member 3's real function
    final_path = run("member2_output.json", checker=check)

Or from the terminal:
    python optimize_agent.py member2_output.json --checker member3_file:check
"""

import argparse
import importlib
import json
import os
import shutil

from qwen_client import optimize_verilog
from yosys_checker import count_cells, synthesize

MAX_ROUNDS = 8
WORK_DIR = "opt_work"

# Key names in Member 2's JSON. Change these to match her real JSON.
VERILOG_KEYS = ["verilog", "verilog_path", "vprev", "design", "rtl"]
TESTBENCH_KEYS = ["testbench", "testbench_path", "tb", "tb_path"]


def _find_key(data, candidates, what):
    for key in candidates:
        if data.get(key):
            return data[key]
    raise KeyError(f"Member 2's JSON has no {what} path. Expected one of {candidates}, "
                   f"got keys {list(data.keys())}")


def read_member2_json(json_path):
    """Return (vprev_path, testbench_path) from Member 2's JSON."""
    with open(json_path) as f:
        data = json.load(f)

    vprev = _find_key(data, VERILOG_KEYS, "Verilog")
    tb = _find_key(data, TESTBENCH_KEYS, "testbench")

    # Relative paths are treated as relative to the JSON file's folder
    base = os.path.dirname(os.path.abspath(json_path))
    vprev = vprev if os.path.isabs(vprev) else os.path.join(base, vprev)
    tb = tb if os.path.isabs(tb) else os.path.join(base, tb)

    for path, what in [(vprev, "Verilog"), (tb, "testbench")]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{what} file from Member 2's JSON not found in this pod: {path}")
    return vprev, tb


def run(json_path, checker, max_rounds=MAX_ROUNDS, work_dir=WORK_DIR):
    """
    json_path : Member 2's JSON with paths to the verified Verilog and its testbench
    checker   : Member 3's function, called as checker(vprev_path, vnew_path, testbench_path).
                Truthy return = approved. None / False = rejected.
    Returns the path to the latest verified Verilog,
    or None if Member 3 rejected (she returns the final circuit in that case).
    """
    # Step 1-2: read Member 2's JSON
    vprev_path, testbench = read_member2_json(json_path)
    print(f"[agent] Verilog:   {vprev_path}\n[agent] Testbench: {testbench}")

    os.makedirs(work_dir, exist_ok=True)

    # Keep our own copy so Member 2's file is never overwritten
    vprev = os.path.join(work_dir, "round_0.v")
    shutil.copy(vprev_path, vprev)

    # Detect the top module once from Member 2's design, then force it for every Vnew
    prev_cells, top = synthesize(vprev)
    if prev_cells is None:
        print("[agent] Yosys can't read the starting file. Returning it unchanged.")
        return vprev_path
    if not top:
        print("[agent] Couldn't detect the top module, so Vnew can't be compared fairly. "
              "Returning the original unchanged.")
        return vprev_path
    print(f"[agent] top module: {top} | start: {prev_cells} cells")

    for rnd in range(1, max_rounds + 1):
        print(f"\n[agent] ---- round {rnd} ----")

        # Step 3-4: ask Qwen for an optimized version, save as Vnew
        with open(vprev) as f:
            new_code = optimize_verilog(f.read())
        if new_code is None:
            print("[agent] Qwen gave no usable Verilog. Stopping.")
            break

        vnew = os.path.join(work_dir, f"round_{rnd}.v")
        with open(vnew, "w") as f:
            f.write(new_code)

        # Step 5: compare cell counts
        new_cells = count_cells(vnew, top=top)
        if new_cells is None:
            print(f"[agent] Vnew doesn't synthesize with top module '{top}'. Stopping.")
            break
        print(f"[agent] Vprev: {prev_cells} cells | Vnew: {new_cells} cells")

        if new_cells >= prev_cells:
            print("[agent] No improvement. Stopping.")
            break

        # Step 6-7: Member 3 checks Vnew against Vprev using the testbench
        try:
            approved = checker(vprev, vnew, testbench)
        except Exception as e:
            print(f"[agent] Member 3's checker raised an error: {e}")
            approved = None

        if not approved:
            # Member 3 handles returning the last working circuit to the user
            print("[agent] Member 3 rejected Vnew. Stopping. Member 3 returns the final circuit.")
            return None

        # Step 8: Vnew is now the verified version
        print(f"[agent] Approved: {prev_cells} -> {new_cells} cells")
        vprev, prev_cells = vnew, new_cells
    else:
        print(f"\n[agent] Hit max rounds ({max_rounds}). Stopping.")

    final_path = os.path.join(work_dir, "final.v")
    shutil.copy(vprev, final_path)
    print(f"\n[agent] Final verified circuit: {final_path} ({prev_cells} cells)")
    return final_path


def _load_checker(spec):
    module_name, func_name = spec.split(":")
    return getattr(importlib.import_module(module_name), func_name)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("member2_json", help="Member 2's JSON with the Verilog and testbench paths")
    p.add_argument("--checker", required=True,
                   help="Member 3's function as module:function, e.g. member3_checker:check")
    p.add_argument("--max-rounds", type=int, default=MAX_ROUNDS)
    args = p.parse_args()
    run(args.member2_json, _load_checker(args.checker), max_rounds=args.max_rounds)