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

Member 3's check runs her comparator.py (or compare_verilog.py), which must sit next to vagent.py.
Default location: same folder as this file. Override with COMPARE_SCRIPT=/path/to/script.py

Use from the team's main program:
    from optimize_agent import run
    final_path = run("member2_output.json")    # None = Member 3 rejected (see opt_work/selected.v)

Or from the terminal:
    python optimize_agent.py member2_output.json
"""

import argparse
import importlib
import json
import os
import shutil
import subprocess
import sys

from qwen_client import optimize_verilog
from yosys_checker import count_cells, synthesize

MAX_ROUNDS = 8
WORK_DIR = "opt_work"

# Member 3's script, found in the same folder as this file as comparator.py or compare_verilog.py.
# Override with:  export COMPARE_SCRIPT=/path/to/script.py
_HERE = os.path.dirname(os.path.abspath(__file__))


def _find_compare_script():
    if os.environ.get("COMPARE_SCRIPT"):
        return os.environ["COMPARE_SCRIPT"]
    for name in ("comparator.py", "compare_verilog.py"):
        path = os.path.join(_HERE, name)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"Member 3's comparator.py / compare_verilog.py not found in {_HERE}")


def member3_check(vprev, vnew, testbench):
    """
    Run Member 3's compare_verilog.py as its own process.
    Exit code 0 = approved.
    Exit code 2 = rejected. Her script then writes the last verified design to selected.v.
    Any other exit code = the checker itself broke -> RuntimeError.
    """
    script = os.path.abspath(_find_compare_script())
    out_dir = os.path.abspath(os.path.dirname(vnew))
    tag = os.path.splitext(os.path.basename(vnew))[0]          # e.g. round_3
    cmd = [
        sys.executable, script,
        "--original", os.path.abspath(vprev),
        "--optimized", os.path.abspath(vnew),
        "--tb", os.path.abspath(testbench),
        "--result", os.path.join(out_dir, f"verification_{tag}.json"),
        "--selected", os.path.join(out_dir, "selected.v"),
    ]
    # Run from her script's folder so it can import vagent.py
    result = subprocess.run(cmd, cwd=os.path.dirname(script),
                            capture_output=True, text=True, timeout=900)
    output = (result.stdout + result.stderr).strip()
    if output:
        print("[member3] " + output.replace("\n", "\n[member3] "))

    if result.returncode == 0:
        return True          # approved
    elif result.returncode == 2:
        return False         # genuine rejection
    else:
        raise RuntimeError(f"Member 3's checker failed with exit code {result.returncode}")

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


def run(json_path, checker=member3_check, max_rounds=MAX_ROUNDS, work_dir=WORK_DIR):
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
            # A technical error is NOT a rejection, so don't pretend it is
            raise RuntimeError(f"Member 3's checker failed: {e}") from e

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
    p.add_argument("--checker", default=None,
                   help="Optional: a different checker function as module:function "
                        "(default: run Member 3's compare_verilog.py)")
    p.add_argument("--max-rounds", type=int, default=MAX_ROUNDS)
    args = p.parse_args()
    checker = _load_checker(args.checker) if args.checker else member3_check
    run(args.member2_json, checker, max_rounds=args.max_rounds)