"""
yosys_checker.py
Counts cells in a Verilog file using Yosys.
Both Vprev and Vnew go through the exact same synthesis script, with the SAME top module,
so the counts are comparable.

Note: these are Yosys generic cell counts, not real chip area. They're for comparing
two versions synthesized with identical settings.
"""

import os
import re
import subprocess
import tempfile

# Paths are quoted so folders with spaces work.
# -flatten gives one total count for the whole design.
# "tee -o" writes the stat report to its own file, so we don't depend on the full log.
YOSYS_SCRIPT = 'read_verilog "{path}"; synth {top_opt} -flatten; tee -q -o "{stats}" stat'


def synthesize(verilog_path, top=None):
    """
    Synthesize and return (cell_count, top_module_name), or (None, None) on failure.
    top=None lets Yosys pick the top module (use this once, on Member 2's Vprev).
    top="name" forces that module as top (use this for every Vnew).
    """
    top_opt = f"-top {top}" if top else "-auto-top"
    fd, stats_path = tempfile.mkstemp(suffix="_stat.txt", dir=os.path.dirname(verilog_path) or ".")
    os.close(fd)
    try:
        script = YOSYS_SCRIPT.format(path=verilog_path, top_opt=top_opt, stats=stats_path)
        result = subprocess.run(["yosys", "-q", "-p", script],
                                capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            err = (result.stderr or result.stdout).strip().splitlines()[-5:]
            print(f"[yosys] failed on {verilog_path}:\n  " + "\n  ".join(err))
            return None, None
        with open(stats_path) as f:
            out = f.read()
    except Exception as e:
        print(f"[yosys] could not run: {e}")
        return None, None
    finally:
        if os.path.exists(stats_path):
            os.remove(stats_path)

    # Older Yosys:  "Number of cells:   42"
    matches = re.findall(r"Number of cells:\s+(\d+)", out)
    # Newer Yosys:  "      42 cells"
    if not matches:
        matches = re.findall(r"^\s*(\d+)\s+cells\s*$", out, re.MULTILINE)
    if not matches:
        print(f"[yosys] couldn't find a cell count for {verilog_path}")
        return None, None

    # After -flatten only the top module is left: "=== name ==="
    names = re.findall(r"^=== (\S+) ===", out, re.MULTILINE)
    found_top = names[-1].lstrip("\\") if names else top  # None if not found and not given
    if not found_top:
        print(f"[yosys] couldn't detect the top module name for {verilog_path}")

    return int(matches[-1]), found_top


def count_cells(verilog_path, top=None):
    """Return just the cell count (or None)."""
    return synthesize(verilog_path, top)[0]