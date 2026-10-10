"""
Show the speed feedback the agent now receives, without running the LLM.

Grades two correct level-4 kernels through agent.grade -- the exact path an agent attempt takes:
the shipped reference (K tile 128, N tile 512) and the same kernel with a 64-row K tile -- and
prints the feedback each one gets.

    python demo_hint.py
"""

import pathlib

import agent

ref = pathlib.Path(__file__).with_name("reference_level4.py").read_text()
variants = {
    "reference (K tile 128, N tile 512)": ref,
    "same kernel, K tile 64, N tile 128": ref.replace(
        "TILE_K = nl.tile_size.pmax  # 128", "TILE_K = 64").replace(
        "TILE_N = nl.tile_size.gemm_moving_fmax  # 512", "TILE_N = 128"),
}

for name, src in variants.items():
    assert name.startswith("reference") or src != ref, "tile edit did not apply"
    reward, parts, feedback = agent.grade(src, 4)
    print(f"=== {name}\nreward {reward:.2f}\nfeedback: {feedback}\n")
