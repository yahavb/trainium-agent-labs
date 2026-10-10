"""kopt16.py -- the team's loop (kopt.py, unchanged), run at 16-bit and scored on in-model time.

What changes against kopt.py is only WHAT IS MEASURED. kopt.py times a standalone float32 kernel.
Here every attempt builds one Qwen3-8B feed-forward block (hidden 4096, intermediate 12288, 512
tokens) whose three matmuls are the blocked kernel at the proposed settings, in bfloat16, traces it
with torch_neuronx, checks it against CPU, and times real inference on one NeuronCore. The score is
that inference time. The checker, the menu, the optimiser and the logging are kopt.py's own.

One setting is shared by the block's three matmuls, so the legal values are those every layer
accepts: M in 1,2,4; N in 1,2,4,8; K in 1,2,4,8,16,32 (72 settings).

    python kopt16.py --arm checker5 --start 1,1,1 --run 0 --base http://<server>:8000/v1 --out runs16/checker5.jsonl
    python kopt16.py --grid "1,1,1;4,4,4"            # measure settings into the cache, no model

Runs in the pod's own Python (it needs httpx); the measurement runs in ./env (torch_neuronx).
"""
import argparse, glob, json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import kopt  # noqa: E402

MODEL = "mlp:512:1"
VENV = os.path.join(HERE, "env", "bin")
M_VALS, N_VALS, K_VALS = (1, 2, 4), (1, 2, 4, 8), (1, 2, 4, 8, 16, 32)
# 3 matmuls: two of K4096 x (M512, N12288) and one of K12288 x (M512, N4096), in 2-byte values.
FLOOR_BYTES = 2 * (2 * (4096 * 512 + 4096 * 12288 + 512 * 12288) + (12288 * 512 + 12288 * 4096 + 512 * 4096))

kopt.CARD = """The model is one feed-forward block of a language model (hidden width 4096, intermediate width 12288) running on 512 tokens at 16-bit precision on an AI accelerator. Its three matrix multiplies all use one blocked kernel. Three lines set the kernel's block sizes:

TILES_IN_BLOCK_M=<int>   how many 128-wide tiles of the left matrix are loaded per transfer
TILES_IN_BLOCK_N=<int>   how many 512-wide tiles of the right matrix are loaded per transfer
TILES_IN_BLOCK_K=<int>   how many 128-deep tiles of the shared dimension are kept on-chip and reused

Larger blocks mean fewer, bigger transfers and more reuse of data already on the chip, but they need more on-chip buffer, which is limited. The time measured is the whole block's inference time. M must be 1, 2 or 4. N must be 1, 2, 4 or 8. K must be 1, 2, 4, 8, 16 or 32."""


class ModelBench(kopt.Bench):
    def __init__(self, cache_path):
        self.shape, self.tiles, self.size = (4096, 512, 12288), (4, 8, 32), "ffblock_bf16"
        self.live, self.cache_path, self.cache = True, cache_path, {}
        if os.path.exists(cache_path):
            for line in open(cache_path):
                if line.strip():
                    r = json.loads(line)
                    self.cache[tuple(r["knobs"])] = r

    def legal(self):
        return [(m, n, k) for m in M_VALS for n in N_VALS for k in K_VALS]

    def measure(self, knobs):
        knobs = tuple(knobs)
        if knobs not in self.cache:
            if not self.live:
                raise KeyError(f"{knobs} is not in the cache and live measurement is off")
            rec = measure_in_model(knobs)
            rec["knobs"] = list(knobs)
            self.cache[knobs] = rec
            open(self.cache_path, "a").write(json.dumps(rec) + "\n")
        return self.cache[knobs]


def measure_in_model(knobs):
    """One setting -> inference time of the block on the chip. Returns kopt's record shape; never raises."""
    env = dict(os.environ, PATH=VENV + ":" + os.environ["PATH"])
    variant = "nkibf16:%d,%d,%d" % knobs
    rec = dict(size="ffblock_bf16", correct=False, error=None, te_busy=0.0, dma_busy=0.0, bytes=0,
               transfers=0, avg_transfer_bytes=0.0, floor_bytes=FLOOR_BYTES)
    try:
        run = subprocess.run([os.path.join(VENV, "python"), os.path.join(HERE, "model_infer.py"), MODEL, variant, "30"],
                             env=env, capture_output=True, text=True, timeout=1500, cwd=HERE)
    except subprocess.TimeoutExpired:
        rec["error"] = "compile or run exceeded 25 minutes"
        return rec
    lines = [l for l in run.stdout.splitlines() if l.startswith("{")]
    out = json.loads(lines[-1]) if lines else {}
    if not out or out.get("error"):
        text = run.stderr + run.stdout
        if "State buffer allocation failed" in text or "exceeds state buffer capacity" in text or "SB allocation" in text:
            rec["error"] = "State buffer allocation failed: the blocks need more on-chip buffer than the chip has."
        else:
            why = [l for l in text.splitlines() if "ERROR" in l or "Error" in l]
            rec["error"] = "compile failed: " + ((why[-1] if why else out.get("error", "no output"))[-300:])
        return rec
    rec["rel_err"] = out.get("rel_err")
    rec["compile_s"] = out.get("compile_s")
    if not out.get("correct"):
        rec["error"] = f"WRONG RESULT on the chip: relative error {out.get('rel_err'):.4f} against a bar of 0.02"
        return rec
    rec["correct"] = True
    rec["time_us"] = out["wall_ms_median"] * 1000.0          # in-model inference time, wall clock
    rec["wall_ms"] = [out["wall_ms_min"], out["wall_ms_median"], out["wall_ms_max"]]
    try:                                                      # one hardware profile, for the checker's line
        p = subprocess.run([os.path.join(VENV, "python"), os.path.join(HERE, "profile_neff.py"), out["neff"], "1"],
                           env=env, capture_output=True, text=True, timeout=600, cwd=HERE)
        prof = json.loads(p.stdout.strip().splitlines()[-1])
        if "error" not in prof:
            rec.update(chip_us=prof["chip_us_median"], te_busy=prof["te_busy"], dma_busy=prof["dma_busy"],
                       bytes=prof["bytes"], transfers=prof["transfers"],
                       avg_transfer_bytes=prof["bytes"] / max(prof["transfers"], 1), compute_util=prof.get("mfu"))
    except Exception as e:
        rec["profile_error"] = str(e)[:200]
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="checker5")
    ap.add_argument("--start", default="1,1,1")
    ap.add_argument("--run", type=int, default=0)
    ap.add_argument("--attempts", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--base", default="")
    ap.add_argument("--cache", default=os.path.join(HERE, "grid16.jsonl"))
    ap.add_argument("--out", default=os.path.join(HERE, "runs16.jsonl"))
    ap.add_argument("--grid", help='measure these settings only: "M,N,K;M,N,K"')
    ap.add_argument("--replay", type=int, default=0,
                    help="no chip, no model: run this many runs of a no-model arm (random, greedy5) from the cache")
    ap.add_argument("--starts", default="2,2,2;1,1,1;1,8,1;4,1,1;1,1,8", help="starts cycled by --replay")
    a = ap.parse_args()
    bench = ModelBench(a.cache)
    if a.replay:
        bench.live = False
        starts = [tuple(int(v) for v in p.split(",")) for p in a.starts.split(";")]
        with open(a.out, "a") as log:
            for run_id in range(a.replay):
                kopt.run_once(a, bench, a.arm, run_id, log, start=starts[run_id % len(starts)])
        return
    if a.grid:
        for part in a.grid.split(";"):
            k = tuple(int(v) for v in part.split(","))
            r = bench.measure(k)
            print(k, r.get("time_us"), r.get("error"), flush=True)
        return
    start = tuple(int(v) for v in a.start.split(","))
    with open(a.out, "a") as log:
        best = kopt.run_once(a, bench, a.arm, a.run, log, start=start)
    print(f"{a.arm} run {a.run} from {start}: best {best['rec']['time_us']:.1f} us at M,N,K={best['knobs']}", flush=True)


if __name__ == "__main__":
    main()
