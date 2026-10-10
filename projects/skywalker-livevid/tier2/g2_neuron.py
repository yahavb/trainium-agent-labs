"""G2: the full 30-block causal Wan DiT with the blocks on Neuron, replaying the CPU reference's inputs and noise.

    python g2_neuron.py --backend neuron     30 traced blocks (bf16) on one NeuronCore
    python g2_neuron.py --backend cpu        same host code with the fp32 block port on CPU (isolates host logic)

Host side (CPU fp32): patch embedding, time embedding / projection, text embedding, head, unpatchify, flow->x0,
re-noising, and the per-row KV bookkeeping (CacheState: sinks, FIFO eviction, attn_bias while filling).

Schedule, identical to the repo's batched stream path (causal_stream_inference.py:261-300):
  * session start: 2 latent frames in ONE chunk, 2 sequential passes (700 then 500). Runs on CPU with the fp32
    port (the traced graph is fixed at 1 frame); its K/V become sink frames 0 and 1 of both rows.
  * every stream call: row 0 = newest frame at the adaptive first step, row 1 = previous frame at step 500. Each
    row has its own 30 caches. Row 1 of the very first stream call is skipped: in the repo it denoises an all-zero
    latent whose output is discarded and whose cache slot is overwritten on the next call.
"""
import argparse
import glob
import json
import time

import torch
import torch.nn.functional as F

from block_port import (ART_DIR, CKPT_GLOB, BlockCfg, CacheState, CausalBlockPort, cosine, max_abs, rope_cos_sin)
from g2_common import G2, NUM_STREAM_CHUNKS, STEPS, use_repo
from g2_compile import artifact

BF16 = torch.bfloat16
NUM_LAYERS = 30
GATE = 0.999


def host_weights():
    path = ART_DIR / "host_fp32.pt"
    if path.exists():
        return torch.load(path)
    sd = torch.load(glob.glob(CKPT_GLOB)[0], map_location="cpu", weights_only=False)["generator"]
    w = {k[len("model."):]: v.float().clone() for k, v in sd.items() if ".blocks." not in k}
    torch.save(w, path)
    return w


class Host:
    """The non-block parts of CausalWanModel._forward_inference (causal_model.py:756-896), fp32 on CPU."""

    def __init__(self, cfg, prompt_embeds):
        use_repo()
        from models.wan.flow_match import FlowMatchScheduler
        self.cfg, self.w = cfg, host_weights()
        self.sched = FlowMatchScheduler(shift=8.0, sigma_min=0.0, extra_one_step=True)  # wan_wrapper.py:163-166
        self.sched.set_timesteps(1000, training=True)
        w = self.w
        text = F.linear(prompt_embeds, w["text_embedding.0.weight"], w["text_embedding.0.bias"])
        self.context = F.linear(F.gelu(text, approximate="tanh"), w["text_embedding.2.weight"],
                                w["text_embedding.2.bias"])

    def patch(self, lat):  # [1, F, 16, h, w] -> [1, F * frame_tokens, dim]
        x = F.conv3d(lat.permute(0, 2, 1, 3, 4), self.w["patch_embedding.weight"], self.w["patch_embedding.bias"],
                     stride=(1, 2, 2))
        return x.flatten(2).transpose(1, 2)

    def time(self, t, frames):
        w, half = self.w, self.cfg.freq_dim // 2
        tt = torch.full((frames,), float(t), dtype=torch.float64)
        s = torch.outer(tt, torch.pow(10000, -torch.arange(half, dtype=torch.float64) / half))
        emb = torch.cat([torch.cos(s), torch.sin(s)], dim=1).float()
        e = F.linear(F.silu(F.linear(emb, w["time_embedding.0.weight"], w["time_embedding.0.bias"])),
                     w["time_embedding.2.weight"], w["time_embedding.2.bias"])
        e0 = F.linear(F.silu(e), w["time_projection.1.weight"], w["time_projection.1.bias"])
        return e.view(1, frames, 1, -1), e0.view(1, frames, 6, -1)

    def head(self, tok, e, frames):  # -> flow [1, F, 16, h, w]
        cfg, w = self.cfg, self.w
        m0, m1 = (w["head.modulation"].unsqueeze(1) + e).chunk(2, dim=2)
        x = F.layer_norm(tok, (cfg.dim,), eps=cfg.eps).view(1, frames, cfg.frame_tokens, cfg.dim) * (1 + m1) + m0
        x = F.linear(x, w["head.head.weight"], w["head.head.bias"])
        x = x.view(frames, cfg.grid_h, cfg.grid_w, 1, 2, 2, 16)
        x = torch.einsum("fhwpqrc->cfphqwr", x).reshape(16, frames, cfg.grid_h * 2, cfg.grid_w * 2)
        return x.permute(1, 0, 2, 3).unsqueeze(0)

    def sigma(self, t):
        return self.sched.sigmas[torch.argmin((self.sched.timesteps - float(t)).abs())]

    def to_x0(self, flow, xt, t):  # wan_wrapper.py:174-198
        return (xt.double() - self.sigma(t).double() * flow.double()).float()

    def renoise(self, x0, noise, t):  # flow_match.py add_noise
        s = self.sigma(t)
        return ((1 - s) * x0 + s * noise).type_as(noise)


class Runner:

    def __init__(self, backend, ref):
        self.backend = backend
        self.cfg = BlockCfg()
        self.cfg2 = BlockCfg(frames_per_chunk=2, num_kv_cache=2, num_sink=0)  # session start: 2 frames, no history
        prompt = torch.load(G2 / "prompt_anime.pt")["prompt_embeds"]
        self.host = Host(self.cfg, prompt)
        self.t = {"dit": 0.0, "host": 0.0}
        t0 = time.time()
        sds = [torch.load(ART_DIR / f"block{i}_fp32.pt")["block"] for i in range(NUM_LAYERS)]
        self.start_blocks = [self._port(self.cfg2, sd) for sd in sds]
        if backend == "cpu":
            self.blocks = [self._port(self.cfg, sd) for sd in sds]
            self.dtype = torch.float32
        else:
            import torch_neuronx  # noqa: F401
            self.blocks = [torch.jit.load(str(artifact(self.cfg, i))) for i in range(NUM_LAYERS)]
            self.dtype = BF16
        self.ctx = self.host.context.to(self.dtype)
        print(f"loaded {backend} blocks in {time.time() - t0:.0f}s", flush=True)

    @staticmethod
    def _port(cfg, sd):
        m = CausalBlockPort(cfg)
        m.load_state_dict(sd, strict=True)
        return m.eval().requires_grad_(False)

    def start(self, noisy, noises):
        """2-frame chunk, steps run one after the other; returns x0 [1, 2, 16, h, w] and the per-layer K/V."""
        host, cfg2 = self.host, self.cfg2
        cos_, sin_ = rope_cos_sin(cfg2, 0)
        dummy = torch.zeros(1, cfg2.l_cache, cfg2.num_heads, cfg2.head_dim)
        bias = torch.zeros(1, 1, 1, cfg2.l_cache)
        xt, x0s = noisy, []
        for i, step in enumerate(STEPS):
            tok = host.patch(xt)
            e, e0 = host.time(step, 2)
            kv = []
            for blk in self.start_blocks:
                tok, k, v = blk(tok, e0, host.context, cos_, sin_, dummy, dummy, bias)
                kv.append((k, v))
            x0 = host.to_x0(host.head(tok, e, 2), xt, step)
            x0s.append(x0)
            if i < len(STEPS) - 1:
                xt = host.renoise(x0.flatten(0, 1), noises[i], STEPS[i + 1]).unflatten(0, x0.shape[:2])
        self.rows = []
        for _ in STEPS:  # one cache set per denoising row (causal_stream_inference.py:224-229)
            states = []
            for k, v in kv:
                st = CacheState(self.cfg, dtype=self.dtype)
                st.k[:, :cfg2.l_chunk] = k.to(self.dtype)
                st.v[:, :cfg2.l_chunk] = v.to(self.dtype)
                st.frames[:2], st.seen = [0, 1], 2
                states.append(st)
            self.rows.append(states)
        self.start_blocks = None
        return x0s

    def forward_row(self, row, lat, t, frame):
        """One DiT pass for one latent frame: lat [1, 1, 16, h, w] -> x0."""
        host, cfg, dt = self.host, self.cfg, self.dtype
        t0 = time.perf_counter()
        tok = host.patch(lat).to(dt)
        e, e0 = host.time(t, 1)
        e0 = e0.to(dt)
        cos_, sin_ = (r.to(dt) for r in rope_cos_sin(cfg, frame))
        t_dit = 0.0
        for blk, st in zip(self.blocks, self.rows[row]):
            bias = st.bias()
            t1 = time.perf_counter()
            tok, k, v = blk(tok, e0, self.ctx, cos_, sin_, st.k, st.v, bias)
            t_dit += time.perf_counter() - t1
            st.commit(k, v)
        x0 = host.to_x0(host.head(tok.float(), e, 1), lat, t)
        self.t["dit"] += t_dit
        self.t["host"] += time.perf_counter() - t0 - t_dit
        return x0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["neuron", "cpu"], default="neuron")
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    ref = torch.load(G2 / "ref.pt")
    run = Runner(args.backend, ref)
    rows_out = []

    t0 = time.time()
    x0s = run.start(ref["start"]["noisy"], ref["start"]["noises"])
    start = {"x0_cos": [cosine(a, b) for a, b in zip(x0s, ref["start"]["x0"])],
             "x0_max_abs": [max_abs(a, b) for a, b in zip(x0s, ref["start"]["x0"])], "seconds": time.time() - t0}
    print(f"start (CPU port, 2 frames): x0 cos per step {start['x0_cos']}  {start['seconds']:.0f}s", flush=True)

    hidden_prev, finals, timing = None, [x0s[-1]], []
    for c, call in enumerate(ref["calls"][:NUM_STREAM_CHUNKS]):
        run.t = {"dit": 0.0, "host": 0.0}
        tc = time.perf_counter()
        frame = call["current_start"] // run.cfg.frame_tokens
        x0_r0 = run.forward_row(0, call["noisy"][:, :1], call["current_step"], frame)
        row = {"chunk": c, "frame_row0": frame, "current_step": call["current_step"],
               "row0_cos": cosine(x0_r0, call["x0"][0]), "row0_max_abs": max_abs(x0_r0, call["x0"][0]),
               "ref_abs_max": call["x0"].abs().max().item()}
        if hidden_prev is not None:
            x0_r1 = run.forward_row(1, hidden_prev, STEPS[1], frame - 1)
            row.update({"row1_cos": cosine(x0_r1, call["x0"][1]), "row1_max_abs": max_abs(x0_r1, call["x0"][1])})
            finals.append(x0_r1)
        hidden_prev = run.host.renoise(x0_r0, call["noise"].view_as(x0_r0), STEPS[1])
        wall = time.perf_counter() - tc
        timing.append({"chunk": c, "passes": 2 if "row1_cos" in row else 1, "wall_s": wall,
                       "dit_s": run.t["dit"], "host_s": run.t["host"]})
        row["slots_row0"] = list(run.rows[0][0].frames)
        rows_out.append(row)
        print(f"chunk {c}: row0 cos={row['row0_cos']:.6f} max_abs={row['row0_max_abs']:.4f}"
              + (f"  row1(final) cos={row['row1_cos']:.6f} max_abs={row['row1_max_abs']:.4f}" if "row1_cos" in row else "")
              + f"  wall {wall:.2f}s dit {run.t['dit']:.2f}s host {run.t['host']:.2f}s  slots {row['slots_row0']}",
              flush=True)

    cosines = [r[k] for r in rows_out for k in ("row0_cos", "row1_cos") if k in r]
    res = {"backend": args.backend, "gate": GATE, "start": start, "chunks": rows_out, "timing": timing,
           "min_cosine": min(cosines), "pass": bool(min(cosines) >= GATE)}
    (G2 / f"parity_{args.backend}.json").write_text(json.dumps(res, indent=2) + "\n")
    torch.save({"finals": torch.cat(finals, dim=1)}, G2 / f"latents_{args.backend}.pt")
    print(f"G2 PARITY ({args.backend}) min_cosine={res['min_cosine']:.6f} gate={GATE} -> "
          f"{'PASS' if res['pass'] else 'FAIL'}", flush=True)


if __name__ == "__main__":
    main()
