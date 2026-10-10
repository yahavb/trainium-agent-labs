"""G2 CPU reference: the ORIGINAL StreamDiffusionV2 v2v pipeline on CPU fp32.

    python g2_ref.py t5      encode the prompt once with the repo's WanTextEncoder -> artifacts/tier2/g2/prompt_<name>.pt
    python g2_ref.py run     run CausalStreamInferencePipeline (prepare + inference_stream) on the test clip

What is original: CausalStreamInferencePipeline, CausalWanDiffusionWrapper / CausalWanModel (all 30 blocks, ring
buffer, SDPA fallbacks since flash_attn is absent), WanVAEWrapper, FlowMatchScheduler. Nothing is patched except
that the text encoder is swapped for the saved embedding. The ~40-line session loop below re-implements
streamv2v/inference.py:188-256 (start_stream_session / run_stream_batch) because that class hard-codes bf16
(inference.py:92, :129, :175), which CPU cannot run at useful speed; this version is fp32.

Settings: 512x512, steps [700, 500], num_kv_cache 6, num_sink_tokens 3, adapt_sink_threshold -1 (off), noise_scale
0.8, batched denoising (the demo default), seed 1234.

Saves everything the Neuron run needs to replay the exact same inputs and noise: g2/ref.pt.
"""
import sys
import time
from types import SimpleNamespace

import torch

from g2_common import (FRAME_TOKENS, G2, HEIGHT, NOISE_SCALE, NUM_STREAM_CHUNKS, PROMPT, PROMPT_NAME, SEED, STEPS,
                       WIDTH, load_clip, noise_scale_and_step, use_repo)
from block_port import CKPT_GLOB

PROMPT_FILE = G2 / f"prompt_{PROMPT_NAME}.pt"


def encode_prompt():
    use_repo()
    from models.wan.wan_wrapper import WanTextEncoder
    t0 = time.time()
    te = WanTextEncoder()
    emb = te([PROMPT])["prompt_embeds"]
    G2.mkdir(parents=True, exist_ok=True)
    torch.save({"prompt": PROMPT, "prompt_embeds": emb.float().clone()}, PROMPT_FILE)
    print(f"T5 OK {tuple(emb.shape)} nonzero_tokens={(emb.abs().sum(-1) > 0).sum().item()} {time.time() - t0:.0f}s")


class SavedTextEncoder(torch.nn.Module):
    def __init__(self, model_type=None):
        super().__init__()
        self.emb = torch.load(PROMPT_FILE)["prompt_embeds"]

    def forward(self, text_prompts):
        return {"prompt_embeds": self.emb.clone()}


class LogRandn:
    """Record every torch.randn_like drawn inside the pipeline call (the scheduler.add_noise noise)."""

    def __enter__(self):
        self.noises, self._orig = [], torch.randn_like
        torch.randn_like = lambda x, **kw: self._log(self._orig(x, **kw))
        return self

    def _log(self, n):
        self.noises.append(n.clone())
        return n

    def __exit__(self, *a):
        torch.randn_like = self._orig


def run():
    import glob
    use_repo()
    import models.wan.causal_stream_inference as csi
    csi.get_text_encoder_wrapper = lambda model_name: SavedTextEncoder
    args = SimpleNamespace(model_type="T2V-1.3B", model_name="wan", generator_name="causal_wan",
                           denoising_step_list=STEPS + [0], t2v=False, warp_denoising_step=False,
                           height=HEIGHT, width=WIDTH, num_kv_cache=6, num_sink_tokens=3, adapt_sink_threshold=-1,
                           num_frame_per_block=1, use_taehv=False)
    torch.manual_seed(SEED)
    t0 = time.time()
    pipe = csi.CausalStreamInferencePipeline(args, device="cpu")
    sd = torch.load(glob.glob(CKPT_GLOB)[0], map_location="cpu", weights_only=False)["generator"]
    pipe.generator.load_state_dict(sd, strict=True)
    del sd
    pipe.eval().requires_grad_(False)
    print(f"pipeline loaded {time.time() - t0:.0f}s", flush=True)

    x0_log = []  # every generator output (pred x0), in call order
    pipe.generator.register_forward_hook(lambda m, a, out: x0_log.append(out.detach().clone()))

    frames = load_clip(5 + 4 * NUM_STREAM_CHUNKS)
    vae = pipe.vae
    vae.model.first_encode = True
    rec = {"prompt": PROMPT, "seed": SEED, "steps": STEPS, "frames": frames, "calls": []}

    def encode_noisy(images, scale):
        t = time.time()
        lat = vae.stream_encode(images, is_scale=False).transpose(2, 1).contiguous()  # inference.py:173-177
        enc_s = time.time() - t
        noise = torch.randn_like(lat)
        rec.setdefault("enc", []).append({"lat": lat.clone(), "noise": noise.clone(), "scale": scale})  # for e2e.py
        return noise * scale + lat * (1 - scale), enc_s

    # ---- session start: 5 frames = 2 latent frames, sequential over the steps (inference.py:188-215) ----
    noisy, enc_s = encode_noisy(frames[:, :, :5], NOISE_SCALE)
    t = time.time()
    with LogRandn() as lg:
        den = pipe.prepare(text_prompts=[PROMPT], device="cpu", dtype=torch.float32, block_mode="input",
                           noise=noisy, current_start=0, current_end=2 * FRAME_TOKENS)
    rec["start"] = {"noisy": noisy, "noises": lg.noises, "x0": [x.clone() for x in x0_log], "denoised": den.clone(),
                    "encode_s": enc_s, "dit_s": time.time() - t}
    print(f"start: prepare {time.time() - t:.1f}s  x0 abs max {den.abs().max():.3f}", flush=True)
    x0_log.clear()

    # ---- stream chunks (inference.py:217-256, batched denoising) ----
    cur_start, cur_end = 2 * FRAME_TOKENS, 3 * FRAME_TOKENS
    last, scale = frames[:, :, [4]], NOISE_SCALE
    for c in range(NUM_STREAM_CHUNKS):
        images = frames[:, :, 5 + 4 * c: 9 + 4 * c]
        scale, step = noise_scale_and_step(torch.cat([last, images], dim=2), scale, NOISE_SCALE)
        noisy, enc_s = encode_noisy(images, scale)
        t = time.time()
        with LogRandn() as lg:
            hidden = pipe.inference_stream(noise=noisy[:, 0].unsqueeze(1), current_start=cur_start,
                                           current_end=cur_end, current_step=step)
        dit_s = time.time() - t
        assert len(x0_log) == 1 and len(lg.noises) == 1
        rec["calls"].append({"chunk": c, "noisy": noisy.clone(), "noise_scale": scale, "current_step": step,
                             "current_start": cur_start, "x0": x0_log[0], "hidden": hidden.clone(),
                             "noise": lg.noises[0], "encode_s": enc_s, "dit_s": dit_s,
                             "ring_slots": [kv["pos"].tolist() for kv in pipe.kv_cache1[:1]]})
        print(f"chunk {c}: step={step} noise_scale={scale:.4f} dit {dit_s:.1f}s  slots(layer0)={pipe.kv_cache1[0]['pos'].tolist()}",
              flush=True)
        x0_log.clear()
        cur_start, cur_end = cur_end, cur_end + FRAME_TOKENS
        last = images[:, :, [-1]]

    torch.save(rec, G2 / "ref.pt")
    print("REF OK", flush=True)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    {"t5": encode_prompt, "run": run}[sys.argv[1]]()
