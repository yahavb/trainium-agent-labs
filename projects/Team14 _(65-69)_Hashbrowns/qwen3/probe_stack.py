#!/usr/bin/env python3
"""
probe_stack.py -- where Qwen3-8B's attention actually runs in this pod's vLLM-neuron stack, so a
custom kernel can be put exactly there and nowhere else. Reads and copies; changes nothing.

    python qwen3/probe_stack.py        # prints a report; writes qwen3/stack_report.txt and
                                       # qwen3/stack_src.tar.gz for reading off the pod

  1. THE SERVER     the running `vllm serve` process: which python it runs, its command line, and
                    its NEURON_* / VLLM_* settings (nothing else from its environment is read)
  2. THE PACKAGES   in THAT python, not this one: versions and locations of vLLM, the Neuron plugin,
                    NxD Inference, NxD, torch-neuronx, neuronx-cc, nki
  3. THE ATTENTION  in NxD Inference's attention module and its Qwen3 model: every call that looks
                    like a kernel (flash / nki / tkg / paged / *_kernel), with file and line
  4. THE CONFIG     what the server log says about how the model was compiled
  5. THE SOURCES    the attention module, the Qwen3 model, the KV-cache module, NxDI's model base and
                    config, the plugin's runner and loader, and every kernel module the attention
                    code imports -> stack_src.tar.gz

If vLLM is not running, section 1 falls back to the `vllm` on PATH.
"""

import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(HERE, "stack_report.txt")
TARBALL = os.path.join(HERE, "stack_src.tar.gz")
LOG = "/tmp/vllm.log"                                  # serve.sh's default LOG
ENV_KEYS = re.compile(r"^(NEURON_|VLLM_|XLA_|PJRT_|VIRTUAL_ENV$|PYTHONPATH$)")
# kernel launches, including NKI's grid form: flash_fwd[2](...)
KERNEL_CALL = re.compile(r"\b\w*(flash|nki|tkg|paged|_kernel|kernel_)\w*\s*(\[[^\]]*\])?\s*\(",
                         re.IGNORECASE)
# the switches that decide which path runs
KERNEL_SWITCH = re.compile(r"kernel_enabled|use_\w*kernel|attn_kernel|flash_decoding|chunked_prefill|"
                           r"is_chunked|tkg_enabled|is_prefix_caching|block_kv", re.IGNORECASE)
LOG_KEYS = re.compile(r"neuron_?config|attn_kernel|attention_kernel|flash|chunked|prefix|tkg|kernel|"
                      r"logical_nc|tp_degree|bucket|seq_len|compil", re.IGNORECASE)

out = io.StringIO()


def say(line=""):
    print(line)
    out.write(line + "\n")


def section(title):
    say(f"\n========== {title}")


def find_server():
    if not os.path.isdir("/proc"):
        return None
    for pid in filter(str.isdigit, os.listdir("/proc")):
        try:
            cmd = open(f"/proc/{pid}/cmdline", "rb").read().split(b"\0")
        except OSError:
            continue
        words = [c.decode(errors="replace") for c in cmd if c]
        if any(w.endswith("vllm") for w in words) and "serve" in words:
            env = {}
            try:
                for kv in open(f"/proc/{pid}/environ", "rb").read().split(b"\0"):
                    k, _, v = kv.decode(errors="replace").partition("=")
                    if ENV_KEYS.match(k):
                        env[k] = v
            except OSError:
                pass
            return pid, os.readlink(f"/proc/{pid}/exe"), words, env
    return None


# Runs inside the server's own python: where everything is, without importing the heavy packages.
CHILD = r'''
import importlib.metadata as md, importlib.util as iu, json, os, re, sys
dists = ["vllm", "vllm-neuron", "vllm_neuron", "neuronx-distributed-inference", "neuronx-distributed",
         "torch-neuronx", "torch", "neuronx-cc", "nki", "transformers", "libneuronxla"]
mods = ["vllm", "vllm_neuron", "neuronx_distributed_inference", "neuronx_distributed", "torch_neuronx",
        "neuronxcc", "nki", "nkilib"]
info = {"python": sys.version.split()[0], "executable": sys.executable, "versions": {}, "roots": {}}
for d in dists:
    try: info["versions"][d] = md.version(d)
    except Exception: pass
for m in mods:
    try:
        s = iu.find_spec(m)
        if s and s.submodule_search_locations: info["roots"][m] = list(s.submodule_search_locations)[0]
    except Exception: pass
files, imports = [], set()
nxdi = info["roots"].get("neuronx_distributed_inference")
if nxdi:
    for rel in ("modules/attention", "models/qwen3", "modules/kvcache"):
        for d, _, fs in os.walk(os.path.join(nxdi, rel)):
            files += [os.path.join(d, f) for f in fs if f.endswith(".py")]
    for rel in ("models/config.py", "models/model_base.py", "models/model_wrapper.py"):
        if os.path.exists(os.path.join(nxdi, rel)): files.append(os.path.join(nxdi, rel))
    for f in [f for f in files if "/modules/attention/" in f or "/models/qwen3/" in f]:
        for line in open(f, errors="replace"):
            m = re.match(r"\s*(?:from\s+([\w.]+)\s+import|import\s+([\w.]+))", line)
            name = m and (m.group(1) or m.group(2))
            if name and re.search(r"kernel|nki|nkilib|flash", name, re.I): imports.add(name)
for name in sorted(imports):
    try:
        s = iu.find_spec(name)
        if s and s.origin and s.origin.endswith(".py"): files.append(s.origin)
        if s and s.submodule_search_locations:
            for d, _, fs in os.walk(list(s.submodule_search_locations)[0]):
                files += [os.path.join(d, f) for f in fs if f.endswith(".py")]
    except Exception: pass
plug = info["roots"].get("vllm_neuron")
if plug:
    for d, _, fs in os.walk(plug):
        files += [os.path.join(d, f) for f in fs
                  if f.endswith(".py") and re.search(r"runner|loader|worker|attention|platform", f)]
info["files"], info["kernel_imports"] = sorted(set(files)), sorted(imports)
print(json.dumps(info))
'''


def main():
    section("1. THE SERVER")
    server = find_server()
    if server:
        pid, python, words, env = server
        say(f"  vllm serve is running: pid {pid}, python {python}")
        say(f"  command: {' '.join(words)[:600]}")
        for k in sorted(env):
            say(f"  {k}={env[k]}")
    else:
        vllm = shutil.which("vllm")
        line = open(vllm).readline().strip() if vllm else ""
        python = line[2:].split()[0] if line.startswith("#!") else sys.executable
        say(f"  vllm serve is NOT running; using the python behind {vllm or 'no vllm on PATH'}: {python}")

    section("2. THE PACKAGES -- as the server's python sees them")
    p = subprocess.run([python, "-c", CHILD], capture_output=True, text=True, timeout=300)
    if p.returncode:
        say(f"  could not probe {python}:\n" + "\n".join("    " + t for t in p.stderr.strip().splitlines()[-15:]))
        sys.exit(1)
    info = json.loads(p.stdout.strip().splitlines()[-1])
    say(f"  python {info['python']} at {info['executable']}")
    for k, v in info["versions"].items():
        say(f"  {k:<32} {v}")
    for k, v in info["roots"].items():
        say(f"  {k:<32} {v}")
    if "neuronx_distributed_inference" not in info["roots"]:
        say("  NxD Inference is not importable from this python, so sections 3 and 5 are empty. The "
            "server may use a different backend; the command line above says which.")

    section("3. THE ATTENTION -- calls that look like kernels, in NxDI attention / Qwen3 / kernel modules")
    say(f"  kernel modules the attention code imports: {', '.join(info['kernel_imports']) or 'none found'}")
    for f in info["files"]:
        if not re.search(r"/modules/attention/|/models/qwen3/", f):
            continue
        hits = [(i, "call  " if KERNEL_CALL.search(l) else "switch", l.strip())
                for i, l in enumerate(open(f, errors="replace"), 1)
                if KERNEL_CALL.search(l) or KERNEL_SWITCH.search(l)]
        if hits:
            say(f"  {f}")
            for i, kind, l in hits[:60]:
                say(f"    {i:>5} {kind}  {l[:150]}")
            if len(hits) > 60:
                say(f"    ... {len(hits) - 60} more")

    section(f"4. THE CONFIG -- lines from {LOG} about how the model was compiled")
    if os.path.exists(LOG):
        lines = [l.rstrip() for l in open(LOG, errors="replace") if LOG_KEYS.search(l)]
        for l in lines[:80]:
            say(f"  {l[:240]}")
        say(f"  ({len(lines)} matching lines; first 80 shown)")
    else:
        say(f"  no {LOG}")

    section("5. THE SOURCES")
    with tarfile.open(TARBALL, "w:gz") as tar:
        for f in info["files"]:
            tar.add(f, arcname=re.sub(r"^.*?/(site|dist)-packages/", "", f))
        report = out.getvalue().encode()
        ti = tarfile.TarInfo("stack_report.txt")
        ti.size = len(report)
        tar.addfile(ti, io.BytesIO(report))
    say(f"  {len(info['files'])} files -> {TARBALL} ({os.path.getsize(TARBALL) // 1024} KB)")
    open(REPORT, "w").write(out.getvalue())


if __name__ == "__main__":
    main()
