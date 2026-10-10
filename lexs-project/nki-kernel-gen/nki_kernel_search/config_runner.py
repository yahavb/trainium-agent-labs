import argparse
import getpass
import json
import os
from pathlib import Path
import re
import sys

from . import cli
from .evaluator import evaluate_local
from .environment import configure_environment


def read_config(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text())
    if not isinstance(config, dict):
        raise ValueError("Config must be a JSON object")
    allowed = {"reference", "initial_kernel", "module_name", "mode", "iterations", "model", "output",
               "timeout", "checkpoint", "command", "prompt_key", "bootstrap_attempts", "wandb", "context_files"}
    unknown = set(config) - allowed
    if unknown:
        raise ValueError(f"Unknown config fields: {sorted(unknown)}")
    if not isinstance(config.get("reference"), str):
        raise ValueError("reference must be a path to a PyTorch task file")
    if "module_name" in config:
        if not isinstance(config["module_name"], str) or not config["module_name"].strip():
            raise ValueError("module_name must be a nonempty string")
        config["module_name"] = config["module_name"].strip()
    for key in ("reference", "initial_kernel", "output", "checkpoint"):
        if key in config:
            if not isinstance(config[key], str):
                raise ValueError(f"{key} must be a path string")
            config[key] = str((path.parent / config[key]).resolve())
    if not Path(config["reference"]).is_file():
        raise ValueError("reference file does not exist")
    context_files = config.get("context_files", [])
    if not isinstance(context_files, list) or any(not isinstance(p, str) for p in context_files):
        raise ValueError("context_files must be a list of file paths")
    config["context_files"] = [str((path.parent / p).resolve()) for p in context_files]
    for document in config["context_files"]:
        if not Path(document).is_file():
            raise ValueError(f"Context file does not exist: {document}")
    config.setdefault("output", str(path.parent / "outputs/search"))
    for key, default in (("iterations", 10), ("timeout", 600), ("bootstrap_attempts", 3)):
        value = config.setdefault(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"{key} must be positive")
        if key != "timeout" and not isinstance(value, int):
            raise ValueError(f"{key} must be an integer")
    if config.get("mode", "hardware") not in ("hardware", "simulate"):
        raise ValueError("mode must be hardware or simulate")
    if config.get("command", "search") not in ("check", "search"):
        raise ValueError("command must be check or search")
    if "prompt_key" in config and not isinstance(config["prompt_key"], bool):
        raise ValueError("prompt_key must be true or false")
    tracking = config.get("wandb", {})
    if not isinstance(tracking, dict) or set(tracking) - {"enabled", "project", "entity", "mode", "name"}:
        raise ValueError("wandb accepts enabled, project, entity, mode and name")
    if tracking.get("mode", "offline") not in ("offline", "online", "disabled"):
        raise ValueError("wandb mode must be offline, online or disabled")
    return config


def bootstrap(config):
    """Generate a starting kernel, then require simulator correctness before evolution."""
    from openai import OpenAI
    configure_environment()
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("Set OPENAI_API_KEY or pass --prompt-key to generate a starting kernel")
    client = OpenAI()
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    messages = [{"role": "system", "content":
        "Write a correct NKI implementation using nki.language and "
        "nki.isa with explicit destination tensors and DMA copies. Do not use legacy "
        "neuronxcc.nki APIs or nl.load/nl.store. Return Python source only. Use a plain undecorated "
        "kernel(*inputs) function, matching the reference positional arguments. "
        "Put all implementation code inside # EVOLVE-BLOCK-START and "
        "# EVOLVE-BLOCK-END. Respect the 128 partition limit and mask tails. "
        "Do not access files, network, environment, reference modules or host tensor libraries. "
        "Implement every shape and case in this trusted task:\n" + Path(config["reference"]).read_text()}]
    from .sdk_prompt import sdk_context
    from .prompt_context import writer_context
    messages[0]["content"] = writer_context(config.get("context_files", [])) + messages[0]["content"] + sdk_context()
    model = config.get("model", "gpt-6-luna")
    for attempt in range(config["bootstrap_attempts"]):
        print(f"Generating initial NKI kernel: attempt {attempt + 1}", flush=True)
        token_param = "max_completion_tokens" if model.startswith(("gpt-5", "gpt-6", "o1", "o3", "o4")) else "max_tokens"
        response = client.chat.completions.create(model=model, messages=messages,
                                                  **{token_param: 8192})
        code = response.choices[0].message.content or ""
        fenced = re.search(r"```(?:python)?\s*\n(.*?)```", code, re.S)
        if fenced:
            code = fenced.group(1)
        candidate = output / f"bootstrap_{attempt + 1}.py"
        candidate.write_text(code)
        result = evaluate_local(candidate, config["reference"], timeout=config["timeout"])
        if "EVOLVE-BLOCK-START" not in code or "EVOLVE-BLOCK-END" not in code:
            result = {"metrics": {"correctness": 0}, "artifacts": {"error": "Missing evolution markers"}}
        (output / f"bootstrap_{attempt + 1}.json").write_text(json.dumps(result, indent=2))
        if result["metrics"]["correctness"]:
            return str(candidate)
        messages += [{"role": "assistant", "content": code},
                     {"role": "user", "content": "Fix the kernel. Evaluation failed:\n" +
                      str(result["artifacts"])[:8000]}]
    raise ValueError("No generated starting kernel passed correctness checks; inspect bootstrap artifacts")


def main():
    p = argparse.ArgumentParser(description="Evolve NKI kernels from a JSON configuration")
    p.add_argument("--config", required=True)
    p.add_argument("--check", action="store_true", help="Validate the initial kernel without evolving")
    p.add_argument("--prompt-key", action="store_true")
    p.add_argument("--prompt-wandb-key", action="store_true")
    args = p.parse_args()
    try:
        config = read_config(args.config)
        configure_environment()
        if args.prompt_key or config.get("prompt_key", False):
            os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API key: ")
        if args.prompt_wandb_key:
            os.environ["WANDB_API_KEY"] = getpass.getpass("W&B API key: ")
        if os.environ.get("WANDB_KEY") and not os.environ.get("WANDB_API_KEY"):
            os.environ["WANDB_API_KEY"] = os.environ["WANDB_KEY"]
        os.environ["NKI_SEARCH_TRACKING"] = json.dumps(config.get("wandb", {}))
        if not config.get("initial_kernel"):
            if args.check or config.get("command") == "check":
                raise ValueError("check requires initial_kernel; it does not make model calls")
            config["initial_kernel"] = bootstrap(config)
        argv = ["nki-search", "check" if args.check else config.get("command", "search"),
                "--task", config["reference"], "--initial", config["initial_kernel"]]
        for key in ("module_name", "mode", "iterations", "model", "output", "timeout", "checkpoint"):
            if key in config:
                argv += ["--" + key.replace("_", "-"), str(config[key])]
        for document in config["context_files"]:
            argv += ["--context-file", document]
        previous = sys.argv
        try:
            sys.argv = argv
            cli.main()
        finally:
            sys.argv = previous
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        p.error(str(exc))
