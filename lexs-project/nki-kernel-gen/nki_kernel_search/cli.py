import argparse
import asyncio
import getpass
import json
import os
import subprocess
from pathlib import Path

from .evaluator import evaluate_local
from .environment import configure_environment


def main():
    configure_environment()
    p = argparse.ArgumentParser(description="Evolve NKI kernels against a trusted PyTorch reference")
    p.add_argument("command", choices=["check", "search"])
    p.add_argument("--task", required=True)
    p.add_argument("--initial", required=True)
    p.add_argument("--mode", choices=["simulate", "hardware"], default="hardware")
    p.add_argument("--iterations", type=int, default=10)
    p.add_argument("--model", default="gpt-6-luna")
    p.add_argument("--module-name", help="Module label used in the default W&B run name")
    p.add_argument("--output", default="outputs/search")
    p.add_argument("--timeout", type=float, default=600)
    p.add_argument("--checkpoint")
    p.add_argument("--context-file", action="append", default=[], help="NKI authoring reference to include in system context")
    p.add_argument("--prompt-key", action="store_true", help="Read API key without echoing or saving it")
    a = p.parse_args()
    if a.timeout <= 0 or a.iterations <= 0:
        p.error("timeout and iterations must be positive")
    if a.module_name is not None and not a.module_name.strip():
        p.error("module-name must be a nonempty string")
    task, initial = Path(a.task).resolve(), Path(a.initial).resolve()
    if not task.is_file() or not initial.is_file():
        p.error("task and initial kernel must exist")
    if a.mode == "hardware":
        try:
            probe = subprocess.run(["neuron-ls"], capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.TimeoutExpired) as exc:
            p.error(f"Cannot query hardware with neuron-ls: {exc}")
        if probe.returncode:
            p.error("neuron-ls could not access hardware. Run on the host outside the sandbox. "
                    + probe.stderr[-2000:])
    result = evaluate_local(initial, task, a.mode, initial, a.timeout)
    print(json.dumps(result, indent=2))
    if not result["metrics"]["correctness"]:
        raise SystemExit("Initial kernel failed validation; search was not started")
    if a.command == "check":
        return
    if a.prompt_key:
        os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API key: ")
    if not os.environ.get("OPENAI_API_KEY"):
        p.error("Set OPENAI_API_KEY or use --prompt-key")
    from openevolve.config import load_config
    from openevolve.controller import OpenEvolve
    config = load_config()
    config.llm.primary_model = a.model
    config.llm.secondary_model = None
    # Rebuild ensemble with the requested model rather than the default models.
    from openevolve.config import LLMModelConfig
    from .llm_audit import init_audited_llm
    config.llm.models = [LLMModelConfig(name=a.model, weight=1.0,
                                       api_key=os.environ["OPENAI_API_KEY"],
                                       api_base="https://api.openai.com/v1", max_tokens=8192,
                                       temperature=0.7, timeout=180, retries=2, retry_delay=5,
                                       init_client=init_audited_llm)]
    config.llm.evaluator_models = config.llm.models.copy()
    config.max_iterations = a.iterations
    config.checkpoint_interval = 5
    config.evaluator.parallel_evaluations = 1
    config.evaluator.cascade_evaluation = False
    config.evaluator.timeout = a.timeout + 60
    config.diff_based_evolution = True
    config.allow_full_rewrites = False
    config.database.feature_dimensions = ["complexity", "diversity"]
    # Preserve OpenEvolve's selected parent/top/diverse history and parent artifacts.
    config.prompt.include_artifacts = True
    config.evaluator.enable_artifacts = True
    config.prompt.system_message = (
        "Optimize the NKI kernel for AWS Trainium2 using nki.language and "
        "nki.isa. Use explicit destination tensors and nisa.dma_copy. Do not use "
        "legacy neuronxcc.nki APIs or nl.load/nl.store. Keep a plain undecorated kernel(*inputs) entry point. "
        "Only edit EVOLVE-BLOCK regions. Preserve shapes, dtypes, numerical semantics "
        "and bounds masks. Respect 128 partition limit and target SDK tile limits. "
        "Improve tiling, fusion, reuse and scheduling. Never access files, environment, "
        "network, task/evaluator or benchmark APIs from candidate code. "
        "Every case must pass before receiving a performance score. Hardware score is "
        "geometric mean speedup over the initial NKI kernel, not CPU PyTorch latency. "
        "Simulation scores certify correctness only and cannot rank performance.\n"
        "Trusted PyTorch task specification:\n" + task.read_text()
    )
    from .sdk_prompt import sdk_context
    from .prompt_context import writer_context
    config.prompt.system_message = writer_context(a.context_file) + config.prompt.system_message + sdk_context()
    config.llm.update_model_params({"system_message": config.prompt.system_message})
    os.environ.update(NKI_SEARCH_TASK=str(task), NKI_SEARCH_BASELINE=str(initial),
                      NKI_SEARCH_MODE=a.mode, NKI_SEARCH_TIMEOUT=str(a.timeout),
                      NKI_SEARCH_OUTPUT=str(Path(a.output).resolve()))
    engine = OpenEvolve(str(initial), str(Path(__file__).with_name("evaluator.py")),
                        config, output_dir=str(Path(a.output).resolve()))
    from .tracking import SearchTracker
    tracking = json.loads(os.environ.get("NKI_SEARCH_TRACKING", "{}"))
    if not tracking.get("name"):
        module_name = (a.module_name or task.stem).strip().replace("_", "-")
        tracking["name"] = f"{module_name}-{a.model}"
    tracker = SearchTracker(a.output, tracking,
                            delegate=getattr(engine, "evolution_tracer", None))
    engine.evolution_tracer = tracker
    tracker.record(0, result["metrics"], initial.read_text(), "baseline")
    try:
        asyncio.run(engine.run(iterations=a.iterations, checkpoint_path=a.checkpoint))
    finally:
        tracker.close()


if __name__ == "__main__":
    main()
