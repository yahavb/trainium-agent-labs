from pathlib import Path
import sys

import pytest

from nki_kernel_search.evaluator import evaluate_local
from nki_kernel_search.worker import check

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "examples/add/task.py"
INITIAL = ROOT / "examples/add/initial_kernel.py"


def test_real_simulator():
    pytest.importorskip("neuronxcc.nki")
    result = evaluate_local(INITIAL, TASK)
    assert result["metrics"]["correctness"] == 1, result
    assert result["metrics"]["cases_passed"] == 12
    assert result["metrics"]["hardware_measured"] == 0
    assert "latency_us" not in result["metrics"]


def test_wrong_candidate(tmp_path):
    pytest.importorskip("neuronxcc.nki")
    candidate = tmp_path / "wrong.py"
    candidate.write_text(INITIAL.read_text().replace("op=nl.add", "op=nl.subtract"))
    result = evaluate_local(candidate, TASK)
    assert result["metrics"]["combined_score"] == 0
    assert "error" in result["artifacts"]


def test_timeout(tmp_path):
    candidate = tmp_path / "hang.py"
    candidate.write_text("while True: pass\n")
    result = evaluate_local(candidate, TASK, timeout=0.1)
    assert result["metrics"]["combined_score"] == 0
    assert "exceeded" in result["artifacts"]["error"]


def test_output_contract():
    import numpy as np
    import torch
    task = type("Task", (), {})()
    with pytest.raises(AssertionError):
        check(np.zeros(2, dtype=np.float64), torch.zeros(2), task)
    with pytest.raises(AssertionError):
        check(np.array([np.nan], dtype=np.float32), torch.zeros(1), task)


def test_search_configuration(monkeypatch, tmp_path):
    from nki_kernel_search import cli
    import openevolve.controller
    seen = {}

    class Engine:
        def __init__(self, initial, evaluator, config, output_dir):
            seen["config"] = config
            assert Path(initial) == INITIAL
            assert Path(evaluator).exists()

        async def run(self, **kwargs):
            seen["run"] = kwargs

    monkeypatch.setattr(openevolve.controller, "OpenEvolve", Engine)
    monkeypatch.setattr(cli, "evaluate_local", lambda *a: {"metrics": {"correctness": 1}})
    monkeypatch.setenv("OPENAI_API_KEY", "test-placeholder")
    monkeypatch.setattr(sys, "argv", ["nki-search", "search", "--task", str(TASK),
                                     "--initial", str(INITIAL), "--mode", "simulate",
                                     "--model", "test-model", "--iterations", "2",
                                     "--output", str(tmp_path)])
    cli.main()
    assert seen["config"].llm.models[0].name == "test-model"
    assert len(seen["config"].llm.models) == 1
    assert seen["config"].evaluator.parallel_evaluations == 1
    assert seen["run"]["iterations"] == 2


def test_json_paths_and_dispatch(monkeypatch, tmp_path):
    import json
    from nki_kernel_search import config_runner
    reference = tmp_path / "reference.py"
    reference.write_text("# reference")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"reference": "reference.py", "initial_kernel": str(INITIAL),
                                  "mode": "simulate", "iterations": 3}))
    seen = []
    monkeypatch.setattr(config_runner.cli, "main", lambda: seen.extend(sys.argv))
    monkeypatch.setattr(sys, "argv", ["nki_evolve.py", "--config", str(config), "--check"])
    config_runner.main()
    assert seen[1] == "check"
    assert str(reference) in seen
    assert "3" in seen
    assert config_runner.read_config(config)["output"] == str(tmp_path / "outputs/search")


def test_json_rejects_unknown_fields(tmp_path):
    from nki_kernel_search.config_runner import read_config
    config = tmp_path / "config.json"
    config.write_text('{"reference": "x.py", "iteratoins": 10}')
    with pytest.raises(ValueError, match="Unknown"):
        read_config(config)


def test_softmax_simulator():
    pytest.importorskip("neuronxcc.nki")
    result = evaluate_local(ROOT / "examples/softmax/initial_kernel.py",
                            ROOT / "examples/softmax/reference_pytorch_softmax.py")
    assert result["metrics"]["correctness"] == 1, result
    assert result["metrics"]["cases_passed"] == 15


@pytest.mark.parametrize("model", ["gpt-4.1", "gpt-6-luna"])
def test_bootstrap_retries(monkeypatch, tmp_path, model):
    from types import SimpleNamespace
    import openai
    from nki_kernel_search import config_runner
    code = INITIAL.read_text()
    calls = []

    def create(**kwargs):
        calls.append({**kwargs, "messages": list(kwargs["messages"])})
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=code))])

    monkeypatch.setattr(openai, "OpenAI", lambda: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    results = iter([{"metrics": {"correctness": 0}, "artifacts": {"error": "wrong output"}},
                    {"metrics": {"correctness": 1}, "artifacts": {}}])
    monkeypatch.setattr(config_runner, "evaluate_local", lambda *a, **k: next(results))
    monkeypatch.setenv("OPENAI_API_KEY", "test-placeholder")
    path = config_runner.bootstrap({"reference": str(TASK), "output": str(tmp_path),
                                    "bootstrap_attempts": 2, "timeout": 10, "model": model})
    assert Path(path).name == "bootstrap_2.py"
    assert len(calls) == 2
    assert "wrong output" in calls[1]["messages"][-1]["content"]
    assert calls[0]["model"] == model
    token_param = "max_completion_tokens" if model == "gpt-6-luna" else "max_tokens"
    assert calls[0][token_param] == 8192
    assert "temperature" not in calls[0]


def test_tracker_correctness_gate(tmp_path):
    import json
    from nki_kernel_search.tracking import SearchTracker
    tracker = SearchTracker(tmp_path)
    tracker.record(0, {"correctness": 1, "hardware_measured": 1, "latency_us": 10,
                       "combined_score": 1}, "baseline")
    tracker.record(1, {"correctness": 0, "latency_us": 1, "combined_score": 0}, "bad")
    tracker.record(2, {"correctness": 1, "hardware_measured": 1, "latency_us": 5,
                       "combined_score": 2}, "winner")
    tracker.close()
    rows = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert "kernel/latency_us" not in rows[1]
    assert rows[2]["kernel/best_latency_us"] == 5
    assert (tmp_path / "best_kernel.py").read_text() == "winner"


@pytest.mark.parametrize("project_env", [None, "env-owner/env-kernels"])
def test_wandb_tracking(monkeypatch, tmp_path, project_env):
    from types import SimpleNamespace
    import wandb
    from nki_kernel_search.tracking import SearchTracker
    options, logged, finished = {}, [], []
    def init(**kwargs):
        options.update(kwargs)
        return SimpleNamespace(id="test-run", url="https://example.test/run",
                               define_metric=lambda *a, **k: None,
                               log=logged.append, finish=lambda: finished.append(True))
    monkeypatch.setattr(wandb, "init", init)
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    if project_env:
        monkeypatch.setenv("WANDB_PROJECT", project_env)
    else:
        monkeypatch.delenv("WANDB_PROJECT", raising=False)
    tracker = SearchTracker(tmp_path, {"enabled": True, "project": "owner/kernels", "mode": "offline"})
    tracker.record(0, {"correctness": 1, "hardware_measured": 1,
                       "latency_us": 25, "combined_score": 1})
    tracker.close()
    tracker.close()
    assert options["entity"] == ("env-owner" if project_env else "owner")
    assert options["project"] == ("env-kernels" if project_env else "kernels")
    if project_env:
        assert __import__("os").environ["WANDB_PROJECT"] == "env-kernels"
    assert logged[0]["kernel/latency_us"] == 25
    assert len(finished) == 1


def test_installed_sdk_prompt():
    pytest.importorskip("nki")
    from nki_kernel_search.sdk_prompt import sdk_context
    prompt = sdk_context()
    assert "nisa.tensor_scalar(dst:" in prompt
    assert "There is no nl.memset" in prompt


def test_writer_context_loading(tmp_path):
    import json
    from nki_kernel_search.prompt_context import writer_context
    from nki_kernel_search.config_runner import read_config
    document = tmp_path / "writer.md"
    document.write_text("---\nmodel: opus\ntools: [Bash]\n---\n# Writer\nUse explicit dst tensors.")
    text = writer_context([document])
    assert "Use explicit dst tensors" in text
    assert "model: opus" not in text
    assert "cannot invoke" in text
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"reference": str(TASK), "context_files": ["writer.md"]}))
    assert read_config(config)["context_files"] == [str(document)]
    config.write_text(json.dumps({"reference": str(TASK), "context_files": ["missing.md"]}))
    with pytest.raises(ValueError, match="Context file does not exist"):
        read_config(config)


def test_credential_environment_aliases(monkeypatch):
    from nki_kernel_search.environment import configure_environment
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_KEY", "test-openai")
    monkeypatch.setenv("WANDB_KEY", "test-wandb")
    configure_environment()
    assert __import__("os").environ["OPENAI_API_KEY"] == "test-openai"
    assert __import__("os").environ["WANDB_API_KEY"] == "test-wandb"
    monkeypatch.setenv("OPENAI_API_KEY", "canonical")
    configure_environment()
    assert __import__("os").environ["OPENAI_API_KEY"] == "canonical"


def test_wandb_environment_normalization(monkeypatch):
    import os
    from nki_kernel_search.environment import configure_environment
    monkeypatch.setenv("WANDB_PROJECT", "org/kernels")
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    configure_environment()
    assert os.environ["WANDB_PROJECT"] == "kernels"
    assert os.environ["WANDB_ENTITY"] == "org"
    configure_environment()
    assert os.environ["WANDB_ENTITY"] == "org"
    monkeypatch.setenv("WANDB_ENTITY", "explicit-org")
    monkeypatch.setenv("WANDB_PROJECT", "other-org/kernels")
    configure_environment()
    assert os.environ["WANDB_ENTITY"] == "explicit-org"


def test_real_openevolve_context_is_logged_unchanged(monkeypatch, tmp_path):
    import asyncio
    import json
    from openevolve.config import PromptConfig
    from openevolve.database import Program
    from openevolve.prompt.sampler import PromptSampler
    from openevolve.llm.openai import OpenAILLM
    from nki_kernel_search.llm_audit import AuditedOpenAILLM
    prompt_config = PromptConfig(system_message="NKI authoring guidance", include_artifacts=True)
    sampler = PromptSampler(prompt_config)
    parent = Program(id="parent", code=INITIAL.read_text(), metrics={"combined_score": 0, "correctness": 0})
    winner = Program(id="winner", code="def faster_kernel(x): return x", metrics={"combined_score": 1.1, "latency_us": 20})
    prompt = sampler.build_prompt(current_program=parent.code, parent_program=parent.code,
                                  program_metrics=parent.metrics, previous_programs=[parent.to_dict()],
                                  top_programs=[winner.to_dict()], inspirations=[],
                                  program_artifacts={"error": "AttributeError: nl.memset does not exist"},
                                  evolution_round=7, diff_based_evolution=True)
    sent = {}
    async def fake_generate(self, system_message, messages, **kwargs):
        sent.update(system=system_message, messages=messages)
        return "test-response"
    monkeypatch.setattr(OpenAILLM, "generate_with_context", fake_generate)
    monkeypatch.setenv("NKI_SEARCH_OUTPUT", str(tmp_path))
    client = AuditedOpenAILLM.__new__(AuditedOpenAILLM)
    client.model = "test-model"
    messages = [{"role": "user", "content": prompt["user"]}]
    assert asyncio.run(client.generate_with_context(prompt["system"], messages)) == "test-response"
    recorded = json.loads(next((tmp_path / "prompts").glob("*.json")).read_text())
    assert recorded["messages"][0]["content"] == sent["system"]
    assert recorded["messages"][1:] == sent["messages"]
    user = recorded["messages"][1]["content"]
    assert "nl.memset does not exist" in user
    assert "faster_kernel" in user
    assert parent.code in user


@pytest.mark.parametrize("model,effort", [("gpt-6-luna", None),
                                          ("gpt-6-luna", "none"),
                                          ("gpt-4.1", None)])
def test_openevolve_chat_request_compatibility(monkeypatch, tmp_path, model, effort):
    import asyncio
    from types import SimpleNamespace
    import openai
    from openevolve.config import LLMModelConfig
    from nki_kernel_search.llm_audit import AuditedOpenAILLM

    sent = {}

    def create(**kwargs):
        sent.update(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="kernel"))])

    monkeypatch.setattr(openai, "OpenAI", lambda **kwargs: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    monkeypatch.setenv("NKI_SEARCH_OUTPUT", str(tmp_path))
    client = AuditedOpenAILLM(LLMModelConfig(
        name=model, api_key="test-placeholder", api_base="https://api.openai.com/v1",
        temperature=0.7, top_p=0.9, max_tokens=8192,
        timeout=30, retries=0, retry_delay=0, reasoning_effort=effort))
    messages = [{"role": "user", "content": "Improve the selected parent kernel."}]
    assert asyncio.run(client.generate_with_context("NKI guidance", messages)) == "kernel"
    assert sent["model"] == model
    assert sent["messages"] == [{"role": "system", "content": "NKI guidance"}] + messages
    if model == "gpt-6-luna":
        assert sent["max_completion_tokens"] == 8192
        assert "max_tokens" not in sent
    else:
        assert sent["max_tokens"] == 8192
    if model == "gpt-6-luna" and effort is None:
        assert "temperature" not in sent
        assert "top_p" not in sent
    else:
        assert sent["temperature"] == 0.7
        assert sent["top_p"] == 0.9
