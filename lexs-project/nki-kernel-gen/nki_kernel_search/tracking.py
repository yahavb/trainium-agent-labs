import json
import math
import os
from pathlib import Path
import time

from .environment import configure_environment, split_wandb_project

class SearchTracker:
    """Main-process trace sink, with optional W&B and an always-local metrics log."""
    def __init__(self, output, options=None, delegate=None):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.delegate = delegate
        self.records = []
        self.best_score = 0
        self.best_latency = math.inf
        self.closed = False
        self.run = None
        options = options or {}
        if options.get("enabled", False):
            configure_environment()
            project = os.environ.get("WANDB_PROJECT") or options.get("project", "trainium-kernels")
            embedded_entity, project = split_wandb_project(project)
            entity = os.environ.get("WANDB_ENTITY") or options.get("entity") or embedded_entity
            import wandb
            self.run = wandb.init(project=project, entity=entity, mode=options.get("mode", "offline"),
                                  name=options.get("name"), dir=str(self.output))
            self.run.define_metric("iteration")
            self.run.define_metric("kernel/*", step_metric="iteration")
            (self.output / "wandb_run.json").write_text(json.dumps(
                {"id": self.run.id, "url": self.run.url}, indent=2))

    def record(self, iteration, metrics, code=None, program_id=None):
        row = {"iteration": iteration, "timestamp": time.time(), "program_id": program_id,
               "kernel/correctness": metrics.get("correctness", 0),
               "kernel/speedup": metrics.get("combined_score", 0),
               "kernel/hardware_measured": metrics.get("hardware_measured", 0)}
        valid = metrics.get("correctness") == 1 and metrics.get("hardware_measured") == 1
        if valid and "latency_us" in metrics:
            row["kernel/latency_us"] = metrics["latency_us"]
            self.best_latency = min(self.best_latency, metrics["latency_us"])
            if metrics["combined_score"] > self.best_score:
                self.best_score = metrics["combined_score"]
                if code:
                    (self.output / "best_kernel.py").write_text(code)
        if math.isfinite(self.best_latency):
            row["kernel/best_latency_us"] = self.best_latency
        row["kernel/best_speedup"] = self.best_score
        if code:
            directory = self.output / "candidates"
            directory.mkdir(exist_ok=True)
            (directory / f"iteration_{iteration}.py").write_text(code)
        self.records.append(row)
        with (self.output / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        if self.run:
            self.run.log({k: v for k, v in row.items() if k not in ("timestamp", "program_id")})
        self.plot()

    def log_trace(self, iteration, parent_program, child_program, **kwargs):
        if self.delegate:
            self.delegate.log_trace(iteration=iteration, parent_program=parent_program,
                                    child_program=child_program, **kwargs)
        self.record(iteration, child_program.metrics, child_program.code, child_program.id)

    def plot(self):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return
        fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
        good = [r for r in self.records if "kernel/latency_us" in r]
        axes[0].plot([r["iteration"] for r in good], [r["kernel/latency_us"] for r in good], "o-", label="Candidate")
        axes[0].plot([r["iteration"] for r in good], [r["kernel/best_latency_us"] for r in good], label="Best so far")
        axes[0].set_ylabel("Mean of shape p50 runtimes (µs)")
        axes[0].legend()
        axes[1].plot([r["iteration"] for r in self.records], [r["kernel/correctness"] for r in self.records], "o")
        axes[1].set_ylabel("Passed correctness")
        axes[1].set_xlabel("Search iteration (0 = baseline)")
        for ax in axes:
            ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(self.output / "runtime_history.png")
        plt.close(fig)

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.delegate:
            self.delegate.close()
        if self.run:
            self.run.finish()
