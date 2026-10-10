"""Run a bounded experiment without loading model weights locally."""
import argparse
import copy
import json
import os
import random
import sys
import time
import unittest
from pathlib import Path

from .bandit import Bandit
from .experiment import BudgetStop, HTTPClient, OfflineClient, OperationalError, episode, INTERVENTION_VERSION
from .problems import generate, load, signature

MODES = ("fixed", "random", "rule", "learned")
HERE = Path(__file__).resolve().parents[1]


def save_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def config_of(a):
    if not a.offline and not a.base:
        raise ValueError("set HEATROD_BASE_URL or pass --base; use --offline for synthetic checks")
    return dict(base=a.base, model=a.model, temperature=0.6, top_p=0.95,
                max_tokens=a.max_tokens, max_calls=7, repairs=3,
                http_timeout=a.http_timeout, symbolic_timeout=a.symbolic_timeout,
                synthetic=a.offline, seed=a.seed, epsilon_horizon=60,
                request_seeds=a.request_seeds, intervention_version=INTERVENTION_VERSION)


def train(a):
    records = load(a.data, "train")
    config = config_of(a)
    output = Path(a.output)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "policy.json"
    if checkpoint.exists() and not a.resume:
        raise ValueError("output already contains a policy; use --resume or a new directory")
    policy = (Bandit.load(a.resume) if a.resume else
              Bandit.warm_start(a.warm_start, config, signature(records)) if a.warm_start else Bandit(a.seed))
    if a.resume and (policy.dataset != signature(records) or policy.config != config):
        raise ValueError("resume requires identical data and generation configuration")
    policy.dataset, policy.config = signature(records), config
    client = OfflineClient() if a.offline else HTTPClient(config)
    started = time.monotonic()
    deadline = started + a.minutes*60
    status, reason, completed = "complete", None, 0
    policy.save(checkpoint)
    with (output / "attempts.jsonl").open("a") as log, (output / "metrics.jsonl").open("a") as metrics:
        try:
            while policy.episodes < a.episodes:
                epoch, offset = divmod(policy.episodes, len(records))
                order = list(records)
                random.Random(a.seed+epoch).shuffle(order)
                epsilon = 0.3-0.25*min(policy.episodes/(config["epsilon_horizon"]-1), 1)
                row = episode(order[offset], client, policy, "train", config,
                              a.seed+policy.episodes, deadline, log, epsilon)
                policy.episodes += 1
                completed += 1
                row.update(episode=policy.episodes, epsilon=epsilon)
                metrics.write(json.dumps(row)+"\n")
                metrics.flush()
                policy.save(checkpoint)
                print(f"episode {policy.episodes}: level {row['level']}, score {row['score']:.1f}, calls {row['http_calls']}", flush=True)
        except (BudgetStop, OperationalError, KeyboardInterrupt) as exc:
            status = "capped" if isinstance(exc, BudgetStop) else "interrupted" if isinstance(exc, KeyboardInterrupt) else "server_error"
            reason = str(exc) or "keyboard interrupt"
        finally:
            # Completed transitions in an interrupted episode are retained. Resume retries
            # that problem; the episode count advances only on complete episodes.
            policy.save(checkpoint)
    summary = dict(status=status, reason=reason, episodes=policy.episodes, completed_this_run=completed,
                   updates=sum(sum(counts) for counts in policy.counts.values()),
                   elapsed=time.monotonic()-started, synthetic=a.offline, config=config,
                   dataset=policy.dataset, policy=str(checkpoint), provenance=policy.provenance)
    save_json(output / "training.json", summary)
    print(json.dumps(summary, indent=2))
    return 2 if status == "server_error" else 0


def evaluate(a):
    records = load(a.data, a.split)
    config = config_of(a)
    policy = Bandit.load(a.policy)
    if policy.config.get("synthetic") != a.offline:
        raise ValueError("cannot mix a synthetic policy and real-model evaluation")
    if policy.config != config:
        raise ValueError("evaluation must use the policy's generation configuration and seed")
    # Validate that held-out physical IDs are not present in the policy's training set.
    training = load(Path(a.data).parent / "train.jsonl", "train")
    if signature(training) != policy.dataset or {r['id'] for r in records} & {r['id'] for r in training}:
        raise ValueError("evaluation dataset is not the policy's disjoint held-out split")
    source_dataset = signature(records)
    if a.one_per_level:
        first = {}
        for record in records:
            first.setdefault(record["level"], record)
        if set(first) != set(range(5)):
            raise ValueError("one-per-level evaluation requires all five levels in the held-out data")
        records = [first[level] for level in range(5)]
    modes = a.controllers
    output = Path(a.output)
    output.mkdir(parents=True, exist_ok=True)
    target = output / "evaluation.json"
    if target.exists():
        raise ValueError("evaluation already exists; choose a new output directory")
    client = OfflineClient() if a.offline else HTTPClient(config)
    started = time.monotonic()
    deadline = started+a.minutes*60
    result = dict(version=1, status="running", reason=None, synthetic=a.offline,
                  config=config, dataset=signature(records), split=a.split,
                  source_dataset=source_dataset, problem_ids=[r["id"] for r in records],
                  controllers=modes, one_per_level=a.one_per_level,
                  seeds=a.seeds, expected=len(records)*len(a.seeds)*len(modes),
                  policy_dataset=policy.dataset, policy_episodes=policy.episodes,
                  policy_signature=signature(dict(values=policy.values, counts=policy.counts)),
                  policy_provenance=policy.provenance,
                  rows=[])
    save_json(target, result)
    with (output / "attempts.jsonl").open("a") as log:
        try:
            for seed in a.seeds:
                for record in records:
                    for mode in modes:
                        frozen = copy.deepcopy(policy)
                        frozen.rng.seed(seed)
                        row = episode(record, client, frozen, mode, config, seed, deadline, log)
                        result["rows"].append(row)
                        save_json(target, result)
                        print(f"{mode}: level {row['level']}, score {row['score']:.1f}, calls {row['http_calls']}", flush=True)
            result["status"] = "complete"
        except (BudgetStop, OperationalError, KeyboardInterrupt) as exc:
            result["status"] = "capped" if isinstance(exc, BudgetStop) else "interrupted" if isinstance(exc, KeyboardInterrupt) else "server_error"
            result["reason"] = str(exc) or "keyboard interrupt"
        finally:
            result["elapsed"] = time.monotonic()-started
            save_json(target, result)
    print(f"{result['status']}: {len(result['rows'])}/{result['expected']} runs saved to {target}")
    return 2 if result["status"] == "server_error" else 0


def summarize(rows):
    count = len(rows)
    if not count:
        return dict(n=0, first_solve_rate=None, solve_rate=None, mean_score=None,
                    mean_calls=None, mean_calculator_requests=None, elapsed=0)
    return dict(n=count, first_solve_rate=sum(r["first_solved"] for r in rows)/count,
                solve_rate=sum(r["solved"] for r in rows)/count,
                mean_score=sum(r["score"] for r in rows)/count,
                mean_calls=sum(r["http_calls"] for r in rows)/count,
                mean_calculator_requests=sum(r["calculator_requests"] for r in rows)/count,
                elapsed=sum(r["elapsed"] for r in rows))


def report(a):
    result = json.loads(Path(a.input).read_text())
    if result.get("version") != 1:
        raise ValueError("unsupported evaluation format")
    modes = result.get("controllers", list(MODES))
    if not modes or len(set(modes)) != len(modes) or any(mode not in MODES for mode in modes):
        raise ValueError("invalid evaluation controllers")
    by_mode = {mode: {} for mode in modes}
    for row in result["rows"]:
        key = (row["problem"], row["seed"])
        table = by_mode[row["mode"]]
        if key in table or row["synthetic"] != result["synthetic"]:
            raise ValueError("duplicate or inconsistent evaluation rows")
        table[key] = row
    common = set.intersection(*(set(table) for table in by_mode.values()))
    summary = dict(synthetic=result["synthetic"], status=result["status"],
                   incomplete=result["status"] != "complete" or len(result["rows"]) != result["expected"],
                   common_problem_seed_pairs=len(common), excluded_runs=len(result["rows"])-len(modes)*len(common),
                   policy_provenance=result.get("policy_provenance"),
                   controllers={}, by_level={})
    for mode, table in by_mode.items():
        rows = [table[key] for key in sorted(common)]
        summary["controllers"][mode] = summarize(rows)
        summary["by_level"][mode] = {str(level): summarize([r for r in rows if r["level"] == level]) for level in range(5)}
    label = "evaluation" if len(modes) == 1 else "comparison"
    lines = [f"# Learning to Repair — {label}", "",
             "**SYNTHETIC OFFLINE DATA — no real-model performance claim.**" if result["synthetic"] else f"Frozen Qwen controller {label}.", "",
             f"Status: {result['status']}. Common completed problem/seed pairs: {len(common)}. "
             f"Excluded unmatched runs: {summary['excluded_runs']}. "
             f"{label.capitalize()} {'incomplete' if summary['incomplete'] else 'complete'}.", "",
             "Maximum per problem: seven HTTP calls, three repair decisions; calculator follow-ups count as HTTP calls. All controllers stop when solved. Server-side request seeds are " +
             ("enabled; server support determines reproducibility." if result["config"].get("request_seeds", True) else
              "disabled for Neuron compatibility; model samples are not reproducibly paired."), "",
             "| Controller | N | First solve | Final solve | Mean score | Mean HTTP calls | Mean calculator requests | Seconds |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    def cells(stats):
        return " | ".join("—" if stats[k] is None else f"{stats[k]:.3f}" for k in
                          ("first_solve_rate", "solve_rate", "mean_score", "mean_calls", "mean_calculator_requests", "elapsed"))
    for mode in modes:
        stats = summary["controllers"][mode]
        lines.append(f"| {mode} | {stats['n']} | {cells(stats)} |")
    if result.get("policy_provenance"):
        lineage = result["policy_provenance"]
        lines += ["", f"The learned controller was warm-started from {lineage['source_episodes']} completed "
                  f"version-1 episodes. {lineage['reset_observations']} calculator observations were reset; "
                  f"{lineage['retained_observations']} other observations were retained. This is a mixed-history "
                  "warm-started policy, not training from scratch entirely under the corrected protocol."]
    lines += ["", "## Results by level", "", "| Level | Controller | N | Final solve | Mean score |", "|---|---|---:|---:|---:|"]
    for level in range(5):
        for mode in modes:
            stats = summary["by_level"][mode][str(level)]
            solve = "—" if stats["solve_rate"] is None else f"{stats['solve_rate']:.3f}"
            score = "—" if stats["mean_score"] is None else f"{stats['mean_score']:.3f}"
            lines.append(f"| {level} | {mode} | {stats['n']} | {solve} | {score} |")
    lines += ["", "This evaluation measures unseen instances within the same five levels. One problem per level is a smoke test, not coverage of every problem type. A learned-only run measures that controller's performance; it does not establish improvement over baselines. Scores use numerical checks, not a proof for every x and t."]
    output = Path(a.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines)+"\n")
    save_json(output.with_suffix(".json"), summary)
    print(f"Report saved to {output}")
    return 0


def positive(value):
    parsed = float(value)
    if not 0 < parsed < float("inf"):
        raise argparse.ArgumentTypeError("must be finite and positive")
    return parsed


def integer(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train a repair bandit around frozen Qwen; no model fine-tuning.")
    subs = ap.add_subparsers(dest="command", required=True)
    subs.add_parser("selftest", help="run local physics and controller regressions")
    g = subs.add_parser("generate")
    g.add_argument("--output", default="data")
    g.add_argument("--seed", type=int, default=42)
    for name, default in (("train", 60), ("validation", 15), ("test", 15)):
        g.add_argument("--"+name, type=integer, default=default)
    for command in ("train", "evaluate"):
        p = subs.add_parser(command)
        p.add_argument("--data", default=f"data/{'train' if command == 'train' else 'test'}.jsonl")
        p.add_argument("--output", default=f"runs/{command}")
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--base", default=os.environ.get("HEATROD_BASE_URL"))
        p.add_argument("--model", default=os.environ.get("HEATROD_MODEL", "Qwen/Qwen3-8B"))
        p.add_argument("--max-tokens", type=integer, default=1200)
        p.add_argument("--http-timeout", type=positive, default=120)
        p.add_argument("--symbolic-timeout", type=positive, default=20)
        p.add_argument("--minutes", type=positive, default=60 if command == "train" else 35)
        p.add_argument("--offline", action="store_true")
        p.add_argument("--request-seeds", action="store_true",
                       help="send per-request seeds only on backends supporting device RNGs; off by default for Neuron")
        if command == "train":
            p.add_argument("--episodes", type=integer, default=60)
            continuation = p.add_mutually_exclusive_group()
            continuation.add_argument("--resume")
            continuation.add_argument("--warm-start", help="import v1 policy into v2, resetting only calculator observations; use a new output directory")
        else:
            p.add_argument("--policy", default="runs/train/policy.json")
            p.add_argument("--split", choices=("validation", "test"), default="test")
            p.add_argument("--seeds", nargs="+", type=int, default=[10])
            p.add_argument("--controllers", nargs="+", choices=MODES, default=list(MODES),
                           help="controllers to evaluate; use --controllers learned to omit baselines")
            p.add_argument("--one-per-level", action="store_true",
                           help="select the first held-out problem at each level 0..4, in level order")
    r = subs.add_parser("report")
    r.add_argument("input", help="evaluation.json from the paired comparison")
    r.add_argument("--output", default="runs/comparison.md")
    a = ap.parse_args(argv)
    try:
        if a.command == "selftest":
            suite = unittest.defaultTestLoader.discover(str(HERE / "tests"))
            return_code = 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1
        elif a.command == "generate":
            print(json.dumps(generate(a.output, a.seed, a.train, a.validation, a.test), indent=2))
            return_code = 0
        else:
            if a.command == "evaluate" and len(set(a.seeds)) != len(a.seeds):
                raise ValueError("evaluation seeds must be unique")
            if a.command == "evaluate" and len(set(a.controllers)) != len(a.controllers):
                raise ValueError("evaluation controllers must be unique")
            return_code = globals()[a.command](a)
    except (ValueError, OSError, KeyError) as exc:
        ap.error(str(exc))
    raise SystemExit(return_code)
