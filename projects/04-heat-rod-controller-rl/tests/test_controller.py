import copy
import contextlib
import io
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from controllerlab.bandit import ACTIONS, Bandit, better, rule, state_of, transition_reward
from controllerlab.cli import main
from controllerlab.experiment import BudgetStop, HTTPClient, OperationalError, episode, http_request, calculator_requests, CALCULATOR
from controllerlab.physics import arithmetic, bounded, checker, problem_of
from controllerlab.problems import generate, load, make


def result(score=0.8, error=0.1, valid=True):
    return dict(reward=score, parts=dict(equation=True, left_bc=True, right_bc=True,
                                       start_shape=score == 1) if valid else {},
                expr="0", start_error=error, feedback="Initial profile fails.")


CONFIG = dict(max_calls=7, repairs=3, symbolic_timeout=2)


class ScriptedClient:
    synthetic = True

    def __init__(self, responses=None, fail=False):
        self.responses = iter(responses or ["u(x,t)=0"]*10)
        self.fail = fail
        self.prompts = []

    def ask(self, prompt, seed, record, action, deadline):
        self.prompts.append(prompt)
        if self.fail:
            raise OperationalError("server down")
        return next(self.responses), None


class ControllerTests(unittest.TestCase):
    def test_learned_only_one_per_level_evaluation_and_report(self):
        def invoke(args):
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as exc:
                main(args)
            self.assertEqual(exc.exception.code, 0)

        def fake_episode(record, client, policy, mode, config, seed, deadline, log, epsilon=0):
            return dict(problem=record["id"], level=record["level"], seed=seed, mode=mode,
                        first_score=1.0, score=1.0, first_solved=True, solved=True,
                        http_calls=1, calculator_requests=0, repairs=0, elapsed=0.1, synthetic=True)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generate(root/"data", train=15)
            with patch("controllerlab.cli.episode", side_effect=fake_episode) as mock_episode:
                invoke(["train", "--offline", "--episodes", "1", "--data", str(root/"data/train.jsonl"), "--output", str(root/"train")])
                source = root/"train/policy.json"
                before = source.read_bytes()
                mock_episode.reset_mock()
                invoke(["evaluate", "--offline", "--policy", str(source), "--data", str(root/"data/test.jsonl"),
                        "--controllers", "learned", "--one-per-level", "--output", str(root/"eval")])
                self.assertEqual(mock_episode.call_count, 5)
                self.assertEqual([call.args[0]["level"] for call in mock_episode.call_args_list], list(range(5)))
                self.assertTrue(all(call.args[3] == "learned" for call in mock_episode.call_args_list))
                self.assertEqual(source.read_bytes(), before)
            result_path = root/"eval/evaluation.json"
            result = json.loads(result_path.read_text())
            self.assertEqual(result["expected"], 5)
            self.assertEqual(result["controllers"], ["learned"])
            self.assertEqual(len(result["problem_ids"]), 5)
            invoke(["report", str(result_path), "--output", str(root/"report.md")])
            summary = json.loads((root/"report.json").read_text())
            self.assertEqual(set(summary["controllers"]), {"learned"})
            self.assertEqual(summary["controllers"]["learned"]["n"], 5)
            self.assertEqual(summary["excluded_runs"], 0)
            result["rows"].pop()
            result["status"] = "capped"
            result_path.write_text(json.dumps(result))
            invoke(["report", str(result_path), "--output", str(root/"partial.md")])
            partial = json.loads((root/"partial.json").read_text())
            self.assertEqual(partial["controllers"]["learned"]["n"], 4)
            self.assertTrue(partial["incomplete"])

    def test_warm_start_retains_other_actions_and_episode_position(self):
        p = Bandit(9)
        p.config = dict(max_tokens=1200, request_seeds=False, intervention_version=1)
        p.dataset, p.episodes = "dataset", 16
        for index, action in enumerate(ACTIONS):
            p.update("s", action, (index-2)/10)
        before = copy.deepcopy(p.values["s"])
        config = dict(p.config, intervention_version=2)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/"old.json"
            p.save(source)
            original = source.read_bytes()
            migrated = Bandit.warm_start(source, config, "dataset")
            self.assertEqual(migrated.episodes, 16)
            self.assertEqual(migrated.rng.getstate(), p.rng.getstate())
            for index, action in enumerate(ACTIONS):
                self.assertEqual(migrated.values["s"][index], 0 if action == "USE_CALCULATOR" else before[index])
                self.assertEqual(migrated.counts["s"][index], 0 if action == "USE_CALCULATOR" else 1)
            self.assertEqual(migrated.provenance["reset_observations"], 1)
            self.assertEqual(migrated.provenance["retained_observations"], 5)
            target = Path(directory)/"new.json"
            migrated.save(target)
            self.assertEqual(Bandit.load(target).provenance, migrated.provenance)
            self.assertEqual(source.read_bytes(), original)
            with self.assertRaises(ValueError):
                Bandit.warm_start(source, config, "different dataset")
            with self.assertRaises(ValueError):
                Bandit.warm_start(source, dict(config, max_tokens=600), "dataset")

    def test_calculator_markdown_protocol(self):
        expression = "Integral((-x**3 + x)*sin(pi*x), (x,0,1))"
        for source in (f"COMPUTE: {expression}", f"**COMPUTE: {expression}**",
                       f"`COMPUTE: {expression}`", f"```\nCOMPUTE: {expression}\n```"):
            self.assertEqual(calculator_requests(source), [expression])
        self.assertEqual(calculator_requests("### Compute: $ c_1 $"), [])
        self.assertEqual(calculator_requests("u(x,t) = x**2"), [])
        self.assertEqual(len(calculator_requests("\n".join(["COMPUTE: 2**3"]*10))), 8)
        self.assertIn("ONLY 1 to 8 plain lines", CALCULATOR)

    def test_bandit_running_mean_and_persistence(self):
        p = Bandit(7)
        p.update("s", "FIX_PDE", 0.2)
        p.update("s", "FIX_PDE", 0.6)
        self.assertAlmostEqual(p.values["s"][1], 0.4)
        p.config, p.dataset, p.episodes = {"model": "qwen"}, "abc", 3
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"policy.json"
            p.save(path)
            loaded = Bandit.load(path)
            self.assertEqual(p.__dict__, loaded.__dict__ | {"rng": p.rng})
            self.assertEqual([p.select("s", 0.3) for _ in range(20)],
                             [loaded.select("s", 0.3) for _ in range(20)])

    def test_state_buckets_and_rule(self):
        self.assertTrue(state_of(result(1, 0), 0.005).endswith("passed"))
        self.assertTrue(state_of(result(error=0.01), 0.005).endswith("near"))
        self.assertTrue(state_of(result(error=0.5), 0.005).endswith("far"))
        self.assertTrue(state_of(result(error=None, valid=False), 0.005).endswith("unavailable"))
        self.assertEqual(rule(result(valid=False)), "FIX_FORMAT")
        r = result()
        r["parts"]["left_bc"] = False
        self.assertEqual(rule(r), "CHANGE_BASIS")
        r["parts"]["left_bc"] = True
        r["parts"]["equation"] = False
        self.assertEqual(rule(r), "FIX_PDE")
        self.assertEqual(rule(result()), "USE_CALCULATOR")
        self.assertEqual(rule(result(), True), "FIX_COEFF")

    def test_reward_and_best_retention(self):
        self.assertAlmostEqual(transition_reward(result(), result(1, 0), 2, 1), 0.678)
        self.assertLess(transition_reward(result(), result(0.4), 1, 0), 0)
        self.assertTrue(better(result(error=0.01), result(error=0.1)))
        self.assertFalse(better(result(0.4, 0), result()))

    def test_disjoint_balanced_reproducible_data(self):
        with tempfile.TemporaryDirectory() as directory:
            generate(directory)
            splits = {s: load(Path(directory)/f"{s}.jsonl", s) for s in ("train", "validation", "test")}
            all_ids = [r["id"] for records in splits.values() for r in records]
            self.assertEqual(len(all_ids), len(set(all_ids)))
            for split, records in splits.items():
                for level in range(5):
                    self.assertEqual(sum(r["level"] == level for r in records), len(records)//5)
                self.assertFalse(any("reference" in r for r in records))
            self.assertEqual(make(4, 2, 2), make(4, 2, 2))
            path = Path(directory)/"train.jsonl"
            path.write_text(path.read_text().replace('"tol": 1e-06', '"tol": 0.1', 1))
            with self.assertRaises(ValueError):
                load(path, "train")

    def test_action_controls_calculator(self):
        record = make(0, 1)
        # Calculator follow-ups count as model calls; repeated non-improving actions
        # still update the bandit from actual results rather than retained best.
        policy = Bandit()
        state = state_of(result(), record["tol"])
        policy.update(state, "USE_CALCULATOR", 10)
        client = ScriptedClient(["u(x,t)=0"] + ["**COMPUTE: Integral(x,(x,0,1))**", "u(x,t)=0"]*3)
        with patch("controllerlab.experiment.check", return_value=result()), patch("controllerlab.experiment.bounded", return_value="1/2") as calc:
            row = episode(record, client, policy, "train", CONFIG, 10, time.monotonic()+30, io.StringIO())
        self.assertEqual(row["http_calls"], 7)
        self.assertEqual(row["calculator_requests"], 3)
        self.assertEqual(calc.call_count, 3)
        self.assertEqual(policy.counts[state][3], 4)
        self.assertLess(policy.values[state][3], 10)
        client = ScriptedClient(["COMPUTE: Integral(x,(x,0,1))"]*4)
        with patch("controllerlab.experiment.check", return_value=result()), patch("controllerlab.experiment.bounded") as calc:
            row = episode(record, client, policy, "fixed", CONFIG, 10, time.monotonic()+30, io.StringIO())
        calc.assert_not_called()
        self.assertEqual(row["http_calls"], 4)

    def test_solved_stop_and_frozen_evaluation(self):
        p = Bandit(12)
        before = copy.deepcopy((p.values, p.counts, p.episodes))
        with patch("controllerlab.experiment.check", side_effect=[result(), result(1, 0)]):
            row = episode(make(0, 1), ScriptedClient(), p, "learned", CONFIG, 10, time.monotonic()+30, io.StringIO())
        self.assertEqual(row["http_calls"], 2)
        self.assertTrue(row["solved"])
        self.assertEqual(before, (p.values, p.counts, p.episodes))

    def test_call_cap_and_actual_negative_update(self):
        p = Bandit()
        cfg = dict(CONFIG, max_calls=2)
        log = io.StringIO()
        with patch("controllerlab.experiment.check", side_effect=[result(), result(0.4)]):
            row = episode(make(0, 1), ScriptedClient(), p, "train", cfg, 1, time.monotonic()+30, log, 1)
        self.assertEqual(row["score"], 0.8)
        self.assertEqual(row["http_calls"], 2)
        transition = next(row for row in map(json.loads, log.getvalue().splitlines()) if row["kind"] == "transition")
        self.assertAlmostEqual(transition["bandit_reward"], -0.41)

    def test_operational_failure_has_no_update(self):
        p = Bandit()
        with self.assertRaises(OperationalError):
            episode(make(0, 1), ScriptedClient(fail=True), p, "train", CONFIG, 1, time.monotonic()+30, io.StringIO())
        self.assertFalse(p.counts)
        with self.assertRaises(BudgetStop):
            episode(make(0, 1), ScriptedClient(), p, "train", CONFIG, 1, time.monotonic()-1, io.StringIO())
        client = ScriptedClient()
        log = io.StringIO()
        with patch.object(client, "ask", side_effect=[("u(x,t)=0", None), OperationalError("repair failed")]), patch("controllerlab.experiment.check", return_value=result()):
            with self.assertRaises(OperationalError):
                episode(make(0, 1), client, p, "train", CONFIG, 1, time.monotonic()+30, log)
        self.assertFalse(p.counts)
        self.assertEqual(json.loads(log.getvalue().splitlines()[-1])["kind"], "request_failure")

    def test_worker_timeout_and_restricted_calculator(self):
        with self.assertRaises(TimeoutError):
            bounded("compute", "Integral(x,(x,0,1))", 0.001)
        self.assertEqual(bounded("compute", "Integral(x,(x,0,1))", 10), "1/2")
        for source in ("__import__('os').system('ls')", "x.__class__", "2**1000", "open('secret')"):
            with self.assertRaises(ValueError):
                arithmetic(source, calculator=True)

    def test_http_settings_and_failure_classification(self):
        cfg = dict(base="https://example.test/v1", model="Qwen/Qwen3-8B",
                   temperature=0.6, top_p=0.95, max_tokens=1200, http_timeout=2)
        from unittest.mock import Mock
        response = Mock()
        response.json.return_value = {"choices": [{"message": {"content": "u(x,t)=0"}}], "usage": {"total_tokens": 2}}
        with patch("controllerlab.experiment.httpx.post", return_value=response) as post:
            answer, usage = http_request(cfg, "prompt", 10, 2)
        self.assertEqual(answer, "u(x,t)=0")
        body = post.call_args.kwargs["json"]
        self.assertFalse(body["chat_template_kwargs"]["enable_thinking"])
        self.assertEqual(body["temperature"], 0.6)
        self.assertNotIn("seed", body)
        with patch("controllerlab.experiment.httpx.post", return_value=response) as post:
            http_request(dict(cfg, request_seeds=True), "prompt", 10, 2)
        self.assertEqual(post.call_args.kwargs["json"]["seed"], 10)
        with patch("controllerlab.experiment.bounded", side_effect=TimeoutError("HTTP deadline")):
            with self.assertRaises(OperationalError):
                HTTPClient(cfg).ask("prompt", 10, {}, None, time.monotonic()+20)

    def test_training_resume_and_configuration_guard(self):
        def invoke(args, expected=0):
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exc:
                main(args)
            self.assertEqual(exc.exception.code, expected)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generate(root/"data", train=15)
            out = root/"train"
            args = ["train", "--offline", "--data", str(root/"data/train.jsonl"), "--output", str(out)]
            with patch("controllerlab.cli.episode", return_value=dict(level=0, score=0.8, http_calls=1)):
                invoke(args+["--episodes", "1"])
                invoke(args+["--episodes", "3", "--resume", str(out/"policy.json")])
                self.assertEqual(Bandit.load(out/"policy.json").episodes, 3)
                invoke(args+["--episodes", "4", "--resume", str(out/"policy.json"), "--max-tokens", "100"], 2)
                self.assertEqual(Bandit.load(out/"policy.json").episodes, 3)
                legacy = Bandit.load(out/"policy.json")
                legacy.config["intervention_version"] = 1
                legacy.update("s", "USE_CALCULATOR", -0.81)
                legacy.update("s", "FIX_COEFF", 0.89)
                legacy.save(root/"legacy.json")
                updated_args = ["train", "--offline", "--data", str(root/"data/train.jsonl"),
                                "--output", str(root/"corrected"), "--episodes", "4",
                                "--warm-start", str(root/"legacy.json")]
                invoke(updated_args)
                corrected = Bandit.load(root/"corrected/policy.json")
                self.assertEqual(corrected.episodes, 4)
                self.assertEqual(corrected.values["s"][3], 0)
                self.assertEqual(corrected.values["s"][2], 0.89)
                self.assertEqual(corrected.provenance["source_episodes"], 3)

    def test_evaluation_rejects_training_split(self):
        with tempfile.TemporaryDirectory() as directory:
            generate(directory, train=15)
            with self.assertRaises(ValueError):
                load(Path(directory)/"train.jsonl", "test")

    def test_partial_report_uses_only_common_rows(self):
        row = dict(problem="a", seed=10, level=0, score=1, first_solved=False,
                   solved=True, http_calls=2, calculator_requests=0, elapsed=1, synthetic=True)
        payload = dict(version=1, status="capped", synthetic=True, expected=8, config={"request_seeds": False},
                       policy_provenance=dict(source_episodes=16, reset_observations=2, retained_observations=12),
                       rows=[dict(row, mode=mode) for mode in ("fixed", "random", "rule", "learned")]+[dict(row, problem="b", mode="fixed")])
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory)/"evaluation.json", Path(directory)/"report.md"
            source.write_text(json.dumps(payload))
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as exc:
                main(["report", str(source), "--output", str(output)])
            self.assertEqual(exc.exception.code, 0)
            stats = json.loads(output.with_suffix(".json").read_text())
            self.assertEqual(stats["common_problem_seed_pairs"], 1)
            self.assertEqual(stats["excluded_runs"], 1)
            self.assertTrue(stats["incomplete"])
            self.assertIn("SYNTHETIC", output.read_text())
            self.assertIn("warm-started from 16", output.read_text())
