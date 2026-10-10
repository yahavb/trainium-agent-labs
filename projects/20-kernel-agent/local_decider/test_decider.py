"""Safety and probability-readout regression tests; no model/network required."""
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx
import numpy as np
import decider
import e2e_check


def payload():
    return {"tokens_predicted": 1, "truncated": False,
            "completion_probabilities": [{"top_logprobs": [
                {"token": letter, "logprob": -i - 1.0} for i, letter in enumerate("ABCDEF")]}]}


class DeciderTests(unittest.TestCase):
    def test_normalizes_only_valid_options(self):
        data = payload()
        data["completion_probabilities"][0]["top_logprobs"].append(
            {"token": "shell_command", "logprob": 0})
        probs = decider.option_probabilities(data)
        self.assertEqual(set(probs), set(decider.OPTIONS))
        self.assertAlmostEqual(sum(probs.values()), 1)
        self.assertEqual(max(probs, key=probs.get), "repair_layout")
        self.assertAlmostEqual(probs['repair_layout'] / probs['repair_api'],
                               math.exp(1 / decider.TEMPERATURE))

    def test_missing_option_is_refused(self):
        data = payload()
        data["completion_probabilities"][0]["top_logprobs"].pop()
        with self.assertRaises(ValueError):
            decider.option_probabilities(data)

    def test_malformed_runtime_response_is_refused(self):
        for data in ([], {"tokens_predicted": 1, "completion_probabilities": ["invalid"]},
                     {"tokens_predicted": 1, "completion_probabilities": [{"top_logprobs": [None]}]}):
            with self.assertRaises(ValueError):
                decider.option_probabilities(data)

    def test_nonfinite_and_duplicate_logits_are_refused(self):
        for value in (float("nan"), float("inf")):
            data = payload()
            data["completion_probabilities"][0]["top_logprobs"][0]["logprob"] = value
            with self.assertRaises(ValueError):
                decider.option_probabilities(data)
        data = payload()
        data["completion_probabilities"][0]["top_logprobs"].append({"token": "A", "logprob": -1})
        with self.assertRaises(ValueError):
            decider.option_probabilities(data)

    def test_truncated_and_multi_token_responses_are_refused(self):
        for change in ({"truncated": True}, {"tokens_predicted": 2}):
            data = payload() | change
            with self.assertRaises(ValueError):
                decider.option_probabilities(data)

    def test_unavailable_service_keeps_existing_rule(self):
        with patch.object(decider, "client", side_effect=httpx.ConnectError("offline")):
            result = decider.decide({"feedback": "module has no attribute dot"})
        self.assertEqual(result["strategy"], "repair_api")
        self.assertEqual(result["source"], "fallback")
        self.assertTrue(result["advisory_only"])

    def test_low_confidence_uses_rule_not_model(self):
        class FakeAPI:
            def __enter__(self): return self
            def __exit__(self, *args): return False

        def fake_post(api, path, body):
            if path == "/tokenize":
                # The test prefix ends with '('; appended letters are one extra token.
                return {"tokens": [1] if body["content"].endswith("(") else [1, 2]}
            return payload()

        with patch.object(decider, "client", return_value=FakeAPI()), patch.object(decider, "post", side_effect=fake_post):
            result = decider.decide({"feedback": "has no attribute"}, threshold=1)
        self.assertEqual(result["suggestion"], "repair_layout")
        self.assertEqual(result["strategy"], "repair_api")
        self.assertEqual(result["source"], "fallback")

    def test_no_credentials_or_tools_in_server_environment(self):
        with patch.dict(decider.os.environ, {"AWS_SECRET_ACCESS_KEY": "secret",
                        "HF_TOKEN": "secret", "LLAMA_ARG_AGENT": "1",
                        "LLAMA_ARG_TOOLS": "all", "HTTPS_PROXY": "proxy"}):
            env = decider.server_environment()
        self.assertFalse(any(k.startswith(("AWS", "HF", "LLAMA")) for k in env))
        self.assertNotIn("HTTPS_PROXY", env)

    def test_bad_model_is_refused_before_process_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "fake.gguf"
            model.write_bytes(b"not a model")
            with patch.object(decider, "MODEL", model), patch.object(decider.subprocess, "Popen") as popen:
                with self.assertRaises(ValueError): decider.start()
                popen.assert_not_called()

    def test_recorded_code_is_data(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "attempts.jsonl"
            log.write_text(json.dumps({"code": "raise RuntimeError('never execute')"}) + "\n")
            self.assertEqual(list(decider.records(log))[0]["code"], "raise RuntimeError('never execute')")

    def test_state_size_is_bounded(self):
        state = decider.bounded_state({"feedback": "x" * 100000, "code": "y" * 100000})
        self.assertEqual(len(state["feedback"]), 1400)
        self.assertEqual(len(state["code_excerpt"]), 2500)
        self.assertNotIn("AWS_SESSION_TOKEN", state)


class E2EFixtureTests(unittest.TestCase):
    def test_numerical_checker_rejects_bad_shape_values_and_nan(self):
        x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        reference = lambda a: a.copy()
        for candidate in (lambda a: a[:, :1], lambda a: a + 1,
                          lambda a: np.full_like(a, np.nan)):
            self.assertFalse(e2e_check.checked(candidate, (x,), reference)[0])

    def test_trusted_softmax_repair_matches_independent_reference(self):
        x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        reference = lambda a: np.exp(a) / np.exp(a).sum(axis=1, keepdims=True)
        self.assertTrue(e2e_check.checked(e2e_check.repaired_softmax, (x,), reference)[0])
        self.assertFalse(e2e_check.checked(e2e_check.bad_api, (x,), reference)[0])

    def test_require_explicitly_rejects_failure(self):
        with self.assertRaisesRegex(AssertionError, "deliberate failure"):
            e2e_check.require(False, "deliberate failure")


if __name__ == "__main__":
    unittest.main()
