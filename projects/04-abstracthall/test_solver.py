"""Offline protocol tests. Mock replies are test fixtures, not experimental data."""
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
import solver


def response(content="Test response", finish="stop"):
    return io.BytesIO(json.dumps({"choices": [{"message": {"content": content},
                                              "finish_reason": finish}],
                                 "usage": {"completion_tokens": 3}}).encode())


class SolverTests(unittest.TestCase):
    def setUp(self):
        self.arm = solver.Arm("http://localhost:8000/v1")
        self.counter_patch = patch.object(solver, "count_input_tokens", return_value=42)
        self.counter = self.counter_patch.start()
        self.addCleanup(self.counter_patch.stop)

    def test_input_boundary_independent_of_output(self):
        self.counter.return_value = 8100
        with patch.object(solver, "urlopen", side_effect=lambda *a, **kw: response()) as send:
            reply = solver.solve("prompt", self.arm)[0]
            self.assertEqual(reply.input_tokens, 8100)
            self.assertEqual(reply.status, "complete")
            self.assertEqual(json.loads(send.call_args.args[0].data)["max_tokens"], 300)
        self.counter.return_value = 8101
        with patch.object(solver, "urlopen") as send:
            replies = solver.solve("prompt", self.arm, 4)
            send.assert_not_called()
        self.assertTrue(all(r.status == "input_too_long" for r in replies))

    def test_real_tokenize_protocol_and_missing_endpoint(self):
        self.counter_patch.stop()
        with patch.object(solver, "urlopen", return_value=io.BytesIO(b'{"count": 123}')) as send:
            self.assertEqual(solver.count_input_tokens("prompt", self.arm), 123)
        request = send.call_args.args[0]
        self.assertEqual(request.full_url, "http://localhost:8000/tokenize")
        body = json.loads(request.data)
        self.assertTrue(body["add_generation_prompt"])
        self.assertFalse(body["chat_template_kwargs"]["enable_thinking"])
        with patch.object(solver, "urlopen", side_effect=HTTPError(request.full_url, 404, "Missing", {}, None)) as send:
            reply = solver.solve("prompt", self.arm)[0]
        self.assertEqual(send.call_count, 1)
        self.assertEqual(reply.status, "error")

    def test_request_and_ordered_batch(self):
        with patch.object(solver, "urlopen", side_effect=lambda *a, **kw: response()) as send:
            replies = solver.solve("Summarise: source text", self.arm, 4)
        self.assertEqual(len(replies), 4)
        self.assertTrue(all(r.status == "complete" for r in replies))
        for call in send.call_args_list:
            request = call.args[0]
            body = json.loads(request.data)
            self.assertEqual(request.full_url, "http://localhost:8000/v1/chat/completions")
            self.assertEqual(body["messages"], [{"role": "user", "content": "Summarise: source text"}])
            self.assertFalse(body["chat_template_kwargs"]["enable_thinking"])
            self.assertEqual(body["max_tokens"], 300)

    def test_finish_statuses_and_reasoning(self):
        for content, finish, status, text in [
            ("<think>hidden</think>Summary", "stop", "complete", "Summary"),
            ("<think>unfinished", "length", "truncated", ""),
            ("", "stop", "empty", ""),
            ("partial", "length", "truncated", "partial"),
            ("unknown ending", None, "incomplete", "unknown ending"),
        ]:
            with self.subTest(finish=finish, content=content):
                with patch.object(solver, "urlopen", return_value=response(content, finish)):
                    reply = solver.solve("prompt", self.arm)[0]
                self.assertEqual((reply.status, reply.text), (status, text))
                self.assertEqual(reply.raw_content, content)

    def test_http_error_is_recorded(self):
        with patch.object(solver, "urlopen", side_effect=HTTPError(self.arm.base_url, 503, "Unavailable", {}, None)):
            reply = solver.solve("prompt", self.arm)[0]
        self.assertEqual(reply.status, "error")
        self.assertIn("503", reply.error)

    def test_bad_payload_is_recorded(self):
        with patch.object(solver, "urlopen", return_value=io.BytesIO(b'{"choices": []}')):
            reply = solver.solve("prompt", self.arm)[0]
        self.assertEqual(reply.status, "error")

    def test_real_dataset(self):
        items = solver.load_items(Path(__file__).parent / "data/abstracts.jsonl")
        self.assertEqual(len(items), 8)
        self.assertTrue(all(item["source"].startswith("https://arxiv.org/abs/") for item in items))

    def test_cli_logs_source_and_rejects_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "run.jsonl"
            args = ["solver.py", "--base", self.arm.base_url, "--id", "yeast_01", "--out", str(output)]
            with patch("sys.argv", args), patch.object(solver, "urlopen", side_effect=lambda *a, **kw: response()):
                self.assertEqual(solver.main(), 0)
                with self.assertRaises(FileExistsError):
                    solver.main()
            record = json.loads(output.read_text())
            self.assertEqual(record["item_id"], "yeast_01")
            self.assertEqual(record["condition"], "baseline")
            self.assertEqual(record["status"], "complete")
            self.assertEqual(len(record["source_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
