"""Local, advisory Decider evaluation. Never executes candidates or changes agent scores."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import shutil
import signal
import socket
import statistics
import subprocess
import tarfile
import time

import httpx

HERE = Path(__file__).resolve().parent
MODELS = HERE / "models"
RUNS = HERE / "runs"
MODEL = MODELS / "decider-4b.v2-Q4_K_M.gguf"
REVISION = "2796fac5cdad018ef6d9a4a004735ff819f424a2"
SHA256 = "f7e2e510ef51d212ea9b7fb8bf27906b5f516d7939ca847428fb91f6a8acfa79"
MODEL_BYTES = 2708804544
URL = f"https://huggingface.co/mindchain/decider-4b-v2-GGUF/resolve/{REVISION}/{MODEL.name}"
TEMPERATURE = 1.935  # upstream v2 config; quantized/NKI-domain confidence is NOT calibrated
BASE = "http://127.0.0.1:8096"
KEY = RUNS / "api-key"
STATE = RUNS / "server.json"
OPTIONS = {
    "repair_layout": "Correct tensor shapes, transpose orientation, slicing or tiling.",
    "repair_api": "Replace an invalid API call or fix its arguments using the installed API card.",
    "repair_memory": "Correct SBUF, PSUM or HBM placement and transfers between them.",
    "reduce_traffic": "Reuse loaded input tiles and remove redundant HBM reads or writes.",
    "change_approach": "Change the repair approach or use a verified building block after repeated failures.",
    "minimal_edit": "Make a small targeted correction using the checker's existing feedback.",
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_model():
    if MODEL.is_symlink() or not MODEL.is_file():
        raise ValueError("Model is missing (or a symlink); run download first.")
    if MODEL.stat().st_size != MODEL_BYTES or sha256(MODEL) != SHA256:
        raise ValueError("Model integrity check failed; refusing to start it.")


def download():
    MODELS.mkdir(exist_ok=True)
    if MODEL.exists():
        verify_model()
        print("Existing model passes size and SHA-256 verification.")
        return
    if shutil.disk_usage(MODELS).free < MODEL_BYTES + 512 * 1024 * 1024:
        raise ValueError("Need at least 3.2 GB of free disk space.")
    partial = MODEL.with_suffix(".partial")
    digest, count, next_report = hashlib.sha256(), 0, 256 * 1024 * 1024
    # Exclusive creation preserves any earlier interrupted download for inspection.
    with partial.open("xb") as output:
        with httpx.Client(follow_redirects=True, trust_env=False, timeout=60) as client:
            with client.stream("GET", URL) as response:
                response.raise_for_status()
                for chunk in response.iter_bytes(1024 * 1024):
                    count += len(chunk)
                    if count > MODEL_BYTES:
                        raise ValueError("Download exceeds pinned model size.")
                    output.write(chunk)
                    digest.update(chunk)
                    if count >= next_report:
                        print(f"Downloaded {count / MODEL_BYTES:.0%}", flush=True)
                        next_report += 256 * 1024 * 1024
    if count != MODEL_BYTES or digest.hexdigest() != SHA256:
        raise ValueError("Download does not match pinned size/hash; partial file is not usable.")
    partial.rename(MODEL)
    print(f"Verified {count:,} bytes. SHA-256: {SHA256}")


def server_environment():
    # Do not inherit AWS/HF credentials, proxy settings, or LLAMA_ARG_TOOLS/AGENT overrides.
    return {k: v for k, v in os.environ.items() if k in ("PATH", "HOME", "TMPDIR", "LANG")}


def start():
    verify_model()
    binary = shutil.which("llama-server")
    if not binary:
        raise ValueError("Install llama.cpp with Homebrew first.")
    with socket.socket() as sock:
        if sock.connect_ex(("127.0.0.1", 8096)) == 0:
            raise ValueError("Port 8096 is already in use. Use status; existing service was not changed.")
    RUNS.mkdir(mode=0o700, exist_ok=True)
    if not KEY.exists():
        fd = os.open(KEY, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as stream:
            stream.write(secrets.token_hex(32))
    elif KEY.is_symlink() or KEY.stat().st_mode & 0o077:
        raise ValueError("API key must be a private regular file (mode 0600).")
    argv = [binary, "-m", str(MODEL), "--alias", "decider-local", "--host", "127.0.0.1",
            "--port", "8096", "-c", "2048", "-np", "1", "-ngl", "99", "--offline",
            "--no-webui", "--no-agent", "--no-ui-mcp-proxy", "--no-jinja",
            "--api-key-file", str(KEY), "--cache-ram", "0"]
    with (RUNS / "server.log").open("ab") as output:
        process = subprocess.Popen(argv, stdout=output, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, start_new_session=True,
                                   cwd=HERE, env=server_environment())
    STATE.write_text(json.dumps({"pid": process.pid, "argv": argv, "revision": REVISION,
                                 "model_sha256": SHA256}, indent=2) + "\n")
    print(f"Started PID {process.pid}; loading model. Run status. Log: {RUNS / 'server.log'}")


def stop():
    state = json.loads(STATE.read_text())
    pid = int(state["pid"])
    if pid <= 1:
        raise ValueError("Invalid stored PID.")
    result = subprocess.run(["ps", "-p", str(pid), "-o", "command="],
                            capture_output=True, text=True, check=False)
    if result.returncode or not result.stdout.strip():
        print("Local Decider server is already stopped.")
        return
    if str(MODEL) not in result.stdout or "--port 8096" not in result.stdout:
        raise ValueError("Stored PID belongs to another process; refusing to stop it.")
    os.kill(pid, signal.SIGTERM)
    print(f"Sent SIGTERM to local Decider server PID {pid}.")


def client():
    # Fixed loopback endpoint: prompts never go to a hosted inference provider.
    return httpx.Client(base_url=BASE, timeout=30, trust_env=False,
                        headers={"Authorization": f"Bearer {KEY.read_text().strip()}"})


def post(api, path, body):
    response = api.post(path, json=body)
    response.raise_for_status()
    return response.json()


def build_prompt(state):
    options = "".join(f"\n({chr(65 + i)}) {text}" for i, text in enumerate(OPTIONS.values()))
    return ("Context:\n" + json.dumps(state, ensure_ascii=False, sort_keys=True)
            + "\n\nQuestion: Which repair strategy best addresses this NKI kernel's failure?"
            + "\nOptions:" + options + "\nAnswer: (")


def option_probabilities(payload):
    if not isinstance(payload, dict):
        raise ValueError("Runtime response must be a JSON object.")
    if payload.get("truncated") or payload.get("tokens_predicted") != 1:
        raise ValueError("Expected one token at an untruncated decision slot.")
    positions = payload.get("completion_probabilities", [])
    if not isinstance(positions, list) or len(positions) != 1 or not isinstance(positions[0], dict):
        raise ValueError("Runtime did not return decision-slot probabilities.")
    rows = positions[0].get("top_logprobs")
    if not isinstance(rows, list) or not 1 <= len(rows) <= 512:
        raise ValueError("Invalid option log probability list.")
    logits = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Invalid option log probability entry.")
        token = row.get("token")
        if isinstance(token, str) and len(token) == 1 and token in "ABCDEF":
            value = float(row["logprob"])
            if not math.isfinite(value) or value > 0 or token in logits:
                raise ValueError("Invalid or duplicate option log probability.")
            logits[token] = value / TEMPERATURE
    if set(logits) != set("ABCDEF"):
        raise ValueError("Not all six option logits returned; refusing partial confidence.")
    peak = max(logits.values())
    weights = {k: math.exp(v - peak) for k, v in logits.items()}
    total = sum(weights.values())
    return {name: weights[chr(65 + i)] / total for i, name in enumerate(OPTIONS)}


def baseline(state):
    feedback = str(state.get("feedback", "")).lower()
    if any(x in feedback for x in ("has no attribute", "unexpected keyword", "not callable")):
        return "repair_api"
    if any(x in feedback for x in ("wrong memory", "must be in", "cannot touch psum")):
        return "repair_memory"
    if any(x in feedback for x in ("hbm traffic", "more than necessary")):
        return "reduce_traffic"
    if any(x in feedback for x in ("shape", "reshape", "partition", "contraction", "transpose")):
        return "repair_layout"
    if state.get("echo") or state.get("repeated_failure"):
        return "change_approach"
    return "minimal_edit"


def bounded_state(record):
    if not isinstance(record, dict):
        raise ValueError("State must be a JSON object.")
    return {"level": record.get("level"), "feedback": str(record.get("feedback", ""))[:1400],
            "code_excerpt": str(record.get("code", ""))[:2500],
            "echo": bool(record.get("echo")), "repeated_failure": bool(record.get("repeated_failure"))}


def decide(state, threshold=0.6):
    fallback = baseline(state)
    started = time.perf_counter()
    result = {"strategy": fallback, "baseline": fallback, "source": "fallback",
              "advisory_only": True, "domain_calibrated": False, "model_revision": REVISION,
              "model_sha256": SHA256}
    try:
        prompt = build_prompt(state)
        with client() as api:
            tokens = post(api, "/tokenize", {"content": prompt, "add_special": False})["tokens"]
            if len(tokens) > 1980:
                raise ValueError("Decision prompt exceeds local context budget.")
            # Verify that option letters each form one token at this exact answer prefix.
            for letter in "ABCDEF":
                combined = post(api, "/tokenize", {"content": prompt + letter, "add_special": False})["tokens"]
                if combined[:-1] != tokens or len(combined) != len(tokens) + 1:
                    raise ValueError("Option letter is not a single token at the decision slot.")
            payload = post(api, "/completion", {"prompt": tokens, "n_predict": 1, "n_probs": 512,
                           "temperature": -1, "post_sampling_probs": False, "cache_prompt": False})
        probabilities = option_probabilities(payload)
        suggestion = max(probabilities, key=probabilities.get)
        confidence = probabilities[suggestion]
        result.update(suggestion=suggestion, confidence=confidence, probabilities=probabilities,
                      prompt_tokens=len(tokens), timings=payload.get("timings"))
        if confidence >= threshold:
            result.update(strategy=suggestion, source="decider")
        else:
            result["reason"] = "Below threshold; retaining deterministic baseline."
    except (httpx.HTTPError, OSError, ValueError, KeyError, TypeError) as error:
        result["reason"] = f"{type(error).__name__}: {error}"
    result["seconds"] = round(time.perf_counter() - started, 4)
    return result


def records(path):
    # Never extract archives or execute their recorded code. Bound archive/member/line sizes.
    def lines(stream):
        while line := stream.readline(1024 * 1024 + 1):
            if len(line) > 1024 * 1024:
                raise ValueError("Log line exceeds 1 MiB.")
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("Attempt log row must be an object.")
                yield value
    if str(path).endswith(".tar.gz"):
        with tarfile.open(path, "r:gz") as archive:
            size = 0
            for member in archive:
                if member.isfile() and member.name.endswith(".jsonl"):
                    size += member.size
                    if member.size > 64 * 1024 * 1024 or size > 512 * 1024 * 1024:
                        raise ValueError("Archive exceeds evaluation size limit.")
                    with archive.extractfile(member) as stream:
                        yield from lines(stream)
    else:
        with path.open("rb") as stream:
            yield from lines(stream)


def replay(path, limit, threshold):
    RUNS.mkdir(mode=0o700, exist_ok=True)
    output = RUNS / f"replay-{time.time_ns()}.jsonl"
    seen, results = set(), []
    with output.open("x") as log:
        for row in records(path):
            if float(row.get("reward", 0)) >= 0.999 or row.get("selected") is False:
                continue
            state = bounded_state(row)
            fingerprint = hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            result = decide(state, threshold)
            result.update(state_sha256=fingerprint, level=row.get("level"))
            log.write(json.dumps(result) + "\n")
            results.append(result)
            print(f"{len(results)}: level {row.get('level')} -> {result['strategy']} "
                  f"({result['source']}, {result['seconds']:.3f}s)", flush=True)
            if len(results) >= limit:
                break
    summary = {"decisions": len(results), "unique_states": len(seen),
               "fallbacks": sum(r["source"] == "fallback" for r in results),
               "different_from_baseline": sum(r["strategy"] != r["baseline"] for r in results),
               "median_seconds": statistics.median(r["seconds"] for r in results) if results else None,
               "speedup_measured": False, "decision_accuracy_measured": False, "output": str(output)}
    output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("download", "start", "stop", "status"):
        sub.add_parser(command)
    decision = sub.add_parser("decide")
    decision.add_argument("--feedback", required=True)
    decision.add_argument("--level", type=int, default=8)
    decision.add_argument("--threshold", type=float, default=0.6)
    replay_parser = sub.add_parser("replay")
    replay_parser.add_argument("log", type=Path)
    replay_parser.add_argument("--limit", type=int, default=16)
    replay_parser.add_argument("--threshold", type=float, default=0.6)
    args = parser.parse_args()
    if hasattr(args, "threshold") and not 0 <= args.threshold <= 1:
        parser.error("threshold must be between 0 and 1")
    if hasattr(args, "limit") and not 1 <= args.limit <= 10000:
        parser.error("limit must be between 1 and 10000")
    if args.command == "download":
        download()
    elif args.command == "start":
        start()
    elif args.command == "stop":
        stop()
    elif args.command == "status":
        with client() as api:
            response = api.get("/health")
            response.raise_for_status()
            print(json.dumps(response.json()))
    elif args.command == "decide":
        print(json.dumps(decide(bounded_state(vars(args)), args.threshold), indent=2))
    else:
        replay(args.log, args.limit, args.threshold)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, httpx.HTTPError) as error:
        raise SystemExit(f"Local Decider: {error}")
