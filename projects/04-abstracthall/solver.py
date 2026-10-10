#!/usr/bin/env python3
"""Text-only model client. No grading, training, or repair policy lives here."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
import uuid

from conditions import CONDITIONS, build_prompt


@dataclass(frozen=True)
class Arm:
    base_url: str
    model: str = "Qwen/Qwen3-8B"
    temperature: float = 0.6
    max_tokens: int = 300
    timeout: float = 900
    # The workshop Qwen server supports this extension. Other servers may not.
    qwen_template: bool = True
    input_limit: int = 8100


@dataclass
class Reply:
    text: str
    raw_content: str
    finish_reason: str | None
    latency_s: float
    prompt_chars: int
    status: str
    error: str | None
    usage: dict
    input_tokens: int | None = None


def completion_url(base_url):
    base_url = base_url.strip().rstrip("/")
    parsed = urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Provide an HTTP(S) API base URL, e.g. http://localhost:8000/v1")
    if parsed.query or parsed.fragment or parsed.username or parsed.password:
        raise ValueError("The API base URL must not contain credentials, query parameters or a fragment")
    return base_url + "/chat/completions"


def clean_content(content):
    # Preserve raw content separately. Never mistake an unfinished reasoning block for a summary.
    text = re.sub(r"<think>.*?</think>", "", content, flags=re.S)
    if "<think>" in text:
        text = text.split("<think>", 1)[0]
    return text.strip()


def ask_once(prompt, arm):
    start = time.perf_counter()
    body = {
        "model": arm.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": arm.temperature,
        "top_p": 0.95 if arm.temperature else 1.0,
        "max_tokens": arm.max_tokens,
    }
    if arm.qwen_template:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    request = Request(completion_url(arm.base_url), data=json.dumps(body).encode(),
                      headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(request, timeout=arm.timeout) as response:
            payload = json.load(response)
        choice = payload["choices"][0]
        raw = choice["message"].get("content") or ""
        if not isinstance(raw, str):
            raise ValueError("Expected text in message.content")
        text = clean_content(raw)
        finish = choice.get("finish_reason")
        status = ("truncated" if finish == "length" else
                  "empty" if not text else
                  "complete" if finish == "stop" else "incomplete")
        return Reply(text, raw, finish, round(time.perf_counter()-start, 3),
                     len(prompt), status, None, payload.get("usage") or {})
    except HTTPError as exc:
        error = f"HTTP {exc.code}: {exc.reason}"
    except (URLError, TimeoutError, OSError, ValueError, KeyError, IndexError, TypeError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    return Reply("", "", None, round(time.perf_counter()-start, 3), len(prompt),
                 "error", error, {})


def count_input_tokens(prompt, arm):
    """Use the serving tokenizer, including the same chat template as generation."""
    base = arm.base_url.strip().rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    body = dict(model=arm.model, messages=[{"role": "user", "content": prompt}],
                add_generation_prompt=True)
    if arm.qwen_template:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    request = Request(base + "/tokenize", data=json.dumps(body).encode(),
                      headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(request, timeout=arm.timeout) as response:
        count = json.load(response)["count"]
    if type(count) is not int or count < 0:
        raise ValueError("Tokenizer returned an invalid token count")
    return count


def solve(prompt: str, arm: Arm, n: int = 1) -> list[Reply]:
    """Stateless boundary for future conditions and checker-driven repair prompts."""
    completion_url(arm.base_url)
    if not prompt.strip():
        raise ValueError("Prompt must not be empty")
    if not 1 <= n <= 4:
        raise ValueError("Use 1 to 4 concurrent samples for the workshop server")
    if arm.max_tokens <= 0 or arm.timeout <= 0 or arm.temperature < 0 or arm.input_limit <= 0:
        raise ValueError("Invalid generation settings")
    start = time.perf_counter()
    try:
        count = count_input_tokens(prompt, arm)
    except (URLError, TimeoutError, OSError, ValueError, KeyError, TypeError) as exc:
        return [Reply("", "", None, round(time.perf_counter()-start, 3), len(prompt),
                      "error", f"Input token check failed ({type(exc).__name__}); no generation sent. "
                      "Check that this server exposes /tokenize.", {}) for _ in range(n)]
    if count > arm.input_limit:
        return [Reply("", "", None, 0, len(prompt), "input_too_long",
                      f"Input has {count} tokens; limit is {arm.input_limit}. Shorten the prompt.",
                      {}, count) for _ in range(n)]
    with ThreadPoolExecutor(max_workers=n) as pool:
        replies = list(pool.map(lambda _: ask_once(prompt, arm), range(n)))
    for reply in replies:
        reply.input_tokens = count
    return replies


def load_items(path):
    items = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
             if line.strip()]
    ids = set()
    for item in items:
        if not isinstance(item.get("id"), str) or not item["id"] or item["id"] in ids:
            raise ValueError("Each abstract needs a unique nonempty string id")
        if not isinstance(item.get("text"), str) or not item["text"].strip():
            raise ValueError(f"Missing abstract text: {item['id']}")
        ids.add(item["id"])
    if not items:
        raise ValueError("Dataset is empty")
    return items


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=root / "data/abstracts.jsonl")
    parser.add_argument("--id", help="Run only this abstract ID, e.g. yeast_01")
    parser.add_argument("--samples", type=int, choices=range(1, 5), default=1)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--condition", choices=[*CONDITIONS, "all"], default="baseline",
                        help="baseline = plain summary; key2 = repeat 2 key sentences; "
                             "random2 = repeat 2 random non-key sentences; all = run all three")
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--input-limit", type=int, default=8100,
                        help="Input-only budget; independent of --max-tokens")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--base", default=(os.getenv("CR_BASE_URL") or
                        os.getenv("KERNEL_AGENT_BASE_URL") or os.getenv("HEATROD_BASE_URL")))
    parser.add_argument("--model", default=(os.getenv("CR_MODEL") or
                        os.getenv("KERNEL_AGENT_MODEL") or "Qwen/Qwen3-8B"))
    parser.add_argument("--no-qwen-template", action="store_true")
    parser.add_argument("--out", type=Path, help="New JSONL file; existing files are never overwritten")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and show prompts, without model calls")
    args = parser.parse_args()
    if args.repeat < 1 or args.max_tokens < 1 or args.timeout <= 0 or args.temperature < 0 or args.input_limit < 1:
        parser.error("repeat, max-tokens and timeout must be positive; temperature must be nonnegative")
    items = load_items(args.data)
    if args.id:
        items = [item for item in items if item["id"] == args.id]
        if not items:
            parser.error("That abstract ID is not in the dataset")
    conditions = list(CONDITIONS) if args.condition == "all" else [args.condition]
    if args.dry_run:
        # Build every prompt now, so a bad abstract (e.g. too few sentences) fails before any model call.
        try:
            for item in items:
                for cond in conditions:
                    for repeat in range(args.repeat):
                        build_prompt(cond, item["id"], item["text"], repeat + 1)
        except ValueError as exc:
            parser.error(str(exc))
        print(f"Validated {len(items)} real abstracts; conditions: {', '.join(conditions)}; "
              f"planned generations: {len(items)*len(conditions)*args.repeat*args.samples}")
        for item in items:
            for cond in conditions:
                built = build_prompt(cond, item["id"], item["text"], 1)
                print(f"\n[{item['id']} | {cond}] repeated sentences: {built['repeated_indices']} "
                      f"({built['context_chars']} extra chars)\n{built['prompt']}")
        return 0
    if not args.base:
        parser.error("No API URL: run inside your pod or pass --base http://localhost:8000/v1")
    arm = Arm(args.base, args.model, args.temperature, args.max_tokens, args.timeout,
              not args.no_qwen_template, args.input_limit)
    completion_url(arm.base_url)
    run_id = uuid.uuid4().hex
    output = args.out or root / "runs" / f"{args.condition}-{run_id}.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    statuses = {}
    print(f"Output: {output}", flush=True)
    print(f"Server: {arm.base_url}  model: {arm.model}  conditions: {', '.join(conditions)}", flush=True)
    with output.open("x", encoding="utf-8") as log:
        for repeat in range(args.repeat):
            for item in items:
                # All conditions for one abstract run back to back, with identical settings.
                for cond in conditions:
                    built = build_prompt(cond, item["id"], item["text"], repeat + 1)
                    prompt = built["prompt"]
                    replies = solve(prompt, arm, args.samples)
                    for sample, reply in enumerate(replies):
                        record = dict(run_id=run_id, condition=cond, round=0,
                                      repeat=repeat+1, sample=sample+1, item_id=item["id"],
                                      source=item.get("source"),
                                      source_sha256=hashlib.sha256(item["text"].encode()).hexdigest(),
                                      prompt=prompt, settings=asdict(arm),
                                      sentence_count=built["sentence_count"],
                                      key_indices=built["key_indices"],
                                      repeated_indices=built["repeated_indices"],
                                      repeated_sentences=built["repeated_sentences"],
                                      context_chars=built["context_chars"], seed=built["seed"],
                                      recorded_at=datetime.now(timezone.utc).isoformat(),
                                      **asdict(reply))
                        log.write(json.dumps(record, ensure_ascii=False)+"\n")
                        statuses[reply.status] = statuses.get(reply.status, 0)+1
                        print(f"{item['id']} {cond} repeat={repeat+1} sample={sample+1}: "
                              f"{reply.status}, {reply.latency_s}s", flush=True)
                        if reply.error:
                            print(reply.error, flush=True)
                    log.flush()
                    if all(reply.status == "error" for reply in replies):
                        print("All requests in this batch failed. Stopping; inspect the saved errors.")
                        return 1
    print(f"Finished: {statuses}. Completion status is not a correctness score.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())