# gpt-oss-20b on AWS Trainium — student kit

A 20-billion-parameter open-weight model (`openai/gpt-oss-20b`) running on AWS Trainium2, behind an
OpenAI-compatible HTTP API. This repo is the shortest path from "I have the URL" to "I am talking to
it and measuring it."

Three files, no framework, no build step:

| file | what it is |
|---|---|
| `chat.py` | terminal chat client — streaming, history, timings. Start here. |
| `web.py` | the same thing in a browser tab, for demos. Zero deps beyond `httpx`. |
| `probe.py` | measures what the endpoint can actually do: real context limit, long-document recall, format adherence under long generation, concurrency ceiling, determinism. |
| `mockserver.py` | a fake endpoint so you can build while offline or while the real one is busy. |

## Setup

```bash
git clone <this repo> && cd nyu-gptoss-kit
pip install httpx
export GPTOSS_BASE_URL="https://..."      # the organisers will give you this
python chat.py
```

That's it. No API key, no auth, no SDK.

```bash
python chat.py                              # interactive
python chat.py --ask "explain a B-tree"     # one shot
python chat.py --endpoint disagg            # the other deployment (see below)
python chat.py --effort high --think        # more reasoning, and show it
python web.py                               # browser UI on :8080
python probe.py                             # measure the endpoint
```

In-chat commands: `/agg` `/disagg` `/new` `/system <text>` `/effort low|medium|high` `/think`
`/budget <n>` `/stats` `/save <file>` `/quit`.

## Talking to it directly

Two endpoints, same model and same weights, differing only in how inference is spread across the
accelerator:

| | base URL | shape |
|---|---|---|
| Aggregated | `$GPTOSS_BASE_URL/agg/v1` | one vLLM server, 8 Trainium devices, tensor-parallel 32 |
| Disaggregated | `$GPTOSS_BASE_URL/disagg/v1` | prefill and decode in separate pods (TP16 each), KV cache moved between them over EFA, router in front |

Model name is exactly `gpt-oss-20b` — not `openai/gpt-oss-20b`.

```bash
curl -sk "$GPTOSS_BASE_URL/agg/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"gpt-oss-20b","messages":[{"role":"user","content":"hi"}],"max_tokens":64}'
```

```python
from openai import OpenAI
import httpx

client = OpenAI(base_url=f"{BASE}/agg/v1", api_key="EMPTY",
                http_client=httpx.Client(verify=False))   # see the TLS note below
print(client.chat.completions.create(
    model="gpt-oss-20b",
    messages=[{"role": "user", "content": "hi"}],
    max_tokens=64).choices[0].message.content)
```

`system_fingerprint` on the response tells you which backend answered: `...tp32...` is aggregated,
`...tp16...` is disaggregated.

---

## Read this before you build. These will bite you.

These are real properties of how the servers are configured, not bugs, and several will look like
model problems if you don't know about them.

**8192 tokens of context, total — prompt *and* completion together.** Asking for 2000 output tokens
leaves you 6000 for everything else. Long conversations, accumulated tool output, and big pasted
documents hit this fast. Over the limit, the request is rejected outright.

**No prefix caching.** Disabled on purpose. Every turn re-processes the entire conversation from
scratch — a long system prompt costs full price on every single message. So the cost of a chat grows
roughly with the *square* of its length. This is the single biggest cost factor in a chat loop, and
it's why `chat.py` trims history to a budget instead of sending everything. Watch `in` climb in the
per-turn stats line.

**Sampling is greedy, server-side. `temperature`, `top_p`, and `seed` are ignored.** Identical input
gives identical output, always. Good for reproducibility. But: no sampling diversity, no
self-consistency voting, no "retry with higher temperature" — those patterns silently do nothing
here. `probe.py` check 7 demonstrates this.

**Four concurrent sequences.** The server is built for `max_num_seqs=4`. A fan-out of 20 parallel
calls doesn't scale — it queues, and you see latency rather than errors. Plan for ~4 in flight per
endpoint. `probe.py` check 9 finds the actual ceiling.

**gpt-oss thinks before it answers, on a separate channel.** Responses can carry a
`reasoning_content` field alongside `content`. On a hard question at `--effort high`, the model can spend
its whole token budget reasoning and return an *empty answer*. If that happens, use `/effort low` or
narrow the question. Both clients handle and surface this.

**The TLS certificate does not match the hostname.** Use `verify=False` in Python or `-k` in curl.
Every script here already does.

**The endpoint is network-restricted.** It only accepts connections from allowlisted networks. From
anywhere else you get a TCP timeout with *no HTTP response at all* — `curl` hangs and returns
nothing. If that happens, it is not the server: ask the organisers to allowlist the address you're
coming from. This is by far the most common "it's broken" report.

**Request timeout is 900s**, so long generations are safe.

**It's shared demo capacity.** If numbers look strange, someone else may be hitting it at the same
time. Check before drawing conclusions.

---

## Which endpoint should I use?

Start with `/agg` — fewer moving parts. Then run the same workload against `/disagg` and compare.

The interesting comparison is **under concurrency**. In the disaggregated deployment, prefill and
decode don't contend for the same devices, so time-to-first-token and tokens/sec should respond
differently as you add load. One request at a time will not show you much. `probe.py` runs both and
prints them side by side.

If you measure a real difference between them, tell us — that comparison is genuinely useful and it's
the kind of result that wins here.

## Measuring, not guessing

```bash
python probe.py                 # both endpoints, full suite, writes probe-results.json
python probe.py --quick         # skip the slow sweeps
python probe.py --path /agg/v1  # one endpoint
```

It answers, with numbers:

1. Is the endpoint reachable at all?
2. Which API surfaces work, and does streaming work?
3. What's the real usable context, and what does exceeding it look like?
4. Can the model find a fact planted at various depths in a ~5000-token document?
5. Can it hold a JSON schema across a 40-record generation, and at what tokens/sec?
6. Five harder tasks with auto-checkable answers (multi-constraint code, stateful probability,
   instruction-following under pressure, a refactor with a trap, tool-call formatting).
7. Is sampling really greedy?
8. Does a repeated long prefix get any cheaper? (It should not.)
9. Where does throughput stop rising as you add concurrency?

The string-based graders in check 6 are crude. **Read the failures before you believe them** — that
habit is worth more than the score.

## Working offline, or while the endpoint is busy

The real endpoint serves **four concurrent sequences for everyone**, and it's network-restricted. So
don't sit waiting on it while you write UI code:

```bash
python mockserver.py                                       # terminal 1
GPTOSS_BASE_URL=http://localhost:8000 python chat.py       # terminal 2
```

The mock imitates the things that break clients — SSE streaming, the separate reasoning channel,
`usage`, the 8192-token rejection, the 4-sequence queue, greedy determinism, and both `/agg` and
`/disagg` path prefixes. It does **not** imitate the model: replies are canned. Build against it,
then switch `GPTOSS_BASE_URL` to the real endpoint to measure anything.

Never report a number you got from the mock.

## Ideas that fit this hardware

Things that work well within 8192 tokens, 4 concurrent sequences, and greedy decoding:

* A focused code assistant — the context limit is fine for a few files
* Structured extraction from documents, with the schema enforced and *measured*
* A short agent loop (3–5 steps) with a tight token budget — and instrument every call
* Anything where determinism is a feature: reproducible evaluation, regression testing, graders
* A head-to-head of `/agg` vs `/disagg` under realistic load

Things that will fight you: long-context work, big parallel fan-out, sampling-diversity tricks
(self-consistency, best-of-n), and chatty agents that accumulate context without trimming.

## License / use

Sample code, provided as-is for the event, free to reuse. The model is
[openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) under its own license.
