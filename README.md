# gpt-oss-20b on AWS Trainium — student kit

A 20-billion-parameter open-weight model (`openai/gpt-oss-20b`) running on AWS Trainium2, behind an
OpenAI-compatible HTTP API. This repo is the shortest path from "I have the URL" to "I am talking to
it and measuring it."

Four files, no framework, no build step:

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
python chat.py --think                      # show the hidden reasoning channel
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

**Ask for at least ~2500 `max_tokens`. This is the number one trap.** gpt-oss writes to a hidden
`reasoning` channel *before* it writes any answer, and if it runs out of budget while thinking you
get **empty `content`** with `finish_reason="length"` — which looks exactly like the model failing
your task. It isn't. You never gave it room to answer.

Measured on this endpoint:

| task | `max_tokens` | result |
|---|---|---|
| "what is 17 × 23?" | 16, 32 | empty content, `finish=length` |
| "what is 17 × 23?" | 64+ | `391` |
| needle in a 4600-token document | 32 | empty content |
| needle in a 4600-token document | 800 | correct, used only 59 tokens |
| write `merge_intervals` + 6 asserts | 900 | **empty content**, 3791 chars of reasoning |
| write `merge_intervals` + 6 asserts | 2500 | correct, used 1442 tokens |

Easy questions need ~64. Hard ones burned through 900 and still had nothing to show. Budget
generously — you're charged for what it uses, not what you allow.

**Note that `Reasoning: low` in the system prompt does not help.** We measured it: on the coding
task it produced *more* reasoning (7080 chars vs 4525), not less. Don't rely on it to control the
budget.

**8192 tokens, checked against your INPUT — and the output is then silently truncated.** This is not
the usual "prompt + completion" ceiling. Send 7435 tokens of prompt with `max_tokens=2000` and the
request is *accepted*; you get 757 tokens of output, `total_tokens=8192`, and
`finish_reason="length"` with no error. So a long prompt doesn't fail loudly — it quietly amputates
your answer. Check `finish_reason` on every response you care about. Past ~8192 input you do get a
clean 400: `Input length (8607) exceeds model's maximum context length (8192)`.

**Long prompts are cheap, and the cost comes in steps rather than proportionally.** Measured
time-to-first-token, 5 reps per size in randomised order:

| prompt | `/agg` median TTFT | `/disagg` median TTFT |
|---|---|---|
| ~200 tok | 1.062s | 0.853s |
| ~2000 tok | 1.064s | 0.857s |
| ~4000 tok | 1.218s | 1.027s |
| ~6000 tok | 1.226s | 1.029s |
| ~7500 tok | 1.224s | 1.041s |

A **37× longer prompt costs about 15–22% more** time to first token. And notice the shape: flat from
200→2000, a jump between 2000 and 4000, then flat again to 7500. That's compiled shape buckets —
within a bucket the prefill is padded to the bucket size, so extra tokens are genuinely free. Two
consequences:

* **Context is cheap. Use it.** Examples, retrieved documents, and history cost you very little
  latency.
* Trim history to stay under the input ceiling and to leave room for the answer — **not** to go
  faster. Trimming for speed buys almost nothing. (`chat.py`'s budget exists for the former reason.)

Measure this yourself in randomised order if you re-run it. Sweeping 200→7500 in sequence confounds
prompt length with warm-up and with whatever else is hitting this shared server, and produces a
tidy rising line that is partly an artifact — we made exactly that mistake first.

**No prefix caching**, so there's no discount for a repeated prefix across turns. Given the padding
above this matters less than you'd expect for latency, but the server does full work every turn,
which is part of why throughput is what it is.

**Sampling is greedy, server-side. `temperature`, `top_p`, and `seed` are ignored.** Identical input
gives identical output, always. Good for reproducibility. But: no sampling diversity, no
self-consistency voting, no "retry with higher temperature" — those patterns silently do nothing
here. `probe.py` check 7 demonstrates this.

**Four concurrent sequences, and throughput tops out around 2 requests/sec.** The server runs
`max_num_seqs=4`. Measured on `/agg`: 0.43 req/s at 1 concurrent request, 1.08 at 4, 2.10 at 16 —
and p50 time-to-first-token degrades from 0.59s to 3.46s as you pile on. Nothing errors; it queues.
Plan around ~4 in flight, and understand that **one endpoint comfortably serves a handful of
simultaneous users, not a roomful.**

**gpt-oss thinks before it answers, on a separate channel.** In this vLLM build the field is called
**`reasoning`** (not `reasoning_content`, which is what other builds and most docs use — the kit
accepts both). It's present on both the non-streaming `message` and the streaming `delta`. See the
`max_tokens` warning above: this channel is what eats your budget.

**The standard `tools=` parameter does not work. If you're building an agent, read this.** Passing
OpenAI-style function definitions with `tool_choice="auto"` returns `tool_calls: []` *and* empty
content — it silently does nothing. You have to hand-roll tool calls as JSON in the prompt.

And when you do, **the phrasing decides whether you get anything back at all**:

```
"...Emit the call."                    → content: ''   (nothing!)
"...Your FINAL ANSWER must be exactly one JSON object... Do not explain."
                                       → content: '{"tool":"get_time","args":{"tz":"Asia/Tokyo"}}'
```

Both `finish_reason: "stop"`. In the first case the model settled the whole thing inside its
reasoning channel and ended its turn without writing an answer. Explicitly demand a **final answer**
in your prompt. This one behaviour will cost an agent-building team hours if they don't know it.

**`/disagg` omits `usage` on streamed responses; `/agg` includes it.** If you're computing tokens/sec
from a stream, you'll silently fall back to guessing on one endpoint and not the other — which makes
the two look different for a reason that has nothing to do with the hardware. `chat.py` prints a `~`
in front of any number it had to estimate. Do something equivalent, or compare non-streaming.

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

There is already a measured difference, and it goes in **opposite directions** for the two things you
care about:

| | `/agg` (TP32, one server) | `/disagg` (prefill + decode, TP16 each) |
|---|---|---|
| TTFT, short prompt | 1.06s | **0.85s** — faster to first token |
| generation rate | **114 tok/s** | 83–89 tok/s — slower to finish |
| peak throughput | 1.9 req/s @ n=8 | 2.2 req/s @ n=16 |

So `/disagg` starts answering sooner but generates more slowly. Which one "wins" depends entirely on
what you're building: a chat UI that feels responsive wants low TTFT, a batch job that emits long
documents wants tokens/sec. Say which you optimised for and why — that reasoning is worth more than
the number.

The comparison gets more interesting **under concurrency**, where prefill and decode no longer
contend for the same devices in the disaggregated setup. `probe.py` check 9 runs both at n=1…16.

If you find a result we haven't, tell us. That's genuinely useful to the team that built this.

## Measuring, not guessing

```bash
python probe.py                 # both endpoints, full suite, writes probe-results.json
python probe.py --quick         # skip the slow sweeps
python probe.py --path /agg/v1  # one endpoint
```

It answers, with numbers:

1. Is the endpoint reachable at all?
2. Which API surfaces work, does streaming work, and what is the reasoning field called?
2b. How much `max_tokens` before you get an answer instead of only reasoning?
3. The real context ceiling, whether it applies to input or input+output, and how it fails.
4. Can the model find a fact planted at various depths in a ~5000-token document?
5. Can it hold a JSON schema across a 40-record generation, and at what tokens/sec?
6. Five harder tasks with auto-checkable answers (multi-constraint code, stateful probability,
   instruction-following under pressure, a refactor with a trap, tool-call formatting).
6b. Tool calling: the native `tools=` parameter, and two hand-rolled prompt phrasings.
7. Is sampling really greedy?
8. Does prompt length change time-to-first-token? (Randomised order, repeated.)
9. Where does throughput stop rising as you add concurrency?

The string-based graders in check 6 are crude. **Read the failures before you believe them** — that
habit is worth more than the score, and it is exactly how the two worst bugs in this very script were
found. Check 6 prints `<NO CONTENT — only reasoning>` when the model never produced an answer, so you
can tell "got it wrong" apart from "was never given room to reply."

Every number in this README came out of this script. If you disagree with one, re-measure it and tell
us — that's the point.

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
