# The shared `gpt-oss-20b` endpoint

A 20-billion-parameter open-weight model (`openai/gpt-oss-20b`) running on AWS Trainium2, behind an
OpenAI-compatible HTTP API, on hardware **separate from your own instance**. This folder is the
shortest path from "I have the URL" to "I am talking to it and measuring it."

Use it when your own chip is busy running your own code — see the device-sharing note in the
[top-level README](../README.md) — or just to talk to a model in your first five minutes.

Everything below was measured against the real endpoint. No framework, no build step:

| file | what it is |
|---|---|
| `chat.py` | terminal chat client — streaming, history, timings. Start here. |
| `web.py` | the same thing in a browser tab, for demos. Zero deps beyond `httpx`. |
| `probe.py` | measures what the endpoint can actually do: real context limit, long-document recall, format adherence under long generation, concurrency ceiling, determinism. |
| `loadtest.py` | closed-loop load test: how many simultaneous users this actually supports. |
| `mockserver.py` | a fake endpoint so you can build while offline or while the real one is busy. |

The kernel-writing challenge that used to live here moved to
[`../projects/02-kernel-agent/`](../projects/02-kernel-agent/).

## Setup

```bash
git clone <this repo> && cd trainium-agent-labs
pip install httpx
export GPTOSS_BASE_URL="https://..."      # the organisers will give you this
python gptoss/chat.py
```

That's it. No API key, no auth, no SDK.

```bash
python gptoss/chat.py                              # interactive; uses BOTH endpoints, with failover
python gptoss/chat.py --ask "explain a B-tree"     # one shot
python gptoss/chat.py --endpoint agg               # pin to one deployment (see below)
python gptoss/chat.py --think                      # show the hidden reasoning channel
python web.py                               # browser UI on :8080
python probe.py                             # measure what the endpoint can do
python loadtest.py                          # how many people can share it
python kernelbench.py --list                # the kernel-agent challenge ladder
python kernelbench.py --selftest            # prove the harness catches planted bugs
```

In-chat commands: `/agg` `/disagg` `/auto` `/new` `/system <text>` `/effort low|medium|high`
`/think` `/budget <n>` `/stats` `/save <file>` `/quit`.

### One URL, two endpoints, automatic failover

`GPTOSS_BASE_URL` is a single base URL, and the two deployments are **paths underneath it** —
`/agg/v1` and `/disagg/v1`. So one environment variable gives you both; you do not need two.

`chat.py` and `web.py` default to **auto** mode, which:

* **round-robins across both endpoints**, starting at a random one so that a room full of clients
  spreads out instead of everyone landing on the same deployment;
* **fails over** to the other endpoint if one times out, refuses the connection, or returns a 5xx —
  you get a `(retrying on ...)` line and an answer instead of a stack trace;
* **does not fail over on a 4xx**, because a bad request will be rejected identically by both;
* **does not fail over once tokens have started arriving** — retrying mid-stream would splice two
  half-answers together;
* marks a failed endpoint unhealthy for 60s so every subsequent turn does not pay the timeout again.

Since the two endpoints are independent hardware, auto mode is also how you get the full ~4 req/s
rather than half of it. Pin with `--endpoint agg` or `/disagg` when you are deliberately comparing
the two — `/stats` then breaks timings down per endpoint.

This matters more than it sounds: these are single-replica demo deployments, so a pod restart takes
one of them away for minutes at a time. Auto mode means that is a hiccup rather than the end of your
afternoon.

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

**But more budget does not always rescue you, and this is the sharp edge.** On a code-generation task
with an enumerated list of rules, the model audited itself against each rule in its reasoning channel
and never began the answer:

| `max_tokens` | reasoning produced | answer produced |
|---|---|---|
| 2500 | 10,194 chars | **0** |
| 7000 | 26,401 chars | **0** |

Nearly 3× the budget bought 2.6× the thinking and still no answer.

**The mechanism is greedy decoding, and it matters.** Sampling here is greedy, so there is no
randomness to break a repeating state. Tell the model what *not* to do and it starts verifying itself;
the verification text becomes self-similar; and it locks into a groove it cannot escape. Caught in the
act:

```
We used loops over columns for max. Good.
We used loops over columns for sum. Good.
We used loops over columns for max. Good.
We used loops over columns for sum. Good.
```

...until the budget died. **You cannot outspend an infinite loop**, which is why 7000 tokens failed the
same way 2500 did.

So: if you get empty content, **shorten the prompt and remove prohibitions — do not raise
`max_tokens`.** The same task asked in one sentence returned working code in 617 tokens. And prefer
phrasing that names one change to make over a list of things to avoid.

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

**Four concurrent sequences per endpoint. Two endpoints. ~4 req/s in total.** Each server runs
`max_num_seqs=4`, so beyond four in flight your request queues rather than failing.

The two endpoints are **fully independent capacity** — separate Trainium devices, and measured at
104% scaling efficiency when hammered simultaneously, with no degradation from sharing the load
balancer. So use both: point half your team at `/agg` and half at `/disagg` and you get double.

Measured with both endpoints loaded at once:

| in flight (total) | `/agg` | `/disagg` | **combined** | p50 latency | p95 latency | p50 TTFT | failures |
|---|---|---|---|---|---|---|---|
| 8 | 1.06 | 0.99 | **1.97 req/s** | 3.7s | 3.7s | 1.2s | 0 |
| 16 | 1.26 | 1.63 | **2.53 req/s** | 4.1s | 4.9s | 1.6s | 0 |
| 32 | 1.63 | 2.18 | **3.27 req/s** | 5.3s | 7.6s | 2.9s | 0 |
| 64 | 1.91 | 2.57 | **3.82 req/s** | 8.1s | 14.6s | 5.9s | 0 |
| 96 | 1.98 | 2.65 | **3.97 req/s** | 11.1s | 21.5s | 9.0s | 0 |

Two things to take from this. **Throughput saturates near 4 req/s** — past ~32 in flight you're
buying latency, not work. And **nothing ever failed**, even at 96 concurrent: it degrades by queueing,
so your code will see slow responses rather than errors. Set a client timeout accordingly (the server
allows 900s).

Also note `/disagg` overtakes `/agg` under load (2.65 vs 1.98 req/s at saturation) despite being
slower on a single request. That's the disaggregation paying off — prefill and decode stop contending
— and it only shows up when you push it.

**Budget your request count, not just your tokens.** A 5-step agent loop is 5 serialised requests; at
p50 5s each that's 25s per iteration before you've done anything clever. Parallelise across the two
endpoints where you can.

### How many people can share this at once?

The burst table above measures raw throughput. What you probably want to know is how many *humans*
it supports, which is a different question, because a person spends most of their time reading and
typing rather than waiting. Measured with a closed-loop simulation — each simulated student asks a
question, waits for the whole answer, then pauses to "read and type" before asking again:

| load | throughput | p50 | p95 | errors |
|---|---|---|---|---|
| 100 students, 40s between messages | 1.54 req/s | 3.5s | **4.7s** | 0 |
| 100 students, 20s between messages | 3.23 req/s | 3.8s | **7.9s** | 0 |
| 100 students, 8s between messages | 4.43 req/s | 8.6s | 17.6s | 0 |
| 150 students, 15s between messages | 4.28 req/s | 10.2s | 22.1s | 0 |
| 200 students, 20s between messages | 4.19 req/s | 13.1s | 26.4s | 0 |
| **25 agent loops, no human pauses** | **5.17 req/s** | 4.0s | 8.5s | 0 |

**Rule of thumb: capacity is ~4.3 req/s, so people supported ≈ 4.3 × (think time + latency).** At a
realistic 20s between messages that's about 100 people with p95 under 8s. Past ~4.3 req/s of demand
nothing fails — latency just inflates as the queue grows. One error in roughly 3,000 requests across
every run here, and that was a timeout, not a rejection.

Three numbers people conflate, all true at once:

* **8 requests are actually being served** at any instant (4 per endpoint × 2 endpoints).
* **~16–25 in flight** is the efficient zone. Past ~32 you buy latency, not work.
* **100–200 humans** can have the app open, because they're mostly not waiting on it.

**One agent loop costs about the same as 4–5 students.** Twenty-five agent loops saturate the whole
thing — and that's one call at a time each; a 5-step agent iteration multiplies it again. If you're
building something agentic, assume you are the load.

### What would change these numbers

Read the above as a starting point, not a specification. Everything here was measured:

* **from a single client machine**, over the load balancer, from one network location. A different
  client, or many clients spread out, will not behave identically.
* **with 220-token answers.** This is the biggest factor you control, and it hits *latency* harder
  than throughput. Measured, same 12 users and same think time, only `max_tokens` changed:

  | `max_tokens` | throughput | p50 | p95 |
  |---|---|---|---|
  | 220 | 0.66 req/s | 3.3s | 3.5s |
  | 1000 | 0.42 req/s | 10.4s | **12.2s** |

  4.5× the answer length cost 3.5× the p95 wait but only 36% of the request rate. So if your users
  care about waiting, cap your answer length — it buys more than any other single change.
* **with short prompts and varied questions.** Identical prompts at high concurrency actually measured
  *worse* than varied ones (4.0 vs 5.2 req/s), so synthetic benchmarks that repeat one prompt will
  mislead you in both directions.
* **on shared demo capacity**, with no guarantee nobody else was using it. If a number looks wrong,
  it might be someone else's traffic.
* **at one moment in time.** The server config can change; a restart with different flags moves all
  of this.

So: **re-measure with your own workload before you trust any of it.** `probe.py` check 9 does the
burst version and check 10 does both endpoints at once. Your generation length and your request
pattern are what decide your numbers, not ours.

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
| TTFT, short prompt, unloaded | 1.06s | **0.85s** — faster to first token |
| generation rate, unloaded | **114 tok/s** | 83–89 tok/s — slower to finish |
| throughput at saturation | 1.98 req/s | **2.65 req/s** — better under load |

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

## Documents bigger than 8192 tokens

You will hit this immediately. A single design document is often well past the limit — the one that
prompted this note was 46 KB, about 11,600 tokens, so it cannot go in at all.

The model also **cannot fetch URLs**. Ask it to read a link and it will tell you so. There is no
browsing tool, and the standard `tools=` parameter doesn't work here (see above), so don't plan
around it.

What to do instead, in increasing order of effort:

1. **Condense it elsewhere first.** Paste the document into ChatGPT or Claude, get it down to a few
   thousand tokens, bring that in. Completely legitimate, costs nothing, and is the right answer under
   time pressure.
2. **Chunk it yourself.** Split into ~4000-token pieces, extract from each, then answer from the
   extracts. Note this costs 4+ requests per question instead of 1 — against the ~4 req/s shared
   ceiling that is roughly what 4–5 chat users cost. And **retrying a bad extract is pointless**:
   sampling is greedy, so you get a byte-identical result. Fix the prompt, never the dice.
3. **Retrieve rather than summarise.** Index the document, pull only the passages a question needs.
   More work, and the honest win: it scales past one document.

Whichever you pick, **hand in the condensed or chunked text as an artifact.** Two reasons. Your result
isn't reproducible without it. And if a frontier model produced the text your system reasons over,
then your reported number is partly measuring *that* model rather than yours — a judge who can't
separate them can't credit either.

The interesting part isn't the mechanics, it's **what you chose to throw away.** A team that can
defend how it got 46 KB down to 6 K has done real work on the actual constraint.

## License / use

Sample code, provided as-is for the event, free to reuse. The model is
[openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) under its own license.
