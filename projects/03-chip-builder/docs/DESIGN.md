# AccelTwin design and implementation contract

## Product statement

AccelTwin is a chip-builder AI: it proposes complete accelerator component
graphs and floorplans for one fixed inference stack, then submits each proposed
chip to a deterministic checker. It answers a bounded counterfactual question:
which connected physical arrangement and resource mix most reduces p99
time-to-first-token for the same model, software, and request trace?

This is an architectural exploration prototype. It does not replace RTL,
physical design, timing closure, power sign-off, or thermal validation.

## Demo output

The user sees the actual Trainium2-like baseline component graph, live or mocked
serving observations, an animated thermal model, a causal TTFT waterfall, and up
to five agent-proposed chip graphs. Each proposal is the full floorplan, not a
scalar patch. Every attempt includes:

- the complete structured hardware design;
- components with positions, dimensions, layers, and resource values, plus
  connections with endpoint IDs and bandwidth;
- a causal graph diff showing moved/added/removed blocks and changed links;
- the agent rationale;
- p50, p95, and p99 TTFT;
- throughput, memory use, power proxy, temperature grid, throttling, and area
  proxy;
- every hard-gate result and actionable checker feedback;
- an append-only record in SQLite and JSONL.

The final output is the lowest-p99 accepted design plus a Markdown report from
`GET /api/reports/{campaign_id}`.

## System boundary

The inference software is held fixed. AccelTwin does not tune vLLM scheduling,
quantization, kernels, prompts, or model weights during a campaign. The search
space contains only these typed fields:

- logical compute cores and aggregate BF16 compute;
- HBM capacity and bandwidth;
- on-chip SBUF capacity;
- DMA engine count;
- power and cooling limits;
- model and KV-memory assumptions used by the checker.
- compute, HBM, SBUF, DMA, and NoC placement and interconnect paths.

The baseline mirrors the hackathon-visible device: four logical NeuronCores,
96 GB of HBM, and 2.9 TB/s HBM bandwidth. This is a simplified public
architecture model, not a representation of proprietary floorplan details.

## Workloads

The simulator ships three deterministic traces:

| Trace | Purpose | Shape |
| --- | --- | --- |
| A | Calibration smoke test | 12 short interactive requests |
| B | Calibration stress test | 8 long-context requests |
| C | Held-out score trace | 48 closely spaced burst requests |

A campaign always evaluates the unchanged baseline and every candidate on the
same selected trace. This prevents invalid comparisons across different request
mixes.

## Digital-twin model

Each request is replayed through a deterministic FCFS queue. Prefill and decode
have separate compute and HBM roofline estimates. Service time uses the slower
roof, then applies a throttling factor when estimated device power exceeds the
smaller of the power and cooling limits.

Memory fit includes weights, the largest request KV allocation, and a bounded
work-buffer allowance. The power estimator is a calibrated activity proxy. The
8x8 temperature field applies a stable center-weighted spatial gradient. The
area proxy is a disclosed linear combination of package overhead, cores, HBM,
SBUF, and DMA engines. These equations are intentionally fast and explainable;
they are not silicon sign-off models.

The simulator target is under 250 ms per candidate. In normal operation it is
substantially faster, so the local LLM is the dominant campaign latency.

## Optimization contract

Primary objective: minimize modeled p99 TTFT on the chosen evaluation trace.

Hard gates:

1. candidate throughput is at least the same-trace baseline;
2. weights, KV cache, and reserve fit in HBM;
3. estimated power is below the configured cap;
4. modeled peak junction temperature is below the configured cap;
5. modeled area proxy is below the configured cap;
6. values pass the `HardwareDesign` schema and every component/link topology
   check, including floorplan bounds, same-layer overlap, resource-total
   consistency, valid connection endpoints, and compute-to-HBM reachability.

The checker—not the LLM—owns acceptance. The agent proposes exactly one complete
design graph per round. The graph and causal diff are included in proposal,
evaluation, and best-candidate events so the renderer can show the model's actual
chip. Checker feedback carries exact topology violations, disconnected IDs, and
resource, thermal, and bandwidth bottlenecks into the next prompt. Responses
must be strict JSON. Invalid output gets one repair attempt;
duplicate designs are rejected without another simulation. The campaign stops
after at most five rounds, when a target is achieved, or after the configured
number of stale rounds.

## Evidence and calibration

`/api/baselines/run` attempts to scrape the local vLLM Prometheus endpoint and
allows an optional Neuron monitor adapter. Raw observations are retained as
`measured_telemetry`. Counterfactual performance, power, temperature, and area
remain labeled modeled. When telemetry is unavailable, the deterministic
baseline keeps the demo operational and the API reports `measurement_source` as
`modeled`.

Prometheus latency histograms should eventually be converted to repeated-trial
quantiles during a longer calibration session. The current one-day build retains
raw series rather than implying that a single scrape is a statistically valid
p99 measurement.

## Runtime architecture

```text
Browser dashboard
  │ REST + server-sent events
  ▼
FastAPI ── CampaignOrchestrator ── OpenAI-compatible vLLM on Trainium
  │                 │
  │                 └─ strict proposal / repair / duplicate controls
  ├─ DigitalTwinSimulator
  ├─ vLLM Prometheus + optional Neuron monitor adapter
  └─ SQLite runs + append-only JSONL audit log
```

The static dashboard has no build step. FastAPI serves it directly. This reduces
demo failure modes and makes the application easy to copy into the assigned pod.

## API

| Method and path | Behavior |
| --- | --- |
| `GET /api/health` | readiness check |
| `GET /api/state` | latest UI snapshot |
| `POST /api/baselines/run` | evaluate and observe a baseline |
| `GET /api/baselines/{id}` | retrieve a baseline |
| `POST /api/calibrations` | attach observed calibration data |
| `POST /api/campaigns` | start a bounded background campaign |
| `GET /api/campaigns/{id}` | retrieve campaign and attempts |
| `GET /api/designs/{attempt_id}` | retrieve a persisted attempt |
| `GET /api/events/{campaign_id}` | stream campaign events with SSE |
| `GET /api/reports/{campaign_id}` | render the decision report |

## Failure handling

- Unavailable vLLM metrics: continue with a clearly modeled baseline.
- Unavailable proposer endpoint in auto mode: use the deterministic bounded
  chip-graph proposer so the demo completes with topology-aware edits.
- Invalid proposal: perform one repair; strict mode fails the campaign if repair
  also fails.
- Candidate simulation failure: record feedback and continue to the next round.
- Process restart: completed simulator runs remain in SQLite/JSONL; active
  in-memory campaigns are not resumed in this prototype.

## Verification

The automated suite covers deterministic replay, percentile ordering, thermal
grid shape, memory overflow, power throttling, persistence, telemetry parsing,
baseline/API retrieval, hard gates, duplicate rejection, one-repair behavior,
SSE completion, and report generation. A complete mock campaign is also run as
an end-to-end smoke test before demo handoff.

## Deliberate follow-ons

- fit coefficients from repeated A/B measurements rather than static defaults;
- ingest Neuron Monitor output through a concrete installed adapter;
- model concurrent batching rather than serial FCFS service;
- persist campaign state for restart/resume;
- add multi-objective Pareto views after the single-metric judging path is solid;
- replace the area and thermal proxies with validated domain models if a hardware
  team supplies them.

