# Comparison reporting and dashboard integration

Run these commands from `projects/03-chipboost` in the integration checkout. Keep the collected comparison output in a dedicated directory; do not mix previous runs or copy infrastructure events into attempt logs.

## Provisional inspection

```powershell
$runDir = '<copied comparison directory>'
python tools/summarize_comparison.py $runDir --allow-partial --json-out reports/comparison-partial.json --markdown-out reports/comparison-partial.md
```

Partial reports are operational snapshots, not evidence for ranking the arms. Their aggregates include only started runs, some of which may be unfinished. They do not fill missing runs with 1x. The actual producer contracts are P3 attempt numbers 1–8 and P2 attempt numbers 0–7.

## Final report gate

Refresh all comparison files, including `state.json`, `acceptance.json`, `infrastructure.jsonl`, and the nine arm JSONL files. Then omit `--allow-partial`:

```powershell
python tools/summarize_comparison.py $runDir --json-out reports/comparison.json --markdown-out reports/comparison.md
if ($LASTEXITCODE -ne 0) { throw 'Comparison report validation failed' }
```

The strict report requires the passed acceptance gate, a completed runner state, pinned commits, nine distinct runs, and eight schema-valid attempts per run. It checks attempt numbering, timing ratios, and state/log consistency. If copied state claims more completed runs than the copied logs contain, refresh the collection instead of editing state. Final report counts must be 72 attempts, 24 per arm, and three runs per arm. Infrastructure records never enter these counts or kernel failure rates.

Run the report tests with `python -m unittest discover -s tools -p test_summarize_comparison.py -v`.

## Build from explicit files

After the report passes, build the dashboard with exactly those comparison logs and the separately verified adversarial result file:

```powershell
$logs = @(Get-ChildItem -LiteralPath $runDir -Filter '*-r*.jsonl' | Sort-Object Name | ForEach-Object FullName)
$adversarial = '<verified adversarial results JSON>'
python dashboard/build.py @logs --results $adversarial --sweep --out dashboard/index.html
if ($LASTEXITCODE -ne 0) { throw 'Dashboard build failed' }
python tools/check_dashboard.py dashboard/index.html reports/comparison.json
```

Explicit logs are necessary because automatic discovery only finds `attempts*.jsonl`. Empty `--sweep` disables discovery of historical sweep data. Explicit `--results` prevents silently adding stale or unrelated result files. Do not include `acceptance.json`, `infrastructure.jsonl`, `state.json`, or the generated comparison report as attempt/results inputs. The adversarial result adapter must provide the dashboard's supported `redteam` or `rows` list; infrastructure failures and unexecuted cases must remain clearly distinguished from cheat verdicts. Keep adversarial suite version/provenance alongside the report.

`check_dashboard.py` checks the embedded attempt count, one timeline mark per attempt, required interaction elements, duplicate IDs, fake-data banners, and external resource dependencies. These are static checks, not visual or interaction QA. Browser preview was blocked by the URL security policy in this session; no alternate browser/server workaround was used. A partial dashboard may be built for static inspection with the same explicit inputs, but it must be distributed with its PARTIAL report and must not be presented as the final comparison.

## Interpretation and version caveats

Each run's best speedup includes only `faster` verdicts, falling back to 1x if none improve. Each arm's summary reports the median and range across runs. Rules/correctness/held-out failures are separate from slower/no-gain outcomes. Correctness on timing shapes does not mean every candidate reached the held-out stage.

P2 searches caps over an expert template; the agent arms start from the reference baseline. Thus all three arms do not share the same candidate prior.

This batch stays pinned to P1 `434e5f9`, P2 `919c6be`, and P3 `2ce9416`. It predates the subsequent representative-feedback fix. The final report must retain that caveat: these results do not measure the corrected feedback implementation. Record the fix commit when available, and keep any later corrected-feedback experiment separate instead of relabeling or combining this batch's records.
