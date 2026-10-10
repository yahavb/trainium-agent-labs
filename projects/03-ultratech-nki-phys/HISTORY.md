# Attempt History: All Available Local Records

Start with [ALL_ATTEMPTS.jsonl](ALL_ATTEMPTS.jsonl): an index of recorded log rows and standalone generation records, preserving their original score/status fields inside `record`. Each entry names its original file and line. Missing scores remain missing or `null`; they are never replaced with invented failures or successes.

The original logs and saved `results.json` reports are copied byte-for-byte under [history/](history/). [history/manifest.json](history/manifest.json) lists source-relative paths and hashes. This covers the available local contact/update/batch, gripping, generation, adaptive-reference, plane-loop, math-agent and benchmark records, including failed/rejected attempts and recovery logs.

**These entries are not a count of unique model proposals.** An execution, individual case grade, certification, generation and controller summary can describe different stages of the same attempt. Some recovery/seat-export logs also overlap. Preserve those records for audit rather than counting every row as another independent run. `record_type` distinguishes original JSONL rows from standalone generation records; stage/file names identify their context. Performance reports without an attempt-log row remain available under `history/`, not assigned fabricated proposal IDs.

The clearest complete experiment views are:

- [Final math-agent four-proposal log](ATTEMPTS.jsonl), with full requests, replies, sources, outputs and timings under `development/`.
- [Gripping six-trial log](gripping-evidence/ATTEMPTS.jsonl), with all per-case diagnostics and available candidates under `gripping-evidence/`.
- Earlier controller logs under `history/plane-loop-*`, `history/full-plane-loop-*` and `history/grip-loop-*`.

## Coverage Limits

This archive contains every matching attempt log, standalone generation record and saved `results.json` report found in the local physics-project data directory at publication. Duplicate generated `submission-math-*` packages are excluded; their original run records are retained. Raw arrays, candidates and model exchanges for every early exploration are not all duplicated into this history archive; the final experiment's full evidence remains separately bundled.

We cannot claim this is every run ever executed on the team seats. Remote-only records not copied locally and the broader engine's raw attempt history were not available for collection. That track is described in our engine PDF; provide its raw logs to complete team-wide coverage. Do not treat this archive as reconstructing missing records or supplying independent scores for the broader prototype.
