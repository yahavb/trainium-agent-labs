# Team 18 submission evidence

The current one-page note is `../ONE_PAGE_NOTE.md` (printable PDF alongside).
`../ATTEMPT_LOG.md` preserves selected and historical experiment summaries.

- `attempt2/`: original candidate/checker/feedback records and summaries from
  commit cd16f7b, original-checker acceptance, Level 1 3/3 in 228.24 seconds.
- `enhanced-checker/`: corresponding original records from commit 31a150d,
  optional validation.grade acceptance, Level 1 3/3 in 187.00 seconds. The
  external adapter independently records actual public original scores.
  Offline failures/reproductions are retained, including the pre-core timeout
  defect repaired in commit 118d5f1. The 187-second model run predates that fix.

Earlier selected baseline/Attempt 1 records were supplied as summaries, not
verified original traces; their evidence gaps remain explicit in the detailed log.
These files do not fabricate historical feedback or establish repeated-run
reliability. The latest enhanced-checker run is additional evidence, not a
replacement for Attempt 2. Public checker weights and tolerances were unchanged.
