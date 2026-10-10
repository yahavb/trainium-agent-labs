# PLAN: project 2 (NKI kernel agent), how the day was run

This is the planner's file. During the day it held the schedule, the owners, the acceptance criteria and the
decisions, in the team's working language; this is its English rewrite. The experiment numbers themselves live in
[NOTES.md](NOTES.md) §0 and in [SUBMISSION.md](SUBMISSION.md). §3 is the part that matters most to a reader: the
rules for keeping or dropping a change, written down before the results they were applied to.

## 1. Who did what

| who | owned | did not touch |
|---|---|---|
| planner (a Claude Code session) | this file, choosing experiments, keep-or-roll-back decisions, SUBMISSION.md | code, seats |
| executor 1 (a Claude Code session) | the only session on seats 115–119: deploys, runs, re-audits, NOTES §0 numbers, code for experiments | the analysis files |
| executor 2 (a Claude Code session) | everything that runs on a laptop: held-out set, token accounting, confidence and calibration, taxonomy, reports, reproduction dry run | seats |
| teoguo | decisions, coordination with liuyq, credentials, the final call on every open question | |
| liuyq | feedback layers v2–v7, the compiler gate and its level-1 rule, levels 9–14, chip runs on seat 115 | |

## 2. The timeline as it ended up

| time | what |
|---|---|
| 10:50–12:43 | baseline: the organizers' agent, `--all --rounds 8 --samples 4 --context 8192 --repeat 5` on seat 116 |
| 13:50 | this plan, first draft |
| 14:07–14:50 | round 2: v7 on levels 1, 3, 4 (seats 117–119), E-F on level 1 (seat 116) |
| 15:05–15:50 | leave-one-out ablation of v7 on levels 3 and 4 |
| 15:15–16:00 | v8 quick tests (skeleton, truncation fix, level-1 call fixes, E-div) |
| 16:09–16:40 | v8.2 (adds E-mix), five runs per level |
| 16:45–17:45 | v8.3 (level-2 error distillation) on level 2 |
| 17:10–18:05 | v8.4 (WARM) on levels 5–7 |
| 18:05 | no new runs |
| 18:05–18:50 | logs pulled and re-audited, tag `final`, final tables, SUBMISSION filled in, credential scan, fresh-clone check |
| 18:50 | pull request (liuyq) |

The original plan froze code at 17:00 and submitted at 18:15. teoguo moved both twice, at 15:55 and 16:00, to
give levels 1 and 2 more time, and the organizers' 18:30 deadline turned out to be soft by up to 30 minutes.
Freezing then became "no new runs after 18:05": every run made on the final commit, or on a commit proven to send
byte-identical requests for that level, counts.

## 3. Experiment protocol and decision rules (fixed in advance)

**Protocol, for every experiment**

1. One change, one commit. A whole feedback version (v3, v7) replacing another counts as one change.
2. `--level X --rounds 8 --samples 4 --context 8192 --repeat 5`, the same vLLM configuration as the baseline
   (TP 2, max-model-len 8192, max-num-seqs 4), one agent process per model server. Seat 115 ran the level 5–7
   runs at max-num-seqs 8; those runs are labelled.
3. Any API name written into a feedback message is first checked with `inspect` on a seat.
4. Every 1.0 is re-audited in a fresh process (`scripts/reaudit.py`), then its held-out verdict is read. Logs made
   before the path-cache fix (c39c0ce) are re-graded in full, not only their 1.0s.
5. Report every run: the rate, never the best run, and from 14:40 on the number of distinct trajectories.

**Keep or roll back** (written at 13:50, before any of the results below)

- More solves than the reference on the target level, every solve re-audited: **keep**.
- The same solves, but the targeted failure mode at least halved, no new failure mode, mean score not lower:
  **keep as groundwork**, with the reason written down.
- Fewer solves, a lower mean, or any run below the reference's lowest score: **roll back**.
- Anything else: not kept; the code stays simple.
- A change to a shared path (prompt template, ledger, repair prompt, stopping rule) also needs a level-2
  regression check: at least 2 of 5 solved.
- The reference is the version currently kept for that level, at first the baseline.

**Added at 14:08, for bundles:** a bundle (v7) is judged as a whole: more solves over levels 1, 3 and 4 combined
and no level with fewer solves. No per-level mixing of versions.

**Added at 15:50, for the final candidate:** never worse than v7 (levels 3 and 4 at least 4/5) and level 2 not
below the baseline.

**A rule that nearly bit.** At 15:18 v7's level 2 looked like a regression (0.30 against the baseline's 3/5), and
the bundle rule would then have rejected v7 despite its new solves on levels 3 and 4. We wrote the conflict down
here before resolving it, and resolved it by finding the cause (v7's sampling settings, §4 at 15:45) rather than
by changing the rule.

**Mixing commits.** The final numbers combine runs from three commits. That is allowed only where the later
commit is proven to send byte-identical requests to the earlier one on that level, with a deterministic server
(96a9fc9 → 5c3aba2 → 15fb0d5; checked twice, independently: analysis/v83_l134_identity.md).

## 4. Decision log

| time | decision | evidence |
|---|---|---|
| 14:00 | teoguo: repair messages may carry code in the model's own variable names (liuyq's method); v7 replaces v3; the pull request goes to the team repo | |
| 14:00 | experiment A (invented API names mapped to real 0.6.0 calls) kept as groundwork | level 1 still 0/5, invented names 80 → 20 |
| 14:09 | simulation target: keep trn2; earlier runs comparable | baseline and experiment A identical under trn2 and trn3 (530ab9a); seats pick trn2 from the hardware |
| 14:35 | the model server is deterministic: report distinct trajectories; splitting runs across seats buys nothing | identical completions for identical requests at temperature 0.7 |
| 14:37 | **v7 adopted**; E-F not carried into v7 (reverted) | v7: level 3 5/5, level 4 5/5 (one trajectory); E-F: level 1 0/5, and v7's level-1 failures are different |
| 15:00 | v8 = v7 + E-div + three switches (skeleton, truncation fix, level-1 calls) | teoguo chose to gamble on the skeleton |
| 15:15 | liuyq's level-1 compiler-gate rule taken into the base | fires on none of our 11 distinct solving kernels (7bc96df) |
| 15:45 | the level-2 regression was v7's sampling settings | round 0: 0/20 with v7's, 2/20 with the original, 2/20 for the baseline re-run |
| 16:00 | **skeleton dropped** | level 4 at 0.62 twice under it, against 5/5 without |
| 16:09 | v8.2 = v8 without skeleton + mixed sampling + level-2 restatement + **E-mix** (planner's proposal: sample 1 repairs, samples 2–4 restart from the first prompt) | level 2 had only ever been solved in round 0 |
| 16:37 | v8.2 is the final version for levels 1, 3, 4 | level 1 5/5 (five kernels), level 3 5/5, level 4 5/5 |
| 16:44 | v8.3 = v8.2 + level-2 error distillation | executor 2's arange probe: 29 of 34 stuck level-2 attempts are one of three readable transforms |
| 17:01 | v8.3 for level 2 | 7 of 9 against v8.2's 5 of 9 on the same seats |
| 17:10 | push for levels 5–7 with WARM (start from the agent's own level-4 kernel) | another team solved them that way |
| 17:30 | the repository must contain no Chinese | teoguo |

## 5. Constraints kept all day

- Thinking off.
- No rule lists in the generation prompt: constraints live in the checker.
- Feedback names what to change, not the answer. The one exception, decided at 14:00 and disclosed in
  SUBMISSION.md §3: for known error classes the change is given as code in the model's own variable names.
- The NKI tutorial kernels, the organizers' reference kernels and liuyq's answers for levels 9–14 never reach a
  prompt (0 lines found by the leak scan, analysis/prompt_leak_check.md).
- Credentials are never written to a file (scanned before the pull request).
- Every number names whether it comes from the simulator, the trn2 compiler or a NeuronCore.
