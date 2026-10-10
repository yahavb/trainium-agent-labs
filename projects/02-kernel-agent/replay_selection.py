"""Read-only replay of recorded candidate choices; never generate or grade code.

Legacy logs lack run/repeat IDs. This accepts a known single-writer log with a
specified sample count, groups contiguous (level, round) records, and resets
history whenever a new round 0 starts. It cannot predict downstream repairs or
measure a changed policy's solve rate from baseline-generated candidates.
"""

import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path

from failure_selection import Candidate, select_candidate


def replay_records(records, samples=4, full_reward=1.0):
    if samples < 1:
        raise ValueError("samples must be positive")
    groups = itertools.groupby(records, key=lambda r: (r['level'], r['round']))
    history = []
    current_level = None
    last_round = None
    choices = []
    skipped = []
    categories = Counter()
    segment = 0
    for (level, rnd), values in groups:
        group = list(values)
        if level != current_level or rnd == 0:
            history = []
            current_level, last_round = level, None
            segment += 1
        if len(group) != samples or (last_round is not None and rnd != last_round + 1) or (
                last_round is None and rnd != 0):
            skipped.append(dict(level=level, round=rnd, records=len(group),
                                reason='incomplete/ambiguous batch or missing round history'))
            # Do not use incomplete history to give a seemingly exact replay.
            history = []
            last_round = None
            continue
        candidates = [Candidate(r['reward'], r['code'], r['feedback']) for r in group]
        legacy = select_candidate(candidates, 'reward')
        diagnostic = select_candidate(candidates, 'diagnostic', history)
        categories.update(d.failure_category for c, d in zip(candidates, diagnostic.diagnostics)
                          if c.reward < full_reward - 1e-9)
        choices.append(dict(segment=segment, level=level, round=rnd,
                            legacy_index=legacy.selected_index,
                            diagnostic_index=diagnostic.selected_index,
                            changed=legacy.selected_index != diagnostic.selected_index,
                            legacy_reward=candidates[legacy.selected_index].reward,
                            diagnostic_reward=candidates[diagnostic.selected_index].reward,
                            distinct_codes=len({e.code_fingerprint for e in diagnostic.evidence}),
                            candidates=[dict(reward=c.reward, **diagnostic.log_metadata(i))
                                        for i, c in enumerate(candidates)]))
        history.append(tuple(c for c in candidates if c.reward < full_reward - 1e-9))
        last_round = rnd
    return dict(rounds_replayed=len(choices), changed_choices=sum(r['changed'] for r in choices),
                failure_counts=dict(sorted(categories.items())), skipped_batches=skipped,
                choices=choices,
                limitation='selection-only replay on baseline-generated candidates; not an improved-agent experiment')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log', type=Path)
    parser.add_argument('--samples', type=int, default=4)
    args = parser.parse_args()
    # Read once, so appended rows do not change the replay while it runs.
    raw = args.log.read_bytes()
    records = []
    incomplete_tail = False
    lines = raw.splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if index == len(lines) - 1 and not raw.endswith(b'\n'):
                incomplete_tail = True
            else:
                raise ValueError(f'malformed JSONL record at line {index + 1}') from None
    report = replay_records(records, args.samples)
    report.update(snapshot_sha256=hashlib.sha256(raw).hexdigest(),
                  complete_records=len(records), ignored_incomplete_tail=incomplete_tail)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
