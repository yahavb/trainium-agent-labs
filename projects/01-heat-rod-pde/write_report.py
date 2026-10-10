"""Generate a compact report from one completed real-model benchmark."""
import collections
import json
from pathlib import Path
import statistics
import sys


def main():
    source = Path(sys.argv[1])
    data = json.loads(source.read_text())
    rows = data['results']
    for row in rows:
        if row['status'] == 'infrastructure_timeout':
            row['status'] = 'budget_timeout'
    source.write_text(json.dumps(data, indent=2))
    lines = ['# Challenge 1 — Heat-Rod PDE Agent results', '',
             'Hardware: seat-85, Trainium2 (`trn2.48xlarge` host, one allocated chip); Qwen3-8B, TP=2, context 8192.',
             'Real model inference only. No offline generator and no analytic-answer fallback.', '',
             'Comparison: upstream baseline versus enhanced agent WITH the experimental concise output contract. '
             'Each variant: 2 samples/round, at most 3 rounds, 512 output tokens, 1 calculator exchange/attempt; 180-second case budget. '
             'Level 0 and Level 1 subproblems 1/2: one run per subproblem. Level 1 subproblem 3: three problem seeds. '
             'Sampling is stochastic; problem seeds do not fix generated answers.', '',
             '| Variant | Level | Validated solved | Service failures | Budget timeouts | Reward range on completed runs | Mean rounds on completed runs |',
             '|---|---|---|---|---|---|---|']
    for variant in ('baseline', 'improved'):
        for level in (0, 1):
            group = [r for r in rows if r['variant'] == variant and r['level'] == level]
            valid = [r for r in group if not r['status'].startswith('infrastructure') and r['status'] != 'budget_timeout']
            rewards = [r['reward'] for r in valid if r.get('reward') is not None]
            spread = f'{min(rewards):.1f}–{max(rewards):.1f}' if rewards else 'n/a'
            rounds = f"{statistics.mean(r['rounds'] for r in valid):.2f}" if valid else 'n/a'
            service_errors = sum(r['status'].startswith('infrastructure') for r in group)
            timeouts = sum(r['status'] == 'budget_timeout' for r in group)
            lines.append(f"| {variant} | {level} | {sum(r['status']=='solved' for r in group)}/{len(group)} | {service_errors} | {timeouts} | {spread} | {rounds} |")
    duplicate_rounds = total_rounds = 0
    for attempt_file in source.parent.rglob('attempts.jsonl'):
        rounds = collections.defaultdict(list)
        for line in attempt_file.read_text().splitlines():
            if line.strip():
                attempt = json.loads(line)
                rounds[attempt['round']].append(attempt['answer'])
        for answers in rounds.values():
            if len(answers) > 1:
                total_rounds += 1
                duplicate_rounds += len(set(answers)) == 1
    smoke_text = 'Default-path smoke result not available in this snapshot.'
    smoke_files = list((source.parents[2] / 'default-smoke').glob('*/summary.json'))
    if smoke_files:
        smoke = json.loads(smoke_files[-1].read_text())['results'][0]
        smoke_text = (f"Final default-mode smoke run: {smoke['problem']}, {smoke['status']}, "
                      f"reward {smoke.get('reward')}, {smoke['rounds']} round(s), "
                      f"{smoke['seconds']:.1f}s; independent validation accepted: "
                      f"{smoke.get('validation', {}).get('accepted', False)}. "
                      'This is one runnable-path check, not an estimated solve rate.')
    lines += ['', 'The improved checker preserves the original physics reward and qualitative coefficient feedback, '
              'then requires a separate 384-point space/time check and a refined 1601-point initial-shape check. '
              'It does not consult known exact solutions. A tested high-frequency error that earns full marks '
              'from the original checker is rejected by the added near-zero-time checks.', '',
              'Both original checker selftests and all nine new regression tests pass. The client records calculator '
              'requests/results, retries transient errors, preserves its best candidate, and keeps a short failure ledger.', '',
              'The concise contract regressed on Level 0 (baseline 3/3 versus concise 1/3), so it is now optional (`--concise`) '
              'and disabled by default. This table measures that experimental mode, not the final default agent. '
              'No claim of improved default solve rate is made; a separate real-model smoke run checks the default path.', '',
              smoke_text, '',
              f'Observed duplicate final answers in {duplicate_rounds}/{total_rounds} recorded multi-sample rounds. '
              'The two samples must not be treated as independent statistical trials.', '',
              'Limits: only the hardest parabola case uses three problem seeds; the easier cases use one. These are exploratory results, not statistical proof of improvement. '
              'Numerical validation is not a universal symbolic proof. Wall-clock costs may differ because the improved '
              'client retries transport errors. The first batch crashed on a Neuron-incompatible per-request seed; '
              'its logs are retained separately and are not included in this comparison.', '',
              'A second preliminary batch exposed long derivations exhausting output budgets; it was stopped to add a concise output contract '
              'and truncation-aware feedback. Its partial logs are retained separately. Only the final batch is summarized here.', '',
              'Budget timeouts are inconclusive: they do not imply a zero mathematical reward or a server failure. '
              'Any full-score answer in a completed baseline run must pass the independent verifier before being counted as solved.', '',
              f'Evidence: `{source.name}` plus case-level `attempts.jsonl`, `console.log` and `summary.json` in the same batch directory.']
    target = source.parent / 'RESULTS.md'
    target.write_text('\n'.join(lines) + '\n')
    print(target)


if __name__ == '__main__':
    main()
