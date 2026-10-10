"""Recover complete attempt records without concealing corrupt or missing evidence."""
import json


def read_attempts(paths):
    rows, errors = [], []
    if not paths:
        errors.append(dict(path=None, line=None, error='Missing attempts.jsonl'))
    for path in paths:
        try:
            with path.open('rb') as stream:
                for number, raw in enumerate(stream, 1):
                    if not raw.strip():
                        continue
                    try:
                        row = json.loads(raw.decode('utf-8'))
                        if (not isinstance(row, dict)
                                or not isinstance(row.get('answer'), str)
                                or type(row.get('round')) is not int or row['round'] < 0
                                or ('executed_answer' in row and
                                    not isinstance(row['executed_answer'], str))
                                or not isinstance(row.get('trace', []), list)
                                or any(not isinstance(turn, dict) or
                                       (turn.get('usage') is not None and
                                        not isinstance(turn['usage'], dict))
                                       for turn in row.get('trace', []))):
                            raise ValueError('Invalid attempt record fields')
                        rows.append(row)
                    except (ValueError, UnicodeError) as exc:
                        errors.append(dict(path=str(path), line=number,
                                           error=f'{type(exc).__name__}: {exc}'))
        except OSError as exc:
            errors.append(dict(path=str(path), line=None,
                               error=f'{type(exc).__name__}: {exc}'))
    return rows, errors


def request_metrics(rows):
    attempted = successful = failed = known_tokens = unknown_usage = truncated = 0
    for row in rows:
        for turn in row.get('trace', []):
            kind = turn.get('type', 'model')  # Original agent traces have no type.
            if kind not in ('model', 'http_error', 'transport_error'):
                continue
            attempted += 1
            successful += kind == 'model'
            failed += kind != 'model'
            count = (turn.get('usage') or {}).get('completion_tokens')
            if type(count) is int and count >= 0:
                known_tokens += count
            else:
                unknown_usage += 1
            truncated += kind == 'model' and turn.get('finish_reason') == 'length'
    return dict(model_requests=attempted, attempted_requests=attempted,
                successful_responses=successful, failed_requests=failed,
                completion_tokens=None if unknown_usage else known_tokens,
                known_completion_tokens=known_tokens, unknown_usage_requests=unknown_usage,
                truncated_replies=truncated)
